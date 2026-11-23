#!/usr/bin/env python
"""Prepare and run a small CPU inference benchmark on a Raspberry Pi.

On the project machine:
    python scripts/raspberry_pi_inference_pilot.py prepare

Copy the generated model state/config and input archive, plus this script, to
the Pi. There:
    python3 raspberry_pi_inference_pilot.py benchmark
"""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_RUN_DIR = Path(
    "results/runs/sensors_only_client_holdout/quantity_skew/"
    "fedavg_normal_only/seed_42"
)
DEFAULT_DATA_CONFIG = "configs/data_sensors_client_holdout_quantity.yaml"
DEFAULT_SCENARIO = "quantity_skew"
DEFAULT_CLIENT = "client_1"


def _prepare(args: argparse.Namespace) -> None:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.experiments.common import ExperimentData
    from src.models.model import build_model

    run_dir = (project_root / args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    model_path = run_dir / "model.pt"
    model_config_path = run_dir / "model_config.json"
    if not model_path.is_file() or not model_config_path.is_file():
        raise FileNotFoundError(f"Expected model.pt and model_config.json under {run_dir}")

    data_config_path = (project_root / args.data_config).resolve()
    data = ExperimentData(data_config_path)
    test_frame = data.clients.rows(data.frame, f"{args.scenario}_test", args.client)
    if test_frame.empty:
        raise ValueError(f"No test records found for {args.scenario}/{args.client}")
    features = data.features(test_frame, data.global_scaler()).numpy()
    features = features[: args.records].astype(np.float32, copy=False)

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model = build_model(model_config, len(data.feature_columns), seed=42)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.inference_mode():
        reconstruction = model(torch.from_numpy(features))
        squared_errors = (reconstruction - torch.from_numpy(features)) ** 2
        top_k = int(np.ceil(features.shape[1] * 0.05))
        reference_scores = torch.topk(squared_errors, k=top_k, dim=1).values.mean(dim=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "obd_model_state.pt")
    (output_dir / "obd_model_config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "obd_inputs.npz",
        x=features,
        reference_scores=reference_scores.numpy(),
    )
    metadata = {
        "source_run": str(run_dir.relative_to(project_root)),
        "scenario": args.scenario,
        "client": args.client,
        "n_records": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "score": "mean of the largest ceil(5% of 105) squared feature errors",
        "preprocessing": "shared training scaler applied on the project machine",
        "model_file": "obd_model_state.pt",
        "model_config_file": "obd_model_config.json",
        "inputs_file": "obd_inputs.npz",
    }
    (output_dir / "pilot_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Prepared {features.shape[0]} records with {features.shape[1]} features.")
    print(f"Copy these files to the Pi: {output_dir / 'obd_model_state.pt'}")
    print(f"                         {output_dir / 'obd_model_config.json'}")
    print(f"                         {output_dir / 'obd_inputs.npz'}")
    print(f"                         {Path(__file__).resolve()}")
    print(f"Metadata: {output_dir / 'pilot_metadata.json'}")


def _build_model(model_config: dict, input_dim: int) -> torch.nn.Module:
    architecture = model_config["architecture"]
    encoder_dims = [input_dim, *architecture["encoder_dims"], architecture["latent_dim"]]
    decoder_dims = encoder_dims[::-1]
    activation = {"relu": torch.nn.ReLU}.get(architecture.get("activation", "relu"))
    if activation is None:
        raise ValueError(f"Unsupported model activation: {architecture.get('activation')}")

    class Autoencoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = self._stack(encoder_dims, activate_last=True)
            self.decoder = self._stack(decoder_dims, activate_last=False)

        @staticmethod
        def _stack(dims: list[int], activate_last: bool) -> torch.nn.Sequential:
            layers: list[torch.nn.Module] = []
            for position, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
                layers.append(torch.nn.Linear(in_dim, out_dim))
                if position < len(dims) - 2 or activate_last:
                    layers.append(activation())
            return torch.nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(inputs))

    return Autoencoder()


def _peak_rss_megabytes() -> float:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak_rss <= 0:
        raise RuntimeError(f"Invalid peak RSS reported by the OS: {peak_rss}")
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    return peak_rss / (1024 if __import__("sys").platform.startswith("linux") else 1024**2)


def _benchmark(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    model_path = Path(args.model).resolve()
    model_config_path = Path(args.model_config).resolve()
    inputs_path = Path(args.inputs).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model state does not exist: {model_path}")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {model_config_path}")
    if not inputs_path.is_file():
        raise FileNotFoundError(f"Input archive does not exist: {inputs_path}")

    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    with np.load(inputs_path) as archive:
        features = archive["x"].astype(np.float32, copy=False)
        reference_scores = archive["reference_scores"].astype(np.float64, copy=False)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2-D input matrix, got {features.shape}")
    if features.shape[1] != 105:
        raise ValueError(f"Expected 105 input features, got {features.shape[1]}")
    if features.shape[0] != reference_scores.shape[0]:
        raise ValueError("Input records and reference scores have different lengths")

    model = _build_model(model_config, features.shape[1])
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    inputs = torch.from_numpy(features)
    with torch.inference_mode():
        for index in range(min(args.warmup, len(inputs))):
            model(inputs[index : index + 1])

        latencies_ms = np.empty(len(inputs), dtype=np.float64)
        scores = np.empty(len(inputs), dtype=np.float64)
        total_start = time.perf_counter()
        for index in range(len(inputs)):
            row = inputs[index : index + 1]
            start = time.perf_counter()
            reconstruction = model(row)
            errors = (reconstruction - row) ** 2
            scores[index] = torch.topk(errors, k=6, dim=1).values.mean().item()
            latencies_ms[index] = (time.perf_counter() - start) * 1000
        total_seconds = time.perf_counter() - total_start

    max_score_delta = float(np.max(np.abs(scores - reference_scores)))
    result = {
        "records": int(len(inputs)),
        "features": int(features.shape[1]),
        "threads": args.threads,
        "warmup_records": min(args.warmup, len(inputs)),
        "latency_median_ms": float(np.median(latencies_ms)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "throughput_records_per_second": float(len(inputs) / total_seconds),
        "peak_process_rss_mb": float(_peak_rss_megabytes()),
        "max_score_abs_delta_vs_reference": max_score_delta,
    }
    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved benchmark results to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a model/input bundle and benchmark CPU inference on a Raspberry Pi."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Export TorchScript model and preprocessed test records on the project machine."
    )
    prepare.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    prepare.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    prepare.add_argument("--scenario", default=DEFAULT_SCENARIO)
    prepare.add_argument("--client", default=DEFAULT_CLIENT)
    prepare.add_argument("--records", type=int, default=128)
    prepare.add_argument("--output-dir", default="results/raspberry_pi_pilot")
    prepare.set_defaults(func=_prepare)

    benchmark = subparsers.add_parser(
        "benchmark", help="Measure the saved model on the Raspberry Pi."
    )
    benchmark.add_argument("--model", default="obd_model_state.pt")
    benchmark.add_argument("--model-config", default="obd_model_config.json")
    benchmark.add_argument("--inputs", default="obd_inputs.npz")
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--threads", type=int, default=1)
    benchmark.add_argument("--output", default="raspberry_pi_benchmark.json")
    benchmark.set_defaults(func=_benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "records", 1) < 1:
        raise ValueError("--records must be positive")
    if getattr(args, "warmup", 0) < 0:
        raise ValueError("--warmup cannot be negative")
    if getattr(args, "threads", 1) < 1:
        raise ValueError("--threads must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
