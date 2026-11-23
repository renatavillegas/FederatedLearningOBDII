#!/usr/bin/env python
"""Export one client's training shard and an unlabeled inference probe for a Pi."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path
from src.experiments.common import (
    ExperimentData,
    fit_normal_real_preprocessing,
    select_normal_real_records,
)


def prepare_client_bundle(
    scenario: str,
    client: str,
    seed: int,
    records: int,
    output_dir: str | Path,
    data_config_path: str | Path,
    model_config_path: str | Path,
) -> Path:
    if records < 1:
        raise ValueError("records must be positive")
    data = ExperimentData(data_config_path)
    if client not in data.client_names(scenario):
        raise ValueError(f"Unknown client {client!r} in scenario {scenario!r}")

    feature_columns, scaler, feature_metadata = fit_normal_real_preprocessing(data)
    client_train = select_normal_real_records(
        data.client_rows(scenario, client),
        data.schema,
    )
    train_features = scaler.transform(
        client_train, dtype=data.config["preprocessing"]["dtype"]
    )
    test_frame = data.clients.rows(data.frame, f"{scenario}_test", client)
    if test_frame.empty:
        raise ValueError(f"No held-out inference records for {scenario}/{client}")
    inference_frame = test_frame.iloc[:records]
    inference_features = scaler.transform(
        inference_frame, dtype=data.config["preprocessing"]["dtype"]
    )

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config_directory = destination / "configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        resolve_path(model_config_path),
        config_directory / "model.yaml",
    )
    np.savez_compressed(
        destination / "edge_client_data.npz",
        train_x=np.ascontiguousarray(train_features, dtype=np.float32),
        inference_x=np.ascontiguousarray(inference_features, dtype=np.float32),
    )
    metadata = {
        "scenario": scenario,
        "client": client,
        "seed": seed,
        "training_records": int(len(train_features)),
        "inference_records": int(len(inference_features)),
        "n_features": int(len(feature_columns)),
        "normal_only_training": True,
        "inference_labels_included": False,
        "scaler": "global normal-real training scaler",
        "source_data_config": str(data_config_path),
        "source_model_config": str(model_config_path),
    }
    (destination / "bundle_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (destination / "feature_metadata.json").write_text(
        json.dumps(feature_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the selected client's local train shard for Raspberry Pi."
    )
    parser.add_argument("--scenario", choices=("iid", "quantity_skew"), default="iid")
    parser.add_argument("--client", default="client_4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--data-config")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument(
        "--output-dir",
        help="Defaults to results/raspberry_flower_client/<scenario>/<client>/",
    )
    args = parser.parse_args()
    data_config = args.data_config or (
        f"configs/data_sensors_client_holdout_{args.scenario}.yaml"
    )
    output_dir = args.output_dir or (
        f"results/raspberry_flower_client/{args.scenario}/{args.client}"
    )
    destination = prepare_client_bundle(
        scenario=args.scenario,
        client=args.client,
        seed=args.seed,
        records=args.records,
        output_dir=output_dir,
        data_config_path=data_config,
        model_config_path=args.model_config,
    )
    metadata = json.loads((destination / "bundle_metadata.json").read_text(encoding="utf-8"))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Bundle directory: {destination}")


if __name__ == "__main__":
    main()
