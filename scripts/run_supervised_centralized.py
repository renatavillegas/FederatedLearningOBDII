#!/usr/bin/env python
"""Supervised centralized baseline: trains a binary MLP on all labelled training data.

The classifier uses the same architecture as the autoencoder encoder with a binary
classification head: Input(105) → Dense(32, ReLU) → Dense(8, ReLU) → Dense(1).

The scaler is fitted on the global normal-real training partition (same as the autoencoder
baseline) so that feature scales are identical across both models for a fair comparison.
All partitions (train, validation, test) include both normal and anomaly records.

Usage:
    python scripts/run_supervised_centralized.py [--scenario iid] [--seed 42]
    python scripts/run_supervised_centralized.py --scenario iid --pretrained-ae results/runs/.../model.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_yaml, set_project_root
from src.experiments.common import (
    EarlyStopping,
    ExperimentData,
    fit_normal_real_preprocessing,
)
from src.models.classifier import build_classifier
from src.models.supervised_trainer import SupervisedTrainer
from src.models.trainer import TrainingConfig, set_seed

ROOT = Path(__file__).resolve().parents[1]


def _features_and_labels(
    data: ExperimentData,
    partition_name: str,
    scaler,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a full partition (all classes) and return scaled features + binary labels."""
    frame = data.partition(partition_name)
    features = data.features(frame, scaler)
    labels = torch.tensor(data.labels(frame), dtype=torch.float32)
    return features, labels


def run(
    scenario: str,
    seed: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    batch_size: int,
    data_config: str,
    model_config_path: str,
    output_dir: Path,
    pretrained_ae: Path | None,
) -> dict:
    set_seed(seed)
    set_project_root(str(ROOT))

    model_config = load_yaml(model_config_path)
    data = ExperimentData(data_config)

    # Scaler is fitted on normal-real training records only — same as the autoencoder baseline —
    # so that both models operate in the same feature space and are directly comparable.
    _, scaler, feature_metadata = fit_normal_real_preprocessing(data)

    train_x, train_y = _features_and_labels(data, "train", scaler)
    val_x, val_y = _features_and_labels(data, "validation", scaler)
    test_x, test_y = _features_and_labels(data, "test", scaler)

    n_features = train_x.shape[1]
    n_anomaly_train = int(train_y.sum().item())
    n_normal_train = int(len(train_y) - n_anomaly_train)
    print(f"Train: {len(train_x)} records ({n_normal_train} normal, {n_anomaly_train} anomaly)")
    print(f"Val:   {len(val_x)} records | Test: {len(test_x)} records")

    model = build_classifier(model_config, n_features, seed)
    print(f"Classifier: {model.config}")

    if pretrained_ae is not None:
        ae_state = torch.load(pretrained_ae, weights_only=True, map_location="cpu")
        copied = model.load_encoder_weights_from_autoencoder(ae_state)
        print(f"[transfer] Copied {copied} encoder weight tensors from autoencoder.")

    train_cfg = TrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer="adam",
        loss="mse",
        device="cpu",
    )
    trainer = SupervisedTrainer(model, train_cfg, seed)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0)

    history: list[dict] = []
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        result = trainer.train(train_x, train_y, epochs=1)
        val_metrics = trainer.evaluate(val_x, val_y)
        improved = early_stopping.update(val_metrics["bce_loss"], epoch)
        if improved:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "train_bce": result.epoch_reconstruction_loss[-1],
                "val_bce": val_metrics["bce_loss"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
        )
        if early_stopping.should_stop:
            break

    total_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    test_metrics = trainer.evaluate(test_x, test_y)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario": scenario,
        "method": "supervised_mlp_centralized",
        "architecture": "105-32-8-1",
        "seed": seed,
        "scaler": "normal_real_train_global",
        "best_epoch": early_stopping.best_step,
        "best_val_bce": early_stopping.best_value,
        "epochs_run": len(history),
        "stopped_early": early_stopping.should_stop,
        "total_seconds": total_seconds,
        "n_train": len(train_x),
        "n_anomaly_train": n_anomaly_train,
        "n_features": n_features,
        "pretrained_ae": str(pretrained_ae) if pretrained_ae else None,
        "test": test_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "history.json").write_text(
        json.dumps({"epochs": history}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "features.json").write_text(
        json.dumps(feature_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save(best_state, output_dir / "model.pt")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("iid", "quantity_skew"), default="iid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--data-config",
        help="Defaults to configs/data_sensors_client_holdout_<scenario>.yaml",
    )
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument(
        "--pretrained-ae",
        help="Path to a pretrained autoencoder model.pt for encoder weight transfer.",
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to results/runs/supervised_centralized/<scenario>/seed_<seed>",
    )
    args = parser.parse_args()

    variant = "transfer_ae" if args.pretrained_ae else "scratch"
    data_config = args.data_config or f"configs/data_sensors_client_holdout_{args.scenario}.yaml"
    output_dir = Path(
        args.output_dir
        or f"results/runs/supervised_centralized/{args.scenario}/{variant}/seed_{args.seed}"
    )
    pretrained_ae = Path(args.pretrained_ae) if args.pretrained_ae else None

    run(
        scenario=args.scenario,
        seed=args.seed,
        max_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        data_config=data_config,
        model_config_path=args.model_config,
        output_dir=output_dir,
        pretrained_ae=pretrained_ae,
    )


if __name__ == "__main__":
    main()
