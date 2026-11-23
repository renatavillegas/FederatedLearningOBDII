#!/usr/bin/env python
"""Visualize FedAvg autoencoder latent codes with UMAP."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path
from src.data.preprocess import Standardizer
from src.experiments.common import ExperimentData
from src.models.model import build_model
from scripts.plot_client_holdout_umap import (
    CLIENT_COLORS,
    SOURCE_CLASS_COLORS,
    _plot_embedding,
    _source_class,
)

SCENARIO_CONFIGS = {
    "iid": "configs/data_sensors_client_holdout_iid.yaml",
    "quantity_skew": "configs/data_sensors_client_holdout_quantity.yaml",
}


def _load_selected_model(
    run_directory: Path,
) -> tuple[torch.nn.Module, Standardizer, list[str]]:
    metadata_paths = {
        "features": run_directory / "features.json",
        "model_config": run_directory / "model_config.json",
        "scaler": run_directory / "scaler.json",
        "weights": run_directory / "model.pt",
    }
    missing = [str(path) for path in metadata_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FedAvg model artifacts are missing: {missing}")

    with metadata_paths["features"].open(encoding="utf-8") as handle:
        feature_metadata = json.load(handle)
    with metadata_paths["model_config"].open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    with metadata_paths["scaler"].open(encoding="utf-8") as handle:
        scaler = Standardizer.from_dict(json.load(handle))

    feature_columns = list(feature_metadata["feature_columns"])
    if feature_columns != scaler.feature_columns:
        raise ValueError("FedAvg feature metadata and scaler feature order differ")
    model = build_model(model_config, len(feature_columns), seed=0)
    state = torch.load(metadata_paths["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, scaler, feature_columns


def _plot_latent_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    categories: pd.Series,
    clients: np.ndarray,
    normal_real_mask: np.ndarray,
    scenario: str,
    method: str,
    seed: int,
    latent_dim: int,
    output_path: Path,
) -> None:
    source_class = np.asarray(
        [
            _source_class(str(category), int(label))
            for category, label in zip(categories, labels)
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    for label, color in SOURCE_CLASS_COLORS.items():
        mask = source_class == label
        if mask.any():
            axes[0].scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=7,
                alpha=0.45,
                color=color,
                label=f"{label} (n={int(mask.sum()):,})",
                linewidths=0,
                rasterized=True,
            )
    axes[0].set_title("Origem e classe (partição de treino)")

    client_names = sorted(
        pd.unique(clients.astype(str)),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    for position, client in enumerate(client_names):
        mask = normal_real_mask & (clients == client)
        axes[1].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.55,
            color=CLIENT_COLORS[position % len(CLIENT_COLORS)],
            label=f"{client} (n={int(mask.sum()):,})",
            linewidths=0,
            rasterized=True,
        )
    axes[1].set_title("Normais reais usados na otimização,\npor cliente")

    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.15, linewidth=0.5)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"Representação latente do autoencoder — {scenario} "
        f"(FedAvg, {method}, seed={seed}, dimensão={latent_dim})",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.015,
        "UMAP ajustado nos códigos latentes dos normais reais de treino do modelo "
        "selecionado; demais registros de treino projetados na mesma referência. "
        "Visualização exploratória, não métrica de detecção. "
        "Cada cenário é projetado separadamente.",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_latent_umap(
    scenario: str,
    seed: int = 42,
    method: str = "fedavg_normal_only",
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    umap_seed: int = 42,
    output_dir: str | Path = (
        "results/figures/sensors_only_client_holdout/umap/latent_fedavg_seed42"
    ),
) -> Path:
    if scenario not in SCENARIO_CONFIGS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if min_dist < 0:
        raise ValueError("min_dist must be non-negative")

    config_path = SCENARIO_CONFIGS[scenario]
    data = ExperimentData(config_path)
    run_directory = resolve_path(
        f"results/runs/sensors_only_client_holdout/{scenario}/{method}/seed_{seed}"
    )
    model, scaler, feature_columns = _load_selected_model(run_directory)
    if feature_columns != data.feature_columns:
        raise ValueError("FedAvg feature list differs from the scenario's feature list")

    train_ids = data.partition_index.loc[
        data.partition_index["partition"].eq("train"), "row_id"
    ].to_numpy()
    frame = data.frame.loc[train_ids].copy()
    frame["row_id"] = frame.index
    assignment = data.clients.load(scenario)
    if assignment["row_id"].duplicated().any() or set(assignment["row_id"]) != set(train_ids):
        raise ValueError(f"{scenario} training assignment does not cover training rows once")
    clients = frame["row_id"].map(assignment.set_index("row_id")["client"]).to_numpy()
    if pd.isna(clients).any():
        raise ValueError("Training rows are missing client assignments")

    categories = data.schema.assign_categories(frame[data.schema.source_column])
    real_categories = {
        category.name
        for category in data.schema.categories
        if category.family == "live_real"
    }
    labels = frame[data.schema.label_column].astype(int).to_numpy()
    normal_real_mask = categories.isin(real_categories).to_numpy() & (
        labels == data.schema.normal_label
    )
    if not normal_real_mask.any():
        raise ValueError(f"No normal-real training records available for {scenario}")

    values = scaler.transform(frame, dtype=data.config["preprocessing"]["dtype"])
    with torch.inference_mode():
        latent = model.encode(torch.from_numpy(np.ascontiguousarray(values))).cpu().numpy()

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=umap_seed,
        n_jobs=1,
        low_memory=True,
    )
    reducer.fit(latent[normal_real_mask])
    embedding = reducer.transform(latent)

    output_path = Path(output_dir).resolve() / f"train_latent_umap_{scenario}.png"
    latent_dim = int(model.config.latent_dim)
    _plot_latent_embedding(
        embedding,
        labels,
        categories,
        clients.astype(str),
        normal_real_mask,
        scenario,
        method,
        seed,
        latent_dim,
        output_path,
    )
    print(f"Saved latent UMAP to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the latent codes of selected normal-only FedAvg models."
    )
    parser.add_argument("--scenario", choices=tuple(SCENARIO_CONFIGS), nargs="+", default=list(SCENARIO_CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="fedavg_normal_only")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="results/figures/sensors_only_client_holdout/umap/latent_fedavg_seed42",
    )
    args = parser.parse_args()
    for scenario in args.scenario:
        plot_latent_umap(
            scenario=scenario,
            seed=args.seed,
            method=args.method,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            umap_seed=args.umap_seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
