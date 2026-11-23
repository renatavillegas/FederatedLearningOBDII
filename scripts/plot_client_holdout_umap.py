#!/usr/bin/env python
"""Visualize training-data neighborhoods and client assignments with UMAP."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.common import ExperimentData

SCENARIO_CONFIGS = {
    "iid": "configs/data_sensors_client_holdout_iid.yaml",
    "quantity_skew": "configs/data_sensors_client_holdout_quantity.yaml",
}
SOURCE_CLASS_COLORS = {
    "Real normal": "#2878b5",
    "Synthetic normal (gray)": "#70ad47",
    "Synthetic normal (cat)": "#8c564b",
    "Synthetic anomaly (cat)": "#e87524",
    "Synthetic anomaly (gray)": "#c43c39",
    "Synthetic real anomaly": "#9467bd",
}
CLIENT_COLORS = (
    "#2878b5",
    "#e87524",
    "#70ad47",
    "#c43c39",
    "#9467bd",
    "#8c564b",
)


def _load_training_data(
    scenario: str,
    config_path: str,
) -> tuple[ExperimentData, pd.DataFrame, pd.Series]:
    data = ExperimentData(config_path)
    train_ids = data.partition_index.loc[
        data.partition_index["partition"].eq("train"), "row_id"
    ].to_numpy()
    frame = data.frame.loc[train_ids].copy()
    frame["row_id"] = frame.index

    assignment = data.clients.load(scenario)
    if assignment["row_id"].duplicated().any():
        raise ValueError(f"{scenario} training assignment contains duplicate row IDs")
    if set(assignment["row_id"]) != set(train_ids):
        raise ValueError(f"{scenario} training assignment does not cover the training partition")

    client_by_row = assignment.set_index("row_id")["client"]
    frame["client"] = frame["row_id"].map(client_by_row)
    if frame["client"].isna().any():
        raise ValueError(f"{scenario} training records are missing client assignments")
    return data, frame, data.schema.assign_categories(frame[data.schema.source_column])


def _source_class(category: str, label: int) -> str:
    if category == "SPLIT_REAL":
        return "Real normal" if label == 0 else "Synthetic real anomaly"
    if category == "SPLIT_GRAY":
        return "Synthetic normal (gray)" if label == 0 else "Synthetic anomaly (gray)"
    if category == "SPLIT_CAT":
        return "Synthetic normal (cat)" if label == 0 else "Synthetic anomaly (cat)"
    raise ValueError(f"Unexpected data category in training partition: {category}")


def _plot_embedding(
    embedding: np.ndarray,
    frame: pd.DataFrame,
    label_column: str,
    categories: pd.Series,
    client_assignments: pd.Series,
    normal_real_mask: np.ndarray,
    scenario: str,
    output_path: Path,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> None:
    labels = frame[label_column].astype(int).to_numpy()
    source_class = np.asarray(
        [_source_class(str(category), int(label)) for category, label in zip(categories, labels)]
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
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    client_names = sorted(
        pd.unique(client_assignments.astype(str)),
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    client_values = client_assignments.astype(str).to_numpy()
    for position, client in enumerate(client_names):
        mask = normal_real_mask & (client_values == client)
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
    axes[1].legend(frameon=False, fontsize=8, loc="best")

    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.15, linewidth=0.5)
    fig.suptitle(
        f"Vizinhanças locais das features OBD-II — {scenario} "
        f"(UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}, seed={seed})",
        fontsize=14,
    )
    fig.text(
        0.5,
        -0.015,
        "A projeção foi ajustada somente em normais reais de treino; "
        "as demais amostras de treino foram projetadas sobre essa referência. "
        "UMAP é exploratório; eixos e distâncias globais não têm interpretação física. "
        "Cada cenário é projetado separadamente.",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot UMAP of training samples and client assignments."
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/sensors_only_client_holdout/umap",
    )
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be at least 2")
    if args.min_dist < 0:
        raise ValueError("--min-dist must be non-negative")

    output_dir = Path(args.output_dir).resolve()
    feature_columns: list[str] | None = None
    for scenario, config_path in SCENARIO_CONFIGS.items():
        data, frame, categories = _load_training_data(scenario, config_path)
        if feature_columns is not None and data.feature_columns != feature_columns:
            raise ValueError("IID and quantity-skew feature lists differ")
        feature_columns = data.feature_columns

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

        features = data.features(frame, data.global_scaler()).numpy()
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric="euclidean",
            random_state=args.seed,
            n_jobs=1,
            low_memory=True,
        )
        reducer.fit(features[normal_real_mask])
        embedding = reducer.transform(features)

        output_path = output_dir / f"train_data_umap_{scenario}.png"
        _plot_embedding(
            embedding,
            frame,
            data.schema.label_column,
            categories,
            frame["client"],
            normal_real_mask,
            scenario,
            output_path,
            args.n_neighbors,
            args.min_dist,
            args.seed,
        )
        print(f"\n{scenario}: normal-real training records by client")
        print(
            frame.loc[normal_real_mask, "client"]
            .value_counts()
            .sort_index()
            .to_string()
        )
        print(f"Saved UMAP figure to {output_path}")


if __name__ == "__main__":
    main()
