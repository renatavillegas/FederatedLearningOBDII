from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/metrics/sensors_only_quantity_skew"
SPLIT_INDEX = ROOT / "data/splits/partition_index.csv"
OUTPUT = ROOT / "results/figures/sensors_only_quantity_skew"
SEED = 42
TARGET_VALIDATION_FPR = 0.05

RUNS = (
    ("Centralized", "all/centralized_normal_only_sensors105_quantity_latent8"),
    ("FedAvg - IID", "iid/fedavg_normal_only"),
    ("FedAvg - quantity-skew", "quantity_skew/fedavg_normal_only"),
)

GROUPS = (
    ("SPLIT_REAL", 0, "Real - normal", "#2878B5"),
    ("SPLIT_GRAY", 0, "Synthetic gray - normal", "#65A765"),
    ("SPLIT_CAT", 1, "Synthetic CAT - anomaly", "#E1812C"),
    ("SPLIT_GRAY", 1, "Synthetic gray - anomaly", "#C44E52"),
)


def threshold_for_fpr(normal_scores: np.ndarray, target_fpr: float) -> float:
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    sorted_scores = np.sort(np.asarray(normal_scores, dtype=float))
    if len(sorted_scores) == 0 or not np.isfinite(sorted_scores).all():
        raise ValueError("normal_scores must contain finite values")

    max_false_alarms = int(np.floor(target_fpr * len(sorted_scores)))
    boundary_index = len(sorted_scores) - max_false_alarms - 1
    boundary = sorted_scores[max(0, boundary_index)]
    return float(np.nextafter(boundary, np.inf))


def load_run(run_path: str) -> pd.DataFrame:
    run_dir = METRICS / run_path / f"seed_{SEED}"
    scores = pd.read_csv(run_dir / "scores.csv")
    if scores.empty:
        raise ValueError(f"No scores found in {run_dir / 'scores.csv'}")
    return scores


def plot_partition(
    score_frames: list[tuple[str, pd.DataFrame, float]], partition: str, x_max: float
) -> tuple[Path, Path]:
    selected = [
        (name, frame, threshold)
        for name, frame, threshold in score_frames
        if frame["partition"].eq(partition).any()
    ]
    if not selected:
        raise ValueError(f"No scores found for partition {partition}")
    x_grid = np.linspace(0.0, x_max, 500)

    density_path = OUTPUT / f"score_density_{partition}_seed_{SEED}.png"
    ecdf_path = OUTPUT / f"score_ecdf_{partition}_seed_{SEED}.png"

    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, all_scores, threshold) in zip(axes, selected):
        frame = all_scores.loc[all_scores["partition"] == partition]
        for category, label, group_name, color in GROUPS:
            values = frame.loc[
                (frame["category"] == category) & (frame["label"] == label), "score"
            ].to_numpy(dtype=float)
            if len(values) < 2:
                continue
            density = gaussian_kde(values)
            ax.plot(x_grid, density(x_grid), label=f"{group_name} (n={len(values)})", color=color)
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"5% FPR threshold = {threshold:.2f}",
        )
        ax.set_title(name)
        ax.set_xlabel("Anomaly score")
        ax.set_xlim(0.0, x_max)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated density")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{partition.title()} anomaly-score distributions by origin and label (seed {SEED})\n"
        "Scores above the pooled 99.5th percentile are outside the displayed range"
    )
    fig.tight_layout()
    fig.savefig(density_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(selected), figsize=(6 * len(selected), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, all_scores, threshold) in zip(axes, selected):
        frame = all_scores.loc[all_scores["partition"] == partition]
        for category, label, group_name, color in GROUPS:
            values = np.sort(
                frame.loc[
                    (frame["category"] == category) & (frame["label"] == label), "score"
                ].to_numpy(dtype=float)
            )
            if len(values) == 0:
                continue
            probability = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, probability, where="post", label=group_name, color=color)
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"5% FPR threshold = {threshold:.2f}",
        )
        ax.set_title(name)
        ax.set_xlabel("Anomaly score")
        ax.set_xlim(0.0, x_max)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Empirical CDF of {partition} anomaly scores by origin and label (seed {SEED})\n"
        "Thresholds target at most 5% false alarms on normal-real validation scores"
    )
    fig.tight_layout()
    fig.savefig(ecdf_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return density_path, ecdf_path


def make_plots() -> list[Path]:
    index = pd.read_csv(SPLIT_INDEX, usecols=["row_id", "category", "partition"])
    loaded = [(name, load_run(path)) for name, path in RUNS]
    score_frames = []
    for name, scores in loaded:
        merged = scores.merge(index, on="row_id", validate="one_to_one")
        if not (merged["partition_x"] == merged["partition_y"]).all():
            raise ValueError(f"Partition mismatch found for {name}")
        merged = merged.drop(columns="partition_y").rename(columns={"partition_x": "partition"})
        validation_normals = merged.loc[
            (merged["partition"] == "validation")
            & (merged["category"] == "SPLIT_REAL")
            & (merged["label"] == 0),
            "score",
        ].to_numpy(dtype=float)
        if len(validation_normals) == 0:
            raise ValueError(f"No normal-real validation scores found for {name}")
        threshold = threshold_for_fpr(validation_normals, TARGET_VALIDATION_FPR)
        score_frames.append((name, merged, threshold))

    plotted_paths = []

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for partition in ("validation", "test"):
        partition_scores = np.concatenate(
            [
                frame.loc[frame["partition"] == partition, "score"].to_numpy(dtype=float)
                for _, frame, _ in score_frames
            ]
        )
        x_max = float(np.quantile(partition_scores, 0.995))
        plotted_paths.extend(plot_partition(score_frames, partition, x_max))
    return plotted_paths


if __name__ == "__main__":
    for plot_path in make_plots():
        print(f"Saved plot: {plot_path.relative_to(ROOT)}")
