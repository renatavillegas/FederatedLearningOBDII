from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/metrics/sensors_only_quantity_skew"
OUTPUT = ROOT / "results/figures/sensors_only_quantity_skew"
SEEDS = (7, 21, 42)
RECALL_GRID = np.linspace(0.0, 1.0, 501)
FPR_GRID = np.linspace(0.0, 1.0, 501)

RUNS = (
    ("Centralized", "all/centralized_normal_only_sensors105_quantity_latent8", "#222222"),
    ("Local - IID", "iid/local_normal_only", "#4C72B0"),
    ("FedAvg - IID", "iid/fedavg_normal_only", "#55A868"),
    ("Local - quantity-skew", "quantity_skew/local_normal_only", "#C44E52"),
    ("FedAvg - quantity-skew", "quantity_skew/fedavg_normal_only", "#8172B2"),
)


def interpolate_precision(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = float(average_precision_score(labels, scores))
    ascending_order = np.argsort(recall, kind="stable")
    recall_ascending = recall[ascending_order]
    precision_ascending = precision[ascending_order]
    unique_recall, inverse = np.unique(recall_ascending, return_inverse=True)
    max_precision = np.zeros_like(unique_recall)
    np.maximum.at(max_precision, inverse, precision_ascending)
    interpolated = np.interp(RECALL_GRID, unique_recall, max_precision)
    return interpolated, average_precision


def interpolate_tpr(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(roc_auc_score(labels, scores))
    interpolated = np.interp(FPR_GRID, fpr, tpr)
    interpolated[0] = 0.0
    interpolated[-1] = 1.0
    return interpolated, roc_auc


def load_seed_curves(run_path: str, seed: int) -> tuple[np.ndarray, float, np.ndarray, float]:
    scores_path = METRICS / run_path / f"seed_{seed}" / "scores.csv"
    if not scores_path.is_file():
        raise FileNotFoundError(f"Missing score file: {scores_path}")

    scores = pd.read_csv(scores_path)
    test = scores.loc[scores["partition"] == "test"]
    if test.empty:
        raise ValueError(f"No test scores found in {scores_path}")

    view_pr_curves = []
    view_average_precisions = []
    view_roc_curves = []
    view_roc_aucs = []
    for _, view in test.groupby("view", sort=True):
        labels = view["label"].to_numpy(dtype=np.int64)
        values = view["score"].to_numpy(dtype=float)
        if np.unique(labels).size != 2:
            raise ValueError(f"Both classes must be present in {scores_path}")
        pr_curve, average_precision = interpolate_precision(labels, values)
        roc_curve_values, roc_auc = interpolate_tpr(labels, values)
        view_pr_curves.append(pr_curve)
        view_average_precisions.append(average_precision)
        view_roc_curves.append(roc_curve_values)
        view_roc_aucs.append(roc_auc)

    return (
        np.mean(view_pr_curves, axis=0),
        float(np.mean(view_average_precisions)),
        np.mean(view_roc_curves, axis=0),
        float(np.mean(view_roc_aucs)),
    )


def plot_precision_recall() -> Path:
    series = []
    for name, run_path, color in RUNS:
        seed_curves = []
        seed_average_precisions = []
        seed_roc_curves = []
        seed_roc_aucs = []
        for seed in SEEDS:
            pr_curve, average_precision, roc_curve_values, roc_auc = load_seed_curves(run_path, seed)
            seed_curves.append(pr_curve)
            seed_average_precisions.append(average_precision)
            seed_roc_curves.append(roc_curve_values)
            seed_roc_aucs.append(roc_auc)
        series.append(
            (
                name,
                color,
                np.asarray(seed_curves),
                np.asarray(seed_average_precisions),
                np.asarray(seed_roc_curves),
                np.asarray(seed_roc_aucs),
            )
        )

    centralized_seed_scores = pd.read_csv(
        METRICS
        / RUNS[0][1]
        / f"seed_{SEEDS[0]}"
        / "scores.csv"
    )
    central_test = centralized_seed_scores.loc[
        centralized_seed_scores["partition"] == "test"
    ]
    prevalence = float(central_test["label"].mean())

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "roc_precision_recall_test_sensor_only_seeds_7_21_42.png"
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(16, 7))

    for name, color, pr_curves, average_precisions, roc_curves, roc_aucs in series:
        mean_pr_curve = pr_curves.mean(axis=0)
        pr_seed_std = pr_curves.std(axis=0, ddof=1)
        mean_ap = float(average_precisions.mean())
        std_ap = float(average_precisions.std(ddof=1))
        pr_ax.plot(
            RECALL_GRID,
            mean_pr_curve,
            color=color,
            linewidth=2.2,
            label=f"{name}: PR-AUC {mean_ap:.3f} ± {std_ap:.3f}",
        )
        pr_ax.fill_between(
            RECALL_GRID,
            np.clip(mean_pr_curve - pr_seed_std, 0.0, 1.0),
            np.clip(mean_pr_curve + pr_seed_std, 0.0, 1.0),
            color=color,
            alpha=0.12,
            linewidth=0,
        )

        mean_roc_curve = roc_curves.mean(axis=0)
        roc_seed_std = roc_curves.std(axis=0, ddof=1)
        mean_roc_auc = float(roc_aucs.mean())
        std_roc_auc = float(roc_aucs.std(ddof=1))
        roc_ax.plot(
            FPR_GRID,
            mean_roc_curve,
            color=color,
            linewidth=2.2,
            label=f"{name}: ROC-AUC {mean_roc_auc:.3f} ± {std_roc_auc:.3f}",
        )
        roc_ax.fill_between(
            FPR_GRID,
            np.clip(mean_roc_curve - roc_seed_std, 0.0, 1.0),
            np.clip(mean_roc_curve + roc_seed_std, 0.0, 1.0),
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    roc_ax.plot([0, 1], [0, 1], color="#777777", linestyle=":", linewidth=1.5, label="No-skill")
    roc_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate (recall)",
        title="ROC curve",
    )
    roc_ax.grid(alpha=0.25)
    roc_ax.legend(loc="lower right", fontsize=8)

    pr_ax.axhline(
        prevalence,
        color="#777777",
        linestyle=":",
        linewidth=1.5,
        label=f"No-skill prevalence ({prevalence:.3f})",
    )
    pr_ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall curve",
    )
    pr_ax.grid(alpha=0.25)
    pr_ax.legend(loc="lower left", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Curves average seeds 7, 21, 42; local methods first macro-average client views. "
        "Shading is ±1 SD across seeds (not a confidence interval).",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(f"Saved ROC and precision-recall curves: {plot_precision_recall().relative_to(ROOT)}")
