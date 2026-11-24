#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.config import load_yaml, resolve_path

BASE_METHODS = ("centralized", "fedavg", "local")
METHODS = (*BASE_METHODS, "fedprox")
METHOD_LABELS = {
    "centralized": "Centralized global",
    "fedavg": "FedAvg global",
    "local": "Local model",
    "fedprox": "FedProx global",
}
COLORS = {
    "centralized": "#2a78d6",
    "fedavg": "#1baf7a",
    "local": "#eb6834",
    "fedprox": "#8f63b8",
}


def _method_family(method: str) -> str:
    if method.startswith("centralized_"):
        return "centralized"
    if method.startswith("fedavg"):
        return "fedavg"
    if method.startswith("fedprox"):
        return "fedprox"
    if method.startswith("local"):
        return "local"
    raise ValueError(f"Unsupported method in client-test metrics: {method}")


def _load_matched_method_metrics(
    metrics_path: Path,
    include_method: str,
) -> pd.DataFrame:
    method_dir = metrics_path.parent / include_method
    method_files = sorted(method_dir.glob("seed_*/client_test_views.csv"))
    if not method_files:
        raise FileNotFoundError(
            f"No per-seed client-test metrics for {include_method} under {method_dir}"
        )
    selected_metrics = pd.concat((pd.read_csv(path) for path in method_files), ignore_index=True)
    if set(selected_metrics["method"]) != {include_method}:
        raise ValueError(f"Found unexpected methods in {method_dir}")
    selected_seeds = set(selected_metrics["seed"])

    baseline_frames = []
    for candidate_dir in sorted(metrics_path.parent.iterdir()):
        if not candidate_dir.is_dir():
            continue
        try:
            family = _method_family(candidate_dir.name)
        except ValueError:
            continue
        if family not in BASE_METHODS:
            continue
        files = sorted(candidate_dir.glob("seed_*/client_test_views.csv"))
        for path in files:
            frame = pd.read_csv(path)
            if frame["seed"].isin(selected_seeds).any():
                baseline_frames.append(frame[frame["seed"].isin(selected_seeds)])
    if not baseline_frames:
        raise FileNotFoundError(
            f"No centralized, FedAvg or local per-seed metrics matching {include_method}"
        )
    baselines = pd.concat(baseline_frames, ignore_index=True)
    return pd.concat((baselines, selected_metrics), ignore_index=True)


def plot_data_distribution(
    client_report_dir: Path,
    figure_dir: Path,
    scenario: str,
) -> Path:
    train = pd.read_csv(client_report_dir / "client_summary.csv").sort_values("client")
    test = pd.read_csv(client_report_dir / "test_client_summary.csv").sort_values("client")
    if not {"client", "n_normal", "n_anomaly", "n_normal_real", "n_vehicles"}.issubset(train.columns):
        raise ValueError("Client training summary is missing required data-count columns")
    if not {"client", "n_normal", "n_anomaly", "n_test_vehicles"}.issubset(test.columns):
        raise ValueError("Client test summary is missing required data-count columns")
    clients = train["client"].tolist()
    test = test.set_index("client").reindex(clients)
    if test[["n_normal", "n_anomaly"]].isna().any().any():
        raise ValueError("Client test summary does not contain every training client")

    positions = np.arange(len(clients))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    train_normal = train["n_normal"].to_numpy()
    train_anomaly = train["n_anomaly"].to_numpy()
    test_normal = test["n_normal"].to_numpy()
    test_anomaly = test["n_anomaly"].to_numpy()

    axes[0].bar(positions, train_normal, color="#2a78d6", label="Normal")
    axes[0].bar(
        positions,
        train_anomaly,
        bottom=train_normal,
        color="#eb6834",
        label="Anomalous",
    )
    axes[0].plot(
        positions,
        train["n_normal_real"],
        color="#1e293b",
        marker="D",
        linestyle="--",
        linewidth=1.5,
        markersize=5,
        label="Real normal records used for training",
    )
    for position, normal, anomaly in zip(positions, train_normal, train_anomaly):
        axes[0].annotate(
            f"{int(normal + anomaly):,}",
            (position, normal + anomaly),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set_title("Client training partition")
    axes[0].set_ylabel("Records")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[0].text(
        0.99,
        0.97,
        "Only real normal records enter model optimization",
        transform=axes[0].transAxes,
        va="top",
        ha="right",
        color="#52514e",
        fontsize=8,
    )

    axes[1].bar(positions, test_normal, color="#2a78d6", label="Normal")
    axes[1].bar(
        positions,
        test_anomaly,
        bottom=test_normal,
        color="#eb6834",
        label="Anomalous",
    )
    for position, normal, anomaly in zip(positions, test_normal, test_anomaly):
        axes[1].annotate(
            f"{int(normal + anomaly):,}",
            (position, normal + anomaly),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[1].set_title("Held-out test partition")
    axes[1].set_ylabel("Records")
    axes[1].legend(frameon=False, loc="upper left")

    for axis in axes:
        axis.set_xticks(positions, clients)
        axis.set_ylim(bottom=0, top=axis.get_ylim()[1] * 1.12)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "client_data_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _metric_summary(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        metrics.groupby(["family", "client"])[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def _plot_metric_panel(
    axis: plt.Axes,
    summary: pd.DataFrame,
    clients: list[str],
    metric: str,
    title: str,
    ylabel: str,
    methods: tuple[str, ...],
    y_max: float = 1.0,
    percent_axis: bool = False,
) -> None:
    width = 0.22 if len(methods) == 3 else 0.18
    offsets = np.linspace(-width, width, len(methods))
    positions = np.arange(len(clients))
    for method, offset in zip(methods, offsets):
        subset = summary[summary["family"] == method].set_index("client").reindex(clients)
        if subset["mean"].isna().all():
            continue
        bars = axis.bar(
            positions + offset,
            subset["mean"],
            width=width,
            yerr=subset["std"].fillna(0),
            capsize=3,
            color=COLORS[method],
            label=METHOD_LABELS[method],
        )
        if percent_axis:
            for bar, value in zip(bars, subset["mean"]):
                if pd.notna(value) and value == 0:
                    axis.annotate(
                        "0",
                        (bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color=COLORS[method],
                    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, clients)
    axis.set_ylim(0, y_max)
    if percent_axis:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot_client_performance(
    metrics_path: Path,
    figure_dir: Path,
    scenario: str,
    include_method: str | None = None,
) -> tuple[Path, Path]:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing client-test metrics at {metrics_path}; run the client-aware training and evaluation first."
        )
    metrics = pd.read_csv(metrics_path)
    if include_method is not None:
        if not re.fullmatch(r"fedprox_mu_[0-9.]+_normal_only", include_method):
            raise ValueError(f"Expected a normal-only FedProx method name, received {include_method!r}")
        metrics = _load_matched_method_metrics(metrics_path, include_method)

    required = {
        "client",
        "method",
        "seed",
        "test_pr_auc",
        "test_roc_auc",
        "test_recall",
        "test_false_positive_rate",
    }
    if not required.issubset(metrics.columns):
        raise ValueError(f"Client-test metrics are missing columns: {sorted(required - set(metrics.columns))}")
    metrics["family"] = metrics["method"].map(_method_family)
    clients = sorted(
        metrics["client"].unique(),
        key=lambda name: int(str(name).rsplit("_", 1)[-1]),
    )
    methods = METHODS if include_method is not None else BASE_METHODS
    if set(metrics["family"]) != set(methods):
        raise ValueError(
            f"Expected client-test results for {methods}; found {sorted(metrics['family'].unique())}"
        )
    if metrics.duplicated(["family", "client", "seed"]).any():
        raise ValueError("Client-test metrics contain duplicate family/client/seed rows")
    expected_pairs = {(family, client) for family in methods for client in clients}
    actual_pairs = set(zip(metrics["family"], metrics["client"]))
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise ValueError(f"Incomplete client coverage; missing={missing}, unexpected={extra}")
    seeds_by_pair = metrics.groupby(["family", "client"])["seed"].agg(lambda values: frozenset(values))
    expected_seeds = seeds_by_pair.iloc[0]
    incomplete = seeds_by_pair[seeds_by_pair != expected_seeds]
    if not incomplete.empty:
        raise ValueError(f"Client/method seed coverage differs: {incomplete.to_dict()}")
    metric_columns = [
        "test_pr_auc",
        "test_roc_auc",
        "test_recall",
        "test_false_positive_rate",
    ]
    if metrics[metric_columns].isna().any().any() or not np.isfinite(metrics[metric_columns].to_numpy()).all():
        raise ValueError("Client-test performance metrics contain missing or non-finite values")
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = METHOD_LABELS.copy()
    if include_method is not None:
        mu = re.fullmatch(r"fedprox_mu_([0-9.]+)_normal_only", include_method).group(1)
        labels["fedprox"] = f"FedProx (mu={mu})"

    pr_auc = _metric_summary(metrics, "test_pr_auc")
    roc_auc = _metric_summary(metrics, "test_roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    _plot_metric_panel(
        axes[0], pr_auc, clients, "test_pr_auc", "Per-client PR-AUC", "PR-AUC", methods
    )
    _plot_metric_panel(
        axes[1], roc_auc, clients, "test_roc_auc", "Per-client ROC-AUC", "ROC-AUC", methods
    )
    for axis in axes:
        handles, labels_found = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, [labels[m] for m in methods], frameon=False, loc="best")
    pr_path = figure_dir / "client_test_auc.png"
    fig.savefig(pr_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    recall = _metric_summary(metrics, "test_recall")
    fpr = _metric_summary(metrics, "test_false_positive_rate")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    _plot_metric_panel(
        axes[0], recall, clients, "test_recall", "Recall at calibrated threshold", "Recall", methods
    )
    fpr_upper = float((fpr["mean"] + fpr["std"].fillna(0)).max())
    fpr_limit = max(0.10, np.ceil(fpr_upper / 0.05) * 0.05)
    _plot_metric_panel(
        axes[1],
        fpr,
        clients,
        "test_false_positive_rate",
        "False-positive rate",
        "FPR",
        methods,
        y_max=fpr_limit,
        percent_axis=True,
    )
    handles, _ = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, [labels[m] for m in methods], frameon=False, loc="best")
    operating_path = figure_dir / "client_test_operating_point.png"
    fig.savefig(operating_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pr_path, operating_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot client-aware split and per-client test results.")
    parser.add_argument("--scenario", choices=("iid", "quantity_skew"), required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument(
        "--experiments-config",
        default="configs/experiments_sensors_client_holdout.yaml",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Plot split sizes before training and client-test metrics are available",
    )
    parser.add_argument(
        "--include-method",
        help="Add one exact normal-only FedProx method to the centralized/FedAvg/local plots",
    )
    parser.add_argument(
        "--output-tag",
        help="Save figures in a named subdirectory rather than overwriting standard plots",
    )
    args = parser.parse_args()

    data_config = load_yaml(args.data_config)
    experiments = load_yaml(args.experiments_config)
    if not 0.0 < args.threshold_quantile < 1.0:
        parser.error("--threshold-quantile must be between 0 and 1")
    quantile_tag = f"p{round(args.threshold_quantile * 100)}"
    report_dir = resolve_path(data_config["outputs"]["client_report_dir"]) / args.scenario
    figure_dir = resolve_path(experiments["outputs"]["figures_dir"]) / args.scenario
    if args.threshold_quantile != 0.99:
        figure_dir /= quantile_tag
    if args.output_tag:
        if args.output_tag in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_tag):
            parser.error("--output-tag may contain only letters, numbers, dots, underscores, and hyphens")
        figure_dir /= args.output_tag
    paths = [plot_data_distribution(report_dir, figure_dir, args.scenario)]
    if args.data_only:
        print(paths[0])
        return
    metrics_dir = resolve_path(experiments["outputs"]["metrics_dir"]) / "client_test"
    if args.threshold_quantile != 0.99:
        metrics_dir /= quantile_tag
    metrics_path = metrics_dir / args.scenario / "summary.csv"
    paths.extend(
        plot_client_performance(
            metrics_path,
            figure_dir,
            args.scenario,
            include_method=args.include_method,
        )
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
