import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_path


REPORTED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
    "pr_auc",
    "roc_auc",
    "mean_score",
)


def update_global_summary(
    global_summary_path: Path,
    scenario: str,
    method: str,
    seed: int,
    views_df: pd.DataFrame,
) -> pd.DataFrame:
    method_name = f"{method}_raspberry_client4"
    n_views = len(views_df)

    row: dict[str, object] = {
        "scenario": scenario,
        "method": method_name,
        "seed": seed,
        "n_views": n_views,
    }
    for metric in REPORTED_METRICS:
        col = f"test_{metric}"
        row[f"{col}_mean"] = float(views_df[col].mean()) if col in views_df else 0.0
        row[f"{col}_std"] = float(views_df[col].std(ddof=0)) if col in views_df else 0.0

    if global_summary_path.is_file():
        existing = pd.read_csv(global_summary_path)
        mask = (
            (existing["scenario"] == scenario)
            & (existing["method"] == method_name)
            & (existing["seed"] == seed)
        )
        if mask.any():
            existing = existing[~mask]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        global_summary_path.parent.mkdir(parents=True, exist_ok=True)
        updated = pd.DataFrame([row])

    updated.to_csv(global_summary_path, index=False)
    return updated


def update_test_summary_table(
    table_path: Path,
    scenario: str,
    method: str,
    seed: int,
    quantile: float,
    views_df: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    mean_record: dict[str, object] = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "threshold_quantile": quantile,
        "scope": "mean_across_clients",
        "client": "all",
        "n_samples": int(views_df["test_n"].sum()),
        "n_anomaly": int(views_df["test_n_anomaly"].sum()),
    }
    for metric in REPORTED_METRICS:
        col = f"test_{metric}"
        mean_record[metric] = float(views_df[col].mean()) if col in views_df else 0.0
    records.append(mean_record)

    for _, row in views_df.iterrows():
        client_name = str(row["client"])
        scope = "edge_raspberry" if client_name == "client_4" else "simulated_client"
        client_record: dict[str, object] = {
            "scenario": scenario,
            "method": method,
            "seed": seed,
            "threshold_quantile": quantile,
            "scope": scope,
            "client": client_name,
            "n_samples": int(row["test_n"]),
            "n_anomaly": int(row["test_n_anomaly"]),
        }
        for metric in REPORTED_METRICS:
            col = f"test_{metric}"
            client_record[metric] = float(row[col]) if col in row else 0.0
        records.append(client_record)

    new_df = pd.DataFrame(records)

    if table_path.is_file():
        existing = pd.read_csv(table_path)
        mask = (
            (existing["scenario"] == scenario)
            & (existing["method"] == method)
            & (existing["seed"] == seed)
            & (existing["threshold_quantile"] == quantile)
        )
        if mask.any():
            existing = existing[~mask]
        updated = pd.concat([existing, new_df], ignore_index=True)
    else:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        updated = new_df

    updated.to_csv(table_path, index=False)
    return updated


def update_edge_performance_table(
    table_path: Path,
    scenario: str,
    method: str,
    seed: int,
    edge_metrics_path: Path,
) -> pd.DataFrame | None:
    if not edge_metrics_path.is_file():
        return None

    df = pd.read_csv(edge_metrics_path)
    if df.empty:
        return None

    row: dict[str, object] = {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "edge_device": "raspberry_pi",
        "client": str(df["client"].iloc[0]),
        "total_rounds": int(len(df)),
        "records_per_round": int(df["records"].iloc[0]),
        "warmup_records": int(df["warmup_records"].iloc[0]),
        "latency_median_ms_mean": float(df["latency_median_ms"].mean()),
        "latency_median_ms_std": float(df["latency_median_ms"].std(ddof=0)),
        "latency_median_ms_min": float(df["latency_median_ms"].min()),
        "latency_median_ms_max": float(df["latency_median_ms"].max()),
        "latency_p95_ms_mean": float(df["latency_p95_ms"].mean()),
        "latency_p95_ms_std": float(df["latency_p95_ms"].std(ddof=0)),
        "throughput_records_per_second_mean": float(df["throughput_records_per_second"].mean()),
        "throughput_records_per_second_std": float(df["throughput_records_per_second"].std(ddof=0)),
        "batch_inference_seconds_mean": float(df["inference_seconds"].mean()),
    }

    new_df = pd.DataFrame([row])
    if table_path.is_file():
        existing = pd.read_csv(table_path)
        mask = (
            (existing["scenario"] == scenario)
            & (existing["method"] == method)
            & (existing["seed"] == seed)
        )
        if mask.any():
            existing = existing[~mask]
        updated = pd.concat([existing, new_df], ignore_index=True)
    else:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        updated = new_df

    updated.to_csv(table_path, index=False)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update metrics/summary.csv and edge performance tables for Raspberry Pi runs."
    )
    parser.add_argument("--scenario", required=True, choices=("iid", "quantity_skew"))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-client-id", default=None, help="Edge client ID (e.g. client_2, client_4, client_6)")
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Path to edge client metrics directory",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Path to edge client runs directory",
    )
    parser.add_argument(
        "--global-summary",
        default="results/metrics/summary.csv",
    )
    args = parser.parse_args()

    edge_id = args.edge_client_id or "client_4"
    metrics_dir = resolve_path(args.metrics_dir or f"results/metrics/sensors_only_client_holdout_raspberry_{edge_id}")
    runs_dir = resolve_path(args.runs_dir or f"results/runs/sensors_only_client_holdout_raspberry_{edge_id}")
    global_summary_path = resolve_path(args.global_summary)
    test_summary_path = resolve_path("results/metrics/raspberry_test_summary.csv")
    edge_perf_path = resolve_path("results/metrics/raspberry_edge_performance.csv")

    p99_views_path = (
        metrics_dir / "client_test" / args.scenario / args.method / f"seed_{args.seed}" / "client_test_views.csv"
    )
    p95_views_path = (
        metrics_dir / "client_test" / "p95" / args.scenario / args.method / f"seed_{args.seed}" / "client_test_views.csv"
    )
    edge_metrics_path = (
        runs_dir / args.scenario / args.method / f"seed_{args.seed}" / "edge_inference_metrics.csv"
    )

    if p99_views_path.is_file():
        views_99 = pd.read_csv(p99_views_path)
        update_global_summary(global_summary_path, args.scenario, args.method, args.seed, views_99)
        update_test_summary_table(test_summary_path, args.scenario, args.method, args.seed, 0.99, views_99)

    if p95_views_path.is_file():
        views_95 = pd.read_csv(p95_views_path)
        update_test_summary_table(test_summary_path, args.scenario, args.method, args.seed, 0.95, views_95)

    update_edge_performance_table(edge_perf_path, args.scenario, args.method, args.seed, edge_metrics_path)


if __name__ == "__main__":
    main()
