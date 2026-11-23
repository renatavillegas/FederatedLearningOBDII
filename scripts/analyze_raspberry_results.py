#!/usr/bin/env python
"""Consolidate edge inference CSVs from all raspberry experiments into a single comparison table.

Reads all edge_inference_metrics.csv files under results/runs/sensors_only_client_holdout_raspberry_*/
and writes results/metrics/raspberry_edge_comparison.csv with per-run aggregate statistics.

Usage:
    python scripts/analyze_raspberry_results.py [--runs-root results/runs] [--output results/metrics/raspberry_edge_comparison.csv]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    with summary_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _aggregate_inference(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}
    agg = {}
    for col in ("latency_median_ms", "latency_p95_ms", "throughput_records_per_second", "inference_seconds"):
        if col not in df.columns:
            continue
        agg[f"{col}_mean"] = float(np.mean(df[col]))
        agg[f"{col}_std"] = float(np.std(df[col]))
        agg[f"{col}_min"] = float(np.min(df[col]))
        agg[f"{col}_max"] = float(np.max(df[col]))
    for col in ("latency_p99_ms", "latency_std_ms"):
        if col not in df.columns:
            continue
        agg[f"{col}_mean"] = float(np.mean(df[col]))
        agg[f"{col}_std"] = float(np.std(df[col]))
    agg["n_rounds_with_inference"] = int(len(df))
    agg["records_per_round"] = int(df["records"].iloc[0]) if "records" in df.columns else None
    agg["warmup_records"] = int(df["warmup_records"].iloc[0]) if "warmup_records" in df.columns else None
    return agg


def collect_runs(runs_root: Path) -> list[dict]:
    rows = []
    pattern = "sensors_only_client_holdout_raspberry_*"
    for exp_dir in sorted(runs_root.glob(pattern)):
        edge_client = exp_dir.name.replace("sensors_only_client_holdout_raspberry_", "")
        for scenario_dir in sorted(exp_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario = scenario_dir.name
            for method_dir in sorted(scenario_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    seed = int(seed_dir.name.replace("seed_", ""))
                    summary = _load_summary(seed_dir)
                    csv_path = seed_dir / "edge_inference_metrics.csv"
                    if not csv_path.is_file():
                        continue
                    inference_agg = _aggregate_inference(csv_path)
                    row = {
                        "edge_client": edge_client,
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                        "n_train_edge": summary.get("clients", {}).get(edge_client, {}).get("n_train"),
                        "n_vehicles_edge": summary.get("clients", {}).get(edge_client, {}).get("n_vehicles"),
                        "best_round": summary.get("best_round"),
                        "best_validation_loss": summary.get("best_validation_loss"),
                        "rounds_run": summary.get("rounds_run"),
                        "stopped_early": summary.get("stopped_early"),
                        "total_seconds": summary.get("total_seconds"),
                        "n_parameters": summary.get("n_parameters"),
                        "parameter_bytes": summary.get("parameter_bytes"),
                        **inference_agg,
                    }
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output", default="results/metrics/raspberry_edge_comparison.csv")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        print(f"Runs root not found: {runs_root}", file=sys.stderr)
        sys.exit(1)

    rows = collect_runs(runs_root)
    if not rows:
        print("No completed raspberry experiments found.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ("edge_client", "scenario", "method", "seed") if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Wrote {len(df)} rows to {output_path}")

    summary_cols = [
        "edge_client", "scenario", "method", "seed",
        "n_train_edge", "rounds_run", "best_validation_loss",
        "latency_median_ms_mean", "latency_p95_ms_mean",
        "throughput_records_per_second_mean", "total_seconds",
    ]
    present = [c for c in summary_cols if c in df.columns]
    print("\n--- Summary ---")
    print(df[present].to_string(index=False))


if __name__ == "__main__":
    main()
