import argparse
from pathlib import Path

import pandas as pd

from src.config import load_yaml, resolve_path
from src.evaluation.report import SCORE_AGGREGATIONS, RunEvaluator, discover_runs
from src.experiments.common import ExperimentData


def _is_client_holdout_federated_method(method: str) -> bool:
    return method in {"local_normal_only", "fedavg_normal_only"} or (
        method.startswith("fedprox_mu_") and method.endswith("_normal_only")
    )


def _is_default_client_test_method(method: str) -> bool:
    return method in {"local_normal_only", "fedavg_normal_only"}


def _matches_method_filter(run_method: str, method_filter: str | None) -> bool:
    return method_filter is None or run_method == method_filter


def run_client_test_report(
    data_config_path: str,
    experiments_config_path: str,
    model_config_path: str,
    scenario: str,
    seeds: list[int],
    score_aggregation: str = "top_5_percent",
    threshold_quantile: float = 0.99,
    method: str | None = None,
    runs_dir: str | None = None,
    metrics_dir: str | None = None,
    edge_client_id: str | None = None,
) -> pd.DataFrame:
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1")
    data = ExperimentData(data_config_path)
    experiments = load_yaml(experiments_config_path)
    model_config = load_yaml(model_config_path)
    evaluator = RunEvaluator(
        data,
        model_config,
        score_aggregation=score_aggregation,
    )
    client_assignment = data.clients.load(f"{scenario}_test")

    raw_runs = runs_dir or experiments["outputs"]["runs_dir"]
    edge_id = edge_client_id or "client_4"
    if "{edge_client_id}" in str(raw_runs):
        resolved_runs = str(raw_runs).format(edge_client_id=edge_id)
    else:
        resolved_runs = str(raw_runs)

    runs = discover_runs(resolved_runs)
    latent_dim = int(model_config["architecture"]["latent_dim"])
    central_method = (
        f"centralized_normal_only_sensors105_client_{scenario}_latent{latent_dim}"
    )
    selected_runs = [
        run
        for run in runs
        if run.seed in seeds
        and _matches_method_filter(run.method, method)
        and (
            (run.kind == "centralized" and run.method == central_method)
            or (
                run.scenario == scenario
                and _is_client_holdout_federated_method(run.method)
                and (
                    method is not None
                    or _is_default_client_test_method(run.method)
                )
            )
        )
    ]
    if not selected_runs:
        raise ValueError(f"No trained runs found for client test evaluation of {scenario}")

    raw_metrics = metrics_dir or experiments["outputs"]["metrics_dir"]
    if "{edge_client_id}" in str(raw_metrics):
        resolved_metrics = str(raw_metrics).format(edge_client_id=edge_id)
    else:
        resolved_metrics = str(raw_metrics)

    target_metrics_dir = resolve_path(resolved_metrics) / "client_test"
    if threshold_quantile != 0.99:
        target_metrics_dir /= f"p{round(threshold_quantile * 100)}"
    target_metrics_dir /= scenario
    result_rows = []
    for run in selected_runs:
        result = evaluator.evaluate_client_test(
            run,
            client_assignment,
            threshold_quantile=threshold_quantile,
        )
        result.insert(0, "scenario", scenario)
        result.insert(1, "score_aggregation", score_aggregation)
        result.insert(2, "method", run.method)
        result.insert(3, "seed", run.seed)
        destination = target_metrics_dir / run.method / f"seed_{run.seed}"
        destination.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination / "client_test_views.csv", index=False)
        result_rows.append(result)
    summary = pd.concat(result_rows, ignore_index=True)
    target_metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        target_metrics_dir / method / "summary.csv"
        if method is not None
        else target_metrics_dir / "summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if method is not None and summary_path.is_file():
        existing = pd.read_csv(summary_path)
        summary = pd.concat((existing, summary), ignore_index=True)
        summary = summary.drop_duplicates(["method", "seed", "client"], keep="last")
    summary.to_csv(summary_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate methods on held-out client test sets.")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--experiments-config", default="configs/experiments_sensors_client_holdout.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--method",
        help="Evaluate only this exact method; filtered results are saved under that method's directory",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=SCORE_AGGREGATIONS,
        default="top_5_percent",
        help="Fixed score aggregation selected using validation data",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.99,
        help="Normal-real validation score quantile used as the threshold (for example 0.95)",
    )
    parser.add_argument("--runs-dir", default=None, help="Override outputs runs_dir")
    parser.add_argument("--metrics-dir", default=None, help="Override outputs metrics_dir")
    parser.add_argument("--edge-client-id", default=None, help="Edge client ID (e.g. client_2, client_6)")
    args = parser.parse_args()
    run_client_test_report(
        args.data_config,
        args.experiments_config,
        args.model_config,
        args.scenario,
        args.seeds or [7, 21, 42],
        score_aggregation=args.score_aggregation,
        threshold_quantile=args.threshold_quantile,
        method=args.method,
        runs_dir=args.runs_dir,
        metrics_dir=args.metrics_dir,
        edge_client_id=args.edge_client_id,
    )


if __name__ == "__main__":
    main()
