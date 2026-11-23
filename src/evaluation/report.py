import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.config import load_yaml, resolve_path
from src.data.preprocess import Standardizer
from src.evaluation.metrics import (
    aggregate_anomaly_scores,
    aggregate_views,
    detection_metrics,
    evaluate_view,
    metrics_by_group,
    per_feature_squared_errors,
    select_threshold_quantile,
    threshold_curve,
)
from src.experiments.common import ExperimentData
from src.models.model import build_model

FEDERATED_PREFIXES = ("fedavg", "fedprox")
CENTRALIZED_METHODS = ("centralized", "centralized_normal_only")
CURVE_QUANTILES = np.linspace(0.01, 0.99, 99)
NORMAL_ONLY_THRESHOLD_QUANTILE = 0.99
REPORTED_METRICS = ("accuracy", "precision", "recall", "f1", "false_positive_rate", "pr_auc", "roc_auc", "mean_score")
SCORE_AGGREGATIONS = ("mean", "top_10_percent", "top_5_percent")


def is_normal_only_method(method: str) -> bool:
    return "normal_only" in method


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    method: str
    seed: int
    directory: Path

    @property
    def kind(self) -> str:
        if self.method in CENTRALIZED_METHODS or self.method.startswith("centralized_"):
            return "centralized"
        if self.method == "local" or self.method.startswith("local_"):
            return "local"
        if self.method.startswith(FEDERATED_PREFIXES):
            return "federated"
        raise ValueError(f"Unknown method directory: {self.method}")


def discover_runs(runs_dir: str | Path) -> list[RunSpec]:
    runs = []
    for summary in sorted(resolve_path(runs_dir).glob("*/*/seed_*/summary.json")):
        directory = summary.parent
        match = re.fullmatch(r"seed_(\d+)", directory.name)
        if match:
            runs.append(RunSpec(directory.parent.parent.name, directory.parent.name, int(match.group(1)), directory))
    return runs


class RunEvaluator:
    """Evaluates the selected model(s) of a run on validation and test.

    A view is one (model, scaler) pair: the centralized model with the global scaler, each local model with
    its client's scaler, or the global federated model with each client's scaler. Thresholds maximize F1
    on the labeled validation partition.
    """

    def __init__(
        self,
        data: ExperimentData,
        model_config: dict[str, Any],
        score_aggregation: str = "mean",
        validation_only: bool = False,
    ) -> None:
        if score_aggregation not in SCORE_AGGREGATIONS:
            raise ValueError(f"score_aggregation must be one of {SCORE_AGGREGATIONS}")
        self.data = data
        self.model_config = model_config
        self.score_aggregation = score_aggregation
        index = data.partition_index.set_index("row_id")
        self.partitions = {}
        partition_names = ("validation",) if validation_only else ("validation", "test")
        for name in partition_names:
            frame = data.partition(name)
            self.partitions[name] = {
                "frame": frame,
                "labels": data.labels(frame),
                "groups": index.loc[frame.index, "category"].astype(str).to_numpy(),
            }

    def evaluate(self, run: RunSpec) -> dict[str, Any]:
        view_rows, group_tables, curves, score_tables = [], [], [], []
        normal_validation = self.data.normal_real_partition("validation") if is_normal_only_method(run.method) else None
        for view, model, scaler in self._views(run):
            scores = {
                name: aggregate_anomaly_scores(
                    per_feature_squared_errors(model, self.data.features(part["frame"], scaler)),
                    self.score_aggregation,
                )
                for name, part in self.partitions.items()
            }
            validation, test = self.partitions["validation"], self.partitions["test"]
            normal_validation_scores = None
            if normal_validation is not None:
                normal_validation_scores = aggregate_anomaly_scores(
                    per_feature_squared_errors(model, self.data.features(normal_validation, scaler)),
                    self.score_aggregation,
                )
            result = evaluate_view(
                view,
                scores["validation"],
                validation["labels"],
                scores["test"],
                test["labels"],
                normal_validation_scores=normal_validation_scores,
                threshold_quantile=NORMAL_ONLY_THRESHOLD_QUANTILE,
            )
            curve_scores = normal_validation_scores if normal_validation_scores is not None else scores["validation"]
            view_rows.append(result.flat())

            groups = metrics_by_group(scores["test"], test["labels"], test["groups"], result.threshold.threshold)
            group_tables.append(groups.assign(view=view))
            thresholds = np.quantile(curve_scores, CURVE_QUANTILES)
            curves.append(threshold_curve(scores["test"], test["labels"], thresholds).assign(view=view, validation_quantile=CURVE_QUANTILES))
            for name, part in self.partitions.items():
                score_tables.append(
                    pd.DataFrame({"row_id": part["frame"].index, "partition": name, "label": part["labels"], "view": view, "score": scores[name]})
                )

        views = pd.DataFrame(view_rows)
        summary = {
            "scenario": run.scenario,
            "method": run.method,
            "seed": run.seed,
            "score_aggregation": self.score_aggregation,
            "threshold_rule": (
                "p99_on_normal_real_validation"
                if is_normal_only_method(run.method)
                else "max_f1_on_validation"
            ),
            "views": views["view"].tolist(),
            "test": aggregate_views(views, [f"test_{name}" for name in REPORTED_METRICS]),
            "validation": aggregate_views(views, [f"validation_{name}" for name in REPORTED_METRICS]),
        }
        return {
            "summary": summary,
            "views": views,
            "groups": pd.concat(group_tables, ignore_index=True),
            "threshold_curve": pd.concat(curves, ignore_index=True),
            "scores": pd.concat(score_tables, ignore_index=True),
        }

    def evaluate_client_test(
        self,
        run: RunSpec,
        client_assignment: pd.DataFrame,
        threshold_quantile: float = NORMAL_ONLY_THRESHOLD_QUANTILE,
    ) -> pd.DataFrame:
        """Evaluate global models on every held-out client and local models on their own client."""
        if "test" not in self.partitions:
            raise ValueError("Client test evaluation requires the test partition")
        required_columns = {"row_id", "client"}
        if not required_columns.issubset(client_assignment.columns):
            raise ValueError(f"client_assignment must include columns {sorted(required_columns)}")
        if client_assignment["row_id"].duplicated().any():
            raise ValueError("Client test assignment contains duplicate row_id values")

        test = self.partitions["test"]
        test_row_ids = pd.Index(test["frame"].index)
        if set(client_assignment["row_id"]) != set(test_row_ids):
            raise ValueError("Client test assignment must cover exactly the held-out test rows")
        client_by_row = client_assignment.set_index("row_id")["client"]
        test_clients = client_by_row.loc[test_row_ids].to_numpy()
        clients = sorted(pd.unique(test_clients), key=lambda client: int(str(client).split("_")[-1]))
        if run.kind != "centralized" and clients != self.data.client_names(run.scenario):
            raise ValueError(f"Client test assignment does not match clients in scenario {run.scenario}")

        validation = self.partitions["validation"]
        normal_validation = (
            self.data.normal_real_partition("validation") if is_normal_only_method(run.method) else None
        )
        rows = []
        for view, model, scaler in self._views(run):
            validation_scores = aggregate_anomaly_scores(
                per_feature_squared_errors(model, self.data.features(validation["frame"], scaler)),
                self.score_aggregation,
            )
            normal_validation_scores = None
            if normal_validation is not None:
                normal_validation_scores = aggregate_anomaly_scores(
                    per_feature_squared_errors(model, self.data.features(normal_validation, scaler)),
                    self.score_aggregation,
                )
            test_scores = aggregate_anomaly_scores(
                per_feature_squared_errors(model, self.data.features(test["frame"], scaler)),
                self.score_aggregation,
            )
            for client in clients:
                if run.kind == "local" and view != client:
                    continue
                mask = test_clients == client
                client_labels = test["labels"][mask]
                result = evaluate_view(
                    view,
                    validation_scores,
                    validation["labels"],
                    test_scores[mask],
                    client_labels,
                    normal_validation_scores=normal_validation_scores,
                    threshold_quantile=threshold_quantile,
                )
                rows.append(
                    {
                        "client": client,
                        "model_view": view,
                        **result.flat(),
                        "test_n_normal": int(np.sum(client_labels == 0)),
                    }
                )
        return pd.DataFrame(rows)

    def score_sweep_validation(self, run: RunSpec) -> pd.DataFrame:
        """Compare predeclared score aggregations using validation data only."""
        validation = self.partitions["validation"]
        normal_validation = self.data.normal_real_partition("validation")
        rows = []
        for view, model, scaler in self._views(run):
            errors = per_feature_squared_errors(model, self.data.features(validation["frame"], scaler))
            normal_errors = per_feature_squared_errors(model, self.data.features(normal_validation, scaler))
            for aggregation in SCORE_AGGREGATIONS:
                scores = aggregate_anomaly_scores(errors, aggregation)
                normal_scores = aggregate_anomaly_scores(normal_errors, aggregation)
                selection = select_threshold_quantile(
                    normal_scores,
                    NORMAL_ONLY_THRESHOLD_QUANTILE,
                )
                metrics = detection_metrics(scores, validation["labels"], selection.threshold)
                rows.append(
                    {
                        "view": view,
                        "score_aggregation": aggregation,
                        "top_k_features": (
                            errors.shape[1]
                            if aggregation == "mean"
                            else int(np.ceil(errors.shape[1] * (0.10 if aggregation == "top_10_percent" else 0.05)))
                        ),
                        "threshold": selection.threshold,
                        "threshold_quantile": selection.quantile,
                        "validation_pr_auc": average_precision_score(validation["labels"], scores),
                        "validation_roc_auc": roc_auc_score(validation["labels"], scores),
                        "validation_f1": metrics["f1"],
                        "validation_precision": metrics["precision"],
                        "validation_recall": metrics["recall"],
                    }
                )
        return pd.DataFrame(rows)

    def _views(self, run: RunSpec) -> list[tuple[str, nn.Module, Standardizer]]:
        if run.kind == "centralized":
            scaler = (
                Standardizer.load(run.directory / "scaler.json")
                if is_normal_only_method(run.method)
                else self.data.global_scaler()
            )
            model = self._load_model(run.directory / "model.pt", run.seed, len(scaler.feature_columns))
            return [("global", model, scaler)]
        clients = self.data.client_names(run.scenario)
        if run.kind == "local":
            views = []
            shared_scaler = (
                Standardizer.load(run.directory / "scaler.json")
                if is_normal_only_method(run.method)
                else None
            )
            for client in clients:
                scaler = (
                    shared_scaler
                    if shared_scaler is not None
                    else self.data.client_scaler(run.scenario, client)
                )
                model = self._load_model(run.directory / client / "model.pt", run.seed, len(scaler.feature_columns))
                views.append((client, model, scaler))
            return views
        if is_normal_only_method(run.method):
            scaler = Standardizer.load(run.directory / "scaler.json")
            model = self._load_model(run.directory / "model.pt", run.seed, len(scaler.feature_columns))
            return [("global", model, scaler)]
        model = self._load_model(run.directory / "model.pt", run.seed)
        return [(client, model, self.data.client_scaler(run.scenario, client)) for client in clients]

    def _load_model(self, path: Path, seed: int, n_features: int | None = None) -> nn.Module:
        input_dim = len(self.data.feature_columns) if n_features is None else n_features
        config_paths = (path.parent / "model_config.json", path.parent.parent / "model_config.json")
        config_path = next((candidate for candidate in config_paths if candidate.exists()), None)
        if config_path is None:
            model_config = self.model_config
        else:
            with config_path.open(encoding="utf-8") as handle:
                model_config = json.load(handle)
        model = build_model(model_config, input_dim, seed)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model


def write_run_report(result: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result["summary"], handle, indent=2, ensure_ascii=False)
    for name in ("views", "groups", "threshold_curve", "scores"):
        result[name].to_csv(directory / f"{name}.csv", index=False)


def consolidated_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {key: summary[key] for key in ("scenario", "method", "seed")}
    row["n_views"] = len(summary["views"])
    for metric, values in summary["test"].items():
        row[f"{metric}_mean"] = values["mean"]
        row[f"{metric}_std"] = values["std"]
    return row


def run_report(
    data_config: str,
    model_config: str,
    experiments_config: str,
    run_dirs: list[str] | None,
    score_aggregation: str = "mean",
    score_sweep_validation: bool = False,
) -> pd.DataFrame | None:
    data = ExperimentData(data_config)
    experiments = load_yaml(experiments_config)
    evaluator = RunEvaluator(data, load_yaml(model_config), score_aggregation, validation_only=score_sweep_validation)
    runs = discover_runs(experiments["outputs"]["runs_dir"])
    if run_dirs:
        selected = {resolve_path(path).resolve() for path in run_dirs}
        runs = [run for run in runs if run.directory.resolve() in selected]
    if not runs:
        raise ValueError("No runs found")

    metrics_dir = resolve_path(experiments["outputs"]["metrics_dir"])
    if score_sweep_validation:
        if len(runs) != 1 or runs[0].kind != "centralized" or not is_normal_only_method(runs[0].method):
            raise ValueError("Validation score sweep requires exactly one centralized normal-only run")
        result = evaluator.score_sweep_validation(runs[0])
        output = metrics_dir / runs[0].scenario / runs[0].method / f"seed_{runs[0].seed}" / "score_sweep_validation.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        return None

    for run in runs:
        write_run_report(evaluator.evaluate(run), metrics_dir / run.scenario / run.method / f"seed_{run.seed}")

    rows = []
    for path in sorted(metrics_dir.glob("*/*/seed_*/summary.json")):
        with path.open(encoding="utf-8") as handle:
            rows.append(consolidated_row(json.load(handle)))
    table = pd.DataFrame(rows).sort_values(["scenario", "method", "seed"]).reset_index(drop=True)
    table.to_csv(metrics_dir / "summary.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained runs on validation and test.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--experiments-config", default="configs/experiments.yaml")
    parser.add_argument("--run-dir", action="append", help="Evaluate only this run directory (repeatable)")
    parser.add_argument(
        "--score-aggregation",
        choices=SCORE_AGGREGATIONS,
        default="mean",
        help="Aggregate per-feature reconstruction errors into a record score",
    )
    parser.add_argument(
        "--score-sweep-validation",
        action="store_true",
        help="Compare score aggregations on validation only; requires one centralized normal-only run",
    )
    args = parser.parse_args()
    run_report(
        args.data_config,
        args.model_config,
        args.experiments_config,
        args.run_dir,
        score_aggregation=args.score_aggregation,
        score_sweep_validation=args.score_sweep_validation,
    )


if __name__ == "__main__":
    main()
