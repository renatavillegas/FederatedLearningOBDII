import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.metrics import (
    aggregate_anomaly_scores,
    aggregate_views,
    anomaly_scores,
    detection_metrics,
    evaluate_view,
    metrics_by_group,
    per_feature_squared_errors,
    select_threshold_max_f1,
    select_threshold_quantile,
    threshold_curve,
)
from src.evaluation.report import (
    RunEvaluator,
    RunSpec,
    consolidated_row,
    discover_runs,
    is_normal_only_method,
)
from src.models.model import Autoencoder, AutoencoderConfig, build_model

SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
LABELS = np.array([0, 0, 0, 1, 1, 1])


def test_detection_metrics_counts():
    metrics = detection_metrics(SCORES, LABELS, threshold=0.3)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (3, 1, 0, 2)
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 * 0.75 / 1.75)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_threshold_uses_greater_or_equal():
    assert detection_metrics(SCORES, LABELS, threshold=0.4)["tp"] == 3


def test_max_f1_threshold_on_separable_data():
    selection = select_threshold_max_f1(SCORES, LABELS)
    assert selection.threshold == pytest.approx(0.4)
    assert selection.f1 == pytest.approx(1.0)


def test_quantile_threshold_uses_only_calibration_scores():
    selection = select_threshold_quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.75)
    assert selection.threshold == pytest.approx(3.25)
    assert selection.quantile == pytest.approx(0.75)
    assert selection.f1 is None


def test_quantile_threshold_rejects_invalid_input():
    with pytest.raises(ValueError):
        select_threshold_quantile(np.array([]), 0.99)
    with pytest.raises(ValueError):
        select_threshold_quantile(np.array([1.0, 2.0]), 1.0)


def test_max_f1_threshold_matches_bruteforce():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    scores = rng.normal(size=200) + labels
    selection = select_threshold_max_f1(scores, labels)
    best = max(detection_metrics(scores, labels, t)["f1"] for t in np.unique(scores))
    assert selection.f1 == pytest.approx(best)
    assert detection_metrics(scores, labels, selection.threshold)["f1"] == pytest.approx(best)


def test_threshold_requires_both_classes():
    with pytest.raises(ValueError):
        select_threshold_max_f1(SCORES, np.zeros(6, dtype=int))


def test_single_class_metrics_have_nan_auc():
    metrics = detection_metrics(SCORES, np.zeros(6, dtype=int), threshold=0.5)
    assert np.isnan(metrics["roc_auc"]) and np.isnan(metrics["recall"])
    assert metrics["false_positive_rate"] == pytest.approx(2 / 6)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        detection_metrics(SCORES, np.array([0, 1, 2, 0, 1, 0]), 0.5)
    with pytest.raises(ValueError):
        detection_metrics(SCORES, LABELS[:-1], 0.5)


def test_metrics_by_group():
    groups = ["REAL", "REAL", "GRAY", "GRAY", "CAT", "CAT"]
    table = metrics_by_group(SCORES, LABELS, groups, threshold=0.3).set_index(["group", "label"])
    assert table.loc[("REAL", 0), "flagged_rate"] == 0.0
    assert table.loc[("GRAY", 0), "flagged_rate"] == 1.0
    assert table.loc[("CAT", 1), "flagged_rate"] == 1.0


def test_threshold_curve():
    curve = threshold_curve(SCORES, LABELS, [0.0, 0.4, 1.0])
    assert curve["recall"].tolist()[:2] == [1.0, 1.0]
    assert np.isnan(curve["precision"].iloc[-1])


def test_evaluate_view_uses_validation_threshold():
    test_scores = np.array([0.35, 0.39, 0.41, 0.5])
    test_labels = np.array([0, 1, 1, 1])
    result = evaluate_view("client_1", SCORES, LABELS, test_scores, test_labels)
    assert result.threshold.threshold == pytest.approx(0.4)
    assert result.test["tp"] == 2 and result.test["fn"] == 1
    row = result.flat()
    assert row["view"] == "client_1" and "test_f1" in row and "validation_f1" in row


def test_evaluate_view_can_calibrate_on_normal_validation_scores():
    normal_scores = np.array([0.1, 0.2, 0.3, 0.4])
    result = evaluate_view(
        "global",
        SCORES,
        LABELS,
        SCORES,
        LABELS,
        normal_validation_scores=normal_scores,
        threshold_quantile=0.75,
    )
    assert result.threshold.threshold == pytest.approx(0.325)
    assert result.threshold.quantile == pytest.approx(0.75)


def test_aggregate_views():
    rows = pd.DataFrame({"test_f1": [0.5, 0.7]})
    aggregated = aggregate_views(rows, ["test_f1"])["test_f1"]
    assert aggregated["mean"] == pytest.approx(0.6)
    assert aggregated["std"] == pytest.approx(0.1)
    assert aggregated["n_views"] == 2


def test_anomaly_scores_match_reconstruction_error():
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderConfig(input_dim=4, encoder_dims=(3,), latent_dim=2))
    features = torch.randn(5, 4)
    expected = ((model(features) - features) ** 2).mean(dim=1).detach().numpy()
    np.testing.assert_allclose(anomaly_scores(model, features), expected, rtol=1e-6)


def test_score_aggregations_mean_and_top_feature_errors():
    errors = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    np.testing.assert_allclose(aggregate_anomaly_scores(errors, "mean"), [3.0, 3.0])
    np.testing.assert_allclose(aggregate_anomaly_scores(errors, "top_10_percent"), [5.0, 5.0])
    np.testing.assert_allclose(aggregate_anomaly_scores(errors, "top_5_percent"), [5.0, 5.0])


def test_top_error_aggregation_uses_ceil_of_feature_fraction():
    errors = np.arange(118, dtype=float).reshape(1, -1)
    assert aggregate_anomaly_scores(errors, "top_10_percent")[0] == pytest.approx(np.mean(np.arange(106, 118)))
    assert aggregate_anomaly_scores(errors, "top_5_percent")[0] == pytest.approx(np.mean(np.arange(112, 118)))


def test_score_aggregation_rejects_invalid_input():
    with pytest.raises(ValueError):
        aggregate_anomaly_scores(np.array([]), "mean")
    with pytest.raises(ValueError):
        aggregate_anomaly_scores(np.ones((2, 3)), "unknown")


def test_per_feature_squared_errors_preserves_model_mode():
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderConfig(input_dim=4, encoder_dims=(3,), latent_dim=2))
    model.train()
    features = torch.randn(5, 4)
    expected = ((model(features) - features) ** 2).detach().numpy()
    np.testing.assert_allclose(per_feature_squared_errors(model, features), expected, rtol=1e-6)
    assert model.training


def test_discover_runs(tmp_path):
    for relative in (
        "all/centralized/seed_42",
        "all/centralized_arch_64_32_16/seed_42",
        "all/centralized_normal_only/seed_42",
        "iid/local/seed_42",
        "iid/local_normal_only/seed_42",
        "iid/fedprox_mu_0.01/seed_7",
        "iid/fedavg_normal_only/seed_42",
        "iid/fedavg/other",
    ):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}")
    runs = discover_runs(tmp_path)
    assert [(r.scenario, r.method, r.seed, r.kind) for r in runs] == [
        ("all", "centralized", 42, "centralized"),
        ("all", "centralized_arch_64_32_16", 42, "centralized"),
        ("all", "centralized_normal_only", 42, "centralized"),
        ("iid", "fedavg_normal_only", 42, "federated"),
        ("iid", "fedprox_mu_0.01", 7, "federated"),
        ("iid", "local", 42, "local"),
        ("iid", "local_normal_only", 42, "local"),
    ]


def test_validation_only_evaluator_does_not_load_test_partition():
    class DataStub:
        partition_index = pd.DataFrame({"row_id": [0], "category": ["SPLIT_REAL"]})

        def __init__(self):
            self.loaded_partitions = []

        def partition(self, name):
            self.loaded_partitions.append(name)
            return pd.DataFrame({"feature": [0.0]}, index=[0])

        @staticmethod
        def labels(frame):
            return np.array([0])

    data = DataStub()
    evaluator = RunEvaluator(data, {}, validation_only=True)
    assert list(evaluator.partitions) == ["validation"]
    assert data.loaded_partitions == ["validation"]


def test_normal_only_report_uses_p99_of_normal_real_validation_scores(monkeypatch):
    class ZeroModel(torch.nn.Module):
        def forward(self, features):
            return torch.zeros_like(features)

    class DataStub:
        def normal_real_partition(self, name):
            assert name == "validation"
            return pd.DataFrame({"feature": np.linspace(1.0, 10.0, 101)})

        @staticmethod
        def features(frame, scaler):
            return torch.tensor(frame[["feature"]].to_numpy(), dtype=torch.float32)

    evaluator = RunEvaluator.__new__(RunEvaluator)
    evaluator.data = DataStub()
    evaluator.score_aggregation = "mean"
    evaluator.partitions = {
        "validation": {
            "frame": pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}),
            "labels": LABELS,
            "groups": np.array(["REAL", "REAL", "GRAY", "GRAY", "CAT", "CAT"]),
        },
        "test": {
            "frame": pd.DataFrame({"feature": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]}),
            "labels": LABELS,
            "groups": np.array(["REAL", "REAL", "GRAY", "GRAY", "CAT", "CAT"]),
        },
    }
    monkeypatch.setattr(evaluator, "_views", lambda run: [("global", ZeroModel(), None)])
    run = RunSpec("all", "centralized_normal_only_sensors105", 42, Path("unused"))

    result = evaluator.evaluate(run)

    expected_threshold = float(np.quantile(np.linspace(1.0, 10.0, 101) ** 2, 0.99))
    assert result["summary"]["threshold_rule"] == "p99_on_normal_real_validation"
    assert result["views"].loc[0, "threshold_quantile"] == pytest.approx(0.99)
    assert result["views"].loc[0, "threshold_threshold"] == pytest.approx(expected_threshold)


def test_evaluator_loads_model_config_saved_with_run(tmp_path):
    run_directory = tmp_path / "all" / "centralized_arch_test" / "seed_42"
    run_directory.mkdir(parents=True)
    saved_config = {
        "architecture": {"encoder_dims": [4], "latent_dim": 2, "activation": "relu"}
    }
    current_config = {
        "architecture": {"encoder_dims": [8, 4], "latent_dim": 2, "activation": "relu"}
    }
    (run_directory / "model_config.json").write_text(json.dumps(saved_config))
    model_path = run_directory / "model.pt"
    torch.save(build_model(saved_config, 6, 42).state_dict(), model_path)

    evaluator = RunEvaluator.__new__(RunEvaluator)
    evaluator.model_config = current_config
    evaluator.data = type("DataStub", (), {"feature_columns": list("abcdef")})()

    loaded = evaluator._load_model(model_path, 42)
    assert loaded.config.encoder_dims == (4,)


def test_evaluator_loads_model_config_from_local_run_directory(tmp_path):
    run_directory = tmp_path / "iid" / "local_normal_only" / "seed_42"
    client_directory = run_directory / "client_1"
    client_directory.mkdir(parents=True)
    saved_config = {
        "architecture": {"encoder_dims": [4], "latent_dim": 2, "activation": "relu"}
    }
    current_config = {
        "architecture": {"encoder_dims": [8, 4], "latent_dim": 2, "activation": "relu"}
    }
    (run_directory / "model_config.json").write_text(json.dumps(saved_config))
    model_path = client_directory / "model.pt"
    torch.save(build_model(saved_config, 6, 42).state_dict(), model_path)

    evaluator = RunEvaluator.__new__(RunEvaluator)
    evaluator.model_config = current_config
    evaluator.data = type("DataStub", (), {"feature_columns": list("abcdef")})()

    loaded = evaluator._load_model(model_path, 42)
    assert loaded.config.encoder_dims == (4,)


def test_unknown_method_kind():
    with pytest.raises(ValueError):
        RunSpec("iid", "other", 1, Path("unused")).kind


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("centralized_normal_only", True),
        ("centralized_normal_only_latent4", True),
        ("local_normal_only", True),
        ("fedavg_normal_only", True),
        ("fedprox_mu_0.01_normal_only", True),
        ("centralized", False),
        ("centralized_arch_latent4", False),
    ],
)
def test_identifies_normal_only_runs(method, expected):
    assert is_normal_only_method(method) is expected


def test_consolidated_row():
    summary = {"scenario": "iid", "method": "fedavg", "seed": 42, "views": ["client_1", "client_2"],
               "test": {"test_f1": {"mean": 0.6, "std": 0.1, "n_views": 2}}}
    row = consolidated_row(summary)
    assert row == {"scenario": "iid", "method": "fedavg", "seed": 42, "n_views": 2, "test_f1_mean": 0.6, "test_f1_std": 0.1}
