from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn


@torch.no_grad()
def per_feature_squared_errors(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    """Squared reconstruction error for each record and feature."""
    was_training = model.training
    model.eval()
    errors = ((model(features) - features) ** 2).cpu().numpy().astype(float)
    model.train(was_training)
    return errors


def aggregate_anomaly_scores(errors: np.ndarray, aggregation: str) -> np.ndarray:
    """Aggregate per-feature errors into one anomaly score per record."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[0] == 0 or errors.shape[1] == 0 or not np.isfinite(errors).all():
        raise ValueError("errors must be a non-empty 2-D array of finite values")
    if aggregation == "mean":
        return errors.mean(axis=1)
    if aggregation == "top_10_percent":
        top_k = int(np.ceil(errors.shape[1] * 0.10))
    elif aggregation == "top_5_percent":
        top_k = int(np.ceil(errors.shape[1] * 0.05))
    else:
        raise ValueError(f"Unknown score aggregation: {aggregation}")
    return np.partition(errors, errors.shape[1] - top_k, axis=1)[:, -top_k:].mean(axis=1)


def anomaly_scores(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    """Per-record mean squared reconstruction error."""
    return aggregate_anomaly_scores(per_feature_squared_errors(model, features), "mean")


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    f1: float | None
    precision: float | None
    recall: float | None
    quantile: float | None = None


def select_threshold_max_f1(scores: np.ndarray, labels: np.ndarray) -> ThresholdSelection:
    """Threshold maximizing F1 on the given (validation) data. A record is an anomaly when score >= threshold."""
    _check_inputs(scores, labels)
    if labels.min() == labels.max():
        raise ValueError("Both classes are required to select a threshold")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    best = int(np.argmax(f1))
    return ThresholdSelection(
        threshold=float(thresholds[best]), f1=float(f1[best]), precision=float(precision[best]), recall=float(recall[best])
    )


def select_threshold_quantile(scores: np.ndarray, quantile: float) -> ThresholdSelection:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a non-empty 1-D array of finite values")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    return ThresholdSelection(
        threshold=float(np.quantile(scores, quantile)),
        f1=None,
        precision=None,
        recall=None,
        quantile=quantile,
    )


def detection_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    _check_inputs(scores, labels)
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    both_classes = labels.min() != labels.max()
    return {
        "n": int(labels.size),
        "n_anomaly": int(positive.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / labels.size,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "pr_auc": float(average_precision_score(labels, scores)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else float("nan"),
        "mean_score": float(scores.mean()),
        "mean_score_normal": float(scores[~positive].mean()) if (~positive).any() else float("nan"),
        "mean_score_anomaly": float(scores[positive].mean()) if positive.any() else float("nan"),
    }


def metrics_by_group(scores: np.ndarray, labels: np.ndarray, groups: Sequence[str], threshold: float) -> pd.DataFrame:
    """Rate of records flagged as anomalies per (group, label); equals FPR for normals and recall for anomalies."""
    frame = pd.DataFrame({"group": list(groups), "label": labels, "flagged": scores >= threshold, "score": scores})
    table = frame.groupby(["group", "label"]).agg(
        n=("flagged", "size"), flagged_rate=("flagged", "mean"), mean_score=("score", "mean")
    )
    return table.reset_index()


def threshold_curve(scores: np.ndarray, labels: np.ndarray, thresholds: Sequence[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = detection_metrics(scores, labels, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                **{key: metrics[key] for key in ("precision", "recall", "f1", "false_positive_rate", "accuracy")},
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ViewEvaluation:
    """Evaluation of one model under one scaler: threshold from validation, metrics on validation and test."""

    view: str
    threshold: ThresholdSelection
    validation: dict[str, float]
    test: dict[str, float]

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"view": self.view, **{f"threshold_{k}": v for k, v in asdict(self.threshold).items()}}
        row.update({f"validation_{k}": v for k, v in self.validation.items()})
        row.update({f"test_{k}": v for k, v in self.test.items()})
        return row


def evaluate_view(
    view: str,
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    normal_validation_scores: np.ndarray | None = None,
    threshold_quantile: float = 0.99,
) -> ViewEvaluation:
    selection = (
        select_threshold_max_f1(validation_scores, validation_labels)
        if normal_validation_scores is None
        else select_threshold_quantile(normal_validation_scores, threshold_quantile)
    )
    return ViewEvaluation(
        view=view,
        threshold=selection,
        validation=detection_metrics(validation_scores, validation_labels, selection.threshold),
        test=detection_metrics(test_scores, test_labels, selection.threshold),
    )


def aggregate_views(rows: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation (ddof = 0) of each metric across views."""
    return {
        column: {"mean": float(rows[column].mean()), "std": float(rows[column].std(ddof=0)), "n_views": len(rows)}
        for column in columns
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _check_inputs(scores: np.ndarray, labels: np.ndarray) -> None:
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays of the same length")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
