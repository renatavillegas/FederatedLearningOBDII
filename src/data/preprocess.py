import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml, resolve_path
from src.data.schema import DataSchema
from src.data.split import PartitionStore


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class FeatureSelector:
    """Selects candidate model inputs, excluding configured groups and training-constant columns."""

    def __init__(self, schema: DataSchema, excluded_prefixes: Sequence[str] = ()) -> None:
        if any(not prefix for prefix in excluded_prefixes):
            raise ValueError("excluded_prefixes must not contain empty strings")
        self.schema = schema
        self.excluded_prefixes = tuple(excluded_prefixes)
        self.feature_columns: list[str] | None = None
        self.dropped_constant: list[str] | None = None

    def fit(self, train_frame: pd.DataFrame) -> "FeatureSelector":
        all_candidates = self.schema.candidate_feature_columns(train_frame)
        unmatched = [
            prefix for prefix in self.excluded_prefixes if not any(column.startswith(prefix) for column in all_candidates)
        ]
        if unmatched:
            raise ValueError(f"Excluded feature prefixes matched no candidate columns: {unmatched}")
        candidates = [column for column in all_candidates if not column.startswith(self.excluded_prefixes)]
        if not candidates:
            raise ValueError("No candidate feature columns remain after exclusions")
        unique_counts = train_frame[candidates].nunique(dropna=False)
        self.dropped_constant = unique_counts[unique_counts <= 1].index.tolist()
        self.feature_columns = [c for c in candidates if c not in set(self.dropped_constant)]
        if not self.feature_columns:
            raise ValueError("No feature columns left after removing constant columns")
        return self

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "dropped_constant": self.dropped_constant,
            "excluded_prefixes": list(self.excluded_prefixes),
        }

    def save(self, path: str | Path) -> None:
        _write_json(self.to_dict(), path)

    @staticmethod
    def load_columns(path: str | Path) -> list[str]:
        return list(_read_json(path)["feature_columns"])

    def _check_fitted(self) -> None:
        if self.feature_columns is None:
            raise RuntimeError("FeatureSelector is not fitted")


class Standardizer:
    """Per-feature standardization (x - center) / scale, fitted on one training set.

    Non-finite values are imputed with the fitted median before scaling.
    Features with zero spread use a scale of `zero_scale`.
    """

    method = ""

    def __init__(self, feature_columns: Sequence[str], zero_scale: float = 1.0) -> None:
        if zero_scale <= 0:
            raise ValueError("zero_scale must be positive")
        self.feature_columns = list(feature_columns)
        self.zero_scale = float(zero_scale)
        self.center: np.ndarray | None = None
        self.spread: np.ndarray | None = None
        self.median: np.ndarray | None = None
        self.n_samples: int | None = None

    @property
    def scale(self) -> np.ndarray:
        self._check_fitted()
        return np.where(self.spread == 0, self.zero_scale, self.spread)

    @property
    def zero_spread_columns(self) -> list[str]:
        self._check_fitted()
        return [c for c, value in zip(self.feature_columns, self.spread) if value == 0]

    def fit(self, frame: pd.DataFrame) -> "Standardizer":
        values = self._finite_values(frame)
        if values.shape[0] == 0:
            raise ValueError(f"Cannot fit {type(self).__name__} on an empty frame")
        empty = np.isnan(values).all(axis=0)
        if empty.any():
            columns = [c for c, flag in zip(self.feature_columns, empty) if flag]
            raise ValueError(f"Columns without finite values: {columns}")
        self.center, self.spread = self._statistics(values)
        self.median = np.nanmedian(values, axis=0)
        self.n_samples = int(values.shape[0])
        return self

    def transform(self, frame: pd.DataFrame, dtype: str = "float32") -> np.ndarray:
        self._check_fitted()
        values = self._finite_values(frame)
        values = np.where(np.isnan(values), self.median, values)
        return ((values - self.center) / self.scale).astype(dtype)

    def fit_transform(self, frame: pd.DataFrame, dtype: str = "float32") -> np.ndarray:
        return self.fit(frame).transform(frame, dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "method": self.method,
            **self._options(),
            "zero_scale": self.zero_scale,
            "n_samples": self.n_samples,
            "feature_columns": self.feature_columns,
            "center": self.center.tolist(),
            "spread": self.spread.tolist(),
            "median": self.median.tolist(),
            "zero_spread_columns": self.zero_spread_columns,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Standardizer":
        scaler_class = SCALERS.get(payload["method"])
        if scaler_class is None:
            raise ValueError(f"Unknown scaler method: {payload['method']}")
        scaler = scaler_class._from_options(payload)
        scaler.center = np.asarray(payload["center"], dtype=float)
        scaler.spread = np.asarray(payload["spread"], dtype=float)
        scaler.median = np.asarray(payload["median"], dtype=float)
        scaler.n_samples = int(payload["n_samples"])
        return scaler

    def save(self, path: str | Path) -> None:
        _write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        return Standardizer.from_dict(_read_json(path))

    def _statistics(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _options(self) -> dict[str, Any]:
        return {}

    @classmethod
    def _from_options(cls, payload: Mapping[str, Any]) -> "Standardizer":
        return cls(payload["feature_columns"], zero_scale=payload["zero_scale"])

    def _finite_values(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        values = frame[self.feature_columns].to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isfinite(values), values, np.nan)

    def _check_fitted(self) -> None:
        if self.center is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")


class ZScoreStandardizer(Standardizer):
    """z-score: (x - mean) / standard deviation (population, ddof = 0)."""

    method = "zscore"

    def _statistics(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.nanmean(values, axis=0), np.nanstd(values, axis=0)


class RobustStandardizer(Standardizer):
    """Robust z-score: (x - median) / IQR."""

    method = "robust_zscore"

    def __init__(
        self,
        feature_columns: Sequence[str],
        quantile_range: tuple[float, float] = (25.0, 75.0),
        zero_scale: float = 1.0,
    ) -> None:
        if not 0.0 <= quantile_range[0] < quantile_range[1] <= 100.0:
            raise ValueError(f"Invalid quantile_range: {quantile_range}")
        super().__init__(feature_columns, zero_scale=zero_scale)
        self.quantile_range = (float(quantile_range[0]), float(quantile_range[1]))

    def _statistics(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = np.nanpercentile(values, self.quantile_range, axis=0)
        return np.nanmedian(values, axis=0), upper - lower

    def _options(self) -> dict[str, Any]:
        return {"quantile_range": list(self.quantile_range)}

    @classmethod
    def _from_options(cls, payload: Mapping[str, Any]) -> "RobustStandardizer":
        return cls(
            payload["feature_columns"],
            quantile_range=tuple(payload["quantile_range"]),
            zero_scale=payload["zero_scale"],
        )


SCALERS: dict[str, type[Standardizer]] = {
    ZScoreStandardizer.method: ZScoreStandardizer,
    RobustStandardizer.method: RobustStandardizer,
}


def build_scaler(feature_columns: Sequence[str], preprocessing: Mapping[str, Any]) -> Standardizer:
    method = preprocessing["scaler"]
    zero_scale = float(preprocessing["zero_scale"])
    if method == ZScoreStandardizer.method:
        return ZScoreStandardizer(feature_columns, zero_scale=zero_scale)
    if method == RobustStandardizer.method:
        return RobustStandardizer(
            feature_columns, quantile_range=tuple(preprocessing["quantile_range"]), zero_scale=zero_scale
        )
    raise ValueError(f"Unknown scaler method: {method}")


def extract_labels(frame: pd.DataFrame, schema: DataSchema) -> np.ndarray:
    labels = frame[schema.label_column]
    allowed = {schema.normal_label, schema.anomaly_label}
    if labels.isna().any() or not set(labels.unique()) <= allowed:
        raise ValueError(f"Labels must be in {sorted(allowed)}")
    return labels.to_numpy(dtype=np.int64)


def to_tensor(values: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(values))


def scaler_compatibility_report(scalers: Mapping[str, Standardizer], reference: Standardizer) -> pd.DataFrame:
    """Compare local scalers with a reference scaler fitted on the pooled training data.

    center_shift is expressed in reference scale units; scale_ratio is local scale / reference scale.
    """
    reference_scale = reference.scale
    rows = []
    for name, scaler in scalers.items():
        if scaler.feature_columns != reference.feature_columns:
            raise ValueError(f"Scaler {name} has a different feature order than the reference")
        if scaler.method != reference.method:
            raise ValueError(f"Scaler {name} uses {scaler.method}, reference uses {reference.method}")
        rows.append(
            pd.DataFrame(
                {
                    "client": name,
                    "feature": scaler.feature_columns,
                    "n_samples": scaler.n_samples,
                    "center": scaler.center,
                    "spread": scaler.spread,
                    "reference_center": reference.center,
                    "reference_spread": reference.spread,
                    "center_shift": (scaler.center - reference.center) / reference_scale,
                    "scale_ratio": scaler.scale / reference_scale,
                    "zero_spread": scaler.spread == 0,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def fit_feature_columns(config_path: str | Path) -> FeatureSelector:
    config = load_yaml(config_path)
    schema = DataSchema.from_config(config)
    frame = pd.read_csv(resolve_path(config["raw_path"]))
    train = PartitionStore(config["outputs"]["splits_dir"]).rows(frame, "train")
    selector = FeatureSelector(
        schema,
        excluded_prefixes=config["preprocessing"].get("exclude_feature_prefixes", ()),
    ).fit(train)
    selector.save(resolve_path(config["preprocessing"]["feature_columns_path"]))
    return selector


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the feature columns on the training partition.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    fit_feature_columns(args.config)


if __name__ == "__main__":
    main()
