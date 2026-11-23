import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml, resolve_path
from src.data.schema import DataSchema, natural_sort_key

UNMAPPED = "UNMAPPED"


class DatasetAuditor:
    """Descriptive audit of the raw dataset. Does not modify the data."""

    def __init__(self, frame: pd.DataFrame, schema: DataSchema) -> None:
        schema.validate_columns(frame)
        self.frame = frame
        self.schema = schema
        self.feature_columns = schema.candidate_feature_columns(frame)
        self.categories = schema.assign_categories(frame[schema.source_column]).fillna(UNMAPPED)

    def summary(self) -> dict[str, Any]:
        return {
            "n_rows": len(self.frame),
            "n_columns": self.frame.shape[1],
            "columns": self._column_summary(),
            "duplicates": self._duplicate_summary(),
            "missing_values": self._missing_summary(),
            "infinite_values_in_features": self._infinite_summary(),
            "constant_feature_columns": self._constant_feature_columns(),
            "labels": self._label_summary(),
            "categories": self._category_summary(),
        }

    def group_summary(self) -> pd.DataFrame:
        source = self.schema.source_column
        label = self.schema.label_column
        frame = pd.DataFrame(
            {
                "category": self.categories,
                "source": self.frame[source].astype(str),
                "label": self.frame[label],
            }
        )
        grouped = frame.groupby(["category", "source"], sort=False)["label"]
        table = grouped.agg(
            n_records="size",
            n_normal=lambda labels: int((labels == self.schema.normal_label).sum()),
            n_anomaly=lambda labels: int((labels == self.schema.anomaly_label).sum()),
        ).reset_index()
        table["n_other_label"] = table["n_records"] - table["n_normal"] - table["n_anomaly"]
        table["family"] = table["category"].map(self._category_attribute("family"))
        table["assignment_unit"] = table["category"].map(self._category_attribute("assignment_unit"))
        table["group_id"] = np.where(
            table["assignment_unit"] == "source", table["source"], pd.NA
        )

        category_order = {name: index for index, name in enumerate(self.schema.category_names)}
        table["_category_order"] = table["category"].map(category_order).fillna(len(category_order))
        table["_source_order"] = table["source"].map(natural_sort_key)
        table = table.sort_values(["_category_order", "_source_order"]).drop(
            columns=["_category_order", "_source_order"]
        )
        columns = [
            "group_id",
            "category",
            "family",
            "source",
            "assignment_unit",
            "n_records",
            "n_normal",
            "n_anomaly",
            "n_other_label",
        ]
        return table[columns].reset_index(drop=True)

    def _category_attribute(self, attribute: str) -> dict[str, str]:
        return {category.name: getattr(category, attribute) for category in self.schema.categories}

    def _column_summary(self) -> dict[str, Any]:
        numeric = [
            column
            for column in self.frame.columns
            if pd.api.types.is_numeric_dtype(self.frame[column])
            and not pd.api.types.is_bool_dtype(self.frame[column])
        ]
        metadata_present = [c for c in self.schema.metadata_columns if c in self.frame.columns]
        metadata_absent = [c for c in self.schema.metadata_columns if c not in self.frame.columns]
        return {
            "n_numeric": len(numeric),
            "non_numeric": [c for c in self.frame.columns if c not in numeric],
            "dtypes": {c: str(t) for c, t in self.frame.dtypes.items()},
            "metadata_present": metadata_present,
            "metadata_absent": metadata_absent,
            "n_candidate_features": len(self.feature_columns),
            "candidate_features": self.feature_columns,
            "non_numeric_non_metadata": self.schema.non_numeric_non_metadata_columns(self.frame),
        }

    def _duplicate_summary(self) -> dict[str, int]:
        return {
            "full_rows": int(self.frame.duplicated().sum()),
            "feature_rows": int(self.frame[self.feature_columns].duplicated().sum()),
        }

    def _missing_summary(self) -> dict[str, Any]:
        per_column = self.frame.isna().sum()
        per_column = per_column[per_column > 0]
        metadata = set(self.schema.metadata_columns)
        return {
            "unit": "cells",
            "total": int(per_column.sum()),
            "in_features": int(per_column[[c for c in per_column.index if c not in metadata]].sum()),
            "in_metadata": int(per_column[[c for c in per_column.index if c in metadata]].sum()),
            "rows_with_any_missing": int(self.frame.isna().any(axis=1).sum()),
            "per_column": {column: int(count) for column, count in per_column.items()},
        }

    def _infinite_summary(self) -> dict[str, Any]:
        values = self.frame[self.feature_columns].to_numpy(dtype=float)
        per_column = pd.Series(np.isinf(values).sum(axis=0), index=self.feature_columns)
        per_column = per_column[per_column > 0]
        return {
            "total": int(per_column.sum()),
            "per_column": {column: int(count) for column, count in per_column.items()},
        }

    def _constant_feature_columns(self) -> list[str]:
        unique_counts = self.frame[self.feature_columns].nunique(dropna=False)
        return unique_counts[unique_counts <= 1].index.tolist()

    def _label_summary(self) -> dict[str, Any]:
        labels = self.frame[self.schema.label_column]
        expected = {self.schema.normal_label, self.schema.anomaly_label}
        counts = labels.value_counts(dropna=False)
        return {
            "counts": {str(value): int(count) for value, count in counts.items()},
            "unexpected_values": sorted(
                str(value) for value in counts.index if pd.isna(value) or value not in expected
            ),
        }

    def _category_summary(self) -> dict[str, Any]:
        groups = self.group_summary()
        result: dict[str, Any] = {}
        for name, table in groups.groupby("category", sort=False):
            entry = {
                "n_records": int(table["n_records"].sum()),
                "n_normal": int(table["n_normal"].sum()),
                "n_anomaly": int(table["n_anomaly"].sum()),
                "n_sources": len(table),
                "sources": table["source"].tolist(),
            }
            if name != UNMAPPED:
                category = self.schema.category(name)
                entry["family"] = category.family
                entry["assignment_unit"] = category.assignment_unit
                if category.assignment_unit == "source":
                    entry["n_groups"] = len(table)
                    mixed = table[(table["n_normal"] > 0) & (table["n_anomaly"] > 0)]
                    entry["groups_with_both_classes"] = mixed["source"].tolist()
            result[name] = entry
        return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(config_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame]:
    config = load_yaml(config_path)
    schema = DataSchema.from_config(config)
    raw_path = resolve_path(config["raw_path"])
    frame = pd.read_csv(raw_path)

    auditor = DatasetAuditor(frame, schema)
    summary = {
        "file": {"path": str(config["raw_path"]), "sha256": file_sha256(raw_path)},
        **auditor.summary(),
    }
    groups = auditor.group_summary()

    output_dir = resolve_path(config["outputs"]["audit_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    groups.to_csv(output_dir / "group_summary.csv", index=False)
    return summary, groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw OBD-II dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    run_audit(args.config)


if __name__ == "__main__":
    main()
