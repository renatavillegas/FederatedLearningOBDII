import argparse
import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_yaml, resolve_path
from src.data.schema import DataSchema, natural_sort_key

PARTITIONS = ("train", "validation", "test")
INDEX_COLUMNS = ["row_id", "group_id", "category", "partition"]
MANIFEST_COLUMNS = [
    "group_id",
    "category",
    "family",
    "vehicle_id",
    "derived_origin_id",
    "n_records",
    "n_normal",
    "n_anomaly",
    "partition",
    "client",
]
CONDITIONS = ("all_classes", "all_categories")


def allocate_counts(total: int, ratios: Sequence[float]) -> list[int]:
    """Split an integer total by ratios using the largest remainder method."""
    raw = np.asarray(ratios, dtype=float) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts.tolist()


@dataclass(frozen=True)
class PartitionRatios:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = self.as_list()
        if any(value <= 0 for value in values) or not np.isclose(sum(values), 1.0):
            raise ValueError(f"Partition ratios must be positive and sum to 1: {values}")

    def as_list(self) -> list[float]:
        return [self.train, self.validation, self.test]


class GroupPartitioner:
    """Assigns every record to train, validation or test.

    Categories with assignment_unit "source" are split by whole sources (vehicles).
    Categories with assignment_unit "record" are split record by record, stratified by label.
    """

    def __init__(self, schema: DataSchema, ratios: PartitionRatios, seed: int) -> None:
        self.schema = schema
        self.ratios = ratios
        self.seed = seed

    def assign(self, frame: pd.DataFrame) -> pd.DataFrame:
        categories = self.schema.assign_categories(frame[self.schema.source_column])
        if categories.isna().any():
            unmapped = sorted(frame.loc[categories.isna(), self.schema.source_column].astype(str).unique())
            raise ValueError(f"Sources without category: {unmapped}")

        parts = []
        for spec in self.schema.categories:
            subset = frame[categories == spec.name]
            if subset.empty:
                raise ValueError(f"Category {spec.name} has no records")
            rng = np.random.default_rng([self.seed, zlib.crc32(spec.name.encode())])
            if spec.assignment_unit == "source":
                parts.append(self._assign_by_source(subset, spec.name, rng))
            else:
                parts.append(self._assign_by_record(subset, spec.name, rng))
        index = pd.concat(parts, ignore_index=True)
        return index.sort_values("row_id").reset_index(drop=True)[INDEX_COLUMNS]

    def _assign_by_source(self, subset: pd.DataFrame, category: str, rng: np.random.Generator) -> pd.DataFrame:
        sources = subset[self.schema.source_column].astype(str)
        groups = np.array(sorted(sources.unique()))
        rng.shuffle(groups)
        counts = allocate_counts(len(groups), self.ratios.as_list())
        mapping = {}
        start = 0
        for partition, count in zip(PARTITIONS, counts):
            mapping.update({group: partition for group in groups[start:start + count]})
            start += count
        return pd.DataFrame(
            {
                "row_id": subset.index.to_numpy(),
                "group_id": sources.to_numpy(),
                "category": category,
                "partition": sources.map(mapping).to_numpy(),
            }
        )

    def _assign_by_record(self, subset: pd.DataFrame, category: str, rng: np.random.Generator) -> pd.DataFrame:
        partition = pd.Series(index=subset.index, dtype="object")
        labels = subset[self.schema.label_column]
        for label in sorted(labels.unique()):
            row_ids = labels.index[labels == label].to_numpy().copy()
            rng.shuffle(row_ids)
            counts = allocate_counts(len(row_ids), self.ratios.as_list())
            start = 0
            for name, count in zip(PARTITIONS, counts):
                partition.loc[row_ids[start:start + count]] = name
                start += count
        return pd.DataFrame(
            {
                "row_id": subset.index.to_numpy(),
                "group_id": [f"{category}_{row_id}" for row_id in subset.index],
                "category": category,
                "partition": partition.to_numpy(),
            }
        )


class PartitionValidator:
    """Integrity checks and descriptive statistics of a partition index."""

    def __init__(self, schema: DataSchema, required_conditions: Sequence[str]) -> None:
        unknown = set(required_conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}")
        self.schema = schema
        self.required_conditions = list(required_conditions)

    def report(self, index: pd.DataFrame, frame: pd.DataFrame, feature_columns: Sequence[str]) -> dict[str, Any]:
        labeled = index.assign(label=frame.loc[index["row_id"], self.schema.label_column].to_numpy())
        checks = {
            "every_row_assigned_once": self._every_row_assigned_once(index, frame),
            "groups_in_single_partition": self._groups_in_single_partition(index),
            "all_classes": self._all_present(labeled, "label", [self.schema.normal_label, self.schema.anomaly_label]),
            "all_categories": self._all_present(labeled, "category", self.schema.category_names),
        }
        return {
            "checks": checks,
            "required_conditions": self.required_conditions,
            "records_per_partition": self._counts(labeled, ["partition"]),
            "records_per_partition_and_class": self._counts(labeled, ["partition", "label"]),
            "records_per_partition_and_category": self._counts(labeled, ["partition", "category"]),
            "records_per_partition_category_and_class": self._counts(labeled, ["partition", "category", "label"]),
            "groups_per_partition_and_category": self._group_counts(labeled),
            "anomaly_fraction_per_partition": labeled.groupby("partition")["label"]
            .apply(lambda labels: float((labels == self.schema.anomaly_label).mean()))
            .to_dict(),
            "feature_rows_shared_across_partitions": self._shared_feature_rows(labeled, frame, feature_columns),
            "derived_synthetic_relations": "not verifiable: synthetic records have no origin identifier",
        }

    def validate(self, report: Mapping[str, Any]) -> None:
        checks = report["checks"]
        failed = [name for name in ("every_row_assigned_once", "groups_in_single_partition") if not checks[name]]
        failed += [name for name in self.required_conditions if not checks[name]]
        if failed:
            raise ValueError(f"Partition checks failed: {failed}")

    @staticmethod
    def _every_row_assigned_once(index: pd.DataFrame, frame: pd.DataFrame) -> bool:
        return bool(
            index["row_id"].is_unique
            and set(index["row_id"]) == set(frame.index)
            and index["partition"].isin(PARTITIONS).all()
        )

    @staticmethod
    def _groups_in_single_partition(index: pd.DataFrame) -> bool:
        return bool((index.groupby("group_id")["partition"].nunique() == 1).all())

    @staticmethod
    def _all_present(labeled: pd.DataFrame, column: str, expected: Sequence[Any]) -> bool:
        present = labeled.groupby("partition")[column].agg(set)
        return all(partition in present and set(expected) <= present[partition] for partition in PARTITIONS)

    @staticmethod
    def _counts(labeled: pd.DataFrame, keys: list[str]) -> dict[str, int]:
        counts = labeled.groupby(keys).size()
        return {"/".join(str(k) for k in np.atleast_1d(key)): int(value) for key, value in counts.items()}

    @staticmethod
    def _group_counts(labeled: pd.DataFrame) -> dict[str, int]:
        counts = labeled.groupby(["partition", "category"])["group_id"].nunique()
        return {f"{partition}/{category}": int(value) for (partition, category), value in counts.items()}

    @staticmethod
    def _shared_feature_rows(labeled: pd.DataFrame, frame: pd.DataFrame, feature_columns: Sequence[str]) -> int:
        features = frame.loc[labeled["row_id"], list(feature_columns)].reset_index(drop=True)
        features["partition"] = labeled["partition"].to_numpy()
        partitions_per_row = features.groupby(list(feature_columns), dropna=False)["partition"].nunique()
        return int((partitions_per_row > 1).sum())


def build_group_manifest(index: pd.DataFrame, frame: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    labels = frame.loc[index["row_id"], schema.label_column].to_numpy()
    labeled = index.assign(label=labels)
    manifest = labeled.groupby(["group_id", "category", "partition"], sort=False).agg(
        n_records=("row_id", "size"),
        n_normal=("label", lambda values: int((values == schema.normal_label).sum())),
        n_anomaly=("label", lambda values: int((values == schema.anomaly_label).sum())),
    ).reset_index()
    units = {spec.name: spec.assignment_unit for spec in schema.categories}
    families = {spec.name: spec.family for spec in schema.categories}
    manifest["family"] = manifest["category"].map(families)
    manifest["vehicle_id"] = manifest["group_id"].where(manifest["category"].map(units) == "source")
    manifest["derived_origin_id"] = pd.NA
    manifest["client"] = pd.NA
    category_order = {name: position for position, name in enumerate(schema.category_names)}
    manifest = manifest.sort_values(
        ["category", "group_id"],
        key=lambda column: column.map(category_order) if column.name == "category" else column.map(natural_sort_key),
    )
    return manifest[MANIFEST_COLUMNS].reset_index(drop=True)


class PartitionStore:
    """Reads and writes the partition index in data/splits/."""

    def __init__(self, splits_dir: str | Path) -> None:
        self.path = resolve_path(splits_dir) / "partition_index.csv"

    def save(self, index: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        index[INDEX_COLUMNS].to_csv(self.path, index=False)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def rows(self, frame: pd.DataFrame, partition: str, index: pd.DataFrame | None = None) -> pd.DataFrame:
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        index = self.load() if index is None else index
        row_ids = index.loc[index["partition"] == partition, "row_id"]
        return frame.loc[row_ids]


def run_split(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    schema = DataSchema.from_config(config)
    split_config = config["split"]
    frame = pd.read_csv(resolve_path(config["raw_path"]))

    partitioner = GroupPartitioner(schema, PartitionRatios(**split_config["ratios"]), int(config["seed"]))
    index = partitioner.assign(frame)

    validator = PartitionValidator(schema, split_config["required_conditions"])
    report = validator.report(index, frame, schema.candidate_feature_columns(frame))
    report["seed"] = int(config["seed"])
    report["ratios"] = split_config["ratios"]

    outputs = config["outputs"]
    report_dir = resolve_path(outputs["split_report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    validator.validate(report)

    PartitionStore(outputs["splits_dir"]).save(index)
    manifests_dir = resolve_path(outputs["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)
    build_group_manifest(index, frame, schema).to_csv(manifests_dir / "groups.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the dataset into train, validation and test.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    run_split(args.config)


if __name__ == "__main__":
    main()
