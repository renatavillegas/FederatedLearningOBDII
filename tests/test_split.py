import numpy as np
import pandas as pd
import pytest

from src.data.schema import DataSchema
from src.data.split import (
    GroupPartitioner,
    PartitionRatios,
    PartitionStore,
    PartitionValidator,
    allocate_counts,
    build_group_manifest,
)

CONFIG = {
    "columns": {"source": "source", "label": "anomaly_label", "metadata": ["source", "anomaly_label"]},
    "labels": {"normal": 0, "anomaly": 1},
    "categories": [
        {"name": "SPLIT_REAL", "family": "live_real", "source_pattern": "live\\d+", "assignment_unit": "source"},
        {"name": "SPLIT_CAT", "family": "synthetic_cat", "source_pattern": "synthetic_cat", "assignment_unit": "record"},
        {"name": "SPLIT_GRAY", "family": "synthetic_gray", "source_pattern": "synthetic_gray", "assignment_unit": "record"},
    ],
}
RATIOS = PartitionRatios(train=0.70, validation=0.15, test=0.15)


@pytest.fixture
def schema() -> DataSchema:
    return DataSchema.from_config(CONFIG)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    real_sources = [f"live{i}" for i in range(1, 21) for _ in range(5)]
    sources = real_sources + ["synthetic_cat"] * 60 + ["synthetic_gray"] * 60
    labels = [0] * 100 + [1] * 60 + [0] * 20 + [1] * 40
    return pd.DataFrame(
        {
            "x": rng.normal(size=len(sources)),
            "y": rng.normal(size=len(sources)),
            "source": sources,
            "anomaly_label": labels,
        }
    )


@pytest.fixture
def index(schema, frame) -> pd.DataFrame:
    return GroupPartitioner(schema, RATIOS, seed=42).assign(frame)


def test_allocate_counts_sums_to_total():
    assert allocate_counts(39, RATIOS.as_list()) == [27, 6, 6]
    assert sum(allocate_counts(3881, RATIOS.as_list())) == 3881


def test_invalid_ratios():
    with pytest.raises(ValueError):
        PartitionRatios(train=0.8, validation=0.15, test=0.15)


def test_every_row_assigned_once(index, frame):
    assert index["row_id"].is_unique
    assert set(index["row_id"]) == set(frame.index)


def test_vehicles_stay_in_one_partition(index):
    real = index[index["category"] == "SPLIT_REAL"]
    assert (real.groupby("group_id")["partition"].nunique() == 1).all()
    assert real.groupby("partition")["group_id"].nunique().to_dict() == {"test": 3, "train": 14, "validation": 3}


def test_synthetic_split_is_stratified(index, frame):
    gray = index[index["category"] == "SPLIT_GRAY"].assign(
        label=frame.loc[index.loc[index["category"] == "SPLIT_GRAY", "row_id"], "anomaly_label"].to_numpy()
    )
    counts = gray.groupby(["partition", "label"]).size().to_dict()
    assert counts[("train", 0)] == 14 and counts[("train", 1)] == 28
    assert counts[("validation", 0)] == 3 and counts[("test", 0)] == 3


def test_assignment_is_reproducible(schema, frame, index):
    again = GroupPartitioner(schema, RATIOS, seed=42).assign(frame)
    pd.testing.assert_frame_equal(index, again)


def test_different_seed_changes_assignment(schema, frame, index):
    other = GroupPartitioner(schema, RATIOS, seed=7).assign(frame)
    assert not index["partition"].equals(other["partition"])


def test_unmapped_source_raises(schema, frame):
    frame.loc[0, "source"] = "unknown"
    with pytest.raises(ValueError):
        GroupPartitioner(schema, RATIOS, seed=42).assign(frame)


def test_validator_passes(schema, frame, index):
    validator = PartitionValidator(schema, ["all_classes", "all_categories"])
    report = validator.report(index, frame, ["x", "y"])
    validator.validate(report)
    assert all(report["checks"].values())
    assert report["feature_rows_shared_across_partitions"] == 0


def test_validator_detects_group_split_across_partitions(schema, frame, index):
    broken = index.copy()
    rows = broken.index[broken["group_id"] == broken.loc[0, "group_id"]]
    other = "test" if broken.loc[rows[0], "partition"] != "test" else "train"
    broken.loc[rows[0], "partition"] = other
    validator = PartitionValidator(schema, [])
    with pytest.raises(ValueError):
        validator.validate(validator.report(broken, frame, ["x", "y"]))


def test_validator_detects_missing_class(schema, frame, index):
    no_anomaly_in_test = index[~((index["partition"] == "test") & index["category"].isin(["SPLIT_CAT", "SPLIT_GRAY"]))]
    subset = frame.loc[no_anomaly_in_test["row_id"]]
    validator = PartitionValidator(schema, ["all_classes"])
    report = validator.report(no_anomaly_in_test, subset, ["x", "y"])
    assert not report["checks"]["all_classes"]
    with pytest.raises(ValueError):
        validator.validate(report)


def test_shared_feature_rows_are_counted(schema, frame, index):
    duplicated = frame.copy()
    train_row = index.loc[index["partition"] == "train", "row_id"].iloc[0]
    test_row = index.loc[index["partition"] == "test", "row_id"].iloc[0]
    duplicated.loc[test_row, ["x", "y"]] = duplicated.loc[train_row, ["x", "y"]].to_numpy()
    report = PartitionValidator(schema, []).report(index, duplicated, ["x", "y"])
    assert report["feature_rows_shared_across_partitions"] == 1


def test_group_manifest(schema, frame, index):
    manifest = build_group_manifest(index, frame, schema)
    assert manifest["group_id"].is_unique
    assert manifest["n_records"].sum() == len(frame)
    real = manifest[manifest["category"] == "SPLIT_REAL"]
    assert len(real) == 20
    assert (real["vehicle_id"] == real["group_id"]).all()
    assert real["group_id"].tolist()[:3] == ["live1", "live2", "live3"]
    assert manifest.loc[manifest["category"] == "SPLIT_CAT", "vehicle_id"].isna().all()


def test_partition_store_roundtrip(tmp_path, frame, index):
    store = PartitionStore(tmp_path)
    store.save(index)
    pd.testing.assert_frame_equal(store.load(), index, check_dtype=False)
    train = store.rows(frame, "train")
    assert set(train.index) == set(index.loc[index["partition"] == "train", "row_id"])
