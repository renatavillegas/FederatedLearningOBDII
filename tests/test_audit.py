import numpy as np
import pandas as pd
import pytest

from src.data.audit import UNMAPPED, DatasetAuditor
from src.data.schema import DataSchema

CONFIG = {
    "columns": {
        "source": "source",
        "label": "anomaly_label",
        "metadata": ["source", "anomaly_label", "is_gray", "TTF_km"],
    },
    "labels": {"normal": 0, "anomaly": 1},
    "categories": [
        {"name": "SPLIT_REAL", "family": "live_real", "source_pattern": "live\\d+", "assignment_unit": "source"},
        {"name": "SPLIT_CAT", "family": "synthetic_cat", "source_pattern": "synthetic_cat", "assignment_unit": "record"},
    ],
}


@pytest.fixture
def schema() -> DataSchema:
    return DataSchema.from_config(CONFIG)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 5.0],
            "a_std": [0.0] * 6,
            "b_max": [1.0, np.inf, 3.0, 4.0, 5.0, 5.0],
            "source": ["live2", "live10", "live2", "synthetic_cat", "synthetic_cat", "other"],
            "anomaly_label": [0, 0, 1, 1, 1, 0],
            "is_gray": [False, False, False, False, True, False],
            "TTF_km": [np.nan, np.nan, np.nan, 1.0, 2.0, np.nan],
        }
    )


def test_candidate_features_exclude_metadata(schema, frame):
    assert schema.candidate_feature_columns(frame) == ["a_mean", "a_std", "b_max"]


def test_category_assignment(schema, frame):
    categories = schema.assign_categories(frame["source"]).fillna(UNMAPPED).tolist()
    assert categories == ["SPLIT_REAL", "SPLIT_REAL", "SPLIT_REAL", "SPLIT_CAT", "SPLIT_CAT", UNMAPPED]


def test_ambiguous_category_raises(frame):
    config = {**CONFIG, "categories": CONFIG["categories"] + [
        {"name": "DUP", "family": "dup", "source_pattern": "live.*", "assignment_unit": "source"}
    ]}
    with pytest.raises(ValueError):
        DataSchema.from_config(config).assign_categories(frame["source"])


def test_summary_counts(schema, frame):
    summary = DatasetAuditor(frame, schema).summary()
    assert summary["n_rows"] == 6
    assert summary["missing_values"]["total"] == 4
    assert summary["missing_values"]["in_features"] == 0
    assert summary["infinite_values_in_features"]["total"] == 1
    assert summary["constant_feature_columns"] == ["a_std"]
    assert summary["duplicates"]["feature_rows"] == 1
    assert summary["categories"]["SPLIT_REAL"]["groups_with_both_classes"] == ["live2"]
    assert summary["categories"][UNMAPPED]["n_records"] == 1


def test_group_summary_order_and_counts(schema, frame):
    groups = DatasetAuditor(frame, schema).group_summary()
    assert groups["source"].tolist() == ["live2", "live10", "synthetic_cat", "other"]
    live2 = groups.set_index("source").loc["live2"]
    assert (live2["n_records"], live2["n_normal"], live2["n_anomaly"]) == (2, 1, 1)
    assert pd.isna(groups.set_index("source").loc["synthetic_cat", "group_id"])
