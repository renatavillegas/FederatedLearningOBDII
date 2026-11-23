import numpy as np
import pandas as pd
import pytest

from src.data.client_holdout import build_client_holdout_split
from src.data.schema import DataSchema
from src.data.split import PartitionRatios

CONFIG = {
    "columns": {"source": "source", "label": "anomaly_label", "metadata": ["source", "anomaly_label"]},
    "labels": {"normal": 0, "anomaly": 1},
    "categories": [
        {"name": "SPLIT_REAL", "family": "live_real", "source_pattern": "live\\d+", "assignment_unit": "source"},
        {"name": "SPLIT_CAT", "family": "synthetic_cat", "source_pattern": "synthetic_cat", "assignment_unit": "record"},
        {"name": "SPLIT_GRAY", "family": "synthetic_gray", "source_pattern": "synthetic_gray", "assignment_unit": "record"},
    ],
}


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sources = [f"live{vehicle}" for vehicle in range(1, 25) for _ in range(2)]
    sources += ["synthetic_cat"] * 28 + ["synthetic_gray"] * 60
    labels = [0] * 48 + [1] * 28 + [0] * 20 + [1] * 40
    return pd.DataFrame(
        {
            "x": rng.normal(size=len(sources)),
            "source": sources,
            "anomaly_label": labels,
        }
    )


@pytest.mark.parametrize(
    ("scenario", "weights"),
    [
        ("iid", [1, 1, 1, 1, 1, 1]),
        ("quantity_skew", [1, 1, 2, 2, 4, 4]),
    ],
)
def test_client_holdout_keeps_rows_and_vehicles_separated(scenario, weights):
    frame = _frame()
    schema = DataSchema.from_config(CONFIG)
    index, mapping = build_client_holdout_split(
        frame,
        schema,
        scenario,
        {"anomaly_fraction": [0.5] * 6, "normal_count_weights": weights},
        n_clients=6,
        ratios=PartitionRatios(train=0.70, validation=0.15, test=0.15),
        seed=42,
    )

    assert index["row_id"].is_unique
    assert mapping["row_id"].is_unique
    assert set(index["row_id"]) == set(frame.index)
    assert set(mapping["row_id"]) == set(frame.index)

    joined = index.merge(mapping[["row_id", "client"]], on="row_id", validate="one_to_one")
    real = joined[joined["category"] == "SPLIT_REAL"]
    assert real.groupby("group_id")["partition"].nunique().eq(1).all()
    assert real.groupby("group_id")["client"].nunique().eq(1).all()

    test = joined[joined["partition"] == "test"].copy()
    test["label"] = frame.loc[test["row_id"], "anomaly_label"].to_numpy()
    counts = test.groupby(["client", "label"]).size().unstack(fill_value=0)
    assert (counts[0] > 0).all()
    assert (counts[1] > 0).all()
    assert (counts[1] / counts.sum(axis=1)).between(0.35, 0.65).all()
    assert real[real["partition"] == "test"].groupby("client")["group_id"].nunique().eq(1).all()
