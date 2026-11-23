import numpy as np
import pandas as pd

from src.data.schema import DataSchema
from src.data.shared_client_holdout import build_shared_client_holdout_splits
from src.data.split import PartitionRatios

CONFIG = {
    "columns": {
        "source": "source",
        "label": "anomaly_label",
        "metadata": ["source", "anomaly_label"],
    },
    "labels": {"normal": 0, "anomaly": 1},
    "categories": [
        {
            "name": "SPLIT_REAL",
            "family": "live_real",
            "source_pattern": "live\\d+",
            "assignment_unit": "source",
        },
        {
            "name": "SPLIT_CAT",
            "family": "synthetic_cat",
            "source_pattern": "synthetic_cat",
            "assignment_unit": "record",
        },
        {
            "name": "SPLIT_GRAY",
            "family": "synthetic_gray",
            "source_pattern": "synthetic_gray",
            "assignment_unit": "record",
        },
    ],
}


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sources = [f"live{vehicle}" for vehicle in range(1, 43) for _ in range(5)]
    sources += ["synthetic_cat"] * 150 + ["synthetic_gray"] * 90
    labels = [0] * 210 + [1] * 150 + [0] * 15 + [1] * 75
    return pd.DataFrame(
        {
            "x": rng.normal(size=len(sources)),
            "source": sources,
            "anomaly_label": labels,
        }
    )


def test_shared_holdout_fixes_validation_and_test_cohorts():
    frame = _frame()
    schema = DataSchema.from_config(CONFIG)
    index, assignments = build_shared_client_holdout_splits(
        frame,
        schema,
        {
            "iid": [1] * 6,
            "quantity_skew": [1, 1, 2, 2, 4, 4],
        },
        PartitionRatios(train=0.70, validation=0.15, test=0.15),
        n_clients=6,
        seed=42,
    )

    assert index["row_id"].is_unique
    assert set(index["row_id"]) == set(frame.index)
    expected_test_rows = set(index.loc[index["partition"] == "test", "row_id"])
    assert set(assignments["iid"]["test"]["row_id"]) == expected_test_rows

    for partition in ("validation", "test"):
        iid = assignments["iid"][partition].set_index("row_id")["client"].sort_index()
        quantity = (
            assignments["quantity_skew"][partition]
            .set_index("row_id")["client"]
            .sort_index()
        )
        pd.testing.assert_series_equal(iid, quantity)

        labeled = assignments["iid"][partition].copy()
        labeled["label"] = frame.loc[labeled["row_id"], "anomaly_label"].to_numpy()
        counts = labeled.groupby(["client", "label"]).size().unstack(fill_value=0)
        assert (counts[0] > 0).all()
        assert (counts[1] > 0).all()
        assert (counts[1] / counts.sum(axis=1)).between(0.45, 0.55).all()

    for scenario in assignments:
        for partition, assignment in assignments[scenario].items():
            expected = set(index.loc[index["partition"] == partition, "row_id"])
            assert assignment["row_id"].is_unique
            assert set(assignment["row_id"]) == expected

    real = index[index["category"] == "SPLIT_REAL"]
    assert real.groupby("group_id")["partition"].nunique().eq(1).all()
    for scenario in assignments:
        all_assignments = pd.concat(assignments[scenario].values(), ignore_index=True)
        joined_real = real.merge(
            all_assignments[["row_id", "client"]],
            on="row_id",
            validate="one_to_one",
        )
        assert joined_real.groupby("group_id")["client"].nunique().eq(1).all()
