import numpy as np
import pandas as pd
import pytest

from src.data.clients import (
    ClientAllocator,
    ClientDistributionReporter,
    ClientStore,
    ScenarioBuilder,
    ScenarioSpec,
    TrainPools,
    client_group_manifest,
)
from src.data.schema import DataSchema
from src.experiments.common import select_normal_real_records

CONFIG = {
    "columns": {"source": "source", "label": "anomaly_label", "metadata": ["source", "anomaly_label"]},
    "labels": {"normal": 0, "anomaly": 1},
    "categories": [
        {"name": "SPLIT_REAL", "family": "live_real", "source_pattern": "live\\d+", "assignment_unit": "source"},
        {"name": "SPLIT_CAT", "family": "synthetic_cat", "source_pattern": "synthetic_cat", "assignment_unit": "record"},
        {"name": "SPLIT_GRAY", "family": "synthetic_gray", "source_pattern": "synthetic_gray", "assignment_unit": "record"},
    ],
}
N_CLIENTS = 3
IID = ScenarioSpec("iid", (0.5, 0.5, 0.5), required_band=(0.45, 0.55))
NON_IID = ScenarioSpec("non_iid", (0.3, 0.5, 0.7), reference_scenario="iid")


@pytest.fixture
def schema() -> DataSchema:
    return DataSchema.from_config(CONFIG)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vehicle_sizes = [40, 35, 30, 30, 25, 20, 20, 15, 10]
    sources = [f"live{i + 1}" for i, size in enumerate(vehicle_sizes) for _ in range(size)]
    labels = [0] * len(sources)
    sources += ["synthetic_cat"] * 150 + ["synthetic_gray"] * 30 + ["synthetic_gray"] * 100
    labels += [1] * 150 + [0] * 30 + [1] * 100
    return pd.DataFrame(
        {"x": rng.normal(size=len(sources)), "y": rng.normal(size=len(sources)), "source": sources, "anomaly_label": labels}
    )


@pytest.fixture
def train_index(schema, frame) -> pd.DataFrame:
    categories = schema.assign_categories(frame["source"])
    group_id = np.where(
        categories == "SPLIT_REAL", frame["source"], categories.astype(str) + "_" + frame.index.astype(str)
    )
    return pd.DataFrame(
        {"row_id": frame.index, "group_id": group_id, "category": categories.astype(str), "partition": "train"}
    )


@pytest.fixture
def pools(schema, frame, train_index) -> TrainPools:
    return TrainPools.from_index(train_index, frame["anomaly_label"], schema)


@pytest.fixture
def assignments(pools) -> dict[str, pd.DataFrame]:
    return ScenarioBuilder(ClientAllocator(N_CLIENTS), seed=42).build([NON_IID, IID], pools)


def fractions(assignment: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    labels = frame.loc[assignment["row_id"], "anomaly_label"].to_numpy()
    return assignment.assign(label=labels).groupby("client")["label"].mean()


def test_pools(pools):
    assert len(pools.vehicle_rows) == 9
    assert pools.n_normal == 225 + 30
    assert pools.n_anomaly == 250
    assert set(pools.normal_pools) == {"SPLIT_GRAY/0"}
    assert set(pools.anomaly_pools) == {"SPLIT_CAT/1", "SPLIT_GRAY/1"}


def test_normal_real_selection_excludes_anomalies_and_synthetic_records(schema, frame):
    selected = select_normal_real_records(frame, schema)
    assert not selected.empty
    assert selected["source"].str.fullmatch(r"live\d+").all()
    assert (selected["anomaly_label"] == schema.normal_label).all()


def test_normal_real_selection_fails_without_matching_rows(schema, frame):
    only_synthetic = frame.loc[frame["source"].str.startswith("synthetic")]
    with pytest.raises(ValueError, match="No normal real records"):
        select_normal_real_records(only_synthetic, schema)


def test_vehicle_with_anomaly_is_rejected(schema, frame, train_index):
    frame.loc[0, "anomaly_label"] = 1
    with pytest.raises(ValueError):
        TrainPools.from_index(train_index, frame["anomaly_label"], schema)


@pytest.mark.parametrize("scenario", ["iid", "non_iid"])
def test_every_train_row_assigned_once(assignments, train_index, scenario):
    assignment = assignments[scenario]
    assert assignment["row_id"].is_unique
    assert set(assignment["row_id"]) == set(train_index["row_id"])


@pytest.mark.parametrize("scenario", ["iid", "non_iid"])
def test_vehicles_in_single_client(assignments, frame, scenario):
    assignment = assignments[scenario]
    sources = frame.loc[assignment["row_id"], "source"].to_numpy()
    real = assignment.assign(source=sources)
    real = real[real["source"].str.startswith("live")]
    assert (real.groupby("source")["client"].nunique() == 1).all()


def test_iid_fractions_in_band(assignments, frame):
    assert fractions(assignments["iid"], frame).between(0.45, 0.55).all()


def test_non_iid_fractions_follow_targets(assignments, frame):
    achieved = fractions(assignments["non_iid"], frame).sort_index()
    assert achieved.is_monotonic_increasing
    assert achieved.iloc[0] < 0.45 and achieved.iloc[-1] > 0.55


def test_normal_count_weights_create_quantity_skew(pools, frame):
    scenario = ScenarioSpec(
        "quantity_skew",
        (0.5, 0.5, 0.5),
        normal_count_weights=(1.0, 2.0, 4.0),
    )
    assignment = ScenarioBuilder(ClientAllocator(N_CLIENTS), seed=42).build([scenario], pools)["quantity_skew"]
    counts = assignment.groupby("client").size().sort_index()
    assert counts.is_monotonic_increasing
    assert counts.iloc[0] < counts.iloc[-1]
    assert assignment["row_id"].is_unique
    assert set(assignment["row_id"]) == set(frame.index)


def test_synthetic_origin_is_proportional(assignments, frame):
    assignment = assignments["iid"]
    sources = frame.loc[assignment["row_id"], "source"].to_numpy()
    labels = frame.loc[assignment["row_id"], "anomaly_label"].to_numpy()
    anomalies = assignment.assign(source=sources, label=labels).query("label == 1")
    cat_share = anomalies.groupby("client")["source"].apply(lambda s: (s == "synthetic_cat").mean())
    assert np.allclose(cat_share, 150 / 250, atol=0.02)


def test_build_is_reproducible(pools, assignments):
    again = ScenarioBuilder(ClientAllocator(N_CLIENTS), seed=42).build([IID, NON_IID], pools)
    for name in ("iid", "non_iid"):
        pd.testing.assert_frame_equal(assignments[name], again[name])


def test_unknown_reference_raises(pools):
    spec = ScenarioSpec("x", (0.5, 0.5, 0.5), reference_scenario="missing")
    with pytest.raises(ValueError):
        ScenarioBuilder(ClientAllocator(N_CLIENTS), seed=42).build([spec], pools)


def test_invalid_fraction_raises():
    with pytest.raises(ValueError):
        ScenarioSpec("x", (0.0, 0.5, 1.0))


def test_reporter_summary_and_checks(schema, frame, train_index, assignments):
    reporter = ClientDistributionReporter(schema, ["x", "y"])
    summary = reporter.client_summary(assignments["iid"], train_index, frame, IID)
    assert summary["client"].tolist() == ["client_1", "client_2", "client_3"]
    assert summary["n_records"].sum() == len(frame)
    assert summary["n_vehicles"].sum() == 9
    checks = reporter.check(assignments["iid"], train_index, summary, IID)
    assert all(checks.values())


def test_reporter_detects_non_train_rows(schema, frame, train_index, assignments):
    reporter = ClientDistributionReporter(schema, ["x", "y"])
    reduced_index = train_index.iloc[1:]
    summary = reporter.client_summary(assignments["iid"], train_index, frame, IID)
    checks = reporter.check(assignments["iid"], reduced_index, summary, IID)
    assert not checks["rows_only_from_train"]


def test_ks_table(schema, frame, assignments):
    reporter = ClientDistributionReporter(schema, ["x", "y"])
    ks = reporter.ks_table(assignments["iid"], frame)
    assert len(ks) == N_CLIENTS * 2
    assert ks["ks_statistic"].between(0, 1).all()
    assert set(reporter.ks_summary(ks)) == {"client_1", "client_2", "client_3"}


def test_client_store_roundtrip(tmp_path, frame, assignments):
    store = ClientStore(tmp_path)
    store.save("iid", assignments["iid"])
    pd.testing.assert_frame_equal(store.load("iid"), assignments["iid"], check_dtype=False)
    rows = store.rows(frame, "iid", "client_1")
    assert set(rows.index) == set(assignments["iid"].query("client == 'client_1'")["row_id"])


def test_client_group_manifest(train_index, assignments):
    groups = train_index.drop_duplicates("group_id")[["group_id", "category", "partition"]].assign(client=pd.NA)
    manifest = client_group_manifest(groups, assignments["iid"], train_index)
    assert manifest["client"].notna().all()
