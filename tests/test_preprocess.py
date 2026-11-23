import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    FeatureSelector,
    RobustStandardizer,
    Standardizer,
    ZScoreStandardizer,
    build_scaler,
    extract_labels,
    scaler_compatibility_report,
    to_tensor,
)
from src.data.schema import DataSchema

CONFIG = {
    "columns": {
        "source": "source",
        "label": "anomaly_label",
        "metadata": ["source", "anomaly_label", "is_gray"],
    },
    "labels": {"normal": 0, "anomaly": 1},
    "categories": [
        {"name": "SPLIT_REAL", "family": "live_real", "source_pattern": "live\\d+", "assignment_unit": "source"},
    ],
}


@pytest.fixture
def schema() -> DataSchema:
    return DataSchema.from_config(CONFIG)


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "constant": [7.0] * 5,
            "b": [10.0, 10.0, 10.0, 10.0, 20.0],
            "c": [0.0, 4.0, 8.0, 12.0, 16.0],
            "source": ["live1"] * 5,
            "anomaly_label": [0, 0, 1, 1, 0],
            "is_gray": [False] * 5,
        }
    )


@pytest.fixture
def scaler(schema, train) -> RobustStandardizer:
    columns = FeatureSelector(schema).fit(train).feature_columns
    return RobustStandardizer(columns).fit(train)


def test_selector_drops_constant_and_metadata(schema, train):
    selector = FeatureSelector(schema).fit(train)
    assert selector.feature_columns == ["a", "b", "c"]
    assert selector.dropped_constant == ["constant"]


def test_selector_can_fit_only_normal_training_records(schema, train):
    normal_train = train.loc[train["anomaly_label"] == schema.normal_label]
    selector = FeatureSelector(schema).fit(normal_train)
    assert selector.feature_columns == ["a", "b", "c"]
    assert selector.dropped_constant == ["constant"]


def test_selector_excludes_configured_prefixes(schema):
    frame = pd.DataFrame(
        {
            "sensor_mean": [1.0, 2.0],
            "TIME_RUN_WITH_MIL_ON ()_mean": [3.0, 4.0],
            "source": ["live1", "live1"],
            "anomaly_label": [0, 0],
        }
    )
    selector = FeatureSelector(schema, excluded_prefixes=["TIME_RUN_WITH_MIL_ON ()"]).fit(frame)
    assert selector.feature_columns == ["sensor_mean"]
    assert selector.to_dict()["excluded_prefixes"] == ["TIME_RUN_WITH_MIL_ON ()"]


def test_selector_rejects_empty_exclusion_prefix(schema):
    with pytest.raises(ValueError, match="empty strings"):
        FeatureSelector(schema, excluded_prefixes=[""])


def test_selector_rejects_unmatched_exclusion_prefix(schema, train):
    with pytest.raises(ValueError, match="matched no candidate columns"):
        FeatureSelector(schema, excluded_prefixes=["unknown_group"]).fit(train)


def test_selector_roundtrip(schema, train, tmp_path):
    selector = FeatureSelector(schema).fit(train)
    selector.save(tmp_path / "features.json")
    assert FeatureSelector.load_columns(tmp_path / "features.json") == ["a", "b", "c"]


def test_robust_statistics(scaler):
    np.testing.assert_allclose(scaler.center, [3.0, 10.0, 8.0])
    np.testing.assert_allclose(scaler.spread, [2.0, 0.0, 8.0])
    np.testing.assert_allclose(scaler.scale, [2.0, 1.0, 8.0])
    assert scaler.zero_spread_columns == ["b"]


def test_zscore_statistics(train):
    scaler = ZScoreStandardizer(["a", "b", "c"]).fit(train)
    np.testing.assert_allclose(scaler.center, [3.0, 12.0, 8.0])
    np.testing.assert_allclose(scaler.spread, [np.sqrt(2.0), 4.0, np.sqrt(32.0)])
    np.testing.assert_allclose(scaler.median, [3.0, 10.0, 8.0])
    transformed = scaler.transform(train).astype(float)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-6)


def test_zscore_zero_spread_uses_zero_scale(train):
    scaler = ZScoreStandardizer(["a", "constant"]).fit(train)
    assert scaler.zero_spread_columns == ["constant"]
    np.testing.assert_allclose(scaler.transform(train)[:, 1], 0.0)


def test_zscore_imputes_with_median(train):
    scaler = ZScoreStandardizer(["b"]).fit(train)
    frame = train.copy()
    frame.loc[0, "b"] = np.nan
    assert scaler.transform(frame)[0, 0] == pytest.approx((10.0 - 12.0) / 4.0)


def test_build_scaler_from_config():
    assert isinstance(build_scaler(["a"], {"scaler": "zscore", "zero_scale": 1.0}), ZScoreStandardizer)
    robust = build_scaler(["a"], {"scaler": "robust_zscore", "zero_scale": 1.0, "quantile_range": [10, 90]})
    assert isinstance(robust, RobustStandardizer) and robust.quantile_range == (10.0, 90.0)
    with pytest.raises(ValueError):
        build_scaler(["a"], {"scaler": "minmax", "zero_scale": 1.0})


def test_transform_values_and_dtype(scaler, train):
    transformed = scaler.transform(train)
    assert transformed.dtype == np.float32
    assert transformed.shape == (5, 3)
    np.testing.assert_allclose(transformed[:, 0], [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_column_order_is_independent_of_input_order(scaler, train):
    shuffled = train[["is_gray", "c", "anomaly_label", "a", "source", "b", "constant"]]
    np.testing.assert_array_equal(scaler.transform(train), scaler.transform(shuffled))


def test_partitions_share_feature_order(scaler, train):
    validation = train.sample(frac=1.0, random_state=0)[list(reversed(train.columns))]
    test = train.iloc[:2][list(train.columns[::2]) + list(train.columns[1::2])]
    expected = scaler.transform(train)
    np.testing.assert_array_equal(scaler.transform(validation), expected[validation.index])
    np.testing.assert_array_equal(scaler.transform(test), expected[:2])


def test_missing_feature_column_raises(scaler, train):
    with pytest.raises(KeyError):
        scaler.transform(train.drop(columns=["b"]))


def test_non_finite_values_are_imputed_with_median(scaler, train):
    frame = train.copy()
    frame.loc[0, "a"] = np.nan
    frame.loc[1, "c"] = np.inf
    transformed = scaler.transform(frame)
    assert transformed[0, 0] == 0.0
    assert transformed[1, 2] == 0.0


def test_fit_ignores_non_finite_values(train):
    frame = train.copy()
    frame.loc[4, "a"] = np.nan
    scaler = RobustStandardizer(["a"]).fit(frame)
    np.testing.assert_allclose(scaler.center, [2.5])
    np.testing.assert_allclose(ZScoreStandardizer(["a"]).fit(frame).center, [2.5])


def test_fit_rejects_column_without_finite_values(train):
    frame = train.copy()
    frame["a"] = np.nan
    with pytest.raises(ValueError):
        RobustStandardizer(["a"]).fit(frame)


@pytest.mark.parametrize("scaler_class", [RobustStandardizer, ZScoreStandardizer])
def test_scaler_roundtrip(scaler_class, train, tmp_path):
    scaler = scaler_class(["a", "b", "c"]).fit(train)
    scaler.save(tmp_path / "scaler.json")
    loaded = Standardizer.load(tmp_path / "scaler.json")
    assert type(loaded) is scaler_class
    np.testing.assert_array_equal(loaded.transform(train), scaler.transform(train))


def test_extract_labels(schema, train):
    np.testing.assert_array_equal(extract_labels(train, schema), [0, 0, 1, 1, 0])
    invalid = train.assign(anomaly_label=[0, 2, 1, 1, 0])
    with pytest.raises(ValueError):
        extract_labels(invalid, schema)


def test_compatibility_report(scaler, train):
    local = RobustStandardizer(scaler.feature_columns).fit(train.iloc[:3])
    report = scaler_compatibility_report({"client_1": local}, scaler)
    assert len(report) == 3
    row = report.set_index("feature").loc["a"]
    assert row["center_shift"] == pytest.approx((2.0 - 3.0) / 2.0)
    assert row["scale_ratio"] == pytest.approx(1.0 / 2.0)


def test_compatibility_report_rejects_different_method(scaler, train):
    local = ZScoreStandardizer(scaler.feature_columns).fit(train)
    with pytest.raises(ValueError):
        scaler_compatibility_report({"client_1": local}, scaler)


def test_compatibility_report_rejects_different_order(scaler, train):
    local = RobustStandardizer(list(reversed(scaler.feature_columns))).fit(train)
    with pytest.raises(ValueError):
        scaler_compatibility_report({"client_1": local}, scaler)


def test_to_tensor(scaler, train):
    torch = pytest.importorskip("torch")
    tensor = to_tensor(scaler.transform(train))
    assert tensor.dtype == torch.float32
    assert tuple(tensor.shape) == (5, 3)
