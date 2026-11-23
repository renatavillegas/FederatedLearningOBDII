import numpy as np
import pandas as pd
import pytest
import torch
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from src.data.preprocess import RobustStandardizer, ZScoreStandardizer
from src.experiments.common import (
    EarlyStopping,
    StoppingConfig,
    fit_with_validation,
    method_name,
    reconstruction_loss,
    weighted_validation_loss,
)
from src.federated.fedavg import build_fedavg, weighted_average
from src.federated.fedprox import build_fedprox
from src.federated.protocol import (
    ARRAYS_KEY,
    METRICS_KEY,
    NUM_EXAMPLES,
    array_record_bytes,
    optimizer_state_from_record,
    optimizer_state_to_record,
    scaler_from_record,
    scaler_payload_bytes,
    scaler_to_record,
)
from src.models.model import Autoencoder, AutoencoderConfig, get_parameters
from src.models.trainer import LocalTrainer, TrainingConfig, set_seed

CONFIG = AutoencoderConfig(input_dim=6, encoder_dims=(4,), latent_dim=2)


@pytest.fixture(autouse=True)
def seed():
    set_seed(0)


@pytest.fixture
def features() -> torch.Tensor:
    return torch.randn(64, 6, generator=torch.Generator().manual_seed(1))


def make_trainer(seed: int = 0) -> LocalTrainer:
    return LocalTrainer(Autoencoder(CONFIG), TrainingConfig(learning_rate=1e-2, batch_size=16), seed)


def test_early_stopping_counts_steps_without_improvement():
    stopping = EarlyStopping(patience=2)
    assert stopping.update(1.0, 1)
    assert not stopping.update(1.0, 2)
    assert not stopping.should_stop
    assert not stopping.update(1.5, 3)
    assert stopping.should_stop
    assert (stopping.best_step, stopping.best_value) == (1, 1.0)


def test_early_stopping_min_delta():
    stopping = EarlyStopping(patience=5, min_delta=0.1)
    stopping.update(1.0, 1)
    assert not stopping.update(0.95, 2)
    assert stopping.update(0.85, 3)


def test_invalid_stopping_config():
    with pytest.raises(ValueError):
        StoppingConfig(max_steps=0, patience=1)


def test_weighted_validation_loss(features):
    model = Autoencoder(CONFIG)
    inputs = {"client_1": features, "client_2": features * 2}
    weighted, per_scaler = weighted_validation_loss(model, inputs, {"client_1": 3.0, "client_2": 1.0})
    expected = (3 * reconstruction_loss(model, features) + reconstruction_loss(model, features * 2)) / 4
    assert weighted == pytest.approx(expected)
    assert set(per_scaler) == {"client_1", "client_2"}


def test_weighted_validation_loss_requires_matching_keys(features):
    with pytest.raises(ValueError):
        weighted_validation_loss(Autoencoder(CONFIG), {"client_1": features}, {"client_2": 1.0})


def test_fit_with_validation_keeps_best_state(features):
    trainer = make_trainer()
    fit = fit_with_validation(trainer, features, features[:16], StoppingConfig(max_steps=30, patience=3))
    losses = [entry["validation_loss"] for entry in fit.history]
    assert fit.best_validation_loss == min(losses)
    assert fit.history[fit.best_epoch - 1]["validation_loss"] == fit.best_validation_loss
    trainer.model.load_state_dict(fit.best_state)
    assert reconstruction_loss(trainer.model, features[:16]) == pytest.approx(fit.best_validation_loss)


def test_flower_aggregation_matches_weighted_average():
    rng = np.random.default_rng(0)
    client_arrays = [[rng.normal(size=(3, 2)), rng.normal(size=2)] for _ in range(3)]
    weights = [10, 30, 60]
    replies = [
        RecordDict({ARRAYS_KEY: ArrayRecord(arrays), METRICS_KEY: MetricRecord({NUM_EXAMPLES: weight})})
        for arrays, weight in zip(client_arrays, weights)
    ]
    aggregated = aggregate_arrayrecords(replies, NUM_EXAMPLES).to_numpy_ndarrays()
    for got, expected in zip(aggregated, weighted_average(client_arrays, weights)):
        np.testing.assert_allclose(got, expected)


def test_global_parameters_change_after_aggregation(features):
    initial = Autoencoder(CONFIG)
    client_arrays, weights = [], []
    for position, size in enumerate((32, 64)):
        model = Autoencoder(CONFIG)
        model.load_state_dict(initial.state_dict())
        LocalTrainer(model, TrainingConfig(learning_rate=1e-2, batch_size=16), position).train(features[:size], epochs=1)
        client_arrays.append(get_parameters(model))
        weights.append(size)
    aggregated = weighted_average(client_arrays, weights)
    assert any(not np.allclose(a, b) for a, b in zip(aggregated, get_parameters(initial)))


def test_strategies_use_all_clients():
    fedavg = build_fedavg(6, 1.0)
    assert (fedavg.min_train_nodes, fedavg.min_available_nodes, fedavg.fraction_evaluate) == (6, 6, 0.0)
    fedprox = build_fedprox(6, 1.0, 0.01)
    assert fedprox.proximal_mu == 0.01
    with pytest.raises(ValueError):
        build_fedprox(6, 1.0, 0.0)


@pytest.mark.parametrize("scaler_class", [ZScoreStandardizer, RobustStandardizer])
def test_scaler_record_roundtrip(scaler_class):
    columns = ["a", "b", "c"]
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)), columns=columns)
    scaler = scaler_class(columns).fit(frame)
    record = scaler_to_record(scaler, "client_2")
    client, restored = scaler_from_record(record, columns)
    assert client == "client_2"
    assert type(restored) is scaler_class
    np.testing.assert_allclose(restored.transform(frame), scaler.transform(frame))
    assert scaler_payload_bytes(record) == 8 * 9
    with pytest.raises(ValueError):
        scaler_from_record(record, columns[:2])


def test_optimizer_state_record_roundtrip(features):
    trainer = make_trainer()
    trainer.train(features, epochs=1)
    restored = optimizer_state_from_record(optimizer_state_to_record(trainer.optimizer_state()))
    other = make_trainer()
    other.load_optimizer_state(restored)
    torch.testing.assert_close(other.optimizer_state()["state"][0]["exp_avg"], trainer.optimizer_state()["state"][0]["exp_avg"])


def test_array_record_bytes():
    model = Autoencoder(CONFIG)
    n_parameters = sum(p.numel() for p in model.parameters())
    assert array_record_bytes(ArrayRecord(model.state_dict())) == 4 * n_parameters


def test_method_name():
    assert method_name("fedavg") == "fedavg"
    assert method_name("fedprox", 0.01) == "fedprox_mu_0.01"
    with pytest.raises(ValueError):
        method_name("fedprox")
