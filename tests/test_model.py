import numpy as np
import pytest
import torch

from src.models.model import (
    Autoencoder,
    AutoencoderConfig,
    count_parameters,
    get_parameters,
    model_size,
    set_parameters,
)
from src.models.trainer import LocalTrainer, TrainingConfig, set_seed

INPUT_DIM = 118
ARCHITECTURE = {"encoder_dims": [32], "latent_dim": 8, "activation": "relu"}


@pytest.fixture(autouse=True)
def seed():
    set_seed(0)


@pytest.fixture
def config() -> AutoencoderConfig:
    return AutoencoderConfig.from_config(ARCHITECTURE, INPUT_DIM)


@pytest.fixture
def model(config) -> Autoencoder:
    return Autoencoder(config)


@pytest.fixture
def features() -> torch.Tensor:
    return torch.randn(100, INPUT_DIM, generator=torch.Generator().manual_seed(1))


def make_trainer(model: Autoencoder, seed: int = 0) -> LocalTrainer:
    return LocalTrainer(model, TrainingConfig(learning_rate=1e-3, batch_size=32), seed=seed)


def test_layer_dims(config):
    assert config.layer_dims == [118, 32, 8, 32, 118]


def test_output_shapes(model, features):
    assert model(features).shape == features.shape
    assert model.encode(features).shape == (100, 8)
    assert model.reconstruction_error(features).shape == (100,)


def test_output_layer_is_linear(model):
    assert isinstance(model.decoder[-1], torch.nn.Linear)


def test_parameter_count_and_size(model):
    expected = (118 * 32 + 32) + (32 * 8 + 8) + (8 * 32 + 32) + (32 * 118 + 118)
    assert count_parameters(model) == expected == 8254
    size = model_size(model)
    assert size["parameter_bytes"] == expected * 4
    assert size["serialized_state_dict_bytes"] > size["parameter_bytes"]


def test_reconstruction_error_matches_mse(model, features):
    expected = ((model(features) - features) ** 2).mean(dim=1).detach()
    torch.testing.assert_close(model.reconstruction_error(features), expected)


def test_invalid_activation():
    with pytest.raises(ValueError):
        AutoencoderConfig(input_dim=4, encoder_dims=(2,), latent_dim=1, activation="tanh")


def test_parameters_roundtrip(config, model):
    other = Autoencoder(config)
    set_parameters(other, get_parameters(model))
    for a, b in zip(model.state_dict().values(), other.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_set_parameters_keeps_parameter_identity(model):
    identities = [id(p) for p in model.parameters()]
    set_parameters(model, [np.zeros_like(a) for a in get_parameters(model)])
    assert identities == [id(p) for p in model.parameters()]
    assert all(torch.count_nonzero(p) == 0 for p in model.parameters())


def test_set_parameters_rejects_wrong_length(model):
    with pytest.raises(ValueError):
        set_parameters(model, get_parameters(model)[:-1])


def test_training_reduces_loss(model, features):
    trainer = make_trainer(model)
    before = trainer.evaluate_loss(features)
    result = trainer.train(features, epochs=20)
    assert trainer.evaluate_loss(features) < before
    assert len(result.epoch_reconstruction_loss) == 20
    assert result.n_steps == 20 * 4
    assert result.n_samples == 100


def test_training_is_reproducible(config, features):
    results = []
    for _ in range(2):
        set_seed(0)
        model = Autoencoder(config)
        make_trainer(model, seed=3).train(features, epochs=2)
        results.append(get_parameters(model))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)


def test_optimizer_state_persists_across_calls(model, features):
    trainer = make_trainer(model)
    trainer.train(features, epochs=1)
    first_step = trainer.optimizer_state()["state"][0]["step"].item()
    set_parameters(model, get_parameters(model))
    trainer.train(features, epochs=1)
    assert trainer.optimizer_state()["state"][0]["step"].item() == 2 * first_step


def test_optimizer_state_roundtrip(config, model, features):
    trainer = make_trainer(model)
    trainer.train(features, epochs=1)
    other = make_trainer(Autoencoder(config))
    other.load_optimizer_state(trainer.optimizer_state())
    assert other.optimizer_state()["state"][0]["step"].item() == trainer.optimizer_state()["state"][0]["step"].item()


def test_proximal_term_keeps_weights_closer_to_reference(config, features):
    distances = []
    for mu in (0.0, 10.0):
        set_seed(0)
        model = Autoencoder(config)
        reference = [p.detach().clone() for p in model.parameters()]
        make_trainer(model).train(features, epochs=5, proximal_mu=mu, reference_parameters=reference)
        distances.append(sum(((p - r) ** 2).sum().item() for p, r in zip(model.parameters(), reference)))
    assert distances[1] < distances[0]


def test_proximal_requires_reference(model, features):
    with pytest.raises(ValueError):
        make_trainer(model).train(features, epochs=1, proximal_mu=0.1)


def test_only_cpu_is_supported():
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=1e-3, batch_size=32, device="cuda")
