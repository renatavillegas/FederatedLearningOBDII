"""Binary MLP classifier for supervised anomaly detection on OBD-II sensor data.

Architecture: Input(105) → Dense(32, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid)

The encoder mirrors the autoencoder encoder to allow weight initialization
from a pretrained autoencoder state dict (transfer learning).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.models.model import ACTIVATIONS, AutoencoderConfig


@dataclass(frozen=True)
class BinaryClassifierConfig:
    input_dim: int
    hidden_dims: tuple[int, ...]
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {self.activation!r}")
        if min((self.input_dim, *self.hidden_dims)) < 1:
            raise ValueError("All dimensions must be positive")

    @classmethod
    def from_autoencoder_config(cls, ae_config: AutoencoderConfig) -> "BinaryClassifierConfig":
        """Mirror the autoencoder encoder as the classifier body."""
        return cls(
            input_dim=ae_config.input_dim,
            hidden_dims=(*ae_config.encoder_dims, ae_config.latent_dim),
            activation=ae_config.activation,
        )

    @classmethod
    def from_config(cls, architecture: Mapping[str, Any], input_dim: int) -> "BinaryClassifierConfig":
        return cls(
            input_dim=int(input_dim),
            hidden_dims=tuple(int(d) for d in architecture["hidden_dims"]),
            activation=architecture.get("activation", "relu"),
        )


class BinaryMLP(nn.Module):
    """Fully connected binary classifier with the same encoder topology as the autoencoder.

    The final layer is a single linear unit (no sigmoid); use BCEWithLogitsLoss.
    """

    def __init__(self, config: BinaryClassifierConfig) -> None:
        super().__init__()
        self.config = config
        dims = [config.input_dim, *config.hidden_dims]
        act = ACTIVATIONS[config.activation]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act())
        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, inputs: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        proba = torch.sigmoid(self(inputs))
        self.train(was_training)
        return proba

    def load_encoder_weights_from_autoencoder(
        self, ae_state_dict: Mapping[str, torch.Tensor]
    ) -> int:
        """Copy matching encoder layer weights from an autoencoder state dict.

        The autoencoder encoder layers are named encoder.0.weight / encoder.0.bias, etc.
        The classifier layers are named network.0.weight / network.0.bias, etc.
        Only layers whose shapes match are copied; mismatches are silently skipped.
        Returns the number of layers successfully copied.
        """
        classifier_state = self.state_dict()
        ae_encoder_keys = sorted(k for k in ae_state_dict if k.startswith("encoder."))
        n_body_layers = len(list(self.network)) - 1
        classifier_body_keys = sorted(
            k for k in classifier_state
            if k.startswith("network.") and int(k.split(".")[1]) < n_body_layers
        )
        copied = 0
        for ae_key, clf_key in zip(ae_encoder_keys, classifier_body_keys):
            ae_tensor = ae_state_dict[ae_key]
            if classifier_state[clf_key].shape == ae_tensor.shape:
                classifier_state[clf_key] = ae_tensor.clone()
                copied += 1
        self.load_state_dict(classifier_state)
        return copied


def build_classifier(
    model_config: Mapping[str, Any],
    input_dim: int,
    seed: int | None = None,
) -> BinaryMLP:
    if seed is not None:
        torch.manual_seed(seed)
    architecture = model_config["architecture"]
    if "hidden_dims" in architecture:
        config = BinaryClassifierConfig.from_config(architecture, input_dim)
    else:
        ae_config = AutoencoderConfig.from_config(architecture, input_dim)
        config = BinaryClassifierConfig.from_autoencoder_config(ae_config)
    return BinaryMLP(config)
