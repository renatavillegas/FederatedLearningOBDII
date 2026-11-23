import argparse
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import load_yaml, resolve_path
from src.data.preprocess import FeatureSelector

ACTIVATIONS = {"relu": nn.ReLU}


@dataclass(frozen=True)
class AutoencoderConfig:
    input_dim: int
    encoder_dims: tuple[int, ...]
    latent_dim: int
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {self.activation}; expected one of {sorted(ACTIVATIONS)}")
        if min((self.input_dim, self.latent_dim, *self.encoder_dims)) < 1:
            raise ValueError("Layer dimensions must be positive")

    @classmethod
    def from_config(cls, architecture: Mapping[str, Any], input_dim: int) -> "AutoencoderConfig":
        return cls(
            input_dim=int(input_dim),
            encoder_dims=tuple(int(dim) for dim in architecture["encoder_dims"]),
            latent_dim=int(architecture["latent_dim"]),
            activation=architecture.get("activation", "relu"),
        )

    @property
    def layer_dims(self) -> list[int]:
        encoder = [self.input_dim, *self.encoder_dims, self.latent_dim]
        return encoder + encoder[-2::-1]


class Autoencoder(nn.Module):
    """Fully connected autoencoder with a symmetric decoder and a linear output layer."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_dims = [config.input_dim, *config.encoder_dims, config.latent_dim]
        decoder_dims = encoder_dims[::-1]
        self.encoder = self._stack(encoder_dims, activate_last=True)
        self.decoder = self._stack(decoder_dims, activate_last=False)

    def _stack(self, dims: Sequence[int], activate_last: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        for position, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if position < len(dims) - 2 or activate_last:
                layers.append(ACTIVATIONS[self.config.activation]())
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))

    @torch.no_grad()
    def reconstruction_error(self, inputs: torch.Tensor) -> torch.Tensor:
        """Per-record mean squared reconstruction error."""
        was_training = self.training
        self.eval()
        errors = ((self(inputs) - inputs) ** 2).mean(dim=1)
        self.train(was_training)
        return errors


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size(model: nn.Module) -> dict[str, int]:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "n_parameters": count_parameters(model),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "serialized_state_dict_bytes": buffer.getbuffer().nbytes,
    }


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy().copy() for tensor in model.state_dict().values()]


def set_parameters(model: nn.Module, arrays: Sequence[np.ndarray]) -> None:
    """Copy arrays into the existing tensors, keeping parameter identity (and optimizer references)."""
    keys = list(model.state_dict().keys())
    if len(keys) != len(arrays):
        raise ValueError(f"Expected {len(keys)} arrays, got {len(arrays)}")
    state = {key: torch.as_tensor(np.asarray(array)) for key, array in zip(keys, arrays)}
    model.load_state_dict(state, strict=True)


def build_model(model_config: Mapping[str, Any], input_dim: int, seed: int) -> Autoencoder:
    torch.manual_seed(seed)
    return Autoencoder(AutoencoderConfig.from_config(model_config["architecture"], input_dim))


def run_model_report(model_config_path: str | Path, data_config_path: str | Path) -> dict[str, Any]:
    model_config = load_yaml(model_config_path)
    data_config = load_yaml(data_config_path)
    feature_columns = FeatureSelector.load_columns(resolve_path(data_config["preprocessing"]["feature_columns_path"]))
    model = build_model(model_config, len(feature_columns), int(data_config["seed"]))

    report = {
        "architecture": asdict(model.config),
        "layer_dims": model.config.layer_dims,
        **model_size(model),
        "training": model_config["training"],
        "modules": str(model),
    }
    output_dir = resolve_path(model_config["outputs"]["model_report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the autoencoder architecture and size.")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    run_model_report(args.model_config, args.data_config)


if __name__ == "__main__":
    main()
