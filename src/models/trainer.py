import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OPTIMIZERS = {"adam": torch.optim.Adam}
LOSSES = {"mse": nn.MSELoss}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    optimizer: str = "adam"
    loss: str = "mse"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {self.optimizer}; expected one of {sorted(OPTIMIZERS)}")
        if self.loss not in LOSSES:
            raise ValueError(f"Unknown loss {self.loss}; expected one of {sorted(LOSSES)}")
        if self.device != "cpu":
            raise ValueError("Only the cpu device is supported")
        if self.learning_rate <= 0 or self.batch_size < 1:
            raise ValueError("learning_rate and batch_size must be positive")

    @classmethod
    def from_config(cls, training: Mapping[str, Any]) -> "TrainingConfig":
        return cls(
            learning_rate=float(training["learning_rate"]),
            batch_size=int(training["batch_size"]),
            optimizer=training.get("optimizer", "adam"),
            loss=training.get("loss", "mse"),
            device=training.get("device", "cpu"),
        )


@dataclass
class TrainingResult:
    n_samples: int
    n_steps: int
    seconds: float
    epoch_reconstruction_loss: list[float] = field(default_factory=list)
    epoch_total_loss: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.epoch_reconstruction_loss[-1]


class LocalTrainer:
    """Trains an autoencoder on one dataset.

    The optimizer is created once and reused by every call to `train`, so its state persists
    across calls. Loading new weights with `set_parameters` keeps the parameter objects and
    therefore the optimizer state. When `proximal_mu > 0`, the FedProx term
    (mu / 2) * ||w - w_reference||^2 is added to the loss.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig, seed: int) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.loss_fn = LOSSES[config.loss]()
        self.optimizer = OPTIMIZERS[config.optimizer](self.model.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(seed)

    def train(
        self,
        features: torch.Tensor,
        epochs: int,
        proximal_mu: float = 0.0,
        reference_parameters: Sequence[torch.Tensor] | None = None,
    ) -> TrainingResult:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if proximal_mu < 0:
            raise ValueError("proximal_mu must be non-negative")
        if proximal_mu > 0 and reference_parameters is None:
            raise ValueError("reference_parameters are required when proximal_mu > 0")
        if len(features) == 0:
            raise ValueError("Cannot train on an empty dataset")

        reference = None
        if proximal_mu > 0:
            reference = [tensor.detach().clone().to(self.config.device) for tensor in reference_parameters]

        loader = DataLoader(
            TensorDataset(features), batch_size=self.config.batch_size, shuffle=True, generator=self.generator
        )
        result = TrainingResult(n_samples=len(features), n_steps=0, seconds=0.0)
        self.model.train()
        start = time.perf_counter()
        for _ in range(epochs):
            reconstruction_sum = 0.0
            total_sum = 0.0
            for (batch,) in loader:
                batch = batch.to(self.config.device)
                self.optimizer.zero_grad()
                reconstruction = self.loss_fn(self.model(batch), batch)
                loss = reconstruction
                if reference is not None:
                    loss = loss + 0.5 * proximal_mu * self._proximal_distance(reference)
                loss.backward()
                self.optimizer.step()
                reconstruction_sum += reconstruction.item() * len(batch)
                total_sum += loss.item() * len(batch)
                result.n_steps += 1
            result.epoch_reconstruction_loss.append(reconstruction_sum / len(features))
            result.epoch_total_loss.append(total_sum / len(features))
        result.seconds = time.perf_counter() - start
        return result

    @torch.no_grad()
    def evaluate_loss(self, features: torch.Tensor) -> float:
        self.model.eval()
        features = features.to(self.config.device)
        loss = self.loss_fn(self.model(features), features).item()
        self.model.train()
        return loss

    def optimizer_state(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(state)

    def _proximal_distance(self, reference: Sequence[torch.Tensor]) -> torch.Tensor:
        parameters = list(self.model.parameters())
        if len(parameters) != len(reference):
            raise ValueError("reference_parameters do not match the model parameters")
        return sum(((p - r) ** 2).sum() for p, r in zip(parameters, reference))
