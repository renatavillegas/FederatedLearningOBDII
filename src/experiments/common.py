import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import load_yaml, resolve_path
from src.data.clients import ClientStore, client_name
from src.data.preprocess import FeatureSelector, Standardizer, build_scaler, extract_labels, to_tensor
from src.data.schema import DataSchema
from src.data.split import PartitionStore
from src.models.trainer import LocalTrainer


@dataclass(frozen=True)
class StoppingConfig:
    max_steps: int
    patience: int
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.patience < 1 or self.min_delta < 0:
            raise ValueError("max_steps and patience must be positive and min_delta non-negative")


class EarlyStopping:
    """Tracks the best validation loss; stops after `patience` steps without a decrease larger than min_delta."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.best_step: int | None = None
        self.steps_without_improvement = 0

    def update(self, value: float, step: int) -> bool:
        improved = value < self.best_value - self.min_delta
        if improved:
            self.best_value = value
            self.best_step = step
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.steps_without_improvement >= self.patience


class ExperimentData:
    """Loads the raw data, partitions, client assignments and scalers produced by phases 1-4."""

    def __init__(self, data_config_path: str | Path = "configs/data.yaml") -> None:
        self.config = load_yaml(data_config_path)
        self.schema = DataSchema.from_config(self.config)
        outputs = self.config["outputs"]
        self.partitions = PartitionStore(outputs["splits_dir"])
        self.clients = ClientStore(outputs["splits_dir"])
        self.scalers_dir = resolve_path(outputs["scalers_dir"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @cached_property
    def frame(self) -> pd.DataFrame:
        return pd.read_csv(resolve_path(self.config["raw_path"]))

    @cached_property
    def feature_columns(self) -> list[str]:
        return FeatureSelector.load_columns(resolve_path(self.config["preprocessing"]["feature_columns_path"]))

    @cached_property
    def partition_index(self) -> pd.DataFrame:
        return self.partitions.load()

    def partition(self, name: str) -> pd.DataFrame:
        return self.partitions.rows(self.frame, name, self.partition_index)

    def normal_real_partition(self, name: str) -> pd.DataFrame:
        return select_normal_real_records(self.partition(name), self.schema)

    def client_names(self, scenario: str) -> list[str]:
        assignment = self.clients.load(scenario)
        n_clients = assignment["client"].nunique()
        return [client_name(position) for position in range(n_clients)]

    def client_rows(self, scenario: str, client: str) -> pd.DataFrame:
        return self.clients.rows(self.frame, scenario, client)

    def client_scaler(self, scenario: str, client: str) -> Standardizer:
        return self._checked(Standardizer.load(self.scalers_dir / scenario / f"{client}.json"))

    def global_scaler(self) -> Standardizer:
        return self._checked(Standardizer.load(self.scalers_dir / "global_train.json"))

    def features(self, frame: pd.DataFrame, scaler: Standardizer) -> torch.Tensor:
        return to_tensor(scaler.transform(frame, dtype=self.config["preprocessing"]["dtype"]))

    def labels(self, frame: pd.DataFrame) -> np.ndarray:
        return extract_labels(frame, self.schema)

    def _checked(self, scaler: Standardizer) -> Standardizer:
        if scaler.feature_columns != self.feature_columns:
            raise ValueError("Scaler feature order differs from feature_columns.json")
        if scaler.method != self.config["preprocessing"]["scaler"]:
            raise ValueError(f"Saved scaler uses {scaler.method}; configs/data.yaml expects {self.config['preprocessing']['scaler']}")
        return scaler


def select_normal_real_records(frame: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    schema.validate_columns(frame)
    real_categories = {category.name for category in schema.categories if category.family == "live_real"}
    if not real_categories:
        raise ValueError("No category with family 'live_real' is configured")

    categories = schema.assign_categories(frame[schema.source_column])
    mask = categories.isin(real_categories) & frame[schema.label_column].eq(schema.normal_label)
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError("No normal real records found in the selected partition")
    return selected


def fit_normal_real_preprocessing(
    data: ExperimentData,
) -> tuple[list[str], Standardizer, dict[str, Any]]:
    train_frame = data.normal_real_partition("train")
    selector = FeatureSelector(
        data.schema,
        excluded_prefixes=data.config["preprocessing"].get("exclude_feature_prefixes", ()),
    ).fit(train_frame)
    feature_metadata = selector.to_dict()
    feature_columns = feature_metadata["feature_columns"]
    scaler = build_scaler(feature_columns, data.config["preprocessing"]).fit(train_frame)
    return feature_columns, scaler, feature_metadata


@torch.no_grad()
def reconstruction_loss(model: nn.Module, features: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    loss = torch.mean((model(features) - features) ** 2).item()
    model.train(was_training)
    return loss


def weighted_validation_loss(
    model: nn.Module,
    validation_features: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Validation MSE of one model under several scalers, averaged with the given weights.

    `validation_features[name]` is the validation partition transformed with the scaler of `name`.
    """
    if set(validation_features) != set(weights):
        raise ValueError("validation_features and weights must have the same keys")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive value")
    per_scaler = {name: reconstruction_loss(model, features) for name, features in validation_features.items()}
    weighted = sum(per_scaler[name] * weights[name] for name in per_scaler) / total_weight
    return weighted, per_scaler


class RunRecorder:
    """Writes the history, summary and selected model of one training run."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = resolve_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: Mapping[str, Any]) -> None:
        with (self.directory / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def save_model(self, state_dict: Mapping[str, torch.Tensor], name: str = "model.pt") -> None:
        torch.save(dict(state_dict), self.directory / name)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def run_directory(runs_dir: str | Path, scenario: str, method: str, seed: int) -> Path:
    return resolve_path(runs_dir) / scenario / method / f"seed_{seed}"


def method_name(method: str, proximal_mu: float | None = None) -> str:
    if method == "fedprox":
        if proximal_mu is None:
            raise ValueError("proximal_mu is required for fedprox")
        return f"fedprox_mu_{proximal_mu:g}"
    return method


@dataclass
class FitResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_loss: float
    best_state: dict[str, torch.Tensor]
    stopped_early: bool
    training_seconds: float


def fit_with_validation(
    trainer: LocalTrainer,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    stopping: StoppingConfig,
) -> FitResult:
    """Train one epoch at a time and keep the weights with the lowest validation loss."""
    early_stopping = EarlyStopping(stopping.patience, stopping.min_delta)
    history: list[dict[str, Any]] = []
    best_state = clone_state(trainer.model)
    training_seconds = 0.0
    for epoch in range(1, stopping.max_steps + 1):
        result = trainer.train(train_features, epochs=1)
        training_seconds += result.seconds
        validation_loss = reconstruction_loss(trainer.model, validation_features)
        if early_stopping.update(validation_loss, epoch):
            best_state = clone_state(trainer.model)
        history.append(
            {
                "epoch": epoch,
                "train_loss": result.final_loss,
                "validation_loss": validation_loss,
                "train_seconds": result.seconds,
            }
        )
        if early_stopping.should_stop:
            break
    return FitResult(
        history=history,
        best_epoch=early_stopping.best_step,
        best_validation_loss=early_stopping.best_value,
        best_state=best_state,
        stopped_early=early_stopping.should_stop,
        training_seconds=training_seconds,
    )
