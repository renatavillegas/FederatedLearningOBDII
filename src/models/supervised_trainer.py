"""Supervised binary classifier trainer for federated and local experiments.

Mirrors the LocalTrainer interface so the same experiment harness can call
both the autoencoder trainer and this supervised trainer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.trainer import TrainingConfig, TrainingResult, set_seed


class SupervisedTrainer:
    """Trains a binary MLP classifier (BCEWithLogitsLoss) on labelled data.

    pos_weight scales the loss for the positive (anomaly) class to handle
    class imbalance. If None, it is computed automatically from the labels.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        seed: int,
        pos_weight: float | None = None,
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.pos_weight_value = pos_weight
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(seed)

    def _loss_fn(self, labels: torch.Tensor) -> nn.BCEWithLogitsLoss:
        if self.pos_weight_value is not None:
            w = torch.tensor([self.pos_weight_value], dtype=torch.float32)
        else:
            n_pos = labels.sum().item()
            n_neg = len(labels) - n_pos
            ratio = n_neg / max(n_pos, 1)
            w = torch.tensor([ratio], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=w.to(self.config.device))

    def train(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int,
        proximal_mu: float = 0.0,
        reference_parameters: Sequence[torch.Tensor] | None = None,
    ) -> TrainingResult:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("Cannot train on empty data")
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")

        reference = None
        if proximal_mu > 0:
            if reference_parameters is None:
                raise ValueError("reference_parameters are required when proximal_mu > 0")
            reference = [t.detach().clone().to(self.config.device) for t in reference_parameters]

        loss_fn = self._loss_fn(labels)
        loader = DataLoader(
            TensorDataset(features, labels),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=self.generator,
        )
        result = TrainingResult(n_samples=len(features), n_steps=0, seconds=0.0)
        self.model.train()
        start = time.perf_counter()
        for _ in range(epochs):
            bce_sum = 0.0
            total_sum = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device).float()
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                bce = loss_fn(logits, batch_y)
                loss = bce
                if reference is not None:
                    prox = sum(((p - r) ** 2).sum() for p, r in zip(self.model.parameters(), reference))
                    loss = loss + 0.5 * proximal_mu * prox
                loss.backward()
                self.optimizer.step()
                bce_sum += bce.item() * len(batch_x)
                total_sum += loss.item() * len(batch_x)
                result.n_steps += 1
            result.epoch_reconstruction_loss.append(bce_sum / len(features))
            result.epoch_total_loss.append(total_sum / len(features))
        result.seconds = time.perf_counter() - start
        return result

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        logits = self.model(features.to(self.config.device))
        proba = torch.sigmoid(logits).cpu().numpy()
        y_true = labels.numpy().astype(int)
        y_pred = (proba >= 0.5).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr = fp / max(fp + tn, 1)
        loss_fn = self._loss_fn(labels)
        bce = loss_fn(logits.cpu(), labels.float()).item()
        self.model.train()
        return {
            "bce_loss": float(bce),
            "accuracy": float((tp + tn) / max(len(y_true), 1)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "n_samples": int(len(y_true)),
            "n_anomaly": int(y_true.sum()),
        }

    def optimizer_state(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(state)
