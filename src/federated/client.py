from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.config import load_yaml, set_project_root
from src.experiments.common import (
    ExperimentData,
    fit_normal_real_preprocessing,
    select_normal_real_records,
)
from src.federated.protocol import (
    ARRAYS_KEY,
    CLIENT_INDEX,
    CONFIG_KEY,
    LOCAL_EPOCHS,
    METRICS_KEY,
    NUM_EXAMPLES,
    OPTIMIZER_STATE_KEY,
    PROXIMAL_MU,
    SCALER_KEY,
    SEED,
    SEND_SCALER,
    SERVER_ROUND,
    TRAIN_LOSS,
    TRAIN_SECONDS,
    EDGE_CLIENT_ID,
    optimizer_state_from_record,
    optimizer_state_to_record,
    scaler_to_record,
)
from src.federated.edge_metrics import benchmark_inference
from src.models.model import build_model
from src.models.trainer import LocalTrainer, TrainingConfig, set_seed

app = ClientApp()


class FederatedClient:
    """Local training of one client; only weights, metrics, and optionally scaler stats are returned."""

    def __init__(
        self,
        client: str,
        scenario: str,
        data_config: str,
        model_config: str,
        normal_only_real: bool = False,
        client_data_dir: str | None = None,
    ) -> None:
        self.client = client
        match = re.fullmatch(r"client_([1-9][0-9]*)", client)
        if match is None:
            raise ValueError(f"Client ID must use client_<positive integer>, got {client!r}")
        self.position = int(match.group(1)) - 1
        self.model_config = load_yaml(model_config)
        self.inference_features: torch.Tensor | None = None
        if client_data_dir is not None:
            if not normal_only_real:
                raise ValueError("Prepared edge clients require normal-real-only training")
            bundle_dir = Path(client_data_dir)
            metadata_path = bundle_dir / "bundle_metadata.json"
            data_path = bundle_dir / "edge_client_data.npz"
            if not metadata_path.is_file() or not data_path.is_file():
                raise FileNotFoundError(
                    f"Edge client bundle requires {metadata_path} and {data_path}"
                )
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("client") != client or metadata.get("scenario") != scenario:
                raise ValueError(
                    f"Edge bundle identity {metadata.get('scenario')}/{metadata.get('client')} "
                    f"does not match {scenario}/{client}"
                )
            if not bool(metadata.get("normal_only_training")):
                raise ValueError("Edge bundle must contain normal-real-only training data")
            with np.load(data_path, allow_pickle=False) as archive:
                train_features = np.asarray(archive["train_x"], dtype=np.float32)
                inference_features = np.asarray(archive["inference_x"], dtype=np.float32)
            if (
                train_features.ndim != 2
                or train_features.shape[0] == 0
                or inference_features.ndim != 2
                or inference_features.shape[0] == 0
                or train_features.shape[1] != inference_features.shape[1]
                or not np.isfinite(train_features).all()
                or not np.isfinite(inference_features).all()
            ):
                raise ValueError("Edge bundle contains invalid train or inference features")
            if int(metadata.get("n_features", -1)) != train_features.shape[1]:
                raise ValueError("Edge bundle feature count does not match its metadata")
            self.n_features = int(train_features.shape[1])
            self.features = torch.from_numpy(np.ascontiguousarray(train_features))
            self.inference_features = torch.from_numpy(
                np.ascontiguousarray(inference_features)
            )
            self.scaler = None
        else:
            data = ExperimentData(data_config)
            if client not in data.client_names(scenario):
                raise ValueError(f"Unknown client {client!r} in scenario {scenario!r}")
            self.position = data.client_names(scenario).index(client)
            client_rows = data.client_rows(scenario, client)
            if normal_only_real:
                feature_columns, self.scaler, _ = fit_normal_real_preprocessing(data)
                client_rows = select_normal_real_records(client_rows, data.schema)
            else:
                feature_columns = data.feature_columns
                self.scaler = data.client_scaler(scenario, client)
            self.n_features = len(feature_columns)
            self.features = data.features(client_rows, self.scaler)

    def train(self, message: Message, context: Context) -> Message:
        config = message.content[CONFIG_KEY]
        server_round = int(config[SERVER_ROUND])
        seed = int(config[SEED])
        proximal_mu = float(config[PROXIMAL_MU]) if PROXIMAL_MU in config else 0.0
        step_seed = seed + 1000 * (self.position + 1) + server_round
        set_seed(step_seed)

        model = build_model(self.model_config, self.n_features, seed)
        model.load_state_dict(message.content[ARRAYS_KEY].to_torch_state_dict())
        inference_metrics: dict[str, float | int] = {}
        if self.inference_features is not None:
            inference_metrics = benchmark_inference(model, self.inference_features)
        trainer = LocalTrainer(model, TrainingConfig.from_config(self.model_config["training"]), step_seed)
        if OPTIMIZER_STATE_KEY in context.state:
            trainer.load_optimizer_state(optimizer_state_from_record(context.state[OPTIMIZER_STATE_KEY]))

        reference = [parameter.detach().clone() for parameter in model.parameters()] if proximal_mu > 0 else None
        result = trainer.train(
            self.features, epochs=int(config[LOCAL_EPOCHS]), proximal_mu=proximal_mu, reference_parameters=reference
        )
        context.state[OPTIMIZER_STATE_KEY] = optimizer_state_to_record(trainer.optimizer_state())

        content = RecordDict(
            {
                ARRAYS_KEY: ArrayRecord(model.state_dict()),
                METRICS_KEY: MetricRecord(
                    {
                        NUM_EXAMPLES: result.n_samples,
                        TRAIN_LOSS: result.final_loss,
                        TRAIN_SECONDS: result.seconds,
                        CLIENT_INDEX: self.position,
                        **inference_metrics,
                    }
                ),
            }
        )
        if bool(config[SEND_SCALER]):
            content[SCALER_KEY] = scaler_to_record(self.scaler, self.client)
        return Message(content=content, reply_to=message)


@lru_cache(maxsize=None)
def _client(
    root: str,
    client: str,
    scenario: str,
    data_config: str,
    model_config: str,
    normal_only_real: bool,
    client_data_dir: str | None,
) -> FederatedClient:
    set_project_root(root)
    return FederatedClient(
        client,
        scenario,
        data_config,
        model_config,
        normal_only_real,
        client_data_dir,
    )


def _setting(context: Context, key: str) -> Any:
    return context.node_config[key] if key in context.node_config else context.run_config[key]


@app.train()
def train(message: Message, context: Context) -> Message:
    root = str(
        context.node_config.get("project-root")
        or os.environ.get("FLOWER_CLIENT_PROJECT_ROOT")
        or context.run_config["project-root"]
    )
    set_project_root(root)
    normal_only_real = context.run_config["normal-only-real"]
    if not isinstance(normal_only_real, bool):
        raise ValueError("normal-only-real must be a boolean")
    client_id = str(context.node_config["client-id"])
    config = message.content[CONFIG_KEY]
    edge_client_id = str(
        config.get(EDGE_CLIENT_ID)
        or context.run_config.get(EDGE_CLIENT_ID, "")
    )
    client_data_dir = (
        str(context.node_config["client-data-dir"])
        if "client-data-dir" in context.node_config
        else os.environ.get("FLOWER_CLIENT_DATA_DIR")
    )
    if edge_client_id and client_id == edge_client_id and client_data_dir is None:
        raise ValueError(
            f"Configured edge client {client_id} is missing client-data-dir node configuration"
        )
    if client_data_dir is not None and client_id != edge_client_id:
        raise ValueError(
            f"Prepared edge data was configured on {client_id}, expected {edge_client_id}"
        )
    client = _client(
        root,
        client_id,
        str(context.run_config["scenario"]),
        str(context.run_config["data-config"]),
        str(context.run_config["model-config"]),
        normal_only_real,
        client_data_dir,
    )
    return client.train(message, context)
