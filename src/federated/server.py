import time
from dataclasses import dataclass
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, UserConfig
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.config import load_yaml, set_project_root
from src.data.clients import client_name
from src.experiments.common import (
    EarlyStopping,
    ExperimentData,
    RunRecorder,
    fit_normal_real_preprocessing,
    method_name,
    run_directory,
    select_normal_real_records,
    weighted_validation_loss,
)
from src.federated.fedavg import build_fedavg
from src.federated.fedprox import build_fedprox
from src.federated.protocol import (
    ARRAYS_KEY,
    CLIENT_INDEX,
    EDGE_CLIENT_ID,
    INFERENCE_LATENCY_MEDIAN_MS,
    INFERENCE_LATENCY_P95_MS,
    INFERENCE_LATENCY_P99_MS,
    INFERENCE_LATENCY_STD_MS,
    INFERENCE_RECORDS,
    INFERENCE_THROUGHPUT_RECORDS_PER_SECOND,
    INFERENCE_TOTAL_SECONDS,
    INFERENCE_WARMUP_RECORDS,
    LOCAL_EPOCHS,
    METRICS_KEY,
    NUM_EXAMPLES,
    SCALER_KEY,
    SEED,
    SEND_SCALER,
    TRAIN_LOSS,
    TRAIN_SECONDS,
    array_record_bytes,
    scaler_from_record,
    scaler_payload_bytes,
)
from src.models.model import build_model, model_size
from src.models.trainer import set_seed

METHODS = ("fedavg", "fedprox")

app = ServerApp()


@dataclass(frozen=True)
class FederatedRunSettings:
    scenario: str
    method: str
    proximal_mu: float
    seed: int | None
    normal_only_real: bool
    data_config: str
    model_config: str
    experiments_config: str
    edge_client_id: str | None
    max_rounds: int | None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method}; expected one of {METHODS}")

    @classmethod
    def from_run_config(cls, run_config: UserConfig) -> "FederatedRunSettings":
        seed = int(run_config["seed"])
        normal_only_real = run_config["normal-only-real"]
        if not isinstance(normal_only_real, bool):
            raise ValueError("normal-only-real must be a boolean")
        return cls(
            scenario=str(run_config["scenario"]),
            method=str(run_config["method"]),
            proximal_mu=float(run_config["proximal-mu"]),
            seed=seed if seed >= 0 else None,
            normal_only_real=normal_only_real,
            data_config=str(run_config["data-config"]),
            model_config=str(run_config["model-config"]),
            experiments_config=str(run_config["experiments-config"]),
            edge_client_id=(
                str(run_config["edge-client-id"])
                if str(run_config["edge-client-id"])
                else None
            ),
            max_rounds=(
                int(run_config["max-rounds"])
                if int(run_config["max-rounds"]) > 0
                else None
            ),
        )


class FederatedServer:
    """Coordinates the federated rounds, evaluates the global model on the validation partition
    with each client's scaler, and keeps the round with the lowest weighted validation loss."""

    def __init__(self, settings: FederatedRunSettings) -> None:
        self.settings = settings
        self.data = ExperimentData(settings.data_config)
        self.model_config = load_yaml(settings.model_config)
        self.experiments = load_yaml(settings.experiments_config)
        self.federated = self.experiments["federated"]
        self.n_clients = int(self.experiments["n_clients"])
        if settings.edge_client_id is not None:
            expected_clients = {
                client_name(position) for position in range(self.n_clients)
            }
            if settings.edge_client_id not in expected_clients:
                raise ValueError(
                    f"Edge client {settings.edge_client_id!r} is not in "
                    f"configured client set {sorted(expected_clients)}"
                )
        self.max_rounds = settings.max_rounds or int(self.federated["max_rounds"])
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.seed = settings.seed if settings.seed is not None else self.data.seed
        if settings.normal_only_real:
            self.feature_columns, self.scaler, self.feature_metadata = fit_normal_real_preprocessing(self.data)
        else:
            self.feature_columns = self.data.feature_columns
            self.scaler = None
            self.feature_metadata = None

    def build_strategy(self) -> FedAvg:
        fraction_train = float(self.federated["fraction_train"])
        if self.settings.method == "fedprox":
            return build_fedprox(self.n_clients, fraction_train, self.settings.proximal_mu)
        return build_fedavg(self.n_clients, fraction_train)

    def run(self, grid: Grid) -> dict[str, Any]:
        set_seed(self.seed)
        model = build_model(self.model_config, len(self.feature_columns), self.seed)
        arrays = ArrayRecord(model.state_dict())
        strategy = self.build_strategy()
        validation = (
            self.data.normal_real_partition("validation")
            if self.settings.normal_only_real
            else self.data.partition("validation")
        )
        early_stopping = EarlyStopping(int(self.federated["patience"]), float(self.federated["min_delta"]))
        timeout = float(self.federated["round_timeout_seconds"])

        if self.settings.normal_only_real:
            if self.scaler is None:
                raise RuntimeError("Normal-real preprocessing was not initialized")
            validation_features: dict[str, torch.Tensor] = {
                "global": self.data.features(validation, self.scaler)
            }
            weights: dict[str, float] = {"global": 1.0}
        else:
            validation_features = {}
            weights = {}
        history: list[dict[str, Any]] = []
        edge_inference_history: list[dict[str, float | int | str]] = []
        best_state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        communication = {"parameter_bytes_down": 0, "parameter_bytes_up": 0, "scaler_bytes_up": 0}
        start = time.perf_counter()

        for server_round in range(1, self.max_rounds + 1):
            round_start = time.perf_counter()
            config = ConfigRecord(
                {
                    LOCAL_EPOCHS: int(self.federated["local_epochs"]),
                    SEND_SCALER: server_round == 1 and not self.settings.normal_only_real,
                    SEED: self.seed,
                    EDGE_CLIENT_ID: self.settings.edge_client_id or "",
                }
            )
            messages = list(strategy.configure_train(server_round, arrays, config, grid))
            replies = list(grid.send_and_receive(messages, timeout=timeout))
            self._check_replies(messages, replies)

            clients: dict[str, dict[str, float]] = {}
            edge_metrics_this_round: set[str] = set()
            round_bytes = {"parameter_bytes_down": array_record_bytes(arrays) * len(messages), "parameter_bytes_up": 0, "scaler_bytes_up": 0}
            for reply in replies:
                metrics = reply.content[METRICS_KEY]
                name = client_name(int(metrics[CLIENT_INDEX]))
                clients[name] = {
                    "num_examples": int(metrics[NUM_EXAMPLES]),
                    "train_loss": float(metrics[TRAIN_LOSS]),
                    "train_seconds": float(metrics[TRAIN_SECONDS]),
                }
                inference_keys = (
                    INFERENCE_RECORDS,
                    INFERENCE_WARMUP_RECORDS,
                    INFERENCE_LATENCY_MEDIAN_MS,
                    INFERENCE_LATENCY_P95_MS,
                    INFERENCE_LATENCY_P99_MS,
                    INFERENCE_LATENCY_STD_MS,
                    INFERENCE_THROUGHPUT_RECORDS_PER_SECOND,
                    INFERENCE_TOTAL_SECONDS,
                )
                has_inference_metrics = any(key in metrics for key in inference_keys)
                if has_inference_metrics:
                    if name != self.settings.edge_client_id:
                        raise ValueError(
                            f"Unexpected inference metrics received from {name}"
                        )
                    missing_inference_metrics = [
                        key for key in inference_keys if key not in metrics
                    ]
                    if missing_inference_metrics:
                        raise ValueError(
                            f"Incomplete inference metrics from {name}: "
                            f"{missing_inference_metrics}"
                        )
                    edge_metrics_this_round.add(name)
                    edge_inference_history.append(
                        {
                            "server_round": server_round,
                            "model_round": server_round - 1,
                            "client": name,
                            "records": int(metrics[INFERENCE_RECORDS]),
                            "warmup_records": int(metrics[INFERENCE_WARMUP_RECORDS]),
                            "latency_median_ms": float(
                                metrics[INFERENCE_LATENCY_MEDIAN_MS]
                            ),
                            "latency_p95_ms": float(
                                metrics[INFERENCE_LATENCY_P95_MS]
                            ),
                            "latency_p99_ms": float(
                                metrics[INFERENCE_LATENCY_P99_MS]
                            ),
                            "latency_std_ms": float(
                                metrics[INFERENCE_LATENCY_STD_MS]
                            ),
                            "throughput_records_per_second": float(
                                metrics[INFERENCE_THROUGHPUT_RECORDS_PER_SECOND]
                            ),
                            "inference_seconds": float(
                                metrics[INFERENCE_TOTAL_SECONDS]
                            ),
                        }
                    )
                    reply.content[METRICS_KEY] = MetricRecord(
                        {
                            key: value
                            for key, value in metrics.items()
                            if key not in inference_keys
                        }
                    )
                round_bytes["parameter_bytes_up"] += array_record_bytes(reply.content[ARRAYS_KEY])
                if SCALER_KEY in reply.content:
                    record = reply.content[SCALER_KEY]
                    scaler_client, scaler = scaler_from_record(record, self.data.feature_columns)
                    if scaler_client != name:
                        raise ValueError(f"Scaler of {scaler_client} received from {name}")
                    validation_features[name] = self.data.features(validation, scaler)
                    weights[name] = float(metrics[NUM_EXAMPLES])
                    round_bytes["scaler_bytes_up"] += scaler_payload_bytes(record)
            if server_round == 1 and not self.settings.normal_only_real and len(validation_features) != len(messages):
                raise RuntimeError("Not every client sent its scaler in the first round")
            if self.settings.edge_client_id is not None and (
                self.settings.edge_client_id not in clients
                or edge_metrics_this_round != {self.settings.edge_client_id}
            ):
                raise RuntimeError(
                    f"Expected one reply with inference metrics from "
                    f"{self.settings.edge_client_id} in round {server_round}"
                )

            aggregated, _ = strategy.aggregate_train(server_round, replies)
            if aggregated is None:
                raise RuntimeError(f"Aggregation failed in round {server_round}")
            arrays = aggregated
            model.load_state_dict(arrays.to_torch_state_dict())

            validation_loss, per_client_loss = weighted_validation_loss(model, validation_features, weights)
            improved = early_stopping.update(validation_loss, server_round)
            if improved:
                best_state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
            for key, value in round_bytes.items():
                communication[key] += value
            history.append(
                {
                    "round": server_round,
                    "participants": len(replies),
                    "validation_loss": validation_loss,
                    "validation_loss_per_client_scaler": per_client_loss,
                    "clients": clients,
                    "train_loss_weighted": sum(c["train_loss"] * c["num_examples"] for c in clients.values())
                    / sum(c["num_examples"] for c in clients.values()),
                    "max_client_train_seconds": max(c["train_seconds"] for c in clients.values()),
                    "round_seconds": time.perf_counter() - round_start,
                    **round_bytes,
                }
            )
            if early_stopping.should_stop:
                break

        name = method_name(self.settings.method, self.settings.proximal_mu if self.settings.method == "fedprox" else None)
        if self.settings.normal_only_real:
            name = f"{name}_normal_only"
        summary = {
            "scenario": self.settings.scenario,
            "method": name,
            "proximal_mu": self.settings.proximal_mu if self.settings.method == "fedprox" else None,
            "seed": self.seed,
            "n_clients": self.n_clients,
            "local_epochs": int(self.federated["local_epochs"]),
            "max_rounds": int(self.federated["max_rounds"]),
            "configured_round_limit": self.max_rounds,
            "patience": int(self.federated["patience"]),
            "min_delta": float(self.federated["min_delta"]),
            "fraction_train": float(self.federated["fraction_train"]),
            "training_population": "normal_real_only" if self.settings.normal_only_real else "all_training_records",
            "validation_population": "normal_real_only" if self.settings.normal_only_real else "all_validation_records",
            "normal_label_used_for_training_selection": self.settings.normal_only_real,
            "scaler": "normal_real_train_global" if self.settings.normal_only_real else "client",
            "validation_weights": weights,
            "best_round": early_stopping.best_step,
            "best_validation_loss": early_stopping.best_value,
            "rounds_run": len(history),
            "stopped_early": early_stopping.should_stop,
            "total_seconds": time.perf_counter() - start,
            "communication": communication,
            **model_size(model),
        }
        if self.settings.normal_only_real:
            client_data = {}
            for client in self.data.client_names(self.settings.scenario):
                client_frame = select_normal_real_records(
                    self.data.client_rows(self.settings.scenario, client), self.data.schema
                )
                client_data[client] = {
                    "n_train": len(client_frame),
                    "n_vehicles": int(client_frame[self.data.schema.source_column].nunique()),
                }
            summary["clients"] = client_data
        if self.settings.edge_client_id is not None:
            summary["edge_client_id"] = self.settings.edge_client_id

        recorder = RunRecorder(
            run_directory(self.experiments["outputs"]["runs_dir"], self.settings.scenario, name, self.seed)
        )
        recorder.save_json("model_config.json", self.model_config)
        if self.settings.normal_only_real:
            if self.scaler is None or self.feature_metadata is None:
                raise RuntimeError("Normal-real preprocessing was not initialized")
            self.scaler.save(recorder.directory / "scaler.json")
            recorder.save_json("features.json", self.feature_metadata)
        recorder.save_json("history.json", {"rounds": history})
        recorder.save_json("summary.json", summary)
        recorder.save_model(best_state)
        if self.settings.edge_client_id is not None:
            import pandas as pd

            inference_path = recorder.directory / "edge_inference_metrics.csv"
            pd.DataFrame(edge_inference_history).to_csv(inference_path, index=False)
            recorder.save_json(
                "edge_inference_metrics.json",
                {
                    "client": self.settings.edge_client_id,
                    "scenario": self.settings.scenario,
                    "seed": self.seed,
                    "measurement": (
                        "single-record reconstruction plus top-5-percent score "
                        "on unlabeled held-out inputs; no gradients or labels"
                    ),
                    "measured_before_local_training": True,
                    "rounds": edge_inference_history,
                },
            )
        return summary

    @staticmethod
    def _check_replies(messages: list[Message], replies: list[Message]) -> None:
        errors = [reply.error.reason for reply in replies if reply.has_error()]
        if errors:
            raise RuntimeError(f"Client errors: {errors}")
        if len(replies) != len(messages):
            raise RuntimeError(f"Expected {len(messages)} replies, received {len(replies)}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    set_project_root(str(context.run_config["project-root"]))
    FederatedServer(FederatedRunSettings.from_run_config(context.run_config)).run(grid)
