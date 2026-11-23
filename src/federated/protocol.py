"""Keys and payload conversions shared by the ServerApp and the ClientApp."""

import io
from collections.abc import Sequence
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord

from src.data.preprocess import Standardizer

ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
SCALER_KEY = "scaler"
OPTIMIZER_STATE_KEY = "optimizer"

NUM_EXAMPLES = "num-examples"
TRAIN_LOSS = "train-loss"
TRAIN_SECONDS = "train-seconds"
CLIENT_INDEX = "client-index"
EDGE_CLIENT_ID = "edge-client-id"
INFERENCE_RECORDS = "inference_records"
INFERENCE_WARMUP_RECORDS = "inference_warmup_records"
INFERENCE_LATENCY_MEDIAN_MS = "inference_latency_median_ms"
INFERENCE_LATENCY_P95_MS = "inference_latency_p95_ms"
INFERENCE_LATENCY_P99_MS = "inference_latency_p99_ms"
INFERENCE_LATENCY_STD_MS = "inference_latency_std_ms"
INFERENCE_THROUGHPUT_RECORDS_PER_SECOND = "inference_throughput_records_per_second"
INFERENCE_TOTAL_SECONDS = "inference_total_seconds"

LOCAL_EPOCHS = "local-epochs"
SEND_SCALER = "send-scaler"
SERVER_ROUND = "server-round"
PROXIMAL_MU = "proximal-mu"
SEED = "seed"


def scaler_to_record(scaler: Standardizer, client: str) -> ConfigRecord:
    payload = scaler.to_dict()
    record = ConfigRecord(
        {
            "client": client,
            "method": payload["method"],
            "n-features": len(payload["feature_columns"]),
            "n-samples": int(payload["n_samples"]),
            "zero-scale": float(payload["zero_scale"]),
            "center": [float(value) for value in payload["center"]],
            "spread": [float(value) for value in payload["spread"]],
            "median": [float(value) for value in payload["median"]],
        }
    )
    if "quantile_range" in payload:
        record["quantile-range"] = [float(value) for value in payload["quantile_range"]]
    return record


def scaler_from_record(record: ConfigRecord, feature_columns: Sequence[str]) -> tuple[str, Standardizer]:
    if int(record["n-features"]) != len(feature_columns):
        raise ValueError("Scaler received with a different number of features")
    payload = {
        "method": record["method"],
        "feature_columns": list(feature_columns),
        "zero_scale": record["zero-scale"],
        "n_samples": record["n-samples"],
        "center": list(record["center"]),
        "spread": list(record["spread"]),
        "median": list(record["median"]),
    }
    if "quantile-range" in record:
        payload["quantile_range"] = list(record["quantile-range"])
    return str(record["client"]), Standardizer.from_dict(payload)


def scaler_payload_bytes(record: ConfigRecord) -> int:
    """Size of the numeric scaler payload (center, spread and median as float64)."""
    return 8 * (len(record["center"]) + len(record["spread"]) + len(record["median"]))


def optimizer_state_to_record(state: dict[str, Any]) -> ConfigRecord:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return ConfigRecord({"state": buffer.getvalue()})


def optimizer_state_from_record(record: ConfigRecord) -> dict[str, Any]:
    return torch.load(io.BytesIO(record["state"]), weights_only=True)


def array_record_bytes(record: ArrayRecord) -> int:
    """Size of the parameter payload (sum of array buffers)."""
    return sum(array.numpy().nbytes for array in record.values())
