from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

BATCH_SIZES_BENCHMARK = (1, 8, 32, 64)


def benchmark_inference(
    model: nn.Module,
    features: torch.Tensor,
    warmup_records: int = 10,
) -> dict[str, float | int]:
    """Single-record reconstruction + top-5-percent scoring on CPU.

    Measures latency for every record individually (single-stream mode per MLPerf Tiny).
    Returns median, p95, p99, std, throughput and total seconds.
    """
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2-D inference features, got {tuple(features.shape)}")
    if warmup_records < 0:
        raise ValueError("warmup_records must be non-negative")
    if features.device.type != "cpu":
        raise ValueError("The edge inference benchmark expects CPU features")

    n_records, n_features = features.shape
    top_k = math.ceil(n_features * 0.05)
    if top_k < 1:
        raise ValueError(f"Expected at least one feature, got {n_features}")

    model_was_training = model.training
    model.eval()
    latencies_ms = np.empty(n_records, dtype=np.float64)
    try:
        with torch.inference_mode():
            for index in range(min(warmup_records, n_records)):
                row = features[index : index + 1]
                reconstruction = model(row)
                errors = (reconstruction - row).square()
                torch.topk(errors, k=top_k, dim=1).values.mean().item()

            total_start = time.perf_counter()
            for index in range(n_records):
                row = features[index : index + 1]
                t0 = time.perf_counter()
                reconstruction = model(row)
                errors = (reconstruction - row).square()
                torch.topk(errors, k=top_k, dim=1).values.mean().item()
                latencies_ms[index] = (time.perf_counter() - t0) * 1000
            elapsed = time.perf_counter() - total_start
    finally:
        model.train(model_was_training)

    if elapsed <= 0:
        raise RuntimeError("Inference benchmark elapsed time must be positive")

    return {
        "inference_records": int(n_records),
        "inference_warmup_records": int(min(warmup_records, n_records)),
        "inference_latency_median_ms": float(np.median(latencies_ms)),
        "inference_latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "inference_latency_p99_ms": float(np.percentile(latencies_ms, 99)),
        "inference_latency_std_ms": float(np.std(latencies_ms)),
        "inference_throughput_records_per_second": float(n_records / elapsed),
        "inference_total_seconds": float(elapsed),
    }


def benchmark_batch_inference(
    model: nn.Module,
    features: torch.Tensor,
    batch_sizes: tuple[int, ...] = BATCH_SIZES_BENCHMARK,
    n_repetitions: int = 20,
) -> list[dict[str, float | int]]:
    """Offline-mode batch throughput benchmark (MLPerf Inference offline mode).

    For each batch_size in batch_sizes, runs n_repetitions forward passes and
    records median latency per batch, throughput (records/s) and latency per record.
    """
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2-D features, got {tuple(features.shape)}")
    if features.device.type != "cpu":
        raise ValueError("batch inference benchmark expects CPU features")

    n_records = features.shape[0]
    model_was_training = model.training
    model.eval()
    results: list[dict[str, float | int]] = []
    try:
        with torch.inference_mode():
            for batch_size in batch_sizes:
                if batch_size > n_records:
                    continue
                batch = features[:batch_size]
                latencies_ms: list[float] = []
                for _ in range(n_repetitions):
                    t0 = time.perf_counter()
                    _ = model(batch)
                    latencies_ms.append((time.perf_counter() - t0) * 1000)
                arr = np.array(latencies_ms, dtype=np.float64)
                median_ms = float(np.median(arr))
                results.append(
                    {
                        "batch_size": int(batch_size),
                        "n_repetitions": int(n_repetitions),
                        "batch_latency_median_ms": median_ms,
                        "batch_latency_p95_ms": float(np.percentile(arr, 95)),
                        "batch_latency_std_ms": float(np.std(arr)),
                        "batch_throughput_records_per_second": float(batch_size / (median_ms / 1000)),
                        "per_record_latency_ms": float(median_ms / batch_size),
                    }
                )
    finally:
        model.train(model_was_training)
    return results


def metric_record_values(
    metrics: dict[str, Any],
    required_keys: tuple[str, ...],
) -> dict[str, float | int]:
    missing = [key for key in required_keys if key not in metrics]
    if missing:
        raise ValueError(f"Edge inference metrics are missing fields: {missing}")
    return {key: metrics[key] for key in required_keys}

