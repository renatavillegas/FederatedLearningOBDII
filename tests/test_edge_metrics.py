import numpy as np
import torch

from src.federated.edge_metrics import benchmark_inference


def test_benchmark_inference_reports_cpu_throughput_and_restores_mode():
    model = torch.nn.Linear(105, 105)
    model.train()
    features = torch.ones((8, 105), dtype=torch.float32)

    metrics = benchmark_inference(model, features, warmup_records=2)

    assert model.training
    assert metrics["inference_records"] == 8
    assert metrics["inference_warmup_records"] == 2
    assert np.isfinite(metrics["inference_latency_median_ms"])
    assert metrics["inference_latency_median_ms"] >= 0
    assert metrics["inference_latency_p95_ms"] >= metrics["inference_latency_median_ms"]
    assert metrics["inference_throughput_records_per_second"] > 0
    assert metrics["inference_total_seconds"] > 0


def test_benchmark_inference_rejects_empty_or_non_matrix_features():
    model = torch.nn.Linear(105, 105)
    try:
        benchmark_inference(model, torch.empty((0, 105)))
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("Expected empty input to be rejected")
