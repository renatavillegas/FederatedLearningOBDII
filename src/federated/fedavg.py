from collections.abc import Sequence

import numpy as np
from flwr.serverapp.strategy import FedAvg

from src.federated.protocol import ARRAYS_KEY, CONFIG_KEY, NUM_EXAMPLES


def build_fedavg(n_clients: int, fraction_train: float) -> FedAvg:
    """FedAvg weighted by the number of local training records; evaluation runs on the server."""
    return FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=0.0,
        min_train_nodes=n_clients if fraction_train == 1.0 else 1,
        min_available_nodes=n_clients,
        weighted_by_key=NUM_EXAMPLES,
        arrayrecord_key=ARRAYS_KEY,
        configrecord_key=CONFIG_KEY,
    )


def weighted_average(client_arrays: Sequence[Sequence[np.ndarray]], weights: Sequence[float]) -> list[np.ndarray]:
    """Reference implementation of the FedAvg aggregation, used to verify the strategy."""
    weights = np.asarray(weights, dtype=float)
    if len(client_arrays) != len(weights) or weights.sum() <= 0:
        raise ValueError("One positive weight per client is required")
    factors = weights / weights.sum()
    return [
        sum(factor * arrays[layer] for factor, arrays in zip(factors, client_arrays))
        for layer in range(len(client_arrays[0]))
    ]
