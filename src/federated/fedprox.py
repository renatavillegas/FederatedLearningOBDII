from flwr.serverapp.strategy import FedProx

from src.federated.protocol import ARRAYS_KEY, CONFIG_KEY, NUM_EXAMPLES


def build_fedprox(n_clients: int, fraction_train: float, proximal_mu: float) -> FedProx:
    """FedProx: FedAvg aggregation; the strategy sends `proximal-mu` and each client adds
    (mu / 2) * ||w - w_global||^2 to its local loss (see LocalTrainer)."""
    if proximal_mu <= 0:
        raise ValueError("proximal_mu must be positive for FedProx")
    return FedProx(
        fraction_train=fraction_train,
        fraction_evaluate=0.0,
        min_train_nodes=n_clients if fraction_train == 1.0 else 1,
        min_available_nodes=n_clients,
        weighted_by_key=NUM_EXAMPLES,
        arrayrecord_key=ARRAYS_KEY,
        configrecord_key=CONFIG_KEY,
        proximal_mu=proximal_mu,
    )
