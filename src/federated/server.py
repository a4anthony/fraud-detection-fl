"""
Flower FL server configuration and strategy.
"""

import flwr as fl
from flwr.server.strategy import FedAvg


def get_fl_strategy(min_clients=3):
    """Return FedAvg strategy for the fraud detection FL setup."""
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )
    return strategy


def start_fl_server(num_rounds=10, min_clients=3, server_address="0.0.0.0:8080"):
    """Launch Flower FL server."""
    strategy = get_fl_strategy(min_clients=min_clients)

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return history
