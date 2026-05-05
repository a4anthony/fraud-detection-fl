"""
FL aggregation helpers and utilities.
"""

import numpy as np
import joblib


def aggregate_models_simple(client_models, weights=None):
    """Simple weighted averaging of ensemble models from multiple clients.

    For tree-based models, this selects the best-performing client's models
    rather than averaging parameters (tree models can't be weight-averaged).

    Args:
        client_models: list of (xgb_model, rf_model, auprc_score) tuples
        weights: optional weight per client (defaults to equal)

    Returns:
        best_xgb, best_rf from the highest-scoring client
    """
    if weights is None:
        weights = [1.0 / len(client_models)] * len(client_models)

    # For tree models, select best client (weighted by performance)
    scores = [m[2] for m in client_models]
    weighted_scores = [s * w for s, w in zip(scores, weights)]
    best_idx = np.argmax(weighted_scores)

    return client_models[best_idx][0], client_models[best_idx][1]


def compute_communication_cost(parameters):
    """Compute bytes transferred for a set of model parameters."""
    total_bytes = 0
    for param in parameters:
        if isinstance(param, np.ndarray):
            total_bytes += param.nbytes
        elif isinstance(param, bytes):
            total_bytes += len(param)
    return total_bytes
