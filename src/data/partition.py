"""
Federated Learning data partitioning.

Creates client datasets for FL simulation where each dataset
represents a different financial institution.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def partition_natural(datasets_dict):
    """Natural partition: each dataset becomes one FL client.

    Args:
        datasets_dict: dict of {name: (X_train, X_test, y_train, y_test, ...)}

    Returns:
        List of (X_train, y_train, X_test, y_test) tuples, one per client.
    """
    clients = []
    for name, data in datasets_dict.items():
        X_train, X_test, y_train, y_test = data[0], data[1], data[2], data[3]
        clients.append({
            "name": name,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        })
    return clients


def partition_iid(X_train, y_train, X_test, y_test, num_clients=3,
                  random_state=42):
    """IID partition: random equal split across clients."""
    rng = np.random.RandomState(random_state)
    train_indices = rng.permutation(len(X_train))
    test_indices = rng.permutation(len(X_test))

    train_splits = np.array_split(train_indices, num_clients)
    test_splits = np.array_split(test_indices, num_clients)

    clients = []
    for i, (tr_idx, te_idx) in enumerate(zip(train_splits, test_splits)):
        clients.append({
            "name": f"client_{i}",
            "X_train": X_train.iloc[tr_idx].reset_index(drop=True),
            "y_train": y_train.iloc[tr_idx].reset_index(drop=True)
                        if hasattr(y_train, "iloc")
                        else pd.Series(y_train).iloc[tr_idx].reset_index(drop=True),
            "X_test": X_test.iloc[te_idx].reset_index(drop=True),
            "y_test": y_test.iloc[te_idx].reset_index(drop=True)
                      if hasattr(y_test, "iloc")
                      else pd.Series(y_test).iloc[te_idx].reset_index(drop=True),
        })
    return clients


def partition_non_iid(X_train, y_train, X_test, y_test, num_clients=3,
                      alpha=0.5, random_state=42):
    """Non-IID partition using Dirichlet distribution for label skew."""
    rng = np.random.RandomState(random_state)
    y_train_s = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    y_test_s = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

    # Partition training data
    label_indices = {l: np.where(y_train_s.values == l)[0]
                     for l in y_train_s.unique()}
    client_train_idx = [[] for _ in range(num_clients)]

    for label, idxs in label_indices.items():
        proportions = rng.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(idxs, split_points)):
            client_train_idx[client_id].extend(chunk.tolist())

    # Split test data proportionally
    test_indices = rng.permutation(len(X_test))
    test_splits = np.array_split(test_indices, num_clients)

    clients = []
    for i in range(num_clients):
        tr_idx = client_train_idx[i]
        te_idx = test_splits[i]
        clients.append({
            "name": f"client_{i}",
            "X_train": X_train.iloc[tr_idx].reset_index(drop=True),
            "y_train": y_train_s.iloc[tr_idx].reset_index(drop=True),
            "X_test": X_test.iloc[te_idx].reset_index(drop=True),
            "y_test": y_test_s.iloc[te_idx].reset_index(drop=True),
        })
    return clients


def get_common_features(datasets_dict):
    """Find common numerical features across all datasets for FL.

    For heterogeneous datasets, aligns to a common feature space
    by padding missing features with zeros.
    """
    all_features = [set(data[0].columns) for data in datasets_dict.values()]
    common = set.intersection(*all_features) if all_features else set()
    return sorted(common)


def align_to_common_features(X, common_features):
    """Align a dataframe to a common feature set, filling missing with 0."""
    aligned = pd.DataFrame(0, index=X.index, columns=common_features)
    shared = [c for c in common_features if c in X.columns]
    aligned[shared] = X[shared]
    return aligned


def save_client_data(clients, output_dir="data/federated"):
    """Save partitioned client data to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for client in clients:
        name = client["name"]
        client["X_train"].to_csv(output_dir / f"{name}_X_train.csv", index=False)
        client["X_test"].to_csv(output_dir / f"{name}_X_test.csv", index=False)
        pd.Series(client["y_train"]).to_csv(
            output_dir / f"{name}_y_train.csv", index=False
        )
        pd.Series(client["y_test"]).to_csv(
            output_dir / f"{name}_y_test.csv", index=False
        )
    print(f"Saved {len(clients)} client datasets to {output_dir}/")
