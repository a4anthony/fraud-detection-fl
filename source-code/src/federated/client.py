"""
Flower FL client for fraud detection ensemble.

Each client trains an XGBoost + Random Forest ensemble on local data
and communicates model parameters to the server for aggregation.
"""

import flwr as fl
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, f1_score
from xgboost import XGBClassifier
import numpy as np
import pickle


class FraudDetectionClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, client_id,
                 xgb_params=None, rf_params=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id

        xgb_p = {
            "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
            "subsample": 0.9, "eval_metric": "aucpr", "random_state": 42,
            "n_jobs": -1,
        }
        rf_p = {
            "n_estimators": 200, "max_depth": 20, "random_state": 42,
            "n_jobs": -1,
        }
        if xgb_params:
            xgb_p.update(xgb_params)
        if rf_params:
            rf_p.update(rf_params)

        self.xgb = XGBClassifier(**xgb_p)
        self.rf = RandomForestClassifier(**rf_p)

    def get_parameters(self, config):
        """Serialize models as byte arrays for FL transport."""
        xgb_bytes = pickle.dumps(self.xgb)
        rf_bytes = pickle.dumps(self.rf)
        return [
            np.frombuffer(xgb_bytes, dtype=np.uint8),
            np.frombuffer(rf_bytes, dtype=np.uint8),
        ]

    def set_parameters(self, parameters):
        """Deserialize aggregated model parameters."""
        if len(parameters) >= 2:
            self.xgb = pickle.loads(parameters[0].tobytes())
            self.rf = pickle.loads(parameters[1].tobytes())

    def fit(self, parameters, config):
        """Train on local data."""
        if parameters:
            self.set_parameters(parameters)

        self.xgb.fit(self.X_train, self.y_train)
        self.rf.fit(self.X_train, self.y_train)

        return self.get_parameters(config), len(self.X_train), {
            "client_id": self.client_id,
        }

    def evaluate(self, parameters, config):
        """Evaluate ensemble on local test data."""
        self.set_parameters(parameters)

        xgb_proba = self.xgb.predict_proba(self.X_test)[:, 1]
        rf_proba = self.rf.predict_proba(self.X_test)[:, 1]
        ensemble_proba = (xgb_proba + rf_proba) / 2
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)

        auprc = average_precision_score(self.y_test, ensemble_proba)
        f1 = f1_score(self.y_test, ensemble_pred)

        return float(1 - auprc), len(self.X_test), {
            "auprc": float(auprc),
            "f1": float(f1),
            "client_id": self.client_id,
        }
