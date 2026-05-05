"""
Centralized baseline models: Logistic Regression, Random Forest, XGBoost,
and XGB-RF Soft Voting Ensemble.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import numpy as np
from pathlib import Path


def get_baseline_models(xgb_params=None, rf_params=None):
    """Return dict of baseline models with default or custom params."""
    xgb_p = {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 0.9, "eval_metric": "aucpr", "random_state": 42,
        "use_label_encoder": False, "n_jobs": -1,
    }
    rf_p = {
        "n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1,
    }

    if xgb_params:
        xgb_p.update(xgb_params)
    if rf_params:
        rf_p.update(rf_params)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(**rf_p),
        "XGBoost": XGBClassifier(**xgb_p),
    }
    return models


def train_baselines(X_train, y_train, models=None, cv_folds=5,
                    output_dir="outputs/models"):
    """Train all baseline and ensemble models with cross-validation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = get_baseline_models()

    results = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
        )
        results[name] = {
            "model": model,
            "cv_auprc_mean": cv_scores.mean(),
            "cv_auprc_std": cv_scores.std(),
            "cv_scores": cv_scores,
        }

        model_file = output_dir / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_file)
        print(f"  CV AUPRC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Soft Voting Ensemble (XGBoost + Random Forest)
    print("Training XGB-RF Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", models["XGBoost"]),
            ("rf", models["Random Forest"]),
        ],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)

    cv_scores = cross_val_score(
        ensemble, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
    )
    results["XGB-RF Ensemble"] = {
        "model": ensemble,
        "cv_auprc_mean": cv_scores.mean(),
        "cv_auprc_std": cv_scores.std(),
        "cv_scores": cv_scores,
    }
    joblib.dump(ensemble, output_dir / "xgb_rf_ensemble.joblib")
    print(f"  CV AUPRC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return results
