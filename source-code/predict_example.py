"""
Example: How to load and use the trained fraud detection models.

Run from the project root:
    python predict_example.py
"""

import joblib
import pandas as pd
import numpy as np
import shap

MODELS_DIR = "outputs/models"
DATA_DIR = "data/processed"

# ──────────────────────────────────────────────
# 1. LOAD A MODEL
# ──────────────────────────────────────────────
# Available models (all trained on BAF dataset, 31 features):
#   logistic_regression.joblib     — simple baseline
#   random_forest.joblib           — RF baseline
#   xgboost.joblib                 — XGBoost baseline
#   xgb_rf_ensemble.joblib         — centralized soft-voting ensemble
#   ga_optimized_ensemble.joblib   — GA-tuned ensemble (best centralized)
#   fl_baf_xgb.joblib              — FL-trained XGBoost (BAF client)
#   fl_baf_rf.joblib               — FL-trained Random Forest (BAF client)

print("Loading model...")
model = joblib.load(f"{MODELS_DIR}/xgboost.joblib")
print(f"  Model: {type(model).__name__}, features: {model.n_features_in_}")

# ──────────────────────────────────────────────
# 2. LOAD TEST DATA
# ──────────────────────────────────────────────
X_test = pd.read_csv(f"{DATA_DIR}/baf_X_test.csv")
y_test = pd.read_csv(f"{DATA_DIR}/baf_y_test.csv").squeeze()
print(f"  Test set: {X_test.shape[0]:,} transactions, {X_test.shape[1]} features")

# ──────────────────────────────────────────────
# 3. PREDICT ON A SINGLE TRANSACTION
# ──────────────────────────────────────────────
transaction = X_test.iloc[[0]]  # first transaction
probability = model.predict_proba(transaction)[0, 1]  # fraud probability
prediction = model.predict(transaction)[0]  # 0 = legitimate, 1 = fraud

print(f"\n--- Single Transaction Prediction ---")
print(f"  Fraud probability: {probability:.4f}")
print(f"  Prediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
print(f"  Actual label: {'FRAUD' if y_test.iloc[0] == 1 else 'LEGITIMATE'}")

# ──────────────────────────────────────────────
# 4. PREDICT ON A BATCH
# ──────────────────────────────────────────────
batch = X_test.iloc[:100]
probabilities = model.predict_proba(batch)[:, 1]
predictions = model.predict(batch)

print(f"\n--- Batch Prediction (100 transactions) ---")
print(f"  Flagged as fraud: {predictions.sum()}")
print(f"  Actual frauds: {y_test.iloc[:100].sum()}")
print(f"  Max fraud probability: {probabilities.max():.4f}")

# ──────────────────────────────────────────────
# 5. EXPLAIN A PREDICTION WITH SHAP
# ──────────────────────────────────────────────
print(f"\n--- SHAP Explanation ---")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(transaction)

# For XGBoost, shap_values is a list [class_0, class_1] — take fraud class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Show top 5 features driving this prediction
feature_names = list(X_test.columns)
sv = shap_values[0]  # first (only) sample
top_indices = np.argsort(np.abs(sv))[-5:][::-1]

print(f"  Top 5 features for this prediction:")
for idx in top_indices:
    direction = "toward FRAUD" if sv[idx] > 0 else "toward LEGIT"
    print(f"    {feature_names[idx]:>35s}: SHAP = {sv[idx]:+.4f} ({direction})")

# ──────────────────────────────────────────────
# 6. USE THE FL ENSEMBLE (XGBoost + RF)
# ──────────────────────────────────────────────
print(f"\n--- FL Ensemble (manual soft voting) ---")
fl_xgb = joblib.load(f"{MODELS_DIR}/fl_baf_xgb.joblib")
fl_rf = joblib.load(f"{MODELS_DIR}/fl_baf_rf.joblib")

xgb_proba = fl_xgb.predict_proba(transaction)[0, 1]
rf_proba = fl_rf.predict_proba(transaction)[0, 1]
ensemble_proba = (xgb_proba + rf_proba) / 2  # soft voting = average probabilities

print(f"  XGBoost says: {xgb_proba:.4f}")
print(f"  RF says:      {rf_proba:.4f}")
print(f"  Ensemble:     {ensemble_proba:.4f}")
print(f"  Prediction:   {'FRAUD' if ensemble_proba >= 0.5 else 'LEGITIMATE'}")

# ──────────────────────────────────────────────
# 7. ADJUST THE THRESHOLD
# ──────────────────────────────────────────────
# Default threshold is 0.5, but you can tune it:
#   Lower threshold (e.g., 0.3) → catch more fraud (higher recall) but more false alarms
#   Higher threshold (e.g., 0.7) → fewer false alarms (higher precision) but miss more fraud

threshold = 0.3
custom_pred = (ensemble_proba >= threshold)
print(f"\n  With threshold={threshold}: {'FRAUD' if custom_pred else 'LEGITIMATE'}")

print("\nDone.")
