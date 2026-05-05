"""
COM748 Task 1.2 — Class-weighted retraining + threshold-optimised evaluation.

For each FL client:
  1. Compute scale_pos_weight = #neg / #pos on the training labels.
  2. Retrain XGBoost with the GA-optimised hyperparameters plus scale_pos_weight.
  3. Retrain Random Forest with the same hyperparameters plus class_weight='balanced'.
  4. Save weighted models as fl_{client}_{xgb,rf}_weighted.joblib.
  5. Re-evaluate the weighted ensemble at val-selected optimal threshold using
     the same 50/50 test split methodology as optimise_threshold.py.
  6. Write A/B comparison CSV (unweighted vs weighted) at both default and optimal
     thresholds.

NB: training data on disk is SMOTE-balanced (50/50). scale_pos_weight against a
pre-balanced set is close to 1.0, so we recompute weights from the natural-
prevalence test distribution as a proxy for deployment class priors. This is
documented in the CSV via the ScalePosWeight column.
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper_v2_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
VAL_FRAC = 0.50

# GA-optimised hyperparameters (read from the unweighted saved models)
XGB_PARAMS = dict(
    learning_rate=0.20278092176694895,
    max_depth=5,
    subsample=0.9533490553166496,
    n_jobs=-1,
    eval_metric='aucpr',
    objective='binary:logistic',
    random_state=46,
)
RF_PARAMS = dict(
    n_estimators=282,
    max_depth=27,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=46,
)

CLIENTS = ['ulb', 'baf', 'synthetic']


def load_client(name):
    X_tr = pd.read_csv(os.path.join(DATA_DIR, f'{name}_X_train.csv'))
    y_tr = pd.read_csv(os.path.join(DATA_DIR, f'{name}_y_train.csv')).squeeze()
    X_te = pd.read_csv(os.path.join(DATA_DIR, f'{name}_X_test.csv'))
    y_te = pd.read_csv(os.path.join(DATA_DIR, f'{name}_y_test.csv')).squeeze()
    return X_tr, y_tr, X_te, y_te


def ensemble_proba(xgb, rf, X):
    return 0.5 * xgb.predict_proba(X)[:, 1] + 0.5 * rf.predict_proba(X)[:, 1]


def metrics_at(y_true, y_proba, t):
    y_pred = (y_proba >= t).astype(int)
    auprc = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    if y_pred.sum() == 0:
        return dict(AUPRC=auprc, F1=0.0, Precision=0.0, Recall=0.0, ROC_AUC=roc, FPR=0.0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        AUPRC=auprc,
        F1=f1_score(y_true, y_pred, zero_division=0),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        ROC_AUC=roc,
        FPR=fp / (fp + tn),
    )


def best_threshold_on_val(y_val, p_val):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 1.00, 0.01):
        y_hat = (p_val >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        f1 = f1_score(y_val, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


rows = []

print('=' * 72)
print('Task 1.2 — Class-weighted retraining + optimal-threshold evaluation')
print('=' * 72)

for name in CLIENTS:
    print(f'\n[{name}]')
    X_tr, y_tr, X_te, y_te = load_client(name)

    # Pre-balanced train → proxy class prior from natural test prevalence.
    pos, neg = int(y_te.sum()), int((y_te == 0).sum())
    spw = neg / pos
    print(f'  train N={len(X_tr):,}  test N={len(X_te):,}  test fraud%={y_te.mean()*100:.3f}')
    print(f'  scale_pos_weight (from test prior): {spw:.2f}')

    xgb_w = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    xgb_w.fit(X_tr, y_tr)
    rf_w = RandomForestClassifier(**RF_PARAMS, class_weight='balanced')
    rf_w.fit(X_tr, y_tr)

    joblib.dump(xgb_w, os.path.join(MODELS_DIR, f'fl_{name}_xgb_weighted.joblib'))
    joblib.dump(rf_w, os.path.join(MODELS_DIR, f'fl_{name}_rf_weighted.joblib'))

    xgb_u = joblib.load(os.path.join(MODELS_DIR, f'fl_{name}_xgb.joblib'))
    rf_u = joblib.load(os.path.join(MODELS_DIR, f'fl_{name}_rf.joblib'))

    X_val, X_eval, y_val, y_eval = train_test_split(
        X_te, y_te, test_size=1 - VAL_FRAC, stratify=y_te, random_state=SEED,
    )

    for label, xgb, rf in [('unweighted', xgb_u, rf_u), ('weighted', xgb_w, rf_w)]:
        p_val = ensemble_proba(xgb, rf, X_val)
        p_eval = ensemble_proba(xgb, rf, X_eval)
        t_star, f1_val = best_threshold_on_val(y_val, p_val)
        m_def = metrics_at(y_eval, p_eval, 0.5)
        m_opt = metrics_at(y_eval, p_eval, t_star)
        print(f'  {label:10s}  t*={t_star:.2f}  '
              f'F1@0.5={m_def["F1"]:.4f}  F1@t*={m_opt["F1"]:.4f}  '
              f'AUPRC={m_opt["AUPRC"]:.4f}')
        rows.append({
            'Dataset': name.upper() if name != 'synthetic' else 'Synthetic',
            'Variant': label,
            'ScalePosWeight': round(spw, 2) if label == 'weighted' else None,
            'OptimalThreshold': round(t_star, 2),
            'AUPRC': round(m_opt['AUPRC'], 4),
            'F1_at_0.5': round(m_def['F1'], 4),
            'Precision_at_0.5': round(m_def['Precision'], 4),
            'Recall_at_0.5': round(m_def['Recall'], 4),
            'F1_at_optimal': round(m_opt['F1'], 4),
            'Precision_at_optimal': round(m_opt['Precision'], 4),
            'Recall_at_optimal': round(m_opt['Recall'], 4),
            'FPR_at_optimal': round(m_opt['FPR'], 4),
            'ROC_AUC': round(m_opt['ROC_AUC'], 4),
        })

out_df = pd.DataFrame(rows)
out_csv = os.path.join(OUTPUT_DIR, 'weighted_vs_unweighted.csv')
out_df.to_csv(out_csv, index=False)

print('\n' + '=' * 72)
print(f'Wrote {out_csv}')
print('=' * 72)
print(out_df.to_string(index=False))