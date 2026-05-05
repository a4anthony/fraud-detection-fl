"""
COM748 Task 1.3 — Ablation study.

Five configurations evaluated per client (ULB, BAF, Synthetic):
  (1) LR baseline       — Logistic regression, default sklearn params.
  (2) XGB solo          — XGBClassifier with library defaults.
  (3) XGB+RF ensemble   — Soft voting (0.5/0.5), library defaults for each base.
  (4) + GA tuning       — Ensemble with GA-optimised hyperparameters.
  (5) + FL (full)       — Pre-trained federated ensemble (loaded from disk).

Metrics reported: AUPRC, F1, Precision, Recall, ROC-AUC on the held-out test set.
AUPRC is threshold-free; F1/P/R reported at the default 0.5 threshold for
consistency with the v2.1 paper's existing ablation table.

All configs train on the same SMOTE-balanced train partition and evaluate on
the same natural-prevalence test partition so the comparison isolates the
modelling choice, not the data pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper_v2_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 46

GA_XGB = dict(
    n_estimators=300, learning_rate=0.20278092176694895, max_depth=5,
    subsample=0.9533490553166496, n_jobs=-1, eval_metric='aucpr',
    objective='binary:logistic', random_state=SEED,
)
GA_RF = dict(
    n_estimators=282, max_depth=27, min_samples_leaf=1,
    n_jobs=-1, random_state=SEED,
)

CLIENTS = ['ulb', 'baf', 'synthetic']


def load_client(name):
    X_tr = pd.read_csv(os.path.join(DATA_DIR, f'{name}_X_train.csv'))
    y_tr = pd.read_csv(os.path.join(DATA_DIR, f'{name}_y_train.csv')).squeeze()
    X_te = pd.read_csv(os.path.join(DATA_DIR, f'{name}_X_test.csv'))
    y_te = pd.read_csv(os.path.join(DATA_DIR, f'{name}_y_test.csv')).squeeze()
    return X_tr, y_tr, X_te, y_te


def proba_ensemble(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def metrics(y, p, t=0.5):
    y_hat = (p >= t).astype(int)
    auprc = average_precision_score(y, p)
    roc = roc_auc_score(y, p)
    if y_hat.sum() == 0:
        return dict(AUPRC=auprc, F1=0.0, Precision=0.0, Recall=0.0, ROC_AUC=roc, FPR=0.0)
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    return dict(
        AUPRC=auprc,
        F1=f1_score(y, y_hat, zero_division=0),
        Precision=precision_score(y, y_hat, zero_division=0),
        Recall=recall_score(y, y_hat, zero_division=0),
        ROC_AUC=roc,
        FPR=fp / (fp + tn),
    )


rows = []

print('=' * 72)
print('Task 1.3 — Ablation study (5 configs x 3 clients, AUPRC@test)')
print('=' * 72)

for name in CLIENTS:
    print(f'\n[{name}]')
    X_tr, y_tr, X_te, y_te = load_client(name)
    print(f'  train N={len(X_tr):,}  test N={len(X_te):,}  test fraud%={y_te.mean()*100:.3f}')

    # (1) LR baseline
    lr = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    m = metrics(y_te, p_lr); m.update(Dataset=name.upper() if name != 'synthetic' else 'Synthetic', Config='1. LR baseline')
    rows.append(m); print(f'  LR            AUPRC={m["AUPRC"]:.4f}  F1={m["F1"]:.4f}  ROC={m["ROC_AUC"]:.4f}')

    # (2) XGB solo — library defaults
    xgb_def = XGBClassifier(eval_metric='aucpr', objective='binary:logistic',
                             n_jobs=-1, random_state=SEED)
    xgb_def.fit(X_tr, y_tr)
    p_xgb = xgb_def.predict_proba(X_te)[:, 1]
    m = metrics(y_te, p_xgb); m.update(Dataset=name.upper() if name != 'synthetic' else 'Synthetic', Config='2. XGB solo')
    rows.append(m); print(f'  XGB solo      AUPRC={m["AUPRC"]:.4f}  F1={m["F1"]:.4f}  ROC={m["ROC_AUC"]:.4f}')

    # (3) XGB+RF ensemble, library defaults
    rf_def = RandomForestClassifier(n_jobs=-1, random_state=SEED)
    rf_def.fit(X_tr, y_tr)
    p_ens = 0.5 * p_xgb + 0.5 * rf_def.predict_proba(X_te)[:, 1]
    m = metrics(y_te, p_ens); m.update(Dataset=name.upper() if name != 'synthetic' else 'Synthetic', Config='3. XGB+RF ensemble')
    rows.append(m); print(f'  XGB+RF        AUPRC={m["AUPRC"]:.4f}  F1={m["F1"]:.4f}  ROC={m["ROC_AUC"]:.4f}')

    # (4) + GA tuning
    xgb_ga = XGBClassifier(**GA_XGB)
    xgb_ga.fit(X_tr, y_tr)
    rf_ga = RandomForestClassifier(**GA_RF)
    rf_ga.fit(X_tr, y_tr)
    p_ga = 0.5 * xgb_ga.predict_proba(X_te)[:, 1] + 0.5 * rf_ga.predict_proba(X_te)[:, 1]
    m = metrics(y_te, p_ga); m.update(Dataset=name.upper() if name != 'synthetic' else 'Synthetic', Config='4. + GA tuning')
    rows.append(m); print(f'  +GA           AUPRC={m["AUPRC"]:.4f}  F1={m["F1"]:.4f}  ROC={m["ROC_AUC"]:.4f}')

    # (5) + FL (load pre-trained federated ensemble)
    fl_xgb = joblib.load(os.path.join(MODELS_DIR, f'fl_{name}_xgb.joblib'))
    fl_rf = joblib.load(os.path.join(MODELS_DIR, f'fl_{name}_rf.joblib'))
    p_fl = 0.5 * fl_xgb.predict_proba(X_te)[:, 1] + 0.5 * fl_rf.predict_proba(X_te)[:, 1]
    m = metrics(y_te, p_fl); m.update(Dataset=name.upper() if name != 'synthetic' else 'Synthetic', Config='5. + FL (full)')
    rows.append(m); print(f'  +FL           AUPRC={m["AUPRC"]:.4f}  F1={m["F1"]:.4f}  ROC={m["ROC_AUC"]:.4f}')

df = pd.DataFrame(rows)
df = df[['Dataset', 'Config', 'AUPRC', 'F1', 'Precision', 'Recall', 'ROC_AUC', 'FPR']]
for c in ['AUPRC', 'F1', 'Precision', 'Recall', 'ROC_AUC', 'FPR']:
    df[c] = df[c].round(4)

out_csv = os.path.join(OUTPUT_DIR, 'ablation_results.csv')
df.to_csv(out_csv, index=False)

print('\n' + '=' * 72)
print(f'Wrote {out_csv}')
print('=' * 72)
print(df.to_string(index=False))

# Pivot: AUPRC per config x client, for paper-ready table
pivot = df.pivot(index='Config', columns='Dataset', values='AUPRC').round(4)
pivot_csv = os.path.join(OUTPUT_DIR, 'ablation_auprc_pivot.csv')
pivot.to_csv(pivot_csv)
print(f'\nPivot AUPRC (paper-ready):  {pivot_csv}')
print(pivot.to_string())
