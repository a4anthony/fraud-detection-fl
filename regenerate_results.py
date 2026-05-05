"""
COM748 — Regenerate per-dataset results + threshold sweep for paper v2.
Produces two CSVs and one PNG for paper integration.

USAGE (from fraud-detection-fl/ directory):
    python regenerate_results.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================
USE_XGB_ONLY = False  # Full XGB+RF ensemble — all RF models are saved on disk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated')

CLIENTS = {
    'ULB': {
        'xgb_model': os.path.join(MODELS_DIR, 'fl_ulb_xgb.joblib'),
        'rf_model':  os.path.join(MODELS_DIR, 'fl_ulb_rf.joblib'),
        'X_test':    os.path.join(DATA_DIR, 'ulb_X_test.csv'),
        'y_test':    os.path.join(DATA_DIR, 'ulb_y_test.csv'),
    },
    'BAF': {
        'xgb_model': os.path.join(MODELS_DIR, 'fl_baf_xgb.joblib'),
        'rf_model':  os.path.join(MODELS_DIR, 'fl_baf_rf.joblib'),
        'X_test':    os.path.join(DATA_DIR, 'baf_X_test.csv'),
        'y_test':    os.path.join(DATA_DIR, 'baf_y_test.csv'),
    },
    'Synthetic': {
        'xgb_model': os.path.join(MODELS_DIR, 'fl_synthetic_xgb.joblib'),
        'rf_model':  os.path.join(MODELS_DIR, 'fl_synthetic_rf.joblib'),
        'X_test':    os.path.join(DATA_DIR, 'synthetic_X_test.csv'),
        'y_test':    os.path.join(DATA_DIR, 'synthetic_y_test.csv'),
    },
}

THRESHOLD_CLIENT = 'BAF'
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper_v2_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def load_data(paths):
    X = pd.read_csv(paths['X_test'])
    y = pd.read_csv(paths['y_test']).squeeze()
    return X, y

def predict_proba_client(paths, X):
    xgb = joblib.load(paths['xgb_model'])
    xgb_proba = xgb.predict_proba(X)[:, 1]
    if USE_XGB_ONLY:
        return xgb_proba
    rf = joblib.load(paths['rf_model'])
    rf_proba = rf.predict_proba(X)[:, 1]
    return 0.5 * xgb_proba + 0.5 * rf_proba

def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    auprc = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    if y_pred.sum() == 0:
        return {
            'AUPRC':     round(auprc, 4),
            'F1':        0.0,
            'Precision': 0.0,
            'Recall':    0.0,
            'ROC_AUC':   round(roc, 4),
            'FPR':       0.0,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUPRC':     round(auprc, 4),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'ROC_AUC':   round(roc, 4),
        'FPR':       round(fp / (fp + tn), 4),
    }

# ============================================================
# STEP 1 — Per-dataset evaluation
# ============================================================
print('=' * 60)
print(f'Per-dataset evaluation (USE_XGB_ONLY={USE_XGB_ONLY})')
print('=' * 60)

per_dataset_rows = []
proba_cache = {}

for name, paths in CLIENTS.items():
    print(f'\n[{name}]')
    try:
        X, y = load_data(paths)
        print(f'  Test set: {len(X):,} rows, {int(y.sum()):,} fraud ({y.mean()*100:.3f}%)')
        y_proba = predict_proba_client(paths, X)
        proba_cache[name] = (y, y_proba)
        metrics = evaluate(y, y_proba, threshold=0.5)
        metrics['Dataset'] = name
        metrics['TestSize'] = len(X)
        metrics['FraudRate_pct'] = round(y.mean() * 100, 3)
        per_dataset_rows.append(metrics)
        for k, v in metrics.items():
            if k not in ('Dataset', 'TestSize'):
                print(f'  {k:12s}: {v}')
    except FileNotFoundError as e:
        print(f'  SKIPPED — missing file: {e}')
    except Exception as e:
        print(f'  ERROR: {e}')

per_dataset_df = pd.DataFrame(per_dataset_rows)
cols = ['Dataset', 'TestSize', 'FraudRate_pct', 'AUPRC', 'F1', 'Precision', 'Recall', 'ROC_AUC', 'FPR']
per_dataset_df = per_dataset_df[[c for c in cols if c in per_dataset_df.columns]]
per_dataset_csv = os.path.join(OUTPUT_DIR, 'per_dataset_results.csv')
per_dataset_df.to_csv(per_dataset_csv, index=False)
print(f'\nWrote {per_dataset_csv}')
print(per_dataset_df.to_string(index=False))

# ============================================================
# STEP 2 — Threshold sweep on BAF
# ============================================================
print('\n' + '=' * 60)
print(f'Threshold sweep on {THRESHOLD_CLIENT}')
print('=' * 60)

if THRESHOLD_CLIENT not in proba_cache:
    print(f'{THRESHOLD_CLIENT} probas not available — threshold sweep skipped.')
else:
    y_true, y_proba = proba_cache[THRESHOLD_CLIENT]
    thresholds = np.arange(0.05, 0.96, 0.05)
    sweep_rows = []
    for t in thresholds:
        m = evaluate(y_true, y_proba, threshold=t)
        m['Threshold'] = round(t, 2)
        sweep_rows.append(m)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df = sweep_df[['Threshold', 'Precision', 'Recall', 'F1', 'AUPRC', 'FPR']]
    sweep_csv = os.path.join(OUTPUT_DIR, 'threshold_analysis.csv')
    sweep_df.to_csv(sweep_csv, index=False)
    print(f'Wrote {sweep_csv}')
    print(sweep_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(sweep_df.Threshold, sweep_df.Precision, label='Precision', marker='o', linewidth=1.8)
    ax.plot(sweep_df.Threshold, sweep_df.Recall,    label='Recall',    marker='s', linewidth=1.8)
    ax.plot(sweep_df.Threshold, sweep_df.F1,        label='F1',        marker='^', linewidth=1.8)
    ax.axvline(0.5, color='grey', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'Precision / Recall / F1 vs. Threshold ({THRESHOLD_CLIENT})')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'threshold_sweep.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Wrote {fig_path}')

print('\n' + '=' * 60)
print('DONE')
print('=' * 60)
print(f'USE_XGB_ONLY was: {USE_XGB_ONLY}')
print(f'Outputs in: {OUTPUT_DIR}')
