"""
COM748 Task 1.1 — Threshold-optimised per-client evaluation.

For each FL client (ULB, BAF, Synthetic):
  1. Load trained XGB+RF ensemble models.
  2. Split the held-out test set 50/50 stratified into (val, eval). The
     training partitions are SMOTE-balanced so cannot serve as a natural-
     prevalence validation signal; splitting test is the standard fallback.
  3. Score the ensemble on val; pick threshold maximising F1 on a fine grid
     (0.01..0.99 step 0.01).
  4. Report metrics on the disjoint eval half at that threshold — no leakage
     since val and eval are disjoint and neither was seen during training.

Outputs:
  paper_v2_outputs/per_dataset_results_optimised.csv
  paper_v2_outputs/pr_curves_per_client.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'federated')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper_v2_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VAL_FRAC = 0.50  # share of TEST used as val for threshold selection
SEED = 42

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


def ensemble_proba(xgb, rf, X):
    return 0.5 * xgb.predict_proba(X)[:, 1] + 0.5 * rf.predict_proba(X)[:, 1]


def metrics_at_threshold(y_true, y_proba, t):
    y_pred = (y_proba >= t).astype(int)
    auprc = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    if y_pred.sum() == 0:
        return {
            'AUPRC': auprc, 'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0,
            'ROC_AUC': roc, 'FPR': 0.0,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUPRC': auprc,
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'ROC_AUC': roc,
        'FPR': fp / (fp + tn),
    }


def best_threshold_on_val(y_val, p_val, grid=np.arange(0.01, 1.00, 0.01)):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (p_val >= t).astype(int)
        if y_hat.sum() == 0:
            continue
        f1 = f1_score(y_val, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


rows = []
pr_data = {}

print('=' * 70)
print('Task 1.1 — Threshold-optimised evaluation (val-selected, eval-reported)')
print(f'Val/eval split of TEST: {VAL_FRAC:.0%} stratified, seed={SEED}')
print('=' * 70)

for name, p in CLIENTS.items():
    print(f'\n[{name}]')
    X_test_full = pd.read_csv(p['X_test'])
    y_test_full = pd.read_csv(p['y_test']).squeeze()

    X_val, X_eval, y_val, y_eval = train_test_split(
        X_test_full, y_test_full, test_size=1 - VAL_FRAC,
        stratify=y_test_full, random_state=SEED,
    )
    print(f'  val  : {len(X_val):,} rows, {int(y_val.sum()):,} fraud ({y_val.mean()*100:.3f}%)')
    print(f'  eval : {len(X_eval):,} rows, {int(y_eval.sum()):,} fraud ({y_eval.mean()*100:.3f}%)')

    xgb = joblib.load(p['xgb_model'])
    rf = joblib.load(p['rf_model'])
    p_val = ensemble_proba(xgb, rf, X_val)
    p_eval = ensemble_proba(xgb, rf, X_eval)

    t_star, f1_val = best_threshold_on_val(y_val, p_val)
    m_default = metrics_at_threshold(y_eval, p_eval, 0.5)
    m_opt = metrics_at_threshold(y_eval, p_eval, t_star)

    print(f'  t*        : {t_star:.2f}  (val F1 {f1_val:.4f})')
    print(f'  eval@0.5  : F1 {m_default["F1"]:.4f}  P {m_default["Precision"]:.4f}  R {m_default["Recall"]:.4f}')
    print(f'  eval@t*   : F1 {m_opt["F1"]:.4f}  P {m_opt["Precision"]:.4f}  R {m_opt["Recall"]:.4f}')

    rows.append({
        'Dataset': name,
        'EvalSize': len(X_eval),
        'FraudRate_pct': round(y_eval.mean() * 100, 3),
        'OptimalThreshold': round(t_star, 2),
        'AUPRC': round(m_opt['AUPRC'], 4),
        'F1_at_optimal': round(m_opt['F1'], 4),
        'Precision_at_optimal': round(m_opt['Precision'], 4),
        'Recall_at_optimal': round(m_opt['Recall'], 4),
        'ROC_AUC': round(m_opt['ROC_AUC'], 4),
        'FPR_at_optimal': round(m_opt['FPR'], 4),
        'F1_at_0.5': round(m_default['F1'], 4),
        'Precision_at_0.5': round(m_default['Precision'], 4),
        'Recall_at_0.5': round(m_default['Recall'], 4),
    })

    prec, rec, thr = precision_recall_curve(y_eval, p_eval)
    pr_data[name] = {
        'precision': prec, 'recall': rec, 'thresholds': thr,
        't_star': t_star,
        'opt_point': (m_opt['Recall'], m_opt['Precision']),
    }

out_df = pd.DataFrame(rows)
cols = [
    'Dataset', 'EvalSize', 'FraudRate_pct', 'OptimalThreshold',
    'AUPRC', 'F1_at_optimal', 'Precision_at_optimal', 'Recall_at_optimal',
    'ROC_AUC', 'FPR_at_optimal',
    'F1_at_0.5', 'Precision_at_0.5', 'Recall_at_0.5',
]
out_df = out_df[cols]
out_csv = os.path.join(OUTPUT_DIR, 'per_dataset_results_optimised.csv')
out_df.to_csv(out_csv, index=False)

print('\n' + '=' * 70)
print(f'Wrote {out_csv}')
print('=' * 70)
print(out_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
for ax, (name, d) in zip(axes, pr_data.items()):
    ax.plot(d['recall'], d['precision'], linewidth=1.8, label='PR curve')
    r_star, p_star = d['opt_point']
    ax.scatter([r_star], [p_star], color='red', zorder=5, s=60,
               label=f't*={d["t_star"]:.2f}\nP={p_star:.2f}, R={r_star:.2f}')
    ax.set_xlabel('Recall')
    ax.set_title(name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
axes[0].set_ylabel('Precision')
fig.suptitle('Per-client PR curves with validation-selected operating point', fontsize=11)
fig.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'pr_curves_per_client.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'\nWrote {fig_path}')
