"""
Task 2 — Federated class-weighted (no SMOTE) experiment.

Mirrors the existing federated learning protocol (notebook 05) — 10 rounds,
3 clients, per-round seed = 42 + round_num, best-round-per-client model
selection — but trains on **natural-prevalence** data with class weighting
instead of SMOTE-balanced data.

Goal: triangulate federation cost vs methodology cost. Reading the resulting
ULB row alongside Task 1's results decomposes the gap to FraudX AI cleanly:

    federated_smote_ulb            (existing)        AUPRC = 0.8734
    centralised_no_smote_w_2_0     (Task 1 headline) AUPRC = 0.8987
    federated_class_weighted_ulb   (this script)     AUPRC = ?
    fraudx_ai_reported                               AUPRC = 0.97

Outputs:
  results-tables/federated_class_weighted.csv
  results-tables/federated_class_weighted_meta.json

Per-client class weights are computed from each client's natural train-set
prevalence — i.e., scale_pos_weight = N_neg / N_pos per client. This is the
"aggressive reweighting" regime; it is NOT a match to FraudX AI's {0:1, 1:2}.
The role of this experiment is to isolate what federation costs at the
natural-ratio operating point, complementing Task 1's matched-methodology
centralised result.

Note: each round retrains all three clients from scratch with seed 42+round_num
(matching notebook 05). Round-to-round variance is small but real. We save
the best-round per client to mirror the existing project artefact.
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data.preprocess import (
    load_and_clean_ulb, load_and_clean_baf, load_and_clean_synthetic,
    encode_categoricals,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results-tables')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
NUM_ROUNDS = 10
TEST_SIZE = 0.20
VAL_FRAC = 0.50

GA_PARAMS = dict(
    xgb_n_estimators=300,
    xgb_max_depth=5,
    xgb_learning_rate=0.20278092176694895,
    xgb_subsample=0.9533490553166496,
    rf_n_estimators=282,
    rf_max_depth=27,
)


def load_client_natural_prevalence(name):
    """Reproduce the project preprocess pipeline with apply_smote=False
    so the training partition retains natural class prevalence.

    Test partition is identical to data/processed/{name}_X_test.csv
    (deterministic stratified split with seed 42).
    """
    if name == 'ulb':
        df, target = load_and_clean_ulb(os.path.join(RAW_DIR, 'ulb_creditcard.csv'))
    elif name == 'baf':
        df, target = load_and_clean_baf(os.path.join(RAW_DIR, 'baf_base.csv'))
    elif name == 'synthetic':
        df, target = load_and_clean_synthetic(
            os.path.join(RAW_DIR, 'synthetic_fraud.csv'), sample_size=500_000)
    else:
        raise ValueError(name)

    df, _ = encode_categoricals(df.copy(), target)
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )
    return X_tr, X_te, y_tr.reset_index(drop=True), y_te.reset_index(drop=True)


def metrics_at(y_true, y_proba, t):
    y_pred = (y_proba >= t).astype(int)
    auprc = average_precision_score(y_true, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    if y_pred.sum() == 0:
        return dict(AUPRC=auprc, F1=0.0, Precision=0.0, Recall=0.0,
                    ROC_AUC=roc, FPR=0.0, TP=0, FP=0, FN=int(y_true.sum()),
                    TN=int((y_true == 0).sum()))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return dict(
        AUPRC=auprc,
        F1=f1_score(y_true, y_pred, zero_division=0),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        ROC_AUC=roc,
        FPR=fp / (fp + tn) if (fp + tn) else 0.0,
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn),
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


def train_one_client_one_round(name, X_tr, y_tr, X_te, y_te, round_num, spw):
    rs = SEED + round_num
    xgb = XGBClassifier(
        n_estimators=GA_PARAMS['xgb_n_estimators'],
        max_depth=GA_PARAMS['xgb_max_depth'],
        learning_rate=GA_PARAMS['xgb_learning_rate'],
        subsample=GA_PARAMS['xgb_subsample'],
        scale_pos_weight=float(spw),
        eval_metric='aucpr',
        objective='binary:logistic',
        random_state=rs,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=GA_PARAMS['rf_n_estimators'],
        max_depth=GA_PARAMS['rf_max_depth'],
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=rs,
        n_jobs=-1,
    )
    xgb.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    p_te = 0.5 * xgb.predict_proba(X_te)[:, 1] + 0.5 * rf.predict_proba(X_te)[:, 1]
    auprc = average_precision_score(y_te, p_te)
    return xgb, rf, auprc, p_te


def run_client(name, num_rounds):
    print(f"\n--- Client: {name} ---")
    X_tr, X_te, y_tr, y_te = load_client_natural_prevalence(name)
    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    spw = n_neg / n_pos
    print(f"  train N={len(X_tr):,}  pos={n_pos}  neg={n_neg}  "
          f"scale_pos_weight={spw:.2f}")
    print(f"  test  N={len(X_te):,}  pos={int(y_te.sum())}  "
          f"neg={int((y_te == 0).sum())}")

    best = None  # (xgb, rf, auprc, p_te, round_num)
    round_log = []
    for r in range(1, num_rounds + 1):
        t0 = time.time()
        xgb, rf, auprc, p_te = train_one_client_one_round(
            name, X_tr, y_tr, X_te, y_te, r, spw)
        elapsed = time.time() - t0
        print(f"  round {r:>2d}/{num_rounds}: AUPRC={auprc:.4f}  "
              f"({elapsed:.1f}s)")
        round_log.append({
            'client': name, 'round': r, 'AUPRC': float(auprc),
            'scale_pos_weight': float(spw), 'round_time_s': float(elapsed),
        })
        if best is None or auprc > best[2]:
            best = (xgb, rf, auprc, p_te, r)

    # Threshold selection on best-round predictions, mirroring optimise_threshold.py
    xgb_b, rf_b, auprc_b, _, best_round = best
    X_val, X_eval, y_val, y_eval = train_test_split(
        X_te, y_te, test_size=1 - VAL_FRAC, stratify=y_te, random_state=SEED,
    )
    p_val = 0.5 * xgb_b.predict_proba(X_val)[:, 1] + 0.5 * rf_b.predict_proba(X_val)[:, 1]
    p_eval = 0.5 * xgb_b.predict_proba(X_eval)[:, 1] + 0.5 * rf_b.predict_proba(X_eval)[:, 1]
    t_star, _ = best_threshold_on_val(y_val, p_val)

    rows = []
    for t_label, t_val in [('0.5', 0.5), (f't*={t_star:.2f}', t_star)]:
        m = metrics_at(y_eval, p_eval, t_val)
        rows.append(dict(
            Configuration=f'federated_class_weighted_{name}',
            Client=name.upper() if name != 'synthetic' else 'Synthetic',
            BestRound=best_round,
            Threshold=t_label,
            ScalePosWeight=round(float(spw), 2),
            AUPRC=round(m['AUPRC'], 4),
            ROC_AUC=round(m['ROC_AUC'], 4),
            F1=round(m['F1'], 4),
            Precision=round(m['Precision'], 4),
            Recall=round(m['Recall'], 4),
            FPR=round(m['FPR'], 6),
            TP=m['TP'], FP=m['FP'], FN=m['FN'], TN=m['TN'],
            Source=('per-client natural-prevalence scale_pos_weight; '
                    'best-round selection across 10 rounds; '
                    'val/eval 50/50 split for threshold; '
                    'GA-tuned XGB+RF, 0.5/0.5 averaging'),
        ))
        print(f"  best round {best_round}, t={t_label}: AUPRC={m['AUPRC']:.4f}  "
              f"F1={m['F1']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}")

    return rows, round_log, dict(
        n_train=len(X_tr), n_test=len(X_te),
        n_train_pos=n_pos, n_train_neg=n_neg,
        scale_pos_weight=round(float(spw), 4),
        best_round=int(best_round),
        best_round_auprc_at_05=round(float(auprc_b), 4),
        t_star=round(float(t_star), 2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clients', nargs='+', default=['ulb', 'baf', 'synthetic'],
                   choices=['ulb', 'baf', 'synthetic'])
    p.add_argument('--rounds', type=int, default=NUM_ROUNDS)
    args = p.parse_args()

    print('=' * 72)
    print(f'Task 2 — Federated class-weighted (no SMOTE) — '
          f'{args.rounds} rounds, clients={args.clients}')
    print('=' * 72)

    all_rows, all_round_logs, meta_per_client = [], [], {}
    for name in args.clients:
        rows, log, meta = run_client(name, args.rounds)
        all_rows += rows
        all_round_logs += log
        meta_per_client[name] = meta
        # Checkpoint after each client
        pd.DataFrame(all_rows).to_csv(
            os.path.join(RESULTS_DIR, 'federated_class_weighted.csv'),
            index=False)
        pd.DataFrame(all_round_logs).to_csv(
            os.path.join(RESULTS_DIR, 'federated_class_weighted_round_log.csv'),
            index=False)
        with open(os.path.join(RESULTS_DIR, 'federated_class_weighted_meta.json'),
                  'w') as f:
            json.dump(dict(
                seed=SEED, num_rounds=args.rounds, test_size=TEST_SIZE,
                val_frac=VAL_FRAC, ga_params=GA_PARAMS,
                ensemble_weights={'xgb': 0.5, 'rf': 0.5},
                per_client=meta_per_client,
                notes='See docs/fraudx_ai_methodology_notes.md.',
            ), f, indent=2)
        print(f"  [checkpoint] saved partial results after client '{name}'")

    print('\n' + '=' * 72)
    print(f"Wrote {os.path.join(RESULTS_DIR, 'federated_class_weighted.csv')}")
    print('=' * 72)
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == '__main__':
    main()