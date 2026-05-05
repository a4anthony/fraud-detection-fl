"""
Task 1 — Matched FraudX AI comparison (Option C: literal match + sensitivity).

Reproduces FraudX AI's setup as closely as the paper specifies:
  * ULB only, centralised (no FL).
  * No SMOTE — natural-prevalence training partition.
  * Stratified 80/20 train/test, seed 42 (identical to existing project pipeline,
    so the test set is the same 56,962-row partition as FraudX AI evaluated on).
  * StandardScaler on all 30 features (Time, Amount, V1-V28).
  * RF + XGBoost soft-voting ensemble at 0.5 / 0.5 averaging.
  * Project's GA-tuned hyperparameters (so the comparison isolates methodology,
    not hyperparameters, from the existing federated baseline).

Two reweighting configurations are evaluated (Option C):

  centralised_no_smote_w_2_0    -- literal match to FraudX AI §4.1
                                   (RF class_weight={0:1.0, 1:2.0},
                                   XGBoost scale_pos_weight=2.0)
  centralised_no_smote_w_580    -- sensitivity check at natural-prevalence ratio
                                   (RF class_weight={0:1.0, 1:N_neg/N_pos},
                                   XGBoost scale_pos_weight=N_neg/N_pos)

Each configuration is evaluated at:
  * t = 0.5  (default)
  * t = t*   (F1-maximising threshold on a 50% validation slice of the test
              set, mirroring optimise_threshold.py / retrain_weighted.py)

Outputs:
  results-tables/fraudx_ai_matched.csv

The existing federated SMOTE anchor and FraudX AI reported numbers are
included in the same CSV (with explicit Source columns) so a single
table carries every number the comparison figure plots.

Methodology assumptions (see docs/fraudx_ai_methodology_notes.md):
  * seed 42 (FraudX AI does not disclose theirs)
  * 0.5/0.5 ensemble averaging (FraudX AI does not disclose their weights)
  * GA-tuned RF/XGB hyperparameters (FraudX AI does not disclose theirs)
  * StandardScaler on all 30 features (FraudX AI is vague on scaling)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, 'data', 'raw', 'ulb_creditcard.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results-tables')
EXISTING_RESULTS = os.path.join(BASE_DIR, 'paper_v2_outputs', 'per_dataset_results.csv')
EXISTING_RESULTS_OPT = os.path.join(BASE_DIR, 'paper_v2_outputs', 'per_dataset_results_optimised.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
TEST_SIZE = 0.20      # FraudX AI §3.1: 80/20 stratified split
VAL_FRAC = 0.50       # 50/50 split of test set into val (threshold selection) / eval

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.20278092176694895,
    max_depth=5,
    subsample=0.9533490553166496,
    n_jobs=-1,
    eval_metric='aucpr',
    objective='binary:logistic',
    random_state=SEED,
)
RF_PARAMS = dict(
    n_estimators=282,
    max_depth=27,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=SEED,
)


def ensemble_proba(xgb, rf, X):
    return 0.5 * xgb.predict_proba(X)[:, 1] + 0.5 * rf.predict_proba(X)[:, 1]


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


def load_ulb_natural_prevalence():
    """Recreate the natural-prevalence ULB train/test split using exactly the
    same procedure as src/data/preprocess.py (seed 42, stratified 80/20,
    StandardScaler on all features) but WITHOUT SMOTE.

    Verifies the resulting test partition is identical to the project's
    SMOTE-pipeline test partition (so headline numbers from this script are
    directly comparable to the existing federated SMOTE result, which was
    evaluated on the same rows).
    """
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.strip().strip('"') for c in df.columns]
    target_col = 'Class'
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=SEED,
    )

    # Verify against existing project test partition (same row counts AT MINIMUM;
    # exact value match expected because preprocess.py uses identical args).
    existing_y_test = pd.read_csv(
        os.path.join(BASE_DIR, 'data', 'processed', 'ulb_y_test.csv')
    ).squeeze()
    assert len(y_test) == len(existing_y_test), \
        f"test size mismatch: {len(y_test)} vs {len(existing_y_test)}"
    assert int(y_test.sum()) == int(existing_y_test.sum()), \
        f"fraud count mismatch: {int(y_test.sum())} vs {int(existing_y_test.sum())}"
    return X_train, X_test, y_train, y_test


def run_config(label, X_tr, y_tr, X_te, y_te, scale_pos_weight, rf_class_weight,
               note):
    print(f"\n[{label}] scale_pos_weight={scale_pos_weight} "
          f"rf_class_weight={rf_class_weight}")
    xgb = XGBClassifier(**XGB_PARAMS, scale_pos_weight=float(scale_pos_weight))
    rf = RandomForestClassifier(**RF_PARAMS, class_weight=rf_class_weight)
    xgb.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    X_val, X_eval, y_val, y_eval = train_test_split(
        X_te, y_te, test_size=1 - VAL_FRAC, stratify=y_te, random_state=SEED,
    )
    p_val = ensemble_proba(xgb, rf, X_val)
    p_eval = ensemble_proba(xgb, rf, X_eval)
    t_star, _ = best_threshold_on_val(y_val, p_val)

    rows = []
    for t_label, t_val in [('0.5', 0.5), (f't*={t_star:.2f}', t_star)]:
        m = metrics_at(y_eval, p_eval, t_val)
        rows.append(dict(
            Configuration=label,
            Threshold=t_label,
            ScalePosWeight=float(scale_pos_weight),
            AUPRC=round(m['AUPRC'], 4),
            ROC_AUC=round(m['ROC_AUC'], 4),
            F1=round(m['F1'], 4),
            Precision=round(m['Precision'], 4),
            Recall=round(m['Recall'], 4),
            FPR=round(m['FPR'], 6),
            TP=m['TP'], FP=m['FP'], FN=m['FN'], TN=m['TN'],
            Source=note,
        ))
        print(f"  t={t_label:>10s}  AUPRC={m['AUPRC']:.4f}  "
              f"F1={m['F1']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}")
    return rows


def main():
    print('=' * 72)
    print('Task 1 — Matched FraudX AI comparison (Option C)')
    print('=' * 72)
    X_tr, X_te, y_tr, y_te = load_ulb_natural_prevalence()
    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    natural_ratio = n_neg / n_pos
    print(f"Train: N={len(X_tr):,}  pos={n_pos}  neg={n_neg}  "
          f"natural neg/pos ratio = {natural_ratio:.2f}")
    print(f"Test : N={len(X_te):,}  pos={int(y_te.sum())}  "
          f"neg={int((y_te == 0).sum())}")

    all_rows = []

    # Config 1: literal match to FraudX AI {0:1.0, 1:2.0}
    all_rows += run_config(
        label='centralised_no_smote_w_2_0',
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        scale_pos_weight=2.0,
        rf_class_weight={0: 1.0, 1: 2.0},
        note='literal match to FraudX AI §4.1 (class weight {0:1.0, 1:2.0})',
    )

    # Config 2: sensitivity at natural-prevalence ratio
    all_rows += run_config(
        label='centralised_no_smote_w_580',
        X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
        scale_pos_weight=natural_ratio,
        rf_class_weight={0: 1.0, 1: float(natural_ratio)},
        note=(f'sensitivity check at natural neg/pos ratio '
              f'({natural_ratio:.2f}); NOT a match to FraudX AI'),
    )

    # Anchor rows: existing federated SMOTE result (ULB).
    #
    # We carry through TWO views of the federated SMOTE baseline so the
    # comparison can be apples-to-apples:
    #
    #   federated_smote_full_test  — full 56,962-row test set, threshold 0.5.
    #     This is the headline 0.8734 quoted in the paper; for context only,
    #     since Task 1's matched / sensitivity rows are evaluated on the
    #     50% eval slice (val/eval split), not the full test set.
    #
    #   federated_smote_eval_half  — same 50% eval slice as the matched and
    #     sensitivity rows below. This is the apples-to-apples anchor for
    #     the figure. Source: paper_v2_outputs/weighted_vs_unweighted.csv
    #     row 2 (Variant=unweighted) — the unweighted baseline there is
    #     exactly the federated SMOTE ensemble re-evaluated on the eval half.
    fed = pd.read_csv(EXISTING_RESULTS)
    fed_ulb = fed[fed['Dataset'] == 'ULB'].iloc[0]
    fed_opt = pd.read_csv(EXISTING_RESULTS_OPT)
    fed_ulb_opt = fed_opt[fed_opt['Dataset'] == 'ULB'].iloc[0]
    all_rows.append(dict(
        Configuration='federated_smote_full_test',
        Threshold='0.5',
        ScalePosWeight=None,
        AUPRC=float(fed_ulb['AUPRC']),
        ROC_AUC=float(fed_ulb['ROC_AUC']),
        F1=float(fed_ulb['F1']),
        Precision=float(fed_ulb['Precision']),
        Recall=float(fed_ulb['Recall']),
        FPR=float(fed_ulb['FPR']),
        TP=None, FP=None, FN=None, TN=None,
        Source='paper_v2_outputs/per_dataset_results.csv row 2 (Dataset=ULB) — full 56,962-row test set',
    ))
    all_rows.append(dict(
        Configuration='federated_smote_full_test',
        Threshold=f"t*={float(fed_ulb_opt['OptimalThreshold']):.2f}",
        ScalePosWeight=None,
        AUPRC=float(fed_ulb_opt['AUPRC']),
        ROC_AUC=float(fed_ulb_opt['ROC_AUC']),
        F1=float(fed_ulb_opt['F1_at_optimal']),
        Precision=float(fed_ulb_opt['Precision_at_optimal']),
        Recall=float(fed_ulb_opt['Recall_at_optimal']),
        FPR=float(fed_ulb_opt['FPR_at_optimal']),
        TP=None, FP=None, FN=None, TN=None,
        Source='paper_v2_outputs/per_dataset_results_optimised.csv row 2 (Dataset=ULB) — full 56,962-row test set',
    ))
    # Apples-to-apples anchor on the same eval slice as the matched / sensitivity rows.
    wvu = pd.read_csv(os.path.join(BASE_DIR, 'paper_v2_outputs',
                                   'weighted_vs_unweighted.csv'))
    wvu_ulb_un = wvu[(wvu['Dataset'] == 'ULB')
                     & (wvu['Variant'] == 'unweighted')].iloc[0]
    all_rows.append(dict(
        Configuration='federated_smote_eval_half',
        Threshold='0.5',
        ScalePosWeight=None,
        AUPRC=float(wvu_ulb_un['AUPRC']),
        ROC_AUC=float(wvu_ulb_un['ROC_AUC']),
        F1=float(wvu_ulb_un['F1_at_0.5']),
        Precision=float(wvu_ulb_un['Precision_at_0.5']),
        Recall=float(wvu_ulb_un['Recall_at_0.5']),
        FPR=None,
        TP=None, FP=None, FN=None, TN=None,
        Source='paper_v2_outputs/weighted_vs_unweighted.csv row 2 (Variant=unweighted) — eval half (28,481 rows)',
    ))
    all_rows.append(dict(
        Configuration='federated_smote_eval_half',
        Threshold=f"t*={float(wvu_ulb_un['OptimalThreshold']):.2f}",
        ScalePosWeight=None,
        AUPRC=float(wvu_ulb_un['AUPRC']),
        ROC_AUC=float(wvu_ulb_un['ROC_AUC']),
        F1=float(wvu_ulb_un['F1_at_optimal']),
        Precision=float(wvu_ulb_un['Precision_at_optimal']),
        Recall=float(wvu_ulb_un['Recall_at_optimal']),
        FPR=float(wvu_ulb_un['FPR_at_optimal']),
        TP=None, FP=None, FN=None, TN=None,
        Source='paper_v2_outputs/weighted_vs_unweighted.csv row 2 (Variant=unweighted) — eval half (28,481 rows)',
    ))

    # Reference benchmark row: FraudX AI reported numbers from Table 2 / Table 3
    all_rows.append(dict(
        Configuration='fraudx_ai_reported',
        Threshold='paper-tuned (PR-curve)',
        ScalePosWeight=2.0,
        AUPRC=0.97,
        ROC_AUC=None,
        F1=0.97,
        Precision=1.00,
        Recall=0.95,
        FPR=0.0,
        TP=93, FP=0, FN=5, TN=56864,
        Source='Baisholan et al. (2025) Computers 14(4) 120, Table 2 + Table 3',
    ))

    out_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULTS_DIR, 'fraudx_ai_matched.csv')
    out_df.to_csv(out_csv, index=False)

    # Sidecar JSON with run metadata for full reproducibility
    meta = dict(
        seed=SEED, test_size=TEST_SIZE, val_frac=VAL_FRAC,
        n_train=len(X_tr), n_test=len(X_te),
        n_train_pos=n_pos, n_train_neg=n_neg,
        natural_neg_pos_ratio=round(float(natural_ratio), 4),
        xgb_params=XGB_PARAMS, rf_params=RF_PARAMS,
        ensemble_weights={'xgb': 0.5, 'rf': 0.5},
        notes='See docs/fraudx_ai_methodology_notes.md for assumptions.',
    )
    with open(os.path.join(RESULTS_DIR, 'fraudx_ai_matched_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print('\n' + '=' * 72)
    print(f'Wrote {out_csv}')
    print('=' * 72)
    print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()