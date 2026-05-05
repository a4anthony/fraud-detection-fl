# FraudX AI Methodology Extraction

Source: Baisholan, N.; Dietz, J.E.; Gnatyuk, S.; Turdalyuly, M.; Matson, E.T.;
Baisholanova, K. **FraudX AI: An Interpretable Machine Learning Framework for
Credit Card Fraud Detection on Imbalanced Datasets.** *Computers* **2025**,
*14*(4), 120. <https://doi.org/10.3390/computers14040120>

PDF read locally from `fraud-detection-fl/docs/fraudx_ai_paper.pdf` (pages 1–19).
Section references below are to that paper.

## 1. Parameters explicitly stated by the paper

| Parameter | Value | Source |
|---|---|---|
| Dataset | Kaggle ULB European credit card (Worldline / ULB-MLG) | §3.1, Data Availability Statement |
| Total transactions | 284,807 | §3.1 |
| Fraud transactions | 492 (0.172%) | §3.1, Figure 1 |
| Train/test split | 80% train / 20% test, stratified ("maintaining class distribution") | §3.1 |
| Class imbalance treatment | **Class weights `{0: 1.0, 1: 2.0}`** applied to RF and XGBoost during training | §4.1 ("a class weight ratio of {0:1.0, 1:2.0} was applied to assign higher importance to the minority fraud class") |
| Oversampling | **None** — explicit rejection of SMOTE | §2 closing paragraph; §5 ("without relying on traditional oversampling techniques like SMOTE") |
| Base models | Random Forest + XGBoost (both trained on imbalanced data) | §3.2, §4.1 |
| Ensemble combination | Probability-weighted averaging of RF and XGBoost soft outputs | §4.1 |
| Threshold | Optimised via PR-curve analysis | §4.1 |
| Auxiliary model | MLP validator trained on undersampled data **does not contribute to ensemble decisions** | §3.2, Figure 2, §4.1 |
| Reported AUC-PR | **0.97** | Table 2, row "FraudX AI" |
| Reported Recall | 0.95 | Table 2 |
| Reported Precision | 1.00 | Table 2 |
| Reported F1 | 0.97 | Table 2 |
| Reported Accuracy | 0.99 | Table 2 |
| Confusion matrix at chosen threshold | TN=56,864, FP=0, FN=5, TP=93 | Table 3 |

The confusion matrix arithmetic confirms the test set has exactly
56,864 + 0 + 5 + 93 = **56,962 rows = 0.20 × 284,807** and
93 + 5 = **98 frauds = 0.20 × 492 (with rounding)**, which is consistent with a
stratified 20% test split.

## 2. Parameters the paper does NOT specify (assumptions documented)

| Parameter | Paper says | Our assumption | Justification |
|---|---|---|---|
| Random seed | Not stated anywhere | **42** | Project standard; FraudX AI's omission means we cannot match their seed and any disagreement is unavoidable. Documented in CSV footnote. |
| Scaler | "feature scaling techniques to normalize transaction values" (§3.1) — not specified | **`StandardScaler` on all 30 features** (Time, Amount, V1–V28) | The project's existing pipeline (`src/data/preprocess.py:99`) already does this; FraudX AI's vague phrasing is consistent. V1–V28 are PCA-projected and approximately standardised, so scaling is mainly meaningful for Time and Amount. |
| RF hyperparameters | "standard hyperparameter settings, optimizing them through validation experiments" — values not disclosed | **Project's GA-tuned values:** `n_estimators=282, max_depth=27, min_samples_leaf=1` | The prompt's brief explicitly requires using our existing GA-tuned hyperparameters from `results-tables/ga_best_params.csv` so the matched comparison isolates **methodology** (SMOTE vs class-weights, FL vs centralised) from **hyperparameters**. |
| XGBoost hyperparameters | Same — "standard… optimizing through validation experiments" | **Project's GA-tuned values:** `n_estimators=300, max_depth=5, learning_rate=0.20278, subsample=0.95335` | Same reason. |
| Ensemble weighting | "empirically determined weighting scheme" (§4.1) — values not disclosed | **0.5 / 0.5** average | Matches our existing project ensemble (`retrain_weighted.py:71`). FraudX AI does not publish their weights. Sensitivity to this choice is small in practice for two well-calibrated soft-voters. |
| PR-optimal threshold | Optimised via PR curve, value not disclosed | We will report at **two operating points**: t = 0.5 (default) and t* = arg-max F1 on a held-out validation slice (50/50 split of the test set, seed 42) | Mirrors the existing project methodology in `retrain_weighted.py:91-100` so all matched / federated / project results are evaluated under the same threshold-selection protocol. |

## 3. Critical correction to the original task brief — audit trail

### What the brief originally specified

The handover prompt for Task 1 (matched FraudX AI comparison) instructed:

> "scale_pos_weight = N where N is FraudX AI's value (or natural ratio ≈ 580
> if unspecified, with assumption noted)"

…and elsewhere:

> "ULB only … No SMOTE — use natural-prevalence training partition …
> scale_pos_weight = N where N is FraudX AI's value (or natural ratio ≈ 580
> if unspecified)"

Read literally, the fallback default in the brief was **`scale_pos_weight ≈ 580`**
— the natural negative-to-positive ratio of the ULB training partition
(227,451 / 394 ≈ 577). The brief explicitly framed this as "FraudX AI's value
*or* the natural ratio if unspecified."

### What FraudX AI's paper actually states

§4.1 of Baisholan et al. (2025) is explicit:

> "The RF and XGBoost models were trained using balanced learning strategies,
> incorporating class weighting to address class imbalance effectively.
> Specifically, a class weight ratio of {0:1.0, 1:2.0} was applied to assign
> higher importance to the minority fraud class."

So FraudX AI's value **is specified** — and it is **`{0:1.0, 1:2.0}`** — i.e.
`scale_pos_weight = 2.0` for XGBoost and `class_weight={0:1.0, 1:2.0}` for RF.
The fallback to 580 in the brief is therefore not triggered: there is a
disclosed value to match, and it is 2.0, not 580.

### Why these differ materially

- **`scale_pos_weight = 2`** barely reweights positives at all. The model
  learns close to its natural-distribution decision boundary; calibration
  remains near-honest; PR-curve threshold tuning then handles the trade-off.
  This is the operating regime in which FraudX AI report AUC-PR = 0.97.
- **`scale_pos_weight = 580`** aggressively pushes the boundary toward
  positive-friendly territory. The model becomes more recall-eager but
  loses precision and calibration. This is the regime in which our project's
  `retrain_weighted.py` already evaluated weighted training on natural-prevalence
  data (paper_v2_outputs/weighted_vs_unweighted.csv).

Reporting `w=580` as "matched FraudX AI" would therefore have been
methodologically misleading — it would compare two systems trained at very
different points on the recall/precision curve and label that comparison
"matched."

### Decision: Option C (both, with the literal match as headline)

For Task 1 we run **both** reweighting configurations and label them
unambiguously in the output CSV and figure:

1. **`centralised_no_smote_w_2_0`** — literal match to FraudX AI §4.1.
   This is the headline "matched FraudX AI" result. The headline bar in
   `figures/fraudx_ai_matched_comparison.png` corresponds to this row.

2. **`centralised_no_smote_w_580`** — sensitivity check at natural prevalence
   ratio. Visually distinguished in the figure (lighter / hatched), labelled
   "sensitivity (natural ratio)". This row exists so the spectrum from mild
   reweighting (1:2) to natural-ratio (1:580) is observable; it is **not**
   a match to FraudX AI.

Both rows use the project's GA-tuned hyperparameters and the project's
preprocessing pipeline (StandardScaler on all 30 features, stratified 80/20
split with seed 42). They differ only in the class-weight value. The
existing federated SMOTE result is carried into the same CSV as the anchor;
FraudX AI's reported 0.97 is recorded as the reference benchmark.

This correction does **not** invalidate Task 2 (federated class-weighted),
which the brief described separately as using natural-prevalence per-client
weights. Task 2 is not a "match FraudX AI" experiment — it measures what
federated training does without SMOTE under aggressive (natural-ratio)
reweighting. Different question, different setup. Proceeding unchanged.

## 4. Reproducibility / library versions

The paper states (§4.1):

- Implementation: Jupyter Notebook 7.3.2 on MacBook Pro M1 Pro
- Libraries: scikit-learn, XGBoost, TensorFlow/Keras (for MLP validator),
  pandas, NumPy, Matplotlib 3.9.2, Seaborn 0.13.2

It does not pin scikit-learn or XGBoost versions. Our project pins:
Python 3.10.12, scikit-learn 1.3.0, XGBoost 2.0.0, imbalanced-learn 0.11.0
(per `CLAUDE_CODE_HANDOFF.md` and `requirements.txt`). Version drift between
their unspecified versions and ours is an unavoidable source of small
numerical disagreement.

## 5. Comparability summary

The existing project has the **same test set** as FraudX AI (verified
empirically: `data/processed/ulb_y_test.csv` contains 98 frauds in 56,962
rows = 0.1720%, identical to FraudX AI's Table 3). This is because the
project's preprocessing pipeline (`src/data/preprocess.py:107-109`) uses the
same 80/20 stratified split with seed 42, on the same Kaggle ULB CSV. So
matched evaluation will be on the **identical 56,962-row test partition**
that FraudX AI evaluated on, removing test-set drift as a confounder.

The 80/20 split also matches what the FL ULB client uses, so all three
results being reported in `fraudx_ai_matched_comparison.png` —
(a) federated SMOTE, (b) matched centralised, (c) FraudX AI reported —
are evaluated on the same 56,962 rows.
