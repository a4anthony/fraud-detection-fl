# Explainable Federated Ensemble Learning for Real-Time Financial Fraud Detection

MSc Thesis Project — COM748, Ulster University

**Author:** Anthony Akro

---

## Overview

This project implements a privacy-preserving fraud detection system that combines:

- **XGBoost–Random Forest soft voting ensemble** for high detection accuracy
- **Federated Learning** (Flower) so institutions collaborate without sharing raw data
- **SHAP explainability** for transparent, auditable predictions
- **Genetic Algorithm** hyperparameter optimization (DEAP)

The system is evaluated across three real-world and synthetic datasets, targeting >95% AUPRC with sub-200ms inference latency.

---

## Project Structure

```
fraud-detection-fl/
├── data/
│   ├── raw/                        # Symlinks to original datasets
│   ├── processed/                  # Cleaned, scaled, SMOTE-resampled splits
│   └── federated/                  # Per-client partitioned data for FL
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, encoding, SMOTE, FL partitioning
│   ├── 03_centralized_baselines.ipynb  # LR, RF, XGBoost, ensemble training
│   ├── 04_ensemble_training.ipynb  # GA hyperparameter optimization
│   ├── 05_federated_learning.ipynb # Flower FL simulation (10 rounds, 3 clients)
│   ├── 06_explainability.ipynb     # SHAP global/local explanations
│   ├── 07_latency_benchmarking.ipynb   # Inference time profiling
│   ├── 08_final_comparison.ipynb   # PR/ROC curves, ablation study, final tables
│   └── 09_diagrams.ipynb          # Architecture, flowcharts, Gantt, risk matrix
├── src/
│   ├── data/
│   │   ├── preprocess.py           # Cleaning, encoding, scaling, SMOTE pipeline
│   │   └── partition.py            # FL client data partitioning (IID, non-IID, natural)
│   ├── models/
│   │   ├── baselines.py            # LR, RF, XGBoost, XGB-RF ensemble training + CV
│   │   └── ga_optimizer.py         # Genetic Algorithm (DEAP) for ensemble tuning
│   ├── federated/
│   │   ├── client.py               # Flower NumPyClient for XGB-RF ensemble
│   │   ├── server.py               # Flower server + FedAvg strategy
│   │   └── utils.py                # Aggregation helpers, communication cost tracking
│   ├── explainability/
│   │   └── shap_analysis.py        # SHAP summary, bar, waterfall, force, dependence
│   └── evaluation/
│       ├── metrics.py              # AUPRC, F1, ROC-AUC, confusion matrix, PR/ROC curves
│       ├── latency.py              # Per-transaction inference benchmarking
│       └── comparison.py           # Final comparison tables, ablation study
├── outputs/
│   ├── figures/                    # All generated plots (39 visuals)
│   ├── tables/                     # CSV result tables
│   └── models/                     # Serialized trained models (.joblib)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Datasets

| Dataset | Transactions | Fraud Rate | Features | Source |
|---------|-------------|------------|----------|--------|
| ULB Credit Card (2013) | 284,807 | 0.17% | 31 (PCA-anonymized) | [Kaggle / Zenodo](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| BAF NeurIPS (2022) | 1,000,000 | 1.10% | 32 (interpretable) | [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) |
| Synthetic Financial (Kaggle) | 5,000,000 | 3.59% | 18 (categorical-rich) | [Kaggle](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection) |

Each dataset represents a different financial institution in the federated learning simulation.

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place raw datasets in data/raw/
#   data/raw/ulb_creditcard.csv
#   data/raw/baf_base.csv
#   data/raw/synthetic_fraud.csv
```

---

## Execution Order

Notebooks must be run sequentially — each depends on outputs from previous steps.

```
01_eda.ipynb                    ← already run (EDA figures)
    ↓
02_preprocessing.ipynb          ← produces data/processed/ and data/federated/
    ↓
03_centralized_baselines.ipynb  ← trains LR, RF, XGBoost, ensemble → outputs/models/
    ↓
04_ensemble_training.ipynb      ← GA optimization → ga_optimized_ensemble.joblib
    ↓
05_federated_learning.ipynb     ← FL simulation → fl_global_xgb.joblib, fl_global_rf.joblib
    ↓
06_explainability.ipynb         ← SHAP plots for XGBoost and RF
    ↓
07_latency_benchmarking.ipynb   ← inference time profiling for all models
    ↓
08_final_comparison.ipynb       ← PR/ROC curves, confusion matrices, ablation study
    ↓
09_diagrams.ipynb               ← architecture diagrams, Gantt, risk matrix, etc.
```

### Automated Execution

To run all notebooks sequentially without opening Jupyter:

```bash
# Run all notebooks in order (outputs saved in-place)
for nb in notebooks/0{1,2,3,4,5,6,7,8,9}_*.ipynb; do
    echo "Running $nb..."
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "$nb"
done
```

Or run a single notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 notebooks/05_federated_learning.ipynb
```

> **Note:** Some notebooks (especially 04 and 05) may take 30+ minutes due to GA optimization and FL training on large datasets. The `--timeout=3600` flag allows up to 1 hour per cell.

---

## Key Results (Generated)

### Tables (`outputs/tables/`)
| File | Description |
|------|-------------|
| `dataset_summary.csv` | Dataset statistics and imbalance ratios |
| `feature_comparison.csv` | Feature type comparison across datasets |
| `baseline_test_results.csv` | Centralized model test set metrics |
| `ga_best_params.csv` | GA-optimized hyperparameters |
| `fl_round_metrics.csv` | Per-round FL convergence metrics |
| `latency_results.csv` | Inference latency (mean, P95, P99) per model |
| `final_comparison.csv` | Master comparison: all models, all metrics |
| `ablation_study.csv` | Ablation: centralized vs FL, single vs ensemble |
| `regulatory_compliance_mapping.csv` | GDPR / EU AI Act / PCI-DSS mapping |

### Figures (`outputs/figures/`) — 39 Visuals

**EDA (V1–V6):** Class distributions, amount distributions, correlation heatmaps, temporal patterns, top features

**Preprocessing (V7–V9):** SMOTE before/after, FL client distribution

**Baselines (V10–V13):** AUPRC comparison, CV box plot, GA convergence, GA params

**Federated Learning (V14–V18):** FL convergence, local vs global, communication cost, FL vs centralized

**Explainability (V19–V23):** SHAP summary, bar, waterfall, force, dependence plots

**Evaluation (V24–V30):** Confusion matrices, PR curves, ROC curves, latency box plot, latency histogram, final comparison, ablation chart

**Diagrams (V31–V38):** System architecture, FL architecture, methodology flowchart, Gantt chart, risk matrix, compliance mapping, ensemble diagram, SMOTE illustration

---

## Methodology

1. **Preprocessing:** StandardScaler normalization, label encoding for categoricals, SMOTE oversampling on training splits only
2. **Baselines:** Logistic Regression, Random Forest, XGBoost, and XGB-RF soft voting ensemble with 5-fold stratified CV
3. **GA Optimization:** DEAP evolutionary search over 6 hyperparameters (30 generations, population 20)
4. **Federated Learning:** 3 clients (one per dataset), 10 communication rounds, best-model aggregation for tree-based ensembles
5. **Explainability:** SHAP TreeExplainer on BAF dataset (interpretable features: income, age, credit_risk_score, velocity)
6. **Evaluation:** AUPRC (primary), F1, ROC-AUC, precision, recall, FPR, confusion matrices, latency profiling against 200ms target

---

## Dependencies

- Python 3.9+
- pandas, numpy, scikit-learn, xgboost
- imbalanced-learn (SMOTE)
- shap (explainability)
- flwr (Flower federated learning)
- deap (genetic algorithm)
- matplotlib, seaborn (visualization)
- jupyter

See `requirements.txt` for pinned minimum versions.

---

## License

Academic use — Ulster University COM748 thesis project.

---

## Repository contents

This GitHub repository mirrors the code, experimental results, and figures from my MSc thesis supporting-material submission.

| Folder | What's in it |
|---|---|
| `src/`, `notebooks/`, root `*.py` scripts | All source code: data prep, modelling, GA, federated learning, SHAP, evaluation |
| `results/` | Curated experimental results (CSV / JSON) referenced in the paper |
| `figures/` | 58 figures generated by the pipeline (EDA, SHAP, FL convergence, confusion matrices, architecture diagrams, etc.) |
| `docs/` | Methodology notes |
| `requirements.txt` | Pinned Python dependencies |

Datasets are **not** included; download links are listed above. Trained model `.joblib` files are also excluded by `.gitignore` (they are large and reproducible from the seeded notebooks).
