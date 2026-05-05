"""
Preprocessing pipeline: cleaning, encoding, normalization, SMOTE.

Handles all three project datasets:
- ULB Credit Card (PCA features V1-V28 + Amount + Time)
- BAF NeurIPS 2022 (mixed numerical/categorical)
- Synthetic Financial Transactions (categorical-heavy)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path


def load_and_clean_ulb(path):
    """Load and clean ULB dataset."""
    df = pd.read_csv(path)
    # No missing values, no cleaning needed beyond scaling
    return df, "Class"


def load_and_clean_baf(path):
    """Load and clean BAF dataset."""
    df = pd.read_csv(path)
    # Replace sentinel -1 values with NaN then median-impute
    sentinel_cols = ["prev_address_months_count", "current_address_months_count",
                     "bank_months_count", "session_length_in_minutes",
                     "device_distinct_emails_8w"]
    for col in sentinel_cols:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)
            df[col] = df[col].fillna(df[col].median())
    return df, "fraud_bool"


def load_and_clean_synthetic(path, sample_size=500_000):
    """Load and clean Synthetic dataset (sampled for tractability)."""
    df = pd.read_csv(path)
    df["is_fraud"] = df["is_fraud"].map({"True": 1, "False": 0, True: 1, False: 0})
    df["is_fraud"] = df["is_fraud"].astype(int)

    # Drop non-predictive columns
    drop_cols = ["transaction_id", "timestamp", "sender_account",
                 "receiver_account", "fraud_type", "ip_address", "device_hash"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Impute missing time_since_last_transaction with median
    if "time_since_last_transaction" in df.columns:
        df["time_since_last_transaction"] = df["time_since_last_transaction"].fillna(
            df["time_since_last_transaction"].median()
        )

    # Stratified sample for tractability
    if len(df) > sample_size:
        fraud_ratio = df["is_fraud"].value_counts(normalize=True)
        samples = []
        for label, frac in fraud_ratio.items():
            subset = df[df["is_fraud"] == label]
            n = int(sample_size * frac)
            samples.append(subset.sample(n=n, random_state=42))
        df = pd.concat(samples).reset_index(drop=True)

    return df, "is_fraud"


def encode_categoricals(df, target_col):
    """Label-encode categorical columns. Returns df and encoder dict."""
    encoders = {}
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def preprocess_dataset(df, target_col, test_size=0.2, apply_smote=True,
                       random_state=42):
    """Full preprocessing pipeline for a single dataset.

    Returns: X_train, X_test, y_train, y_test, scaler
    """
    # Encode categoricals
    df, encoders = encode_categoricals(df.copy(), target_col)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Scale all features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if apply_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"  Original train: {dict(pd.Series(y_train).value_counts())}")
        print(f"  After SMOTE:    {dict(pd.Series(y_train_res).value_counts())}")
    else:
        X_train_res, y_train_res = X_train, y_train

    print(f"  Test set:       {dict(pd.Series(y_test).value_counts())}")

    return X_train_res, X_test, y_train_res, y_test, scaler, encoders


def preprocess_all_datasets(raw_dir="data/raw", output_dir="data/processed",
                            synth_sample_size=500_000):
    """Run full preprocessing on all three datasets."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ULB
    print("Processing ULB dataset...")
    df_ulb, target_ulb = load_and_clean_ulb(raw_dir / "ulb_creditcard.csv")
    results["ulb"] = preprocess_dataset(df_ulb, target_ulb)

    # BAF
    print("\nProcessing BAF dataset...")
    df_baf, target_baf = load_and_clean_baf(raw_dir / "baf_base.csv")
    results["baf"] = preprocess_dataset(df_baf, target_baf)

    # Synthetic
    print(f"\nProcessing Synthetic dataset (sample={synth_sample_size:,})...")
    df_synth, target_synth = load_and_clean_synthetic(
        raw_dir / "synthetic_fraud.csv", sample_size=synth_sample_size
    )
    results["synthetic"] = preprocess_dataset(df_synth, target_synth)

    # Save processed data
    for name, (X_train, X_test, y_train, y_test, scaler, encoders) in results.items():
        prefix = output_dir / name
        X_train.to_csv(f"{prefix}_X_train.csv", index=False)
        X_test.to_csv(f"{prefix}_X_test.csv", index=False)
        pd.Series(y_train).to_csv(f"{prefix}_y_train.csv", index=False)
        pd.Series(y_test).to_csv(f"{prefix}_y_test.csv", index=False)
        joblib.dump(scaler, f"{prefix}_scaler.joblib")
        joblib.dump(encoders, f"{prefix}_encoders.joblib")
        print(f"  Saved {name} processed data to {output_dir}/")

    return results


if __name__ == "__main__":
    preprocess_all_datasets()
