"""
SHAP-based explainability analysis for fraud detection models.
Generates global and local explanations.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def generate_shap_explanations(model, X_test, y_test=None,
                               feature_names=None,
                               output_dir="outputs/figures",
                               model_name="ensemble"):
    """Generate full SHAP analysis suite.

    Args:
        model: trained tree-based model (XGBoost, RF, or ensemble)
        X_test: test features (DataFrame or array)
        y_test: test labels (for finding fraud examples)
        feature_names: list of feature names
        output_dir: where to save plots
        model_name: prefix for filenames

    Returns:
        shap_values: computed SHAP values
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if feature_names is None:
        feature_names = (list(X_test.columns)
                         if hasattr(X_test, "columns")
                         else [f"feature_{i}" for i in range(X_test.shape[1])])

    # Use a subsample for speed if dataset is large
    max_samples = 1000
    if len(X_test) > max_samples:
        sample_idx = np.random.RandomState(42).choice(
            len(X_test), max_samples, replace=False
        )
        X_explain = (X_test.iloc[sample_idx]
                     if hasattr(X_test, "iloc")
                     else X_test[sample_idx])
        y_explain = (y_test.iloc[sample_idx]
                     if y_test is not None and hasattr(y_test, "iloc")
                     else y_test[sample_idx] if y_test is not None else None)
    else:
        X_explain = X_test
        y_explain = y_test

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)

    # Handle binary classification output (take fraud class = index 1)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # --- SHAP Summary Plot (Global Feature Importance) ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_explain, feature_names=feature_names,
        show=False, max_display=20,
    )
    plt.title(f"SHAP Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_summary_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved shap_summary_{model_name}.png")

    # --- SHAP Bar Plot (Mean Absolute SHAP Values) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_explain, feature_names=feature_names,
        plot_type="bar", show=False, max_display=15,
    )
    plt.title(f"Mean |SHAP| Feature Importance — {model_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_bar_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved shap_bar_{model_name}.png")

    # --- SHAP Waterfall Plot (Single Fraud Prediction) ---
    if y_explain is not None:
        fraud_indices = np.where(
            np.array(y_explain) == 1
        )[0]
        if len(fraud_indices) > 0:
            idx = fraud_indices[0]
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

            explanation = shap.Explanation(
                values=shap_values[idx],
                base_values=expected_value,
                data=(X_explain.iloc[idx].values
                      if hasattr(X_explain, "iloc")
                      else X_explain[idx]),
                feature_names=feature_names,
            )
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(explanation, show=False, max_display=15)
            plt.title(f"SHAP Waterfall — Fraud Example ({model_name})")
            plt.tight_layout()
            plt.savefig(output_dir / f"shap_waterfall_{model_name}.png",
                        dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved shap_waterfall_{model_name}.png")

            # --- SHAP Force Plot (HTML) ---
            force_plot = shap.force_plot(
                expected_value, shap_values[idx],
                X_explain.iloc[idx] if hasattr(X_explain, "iloc") else X_explain[idx],
                feature_names=feature_names,
            )
            shap.save_html(
                str(output_dir / f"shap_force_{model_name}.html"), force_plot
            )
            print(f"  Saved shap_force_{model_name}.html")

    # --- SHAP Dependence Plot (Top Feature) ---
    top_feature_idx = np.abs(shap_values).mean(axis=0).argmax()
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature_idx, shap_values, X_explain,
        feature_names=feature_names, show=False,
    )
    plt.title(f"SHAP Dependence — {feature_names[top_feature_idx]} ({model_name})")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_dependence_{model_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved shap_dependence_{model_name}.png")

    return shap_values
