"""
Comprehensive evaluation metrics and visualization for fraud detection.
"""

from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def full_evaluation(y_true, y_pred, y_proba, model_name,
                    output_dir="outputs/figures"):
    """Run all evaluation metrics and generate confusion matrix plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "Model": model_name,
        "AUPRC": average_precision_score(y_true, y_proba),
        "F1-Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "FPR": cm[0, 1] / cm[0].sum() if cm[0].sum() > 0 else 0.0,
    }

    # Confusion Matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=300)
    plt.close()

    return metrics


def plot_precision_recall_curves(models_dict, y_true, output_dir="outputs/figures"):
    """Plot PR curves for all models on one figure.

    Args:
        models_dict: {model_name: y_proba_array}
        y_true: true labels
    """
    output_dir = Path(output_dir)
    plt.figure(figsize=(10, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, (name, y_proba) in enumerate(models_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auprc = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, color=colors[i % len(colors)],
                 linewidth=2, label=f"{name} (AUPRC={auprc:.4f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves — All Models", fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves_all.png", dpi=300)
    plt.close()


def plot_roc_curves(models_dict, y_true, output_dir="outputs/figures"):
    """Plot ROC curves for all models on one figure."""
    output_dir = Path(output_dir)
    plt.figure(figsize=(10, 7))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, (name, y_proba) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 linewidth=2, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — All Models", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_all.png", dpi=300)
    plt.close()
