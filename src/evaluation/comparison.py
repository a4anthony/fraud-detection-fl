"""
Cross-model comparison tables and ablation study generation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def build_comparison_table(all_metrics, latency_results=None,
                           output_dir="outputs/tables"):
    """Build and save the master comparison table.

    Args:
        all_metrics: list of dicts from full_evaluation()
        latency_results: optional dict {model_name: mean_latency_ms}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_metrics)
    df = df.set_index("Model")

    if latency_results:
        df["Latency (ms)"] = df.index.map(
            lambda x: latency_results.get(x, np.nan)
        )

    df = df.round(4)
    df.to_csv(output_dir / "final_comparison.csv")
    print("Final comparison table:")
    print(df.to_string())
    return df


def build_ablation_table(ablation_data, output_dir="outputs/tables"):
    """Build and save ablation study table.

    Args:
        ablation_data: list of dicts with keys:
            Configuration, AUPRC, F1-Score, Recall, Latency, Privacy, Explainable
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(ablation_data)
    df.to_csv(output_dir / "ablation_study.csv", index=False)
    print("Ablation study table:")
    print(df.to_string(index=False))
    return df


def plot_ablation_bar_chart(ablation_df, output_dir="outputs/figures"):
    """Grouped bar chart for ablation study results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_cols = ["AUPRC", "F1-Score", "Recall"]
    available_cols = [c for c in metrics_cols if c in ablation_df.columns]

    if not available_cols:
        print("No metric columns found for ablation chart.")
        return

    x = np.arange(len(ablation_df))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, col in enumerate(available_cols):
        values = pd.to_numeric(ablation_df[col], errors="coerce")
        ax.bar(x + i * width, values, width, label=col, color=colors[i],
               edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Model Configuration Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(ablation_df["Configuration"], rotation=30, ha="right",
                       fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_bar_chart.png", dpi=300,
                bbox_inches="tight")
    plt.close()
