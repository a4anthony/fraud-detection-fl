"""
Inference latency benchmarking for fraud detection models.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def benchmark_latency(model, X_test, n_iterations=10000):
    """Measure inference latency per transaction.

    Returns:
        results: dict with mean, median, p95, p99, max latency in ms
        times: list of individual latency measurements
    """
    single_times = []
    n_samples = len(X_test)

    for i in range(n_iterations):
        sample = X_test.iloc[[i % n_samples]]
        start = time.perf_counter()
        model.predict_proba(sample)
        end = time.perf_counter()
        single_times.append((end - start) * 1000)  # ms

    results = {
        "mean_ms": np.mean(single_times),
        "median_ms": np.median(single_times),
        "p95_ms": np.percentile(single_times, 95),
        "p99_ms": np.percentile(single_times, 99),
        "max_ms": np.max(single_times),
        "min_ms": np.min(single_times),
    }

    return results, single_times


def plot_latency_boxplot(all_latencies, model_names,
                         output_dir="outputs/figures"):
    """Box plot of inference latency across models."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        all_latencies, labels=model_names, patch_artist=True,
        showfliers=False,
    )

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=200, color="red", linestyle="--", alpha=0.5,
               label="200ms Target")
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_title("Inference Latency per Transaction", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "latency_boxplot.png", dpi=300)
    plt.close()


def plot_latency_histogram(latencies, model_name="Federated Ensemble",
                           output_dir="outputs/figures"):
    """Latency distribution histogram for a single model."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(latencies, bins=50, color="#3498db", alpha=0.7, edgecolor="black")
    ax.axvline(x=200, color="red", linestyle="--", label="200ms Target")
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Latency Distribution — {model_name}")
    ax.legend()
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(output_dir / f"latency_histogram_{safe_name}.png", dpi=300)
    plt.close()
