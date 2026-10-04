"""
Diagnostic plots for clustering evaluation.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cluster_size_distribution(
    cluster_summary_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> plt.Figure:
    """Plot histogram of cluster sizes for non-noise clusters."""
    non_noise = cluster_summary_df[~cluster_summary_df["is_noise_cluster"]].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    if non_noise.empty:
        ax.text(0.5, 0.5, "No non-noise clusters were detected.", ha="center", va="center")
        ax.axis("off")
    else:
        sizes = non_noise["image_count"].to_numpy()
        bins = min(50, max(5, int(np.sqrt(len(sizes)))))
        ax.hist(sizes, bins=bins)
        ax.set_xlabel("Images per cluster")
        ax.set_ylabel("Number of clusters")
        ax.set_title("Distribution of HDBSCAN cluster sizes")

    fig.tight_layout()
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi)

    return fig


def plot_probability_and_outlier_distributions(
    cluster_manifest_df: pd.DataFrame,
    low_probability_threshold: float = 0.35,
    high_outlier_threshold: float = 0.95,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 160,
) -> plt.Figure:
    """Plot side-by-side distributions of cluster membership probabilities and outlier scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    probs = cluster_manifest_df["cluster_probability"].dropna().to_numpy()
    axes[0].hist(probs, bins=50)
    axes[0].axvline(low_probability_threshold, linestyle="--", color="red")
    axes[0].set_xlabel("Cluster membership probability")
    axes[0].set_ylabel("Image count")
    axes[0].set_title("Membership probability distribution")

    outliers = cluster_manifest_df["outlier_score"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    axes[1].hist(outliers, bins=50)
    axes[1].axvline(high_outlier_threshold, linestyle="--", color="red")
    axes[1].set_xlabel("Outlier score")
    axes[1].set_ylabel("Image count")
    axes[1].set_title("Outlier-score distribution")

    fig.tight_layout()
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi)

    return fig
