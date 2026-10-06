"""
HDBSCAN clustering execution supporting both `hdbscan` and `sklearn.cluster.HDBSCAN`.
"""

import os
from typing import Tuple

import numpy as np

from ..config import validate_parameters
from ..exceptions import ClusteringError


def fit_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.50,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Run HDBSCAN clustering over normalized embeddings.

    Returns:
        (labels, probabilities, outlier_scores, backend_used)
    """
    validate_parameters(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
    )
    if embeddings.ndim != 2 or len(embeddings) < 2 or not np.isfinite(embeddings).all():
        raise ClusteringError("Clustering requires at least two finite embedding rows.")
    backend = "hdbscan"
    try:
        import hdbscan
    except Exception:
        hdbscan = None
        backend = "sklearn"

    if backend == "hdbscan":
        n_jobs = max(1, min(8, os.cpu_count() or 1))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=False,
            core_dist_n_jobs=n_jobs,
        )
        labels = clusterer.fit_predict(embeddings).astype(int)

        probabilities = getattr(clusterer, "probabilities_", None)
        if probabilities is None:
            probabilities = np.ones(len(labels), dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        outlier_scores = getattr(clusterer, "outlier_scores_", None)
        if outlier_scores is None:
            outlier_scores = np.where(labels == -1, 1.0, 1.0 - probabilities).astype(np.float32)
        outlier_scores = np.asarray(outlier_scores, dtype=np.float32)

    else:
        try:
            from sklearn.cluster import HDBSCAN
        except Exception as exc:
            raise ClusteringError("Neither hdbscan nor sklearn.cluster.HDBSCAN is available.") from exc

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            # sklearn counts the point itself; contrib hdbscan does not.
            min_samples=min(min_samples + 1, len(embeddings)),
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(embeddings).astype(int)

        probabilities = getattr(clusterer, "probabilities_", None)
        if probabilities is None:
            probabilities = np.ones(len(labels), dtype=np.float32)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        outlier_scores = np.where(labels == -1, 1.0, 1.0 - probabilities).astype(np.float32)

    if len(labels) != embeddings.shape[0]:
        raise ClusteringError("Labels count does not match input row count.")

    return labels, probabilities, outlier_scores, backend
