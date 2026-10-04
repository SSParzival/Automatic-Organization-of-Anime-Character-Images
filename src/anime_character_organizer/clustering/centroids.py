"""
Cluster centroid calculation and Euclidean distance metrics.
"""

from typing import Dict
import numpy as np


def compute_cluster_centroids(embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute L2-normalized centroids for each non-noise cluster label (label != -1).
    
    Returns:
        Dictionary mapping cluster_label (int) -> normalized centroid vector (np.ndarray float32).
    """
    centroids: Dict[int, np.ndarray] = {}

    for label in sorted(set(labels)):
        if label == -1:
            continue

        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue

        centroid = embeddings[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        centroids[int(label)] = centroid.astype(np.float32)

    return centroids


def euclidean_distance_to_centroid(embeddings: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance from each row vector in embeddings to a centroid vector."""
    return np.linalg.norm(embeddings - centroid.reshape(1, -1), axis=1)
