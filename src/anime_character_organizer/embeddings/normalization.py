"""
Vector normalization functions.
"""

from typing import Union

import numpy as np


def l2_normalize_matrix(matrix: Union[np.ndarray, list], eps: float = 1e-12) -> np.ndarray:
    """
    Perform L2 normalization along rows of a 2D matrix.

    Args:
        matrix: 2D numpy array of shape (N, D).
        eps: Small epsilon to prevent division by zero.

    Returns:
        L2-normalized float32 numpy array of identical shape.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for normalization, got shape {arr.shape}.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms
