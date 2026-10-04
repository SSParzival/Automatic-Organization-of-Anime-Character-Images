"""
CCIP feature extraction with batching, graceful single-item fallback, and memory management.
"""

import gc
from typing import Any, Dict, Generator, Iterable, List, Union
import numpy as np
from tqdm.auto import tqdm

from ..exceptions import EmbeddingError


def batch_iterable(values: Iterable[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """Yield successive slices of length batch_size from an iterable."""
    items = list(values)
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def warmup_ccip_model(sample_path: str, model: str, size: int) -> np.ndarray:
    """Warm up the CCIP model by extracting features for one sample image."""
    try:
        from imgutils.metrics import ccip_extract_feature
        feat = ccip_extract_feature(sample_path, model=model, size=size)
        arr = np.asarray(feat, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D feature vector, got shape {arr.shape}.")
        return arr
    except Exception as exc:
        raise EmbeddingError(
            f"CCIP warmup failed with model={model}: {exc}. Ensure model weights are accessible."
        ) from exc


def extract_batch_with_fallback(
    paths: List[str],
    model: str,
    size: int,
) -> List[Dict[str, Any]]:
    """
    Extract features for a batch of image paths via imgutils.metrics.
    If batch extraction fails, falls back to single-image extraction per path.
    """
    from imgutils.metrics import ccip_batch_extract_features, ccip_extract_feature

    paths_str = [str(p) for p in paths]

    try:
        features = ccip_batch_extract_features(paths_str, model=model, size=size)
        features = np.asarray(features, dtype=np.float32)

        if features.ndim != 2 or features.shape[0] != len(paths_str):
            raise ValueError(
                f"Feature matrix mismatch: expected ({len(paths_str)}, D), got {features.shape}"
            )

        return [
            {
                "path": path,
                "status": "ok",
                "error_type": None,
                "error_message": None,
                "feature": features[idx],
            }
            for idx, path in enumerate(paths_str)
        ]

    except Exception:
        # Fall back to single-image extraction
        records: List[Dict[str, Any]] = []
        for path in paths_str:
            try:
                feature = ccip_extract_feature(path, model=model, size=size)
                feature = np.asarray(feature, dtype=np.float32)
                if feature.ndim != 1:
                    raise ValueError(f"Expected 1D feature vector, got shape {feature.shape}.")
                records.append({
                    "path": path,
                    "status": "ok",
                    "error_type": None,
                    "error_message": None,
                    "feature": feature,
                })
            except Exception as single_exc:
                records.append({
                    "path": path,
                    "status": "error",
                    "error_type": type(single_exc).__name__,
                    "error_message": str(single_exc),
                    "feature": None,
                })
        return records


def extract_all_embeddings(
    crop_paths: List[str],
    model: str = "ccip-caformer-24-randaug-pruned",
    image_size: int = 384,
    batch_size: int = 16,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Extract CCIP embeddings across all crop paths with garbage collection."""
    all_records: List[Dict[str, Any]] = []
    batches = list(batch_iterable(crop_paths, batch_size))
    iterator = tqdm(batches, desc="Extracting CCIP embeddings") if show_progress else batches

    for batch in iterator:
        batch_records = extract_batch_with_fallback(batch, model=model, size=image_size)
        all_records.extend(batch_records)
        gc.collect()

    return all_records
