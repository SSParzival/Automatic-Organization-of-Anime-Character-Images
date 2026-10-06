"""Deterministic offline model substitutes for notebook and workflow smoke tests.

These vectors and tags are fixtures, not predictions or quality measurements.
"""

import re
from unittest.mock import patch

import numpy as np


def synthetic_feature(path, model=None, size=None):
    match = re.search(r"img_(\d+)", str(path))
    index = int(match.group(1)) if match else 0
    center = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if index >= 5:
        center = -center
    return center + np.random.default_rng(index).normal(0, 0.01, len(center)).astype(np.float32)


def synthetic_embeddings(crop_paths, **kwargs):
    return [
        {"path": p, "status": "ok", "error_type": None, "error_message": None, "feature": synthetic_feature(p)}
        for p in crop_paths
    ]


def synthetic_tags(*args, **kwargs):
    return {"tagger": "synthetic", "character_tags": {"fixture_character": 0.95}, "general_tags": {}}, None, None


def install_substitutes():
    """Start explicitly scoped model patches; callers stop the returned handles."""
    patches = [
        patch("anime_character_organizer.preprocessing.cropping.detect_heads_safe", return_value=[]),
        patch("anime_character_organizer.preprocessing.cropping.detect_persons_safe", return_value=[]),
        patch("anime_character_organizer.workflows.embed.warmup_ccip_model", side_effect=synthetic_feature),
        patch("anime_character_organizer.workflows.embed.extract_all_embeddings", side_effect=synthetic_embeddings),
        patch("anime_character_organizer.workflows.naming.extract_tags_with_fallback", side_effect=synthetic_tags),
    ]
    for handle in patches:
        handle.start()
    return patches
