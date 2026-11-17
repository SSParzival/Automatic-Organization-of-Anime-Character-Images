"""Deterministic offline model substitutes for workflow tests.

These vectors and tags are fixtures, not predictions or quality measurements.
"""

import re
from unittest.mock import patch

import numpy as np


def image_index(path):
    match = re.search(r"img_(\d+)", str(path))
    return int(match.group(1)) if match else 0


def synthetic_feature(path, model=None, size=None):
    index = image_index(path)
    center = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if index >= 5:
        center = -center
    return center + np.random.default_rng(index).normal(0, 0.01, len(center)).astype(np.float32)


def synthetic_embeddings(crop_paths, **kwargs):
    return [
        {"path": p, "status": "ok", "error_type": None, "error_message": None, "feature": synthetic_feature(p)}
        for p in crop_paths
    ]


def synthetic_tags(path):
    """Images 0-4 show character_a and images 5-9 character_b, matching the two embedding groups."""
    return {"character_a": 0.95} if image_index(path) < 5 else {"character_b": 0.90, "character_a": 0.30}


def offline_clients():
    """Online services that fail like an unreachable network, so tests never send requests."""
    from anime_character_organizer.exceptions import TaggingError

    class Unreachable:
        can_search_images = False

        def get(self, *args, **kwargs):
            raise TaggingError("offline test")

        posts = posts_by_md5 = get

    return {"danbooru": Unreachable(), "saucenao": None}


def install_substitutes():
    """Start explicitly scoped model patches; callers stop the returned handles."""
    patches = [
        patch("anime_character_organizer.preprocessing.cropping.detect_heads_safe", return_value=[]),
        patch("anime_character_organizer.preprocessing.cropping.detect_persons_safe", return_value=[]),
        patch("anime_character_organizer.workflows.embed.warmup_ccip_model", side_effect=synthetic_feature),
        patch("anime_character_organizer.workflows.embed.extract_all_embeddings", side_effect=synthetic_embeddings),
        patch("anime_character_organizer.workflows.naming.load_character_tagger"),
        patch("anime_character_organizer.workflows.naming.extract_character_tags", side_effect=synthetic_tags),
    ]
    for handle in patches:
        handle.start()
    return patches
