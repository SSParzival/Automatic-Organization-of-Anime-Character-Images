"""
Character tagging with PixAI Tagger v1.0 (8,308 character tags, May 2026 cutoff) through ONNX Runtime.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from .normalization import normalize_tag_name

# ONNX export of pixai-labs/pixai-tagger-v1.0, pinned to an audited commit.
TAGGER_REPO = "noaione/pixai-tagger-v1.0-onnx"
TAGGER_REVISION = "68e8f4f02dd56a5f40c1b7474489fa0f599dec34"
TAGGER_NAME = "pixai-tagger-v1.0"
INPUT_SIZE = 1008
# Calibrated per-category threshold published with the model; weaker candidates are discarded.
CHARACTER_THRESHOLD = 0.27


@lru_cache(maxsize=1)
def load_character_tagger() -> Tuple[Any, int, int, List[str]]:
    """Download (once) and open the tagger; returns (session, offset, count, character tags)."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> str:
        return hf_hub_download(TAGGER_REPO, name, revision=TAGGER_REVISION)

    with open(fetch("tags.json"), encoding="utf-8") as f:
        tag_map = json.load(f)
    fetch("model.onnx.data")
    # With the default memory arena, resident memory grows to about 8 GB after two 1008 px inferences
    # and is never released; without it, memory returns to about 2 GB between inferences (4.8 GB peak).
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(fetch("model.onnx"), options, providers=ort.get_available_providers())
    character = next(c for c in tag_map["categories"] if c["name"] == "character")
    tags = [normalize_tag_name(t) for t in character["tags"]]
    if len(tags) != int(character["count"]):
        raise ValueError(f"Corrupt tag map in {TAGGER_REPO}@{TAGGER_REVISION}.")
    return session, int(character["offset"]), len(tags), tags


def preprocess_image(image_path: Union[str, Path], size: int = INPUT_SIZE) -> np.ndarray:
    """Reproduce the upstream RescalePadProcessor: white alpha matte, aspect-preserving resize, black pad, [-1, 1]."""
    with Image.open(image_path) as source:
        image = source.convert("RGBA") if source.mode != "RGB" else source.copy()
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    width, height = image.size
    if (width, height) != (size, size):
        ratio = min(size / height, size / width)
        new_w, new_h = int(width * ratio), int(height * ratio)
        padded = Image.new("RGB", (size, size))
        padded.paste(
            image.resize((new_w, new_h), Image.Resampling.BILINEAR), ((size - new_w) // 2, (size - new_h) // 2)
        )
        image = padded
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose((array - 0.5) / 0.5, (2, 0, 1))[np.newaxis]


def extract_character_tags(image_path: Union[str, Path]) -> Dict[str, float]:
    """Return character tags scoring at least CHARACTER_THRESHOLD, sorted by descending probability."""
    session, offset, count, tags = load_character_tagger()
    logits = session.run(None, {session.get_inputs()[0].name: preprocess_image(image_path)})[0][0]
    probabilities = 1.0 / (1.0 + np.exp(-logits[offset : offset + count].astype(np.float64)))
    selected = np.flatnonzero(probabilities >= CHARACTER_THRESHOLD)
    ranked = selected[np.argsort(-probabilities[selected])]
    return {tags[i]: float(probabilities[i]) for i in ranked}
