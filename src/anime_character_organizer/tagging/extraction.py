"""
Inference wrappers for anime image taggers (PixAI and WD14) with fallback.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .normalization import normalize_tag_name


def extract_pixai_tags(
    image_path: Union[str, Path],
    model_name: str = "v0.9",
) -> Dict[str, Any]:
    """Extract character and general tags using PixAI model from dghs-imgutils."""
    from imgutils.tagging import get_pixai_tags

    general, character, ips, ips_mapping = get_pixai_tags(
        str(image_path),
        model_name=model_name,
        thresholds={"character": 0.0},
        fmt=("general", "character", "ips", "ips_mapping"),
    )

    general_dict = {normalize_tag_name(k): float(v) for k, v in dict(general or {}).items()}
    character_dict = {normalize_tag_name(k): float(v) for k, v in dict(character or {}).items()}

    return {
        "tagger": "pixai",
        "general_tags": general_dict,
        "character_tags": character_dict,
        "ips": [normalize_tag_name(x) for x in list(ips or [])],
        "ips_mapping": {
            normalize_tag_name(k): [normalize_tag_name(x) for x in v] for k, v in dict(ips_mapping or {}).items()
        },
    }


def extract_wd14_tags(
    image_path: Union[str, Path],
    model_name: str = "SwinV2_v3",
) -> Dict[str, Any]:
    """Extract character and general tags using WD14 model from dghs-imgutils."""
    from imgutils.tagging import get_wd14_tags

    rating, general, character = get_wd14_tags(
        str(image_path),
        model_name=model_name,
        character_threshold=0.0,
        fmt=("rating", "general", "character"),
    )

    general_dict = {normalize_tag_name(k): float(v) for k, v in dict(general or {}).items()}
    character_dict = {normalize_tag_name(k): float(v) for k, v in dict(character or {}).items()}

    return {
        "tagger": "wd14",
        "rating_tags": {normalize_tag_name(k): float(v) for k, v in dict(rating or {}).items()},
        "general_tags": general_dict,
        "character_tags": character_dict,
        "ips": [],
        "ips_mapping": {},
    }


def extract_tags_with_fallback(
    image_path: Union[str, Path],
    primary_tagger: str = "pixai",
    pixai_model: str = "v0.9",
    wd14_model: str = "SwinV2_v3",
    use_wd14_fallback: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Extract tags using primary tagger, falling back to WD14 if primary fails.

    Returns:
        (tags_dict, error_type, error_message)
    """
    try:
        if primary_tagger == "pixai":
            return extract_pixai_tags(image_path, model_name=pixai_model), None, None
        elif primary_tagger == "wd14":
            return extract_wd14_tags(image_path, model_name=wd14_model), None, None
        else:
            raise ValueError(f"Unknown primary tagger: {primary_tagger}")
    except Exception as primary_exc:
        if use_wd14_fallback and primary_tagger != "wd14":
            try:
                return (
                    extract_wd14_tags(image_path, model_name=wd14_model),
                    type(primary_exc).__name__,
                    str(primary_exc),
                )
            except Exception as fallback_exc:
                return None, type(fallback_exc).__name__, str(fallback_exc)
        return None, type(primary_exc).__name__, str(primary_exc)
