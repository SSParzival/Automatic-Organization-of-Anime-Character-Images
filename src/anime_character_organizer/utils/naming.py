"""
Filesystem naming sanitization helpers.
"""

from typing import Any


def sanitize_filename_component(value: Any, fallback: str = "item", max_length: int = 140) -> str:
    """
    Sanitize an arbitrary string into a safe, valid filesystem filename component.
    Replaces forbidden characters (<>:"/\\|?*), unprintable characters, and strips trailing periods.
    """
    value = str(value).strip()
    if not value:
        value = fallback

    forbidden = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in forbidden else ch for ch in value)
    cleaned = "".join(ch if ch.isprintable() else "_" for ch in cleaned)
    cleaned = cleaned.strip().strip(".")
    cleaned = "_".join(cleaned.split())

    if not cleaned:
        cleaned = fallback

    return cleaned[:max_length]


def sanitize_folder_name(value: Any, fallback: str = "_needs_review", max_length: int = 120) -> str:
    """Sanitize folder name ensuring safe path segment creation."""
    return sanitize_filename_component(value, fallback=fallback, max_length=max_length)
