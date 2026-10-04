"""
Tag name normalization and formatting helpers.
"""


def normalize_tag_name(tag: str) -> str:
    """Normalize anime character tag: strip backslashes, convert slashes and spaces to underscores."""
    tag = str(tag).strip()
    tag = tag.replace("\\", "")
    tag = tag.replace("/", "_")
    tag = tag.replace(" ", "_")
    tag = tag.strip("_")
    return tag


def display_tag_name(tag: str) -> str:
    """Format tag name for display."""
    return normalize_tag_name(tag)
