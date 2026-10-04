"""
Image file validation functions using PIL.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000


def validate_image_file(path: Union[str, Path]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate that an image file exists, is non-empty, can be opened and verified,
    and has positive dimensions.

    Returns:
        (is_valid, error_type, error_message)
    """
    path = Path(path)
    if not path.exists():
        return False, "FileNotFoundError", "Image file does not exist."
    if not path.is_file():
        return False, "NotAFileError", "Path exists but is not a regular file."

    try:
        if path.stat().st_size <= 0:
            return False, "EmptyFile", "File has zero bytes."

        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img.convert("RGB"))
            width, height = img.size

        if width <= 0 or height <= 0:
            return False, "InvalidImageSize", "Image has non-positive dimensions."

        return True, None, None

    except Exception as exc:
        return False, type(exc).__name__, str(exc)
