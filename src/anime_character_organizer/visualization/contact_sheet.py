"""
Contact sheet rendering for visual inspection of image clusters, crops, and review groups.
"""

import math
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000


def create_contact_sheet(
    image_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    thumb_size: int = 180,
    columns: int = 6,
    title: Optional[str] = None,
    quality: int = 92,
) -> bool:
    """
    Render a grid contact sheet from a list of image paths.
    
    Args:
        image_paths: Paths to image files to display.
        output_path: Destination path for the saved JPEG sheet.
        thumb_size: Pixel width and height for each grid thumbnail.
        columns: Number of thumbnail columns in the grid.
        title: Optional title string rendered in a banner across the top.
        quality: JPEG compression quality (0-100).
        
    Returns:
        True if at least one image was rendered and saved, False otherwise.
    """
    valid_paths = [Path(p) for p in image_paths if pd.notna(p) and Path(p).exists()]
    if not valid_paths:
        return False

    rows = int(math.ceil(len(valid_paths) / columns))
    title_height = 34 if title else 0

    sheet_width = columns * thumb_size
    sheet_height = rows * thumb_size + title_height

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    if title:
        draw.rectangle((0, 0, sheet_width, title_height), fill=(245, 245, 245))
        draw.text((10, 9), title[:180], fill=(20, 20, 20))

    rendered_count = 0
    for idx, path in enumerate(valid_paths):
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img.convert("RGB"))
                img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

                x = (idx % columns) * thumb_size + (thumb_size - img.width) // 2
                y = title_height + (idx // columns) * thumb_size + (thumb_size - img.height) // 2

                sheet.paste(img, (x, y))
                rendered_count += 1
        except Exception:
            continue

    if rendered_count == 0:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=quality, optimize=True)
    return True
