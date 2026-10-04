#!/usr/bin/env python3
"""
CLI script to run Stage 2: Anime Head/Person Detection and Crop Preparation.
"""

import argparse
import sys
from pathlib import Path

from anime_character_organizer.workflows.crop import run_crop_preparation


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Filter duplicates, detect heads/persons, and create identity crops."
    )
    parser.add_argument(
        "--project-dir",
        "-p",
        default=Path("./anime_character_pipeline"),
        type=Path,
        help="Base pipeline directory (default: ./anime_character_pipeline).",
    )
    parser.add_argument(
        "--previous-run-dir",
        default=None,
        type=Path,
        help="Path to previous 01_dataset_audit_* run directory (default: latest).",
    )
    parser.add_argument(
        "--filter-perceptual-duplicates",
        action="store_true",
        help="Exclude perceptual duplicate non-representatives (default: False, to avoid dropping images before clustering).",
    )
    parser.add_argument(
        "--crop-size",
        default=512,
        type=int,
        help="Output crop dimensions in pixels (default: 512).",
    )
    parser.add_argument(
        "--crop-format",
        default="JPEG",
        choices=["JPEG", "PNG"],
        help="Format for saved crop files (default: JPEG).",
    )

    args = parser.parse_args()

    try:
        res = run_crop_preparation(
            project_dir=args.project_dir,
            previous_run_dir=args.previous_run_dir,
            use_perceptual_representatives=args.filter_perceptual_duplicates,
            crop_output_size=(args.crop_size, args.crop_size),
            crop_format=args.crop_format,
        )
        print("\nCrop preparation completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Successful crops: {res['summary']['successful_crops']}")
        print(f"Crop errors: {res['summary']['crop_errors']}")
        return 0
    except Exception as exc:
        print(f"Error during crop preparation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
