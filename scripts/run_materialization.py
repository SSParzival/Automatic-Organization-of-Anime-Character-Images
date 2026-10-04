#!/usr/bin/env python3
"""
CLI script to run Stage 5: Non-Destructive Folder Materialization.
"""

import argparse
import sys
from pathlib import Path

from anime_character_organizer.workflows.materialize import run_folder_materialization


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5: Non-destructively organize images into destination cluster directories."
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
        help="Path to previous 04_hdbscan_clustering_* run directory (default: latest).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        type=Path,
        help="Custom destination directory for organized folder tree.",
    )
    parser.add_argument(
        "--mode",
        "--link-mode",
        "-m",
        dest="mode",
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="Materialization link mode (default: hardlink).",
    )
    parser.add_argument(
        "--no-copy-fallback",
        action="store_true",
        help="Fail if hardlink cannot be created rather than falling back to copy.",
    )

    args = parser.parse_args()

    try:
        res = run_folder_materialization(
            project_dir=args.project_dir,
            previous_run_dir=args.previous_run_dir,
            final_output_dir=args.output_dir,
            materialization_mode=args.mode,
            allow_hardlink_fallback_to_copy=not args.no_copy_fallback,
        )
        print("\nMaterialization completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Organized output: {res['final_output_dir']}")
        print(f"Total files organized: {res['summary']['total_files']}")
        print(f"Total folders created: {res['summary']['total_folders']}")
        print(f"Successful operations: {res['summary']['successful_operations']}")
        return 0
    except Exception as exc:
        print(f"Error during materialization: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
