#!/usr/bin/env python3
"""
CLI script to run Stage 3: CCIP Embedding Extraction and Normalization.
"""

import argparse
from pathlib import Path
import sys

from anime_character_organizer.workflows.embed import run_embedding_extraction


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Extract CCIP visual identity embeddings from character crops."
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
        help="Path to previous 02_crop_preparation_* run directory (default: latest).",
    )
    parser.add_argument(
        "--model",
        default="ccip-caformer-24-randaug-pruned",
        help="CCIP feature extractor model name (default: ccip-caformer-24-randaug-pruned).",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="Inference batch size (default: 16).",
    )

    args = parser.parse_args()

    try:
        res = run_embedding_extraction(
            project_dir=args.project_dir,
            previous_run_dir=args.previous_run_dir,
            ccip_model=args.model,
            batch_size=args.batch_size,
        )
        print("\nEmbedding extraction completed successfully!")
        print(f"Run directory: {res['run_dir']}")
        print(f"Successful embeddings: {res['summary']['successful_embeddings']}")
        print(f"Embedding dimension: {res['summary']['embedding_dim']}")
        return 0
    except Exception as exc:
        print(f"Error during embedding extraction: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
