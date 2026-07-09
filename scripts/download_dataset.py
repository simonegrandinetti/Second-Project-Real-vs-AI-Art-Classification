#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import kagglehub

from ai_art_detection.data import (
    scan_dataset,
    validate_dataset_inventory,
    validate_image_readability,
)

DATASET_HANDLE = "ravidussilva/real-ai-art/versions/5"
MINIMUM_FREE_BYTES = 25 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate the pinned AI-ArtBench dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/real-ai-art"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"At least {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB free is required; "
            f"only {free_bytes / 1024**3:.1f} GiB is available."
        )
    if "KAGGLE_API_TOKEN" not in os.environ:
        print(
            "KAGGLE_API_TOKEN is not set; attempting Kaggle's anonymous "
            "public-dataset access."
        )
    downloaded = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(destination),
        force_download=args.force,
    )
    print(f"Downloaded to: {downloaded}")

    frame = scan_dataset(destination)
    validate_dataset_inventory(frame)
    validate_image_readability(frame)
    print(
        f"Validated {len(frame):,} images across "
        f"{frame['official_split'].nunique()} official splits, "
        f"{frame['source_label'].nunique()} sources, and "
        f"{frame['style_label'].nunique()} styles."
    )


if __name__ == "__main__":
    main()
