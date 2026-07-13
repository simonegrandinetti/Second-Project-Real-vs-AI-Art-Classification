#!/usr/bin/env python3
"""Download and validate the pinned AI-ArtBench dataset.

This script is intentionally small and linear because it is the first step of
the coursework workflow: choose the destination, check that the machine has
enough free space, download the exact Kaggle version, then validate the
extracted images before any experiment is allowed to run.
"""

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
    """Download AI-ArtBench version 5 and validate the extracted inventory.

    The command creates the destination directory, checks that its filesystem has at
    least 25 GiB free, asks kagglehub for the pinned dataset version, then scans every
    image and verifies both official counts and Pillow readability. Existing content
    is redownloaded only when ``--force`` is supplied.

    Raises:
        RuntimeError: If the destination filesystem does not have enough free space.
        ValueError: If downloaded metadata, counts, or image readability fail project
            validation.
    """
    # 1) Read the command line choices.  The default matches the README and
    # notebook, so most users can simply run `python scripts/download_dataset.py`.
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

    # 2) Refuse to start a large download if the destination volume is too
    # small.  This catches the most common failure before Kaggle work begins.
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            f"At least {MINIMUM_FREE_BYTES / 1024**3:.0f} GiB free is required; "
            f"only {free_bytes / 1024**3:.1f} GiB is available."
        )

    # 3) Kaggle public access may work anonymously, but authenticated downloads
    # should pass credentials through the environment, never through the repo.
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

    # 4) The dataset layout is part of the experimental protocol.  Validation
    # fails loudly if counts, aliases, or image readability do not match.
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
