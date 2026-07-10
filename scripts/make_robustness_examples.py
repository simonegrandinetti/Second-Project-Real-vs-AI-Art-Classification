#!/usr/bin/env python3
"""Create the report panel showing the robustness perturbations.

The script intentionally writes only a figure artifact. It does not recompute
metrics or modify the training outputs. The transforms mirror the robustness
evaluation path: perturb first, then resize to the model input size.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from ai_art_detection.evaluation import (
    ROBUSTNESS_CONDITIONS,
    apply_robustness_perturbation,
    robustness_plot_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("outputs/tables/test_split.csv"),
        help="Split CSV used to choose a reproducible example image.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Optional explicit image path. Overrides --split-csv selection.",
    )
    parser.add_argument(
        "--source-label",
        default="Human",
        help="Preferred source_label when selecting from --split-csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/generated_robustness_examples.png"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def select_source_image(split_csv: Path, source_label: str) -> tuple[Path, str]:
    """Choose the first existing image for the requested source label."""
    frame = pd.read_csv(split_csv)
    if "image_path" not in frame.columns:
        raise ValueError(f"{split_csv} must contain an image_path column.")

    candidates = frame
    if "source_label" in frame.columns:
        preferred = frame.loc[frame["source_label"] == source_label]
        if not preferred.empty:
            candidates = preferred

    for _, row in candidates.iterrows():
        image_path = Path(str(row["image_path"]))
        if image_path.exists():
            style = str(row.get("style_label", "unknown")).replace("_", " ")
            source = str(row.get("source_label", source_label))
            return image_path, f"Source: {source}, {style}"

    raise FileNotFoundError(
        "No existing image path was found in the split CSV. "
        "Use --image-path to provide one explicitly."
    )


def resize_to_model_input(image: Image.Image, image_size: int) -> Image.Image:
    """Resize with the bicubic interpolation used for the visual panel."""
    return image.resize((image_size, image_size), Image.Resampling.BICUBIC)


def build_panels(image: Image.Image, image_size: int) -> list[tuple[str, Image.Image]]:
    """Return clean and perturbed versions of the same source image."""
    panels = []
    for condition in ROBUSTNESS_CONDITIONS:
        perturbed = apply_robustness_perturbation(
            image, condition.kind, condition.value
        )
        panels.append(
            (
                robustness_plot_label(condition.kind, condition.value),
                resize_to_model_input(perturbed, image_size),
            )
        )
    return panels


def save_panel(
    panels: list[tuple[str, Image.Image]],
    caption: str,
    output_path: Path,
) -> None:
    """Save a compact comparison panel for the report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 5
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12.0, rows * 2.55), dpi=220)
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for axis, (title, panel) in zip(axes_flat, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=8.5, pad=6)
        axis.axis("off")
    for axis in axes_flat[len(panels) :]:
        axis.axis("off")
    figure.text(0.5, 0.025, caption, ha="center", fontsize=9)
    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.94,
        bottom=0.08,
        wspace=0.06,
        hspace=0.28,
    )
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.image_path is not None:
        source_path = args.image_path
        caption = f"Source image: {source_path.name}"
    else:
        source_path, caption = select_source_image(args.split_csv, args.source_label)

    image = Image.open(source_path).convert("RGB")
    panels = build_panels(image, args.image_size)
    save_panel(panels, caption, args.output)
    print(f"Wrote {args.output}")
    print(f"Source image: {source_path}")


if __name__ == "__main__":
    main()
