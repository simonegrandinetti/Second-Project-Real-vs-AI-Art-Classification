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
    """Parse source-image selection and panel output settings.

    Returns:
        Parsed command-line namespace. An explicit image path takes precedence over
        selection from the saved test split.
    """
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
    """Choose a reproducible existing image from a saved split.

    Args:
        split_csv: Metadata CSV containing at least an ``image_path`` column.
        source_label: Preferred source when that optional column is available.

    Returns:
        The first existing candidate path and a short source/style caption. If the
        requested source has no rows, selection falls back to the complete split.

    Raises:
        ValueError: If ``image_path`` is missing from the CSV.
        FileNotFoundError: If none of the candidate paths exists locally.
    """
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
    """Resize one displayed example to the model's square input geometry.

    Args:
        image: Clean or perturbed Pillow image.
        image_size: Desired width and height.

    Returns:
        A bicubically resized Pillow image. Normalization is omitted because this
        function prepares a human-viewable figure rather than a model tensor.
    """
    return image.resize((image_size, image_size), Image.Resampling.BICUBIC)


def build_panels(image: Image.Image, image_size: int) -> list[tuple[str, Image.Image]]:
    """Render every registered robustness condition on one source image.

    Args:
        image: Shared Pillow source image.
        image_size: Square display size applied after each perturbation.

    Returns:
        ``(label, image)`` pairs in the same order as ``ROBUSTNESS_CONDITIONS``.
    """
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
    """Save robustness examples in a compact five-column figure.

    Args:
        panels: Ordered display labels and rendered Pillow images.
        caption: Shared source caption placed beneath the grid.
        output_path: Destination image path. Missing parents are created.

    Note:
        Unused cells in the final row remain blank, and the Matplotlib figure is
        closed after saving.
    """
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
    """Select one artwork and write its complete robustness comparison panel.

    This command creates only a report figure. It does not evaluate a checkpoint,
    recompute metrics, or modify experiment outputs.
    """
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
