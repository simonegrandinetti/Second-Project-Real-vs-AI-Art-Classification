"""Evaluation metrics, diagnostic plots, and robustness transforms."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .data import ArtBinaryDataset, IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class RobustnessCondition:
    """One post-training perturbation used to probe model stability."""

    kind: str
    value: float


ROBUSTNESS_CONDITIONS: tuple[RobustnessCondition, ...] = (
    RobustnessCondition("clean", 1.0),
    RobustnessCondition("contrast", 0.5),
    RobustnessCondition("contrast", 1.5),
    RobustnessCondition("contrast", 2.0),
    RobustnessCondition("brightness", 0.7),
    RobustnessCondition("brightness", 1.3),
    RobustnessCondition("saturation", 0.5),
    RobustnessCondition("saturation", 1.5),
    RobustnessCondition("blur", 1.0),
    RobustnessCondition("blur", 2.0),
    RobustnessCondition("noise", 0.03),
    RobustnessCondition("noise", 0.08),
    RobustnessCondition("jpeg", 90.0),
    RobustnessCondition("jpeg", 50.0),
    RobustnessCondition("jpeg", 20.0),
    RobustnessCondition("resample", 128.0),
    RobustnessCondition("resample", 64.0),
)


def robustness_condition_label(condition: str) -> str:
    """Return the report/table label for one robustness condition."""
    labels = {
        "clean": "Clean",
        "contrast": "Contrast",
        "brightness": "Brightness",
        "saturation": "Saturation",
        "blur": "Blur",
        "noise": "Noise",
        "jpeg": "JPEG",
        "resample": "Resample",
    }
    return labels.get(condition, condition.title())


def robustness_value_label(condition: str, value: float) -> str:
    """Return the report/table value label for one robustness setting."""
    if condition == "clean":
        return "--"
    if condition in {"contrast", "brightness", "saturation"}:
        return f"{value:.1f}x"
    if condition == "blur":
        return f"r={value:g} px"
    if condition == "noise":
        return f"sigma={value:g}"
    if condition == "jpeg":
        return f"Q{int(value)}"
    if condition == "resample":
        return f"{int(value)} px"
    return f"{value:g}"


def robustness_plot_label(condition: str, value: float) -> str:
    """Return a compact x-axis label for robustness plots."""
    value_label = robustness_value_label(condition, value)
    if condition == "clean":
        return "clean"
    return f"{robustness_condition_label(condition)} {value_label}"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Stable sigmoid for model logits stored in NumPy/Pandas objects."""
    logits = np.clip(np.asarray(logits, dtype=float), -50, 50)
    return 1 / (1 + np.exp(-logits))


def binary_metrics(
    labels: np.ndarray | list,
    logits: np.ndarray | list,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute thresholded binary metrics plus ROC-AUC when both classes exist."""
    labels = np.asarray(labels, dtype=int)
    probabilities = sigmoid(np.asarray(logits))
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    roc_auc = float("nan")
    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(labels, probabilities)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a model and return aggregate metrics plus per-image predictions."""
    model.eval()
    loss_total = 0.0
    labels_all: list[float] = []
    logits_all: list[float] = []
    paths: list[str] = []
    sources: list[str] = []
    styles: list[str] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images).flatten()
        loss_total += criterion(logits, labels).item() * len(images)

        labels_all.extend(labels.cpu().tolist())
        logits_all.extend(logits.cpu().tolist())
        paths.extend(batch["path"])
        sources.extend(batch["source_label"])
        styles.extend(batch["style_label"])

    metrics = binary_metrics(labels_all, logits_all, threshold)
    metrics["loss"] = loss_total / len(loader.dataset)
    predictions = pd.DataFrame(
        {
            "image_path": paths,
            "label": np.asarray(labels_all, dtype=int),
            "logit": logits_all,
            "prob_fake": sigmoid(np.asarray(logits_all)),
            "source_label": sources,
            "style_label": styles,
        }
    )
    predictions["pred"] = (predictions["prob_fake"] >= threshold).astype(int)
    predictions["correct"] = predictions["label"] == predictions["pred"]
    return metrics, predictions


def metrics_by_group(predictions: pd.DataFrame, group: str) -> pd.DataFrame:
    """Compute full binary metrics independently for each subgroup."""
    rows = []
    for name, part in predictions.groupby(group, dropna=False):
        values = binary_metrics(part["label"], part["logit"])
        rows.append({group: name, "count": len(part), **values})
    return pd.DataFrame(rows)


def style_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return full binary metrics for each style, which contains both classes."""
    return metrics_by_group(predictions, "style_label")


def source_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report class-appropriate errors for source groups containing one label."""
    rows = []
    for source, part in predictions.groupby("source_label", dropna=False):
        labels = set(part["label"].astype(int))
        if len(labels) != 1:
            raise ValueError(
                f"Source group {source!r} contains labels {sorted(labels)}; "
                "source error summaries require one binary class per source."
            )
        label = labels.pop()
        error_type = "false_positive_rate" if label == 0 else "false_negative_rate"
        error_rate = float((part["pred"].astype(int) != label).mean())
        rows.append(
            {
                "source_label": source,
                "count": len(part),
                "accuracy": 1.0 - error_rate,
                "error_type": error_type,
                "error_rate": error_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("source_label").reset_index(drop=True)


def plot_confusion(
    predictions: pd.DataFrame, output_path: Path, title: str = "Test confusion matrix"
) -> None:
    """Save a two-class confusion matrix figure."""
    matrix = confusion_matrix(predictions["label"], predictions["pred"])
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        matrix, display_labels=["Real/Human", "Fake/AI"]
    ).plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_roc(
    predictions: pd.DataFrame, output_path: Path, title: str = "Test ROC curve"
) -> None:
    """Save a ROC curve figure from per-image fake probabilities."""
    false_positive, true_positive, _ = roc_curve(
        predictions["label"], predictions["prob_fake"]
    )
    score = auc(false_positive, true_positive)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(false_positive, true_positive, label=f"AUC = {score:.3f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


class ContrastTransform:
    """Pillow contrast perturbation used for robustness checks."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self.factor)


class BrightnessTransform:
    """Pillow brightness perturbation used for robustness checks."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(image).enhance(self.factor)


class SaturationTransform:
    """Pillow colour-saturation perturbation used for robustness checks."""

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Color(image).enhance(self.factor)


class GaussianBlurTransform:
    """Apply Gaussian blur with a fixed pixel radius."""

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


class DeterministicGaussianNoiseTransform:
    """Add image-content-seeded Gaussian noise before normalization.

    The seed is derived from the image bytes, so the same image receives the
    same perturbation independent of dataloader order or worker count.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        digest = hashlib.blake2b(rgb.tobytes(), digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=self.sigma, size=pixels.shape)
        noisy = np.clip(pixels + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255).round().astype(np.uint8), mode="RGB")


class JPEGCompressionTransform:
    """Round-trip an image through JPEG compression at a fixed quality."""

    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class CommonResampleTransform:
    """Apply the same low-resolution bottleneck to every source."""

    def __init__(self, intermediate_size: int):
        self.intermediate_size = intermediate_size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(
            (self.intermediate_size, self.intermediate_size),
            Image.Resampling.BICUBIC,
        )


def robustness_perturbation(kind: str, value: float):
    """Build only the PIL-space perturbation for a robustness condition."""
    if kind == "clean":
        return None
    if kind == "contrast":
        return ContrastTransform(value)
    if kind == "brightness":
        return BrightnessTransform(value)
    if kind == "saturation":
        return SaturationTransform(value)
    if kind == "blur":
        return GaussianBlurTransform(value)
    if kind == "noise":
        return DeterministicGaussianNoiseTransform(value)
    if kind == "jpeg":
        return JPEGCompressionTransform(int(value))
    if kind == "resample":
        return CommonResampleTransform(int(value))
    raise ValueError(f"Unknown robustness condition: {kind}")


def apply_robustness_perturbation(
    image: Image.Image, kind: str, value: float
) -> Image.Image:
    """Apply a PIL-space robustness perturbation to one image."""
    perturbation = robustness_perturbation(kind, value)
    if perturbation is None:
        return image
    return perturbation(image)


def robustness_transform(kind: str, value: float, image_size: int):
    """Build the evaluation transform for one robustness condition."""
    perturbation = robustness_perturbation(kind, value)
    perturbations = [] if perturbation is None else [perturbation]
    return transforms.Compose(
        [
            *perturbations,
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluate_robustness(
    model: nn.Module,
    test_frame: pd.DataFrame,
    criterion: nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate the fixed model under clean and perturbed test transforms."""
    rows = []
    for condition in ROBUSTNESS_CONDITIONS:
        # The perturbation is applied before the shared resize/normalization
        # pipeline so every source sees the same final tensor shape.
        dataset = ArtBinaryDataset(
            test_frame,
            robustness_transform(condition.kind, condition.value, image_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics, _ = evaluate(
            model, loader, criterion, device, threshold=threshold
        )
        rows.append(
            {"condition": condition.kind, "value": condition.value, **metrics}
        )
    return pd.DataFrame(rows)
