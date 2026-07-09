from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
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


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=float), -50, 50)
    return 1 / (1 + np.exp(-logits))


def binary_metrics(
    labels: np.ndarray | list,
    logits: np.ndarray | list,
    threshold: float = 0.5,
) -> dict[str, float]:
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
    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(self.factor)


class JPEGCompressionTransform:
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


def robustness_transform(kind: str, value: float, image_size: int):
    perturbations = []
    if kind == "contrast":
        perturbations.append(ContrastTransform(value))
    elif kind == "jpeg":
        perturbations.append(JPEGCompressionTransform(int(value)))
    elif kind == "resample":
        perturbations.append(CommonResampleTransform(int(value)))
    elif kind != "clean":
        raise ValueError(f"Unknown robustness condition: {kind}")
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
    conditions = [
        ("clean", 1.0),
        ("contrast", 0.5),
        ("contrast", 1.5),
        ("contrast", 2.0),
        ("jpeg", 90),
        ("jpeg", 50),
        ("jpeg", 20),
        ("resample", 128),
    ]
    rows = []
    for kind, value in conditions:
        dataset = ArtBinaryDataset(
            test_frame, robustness_transform(kind, value, image_size)
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
        rows.append({"condition": kind, "value": value, **metrics})
    return pd.DataFrame(rows)
