"""Evaluate binary classifiers and probe their behavior after training.

Models in this project emit one logit: positive values support the AI-generated
class and negative values support the human class. This module converts those
scores into probabilities and metrics, retains image-level predictions for
subgroup analysis, creates diagnostic plots, and defines the shared robustness
conditions used by experiments and the report.
"""

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
    """Describe one deterministic post-training stability check.

    Attributes:
        kind: Perturbation family understood by :func:`robustness_perturbation`.
        value: Family-specific setting, such as a contrast factor, JPEG quality,
            noise standard deviation, or intermediate resampling size.
    """

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
    """Convert a robustness identifier into a readable family name.

    Args:
        condition: Internal condition identifier such as ``"jpeg"`` or ``"noise"``.

    Returns:
        The preferred report label. Unknown identifiers fall back to title case so
        exploratory conditions can still be displayed.
    """
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
    """Format a robustness value with units meaningful for its family.

    Args:
        condition: Internal perturbation-family identifier.
        value: Numeric setting associated with the condition.

    Returns:
        A compact label such as ``"0.5x"``, ``"Q20"``, or ``"64 px"``. Unknown
        conditions use the general numeric representation.
    """
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
    """Combine a perturbation family and setting for plot axes.

    Args:
        condition: Internal perturbation-family identifier.
        value: Numeric setting associated with the condition.

    Returns:
        ``"clean"`` for the unmodified baseline, otherwise a family label followed
        by its formatted value.
    """
    value_label = robustness_value_label(condition, value)
    if condition == "clean":
        return "clean"
    return f"{robustness_condition_label(condition)} {value_label}"


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Convert binary logits to fake-class probabilities without overflow.

    Args:
        logits: Scalar or array-like one-logit model outputs.

    Returns:
        A floating NumPy array of the same shape with values in ``[0, 1]``. Inputs
        are clipped to ``[-50, 50]`` before exponentiation for numerical stability.
    """
    logits = np.clip(np.asarray(logits, dtype=float), -50, 50)
    return 1 / (1 + np.exp(-logits))


def binary_metrics(
    labels: np.ndarray | list,
    logits: np.ndarray | list,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the common metrics for one-logit binary predictions.

    Args:
        labels: Ground-truth labels, where ``0`` is human and ``1`` is AI-generated.
        logits: One model logit for each label.
        threshold: Fake-class probability at or above which a row is predicted as
            AI-generated.

    Returns:
        Accuracy, positive-class precision, recall, F1, and ROC-AUC. ROC-AUC is
        ``NaN`` when the supplied labels contain only one class; precision, recall,
        and F1 use zero rather than warning when no positive predictions are present.
    """
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
    """Evaluate a model without gradients and retain image-level predictions.

    Args:
        model: Binary classifier returning one logit per image.
        loader: Non-empty loader whose batches contain ``image``, ``label``, ``path``,
            ``source_label``, and ``style_label``.
        criterion: Loss accepting flattened logits and float binary labels.
        device: Device on which inference is performed.
        threshold: Probability threshold used for the ``pred`` column and metrics.

    Returns:
        A pair containing the aggregate metric dictionary and a DataFrame with
        ``image_path``, ``label``, ``logit``, ``prob_fake``, ``source_label``,
        ``style_label``, ``pred``, and ``correct`` columns. The metric dictionary
        also contains the sample-weighted mean loss.
    """
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
    """Compute binary metrics independently within each subgroup.

    Args:
        predictions: Image-level table containing ``label`` and ``logit`` columns.
        group: Column whose distinct values define the subgroups.

    Returns:
        One row per subgroup with its name, sample count, and binary metrics. The
        standard fixed probability threshold of ``0.5`` is used.
    """
    rows = []
    for name, part in predictions.groupby(group, dropna=False):
        values = binary_metrics(part["label"], part["logit"])
        rows.append({group: name, "count": len(part), **values})
    return pd.DataFrame(rows)


def style_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize binary performance within each artistic style.

    Args:
        predictions: Image-level predictions containing ``style_label``, ``label``,
            and ``logit`` columns.

    Returns:
        Full binary metrics by style. The fixed sampling protocol places both human
        and AI-generated images in each style, making these metrics meaningful.
    """
    return metrics_by_group(predictions, "style_label")


def source_error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report the meaningful error direction for each single-class source.

    Human rows can only produce false positives, while Latent Diffusion and Stable
    Diffusion rows can only produce false negatives. Reporting those rates avoids
    misleading precision, recall, or ROC-AUC values for one-class groups.

    Args:
        predictions: Image-level table containing ``source_label``, ``label``, and
            thresholded ``pred`` columns.

    Returns:
        Source, count, accuracy, applicable error type, and error rate, sorted by
        source label.

    Raises:
        ValueError: If any source group contains more than one binary label.
    """
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
    """Save a labelled two-class confusion matrix.

    Args:
        predictions: Table containing integer ``label`` and ``pred`` columns.
        output_path: Destination image path. Its parent directory must already exist.
        title: Text shown above the matrix.

    Note:
        The Matplotlib figure is closed after writing so repeated experiment runs do
        not retain plotting state.
    """
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
    """Save a receiver operating characteristic curve and its area.

    Args:
        predictions: Table containing binary ``label`` and ``prob_fake`` columns with
            both classes represented.
        output_path: Destination image path. Its parent directory must already exist.
        title: Text shown above the curve.
    """
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
    """Scale Pillow image contrast around its mean luminance.

    Attributes:
        factor: Contrast multiplier. ``1`` preserves contrast, values below ``1``
            flatten it, and values above ``1`` strengthen it.
    """

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return ``image`` with contrast scaled by the configured factor."""
        return ImageEnhance.Contrast(image).enhance(self.factor)


class BrightnessTransform:
    """Scale all Pillow image intensities by a fixed brightness factor.

    Attributes:
        factor: Brightness multiplier, with ``1`` representing the original image.
    """

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return ``image`` with brightness scaled by the configured factor."""
        return ImageEnhance.Brightness(image).enhance(self.factor)


class SaturationTransform:
    """Scale Pillow colour saturation while retaining image geometry.

    Attributes:
        factor: Colour multiplier. ``0`` produces greyscale and ``1`` preserves the
            original saturation.
    """

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return ``image`` with colour scaled by the configured saturation."""
        return ImageEnhance.Color(image).enhance(self.factor)


class GaussianBlurTransform:
    """Suppress fine spatial detail using Gaussian blur.

    Attributes:
        radius: Pillow Gaussian-blur radius measured in input-image pixels.
    """

    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a Gaussian-blurred copy of ``image``."""
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


class DeterministicGaussianNoiseTransform:
    """Add image-content-seeded Gaussian noise before normalization.

    The seed is derived from the image bytes, so the same image receives the
    same perturbation independent of dataloader order or worker count.

    Attributes:
        sigma: Noise standard deviation on RGB intensities scaled to ``[0, 1]``.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return an RGB copy with deterministic Gaussian noise added."""
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        digest = hashlib.blake2b(rgb.tobytes(), digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=self.sigma, size=pixels.shape)
        noisy = np.clip(pixels + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255).round().astype(np.uint8), mode="RGB")


class JPEGCompressionTransform:
    """Round-trip an image through in-memory JPEG compression.

    Attributes:
        quality: Pillow JPEG quality setting. Lower values create stronger block and
            quantization artifacts.
    """

    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return the RGB image recovered from an in-memory JPEG round trip."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class CommonResampleTransform:
    """Downsample every source through the same bicubic resolution bottleneck.

    The later shared evaluation transform resizes this intermediate image to the model
    input size. Applying the bottleneck equally avoids mixing source-specific native
    resolutions with the robustness comparison.

    Attributes:
        intermediate_size: Square width and height produced by this transform.
    """

    def __init__(self, intermediate_size: int):
        self.intermediate_size = intermediate_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a bicubically downsampled square copy of ``image``."""
        return image.resize(
            (self.intermediate_size, self.intermediate_size),
            Image.Resampling.BICUBIC,
        )


def robustness_perturbation(kind: str, value: float):
    """Construct the image-space part of a robustness condition.

    Args:
        kind: One of ``clean``, ``contrast``, ``brightness``, ``saturation``,
            ``blur``, ``noise``, ``jpeg``, or ``resample``.
        value: Family-specific perturbation setting.

    Returns:
        A callable Pillow transform, or ``None`` for the clean condition. Resizing to
        model input size and ImageNet normalization are intentionally not included.

    Raises:
        ValueError: If ``kind`` is not a registered perturbation family.
    """
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
    """Apply one registered perturbation to a Pillow image.

    Args:
        image: Source image before model resizing or normalization.
        kind: Registered perturbation family.
        value: Family-specific perturbation setting.

    Returns:
        The perturbed Pillow image. The clean condition returns the original object.

    Raises:
        ValueError: If ``kind`` is not recognized.
    """
    perturbation = robustness_perturbation(kind, value)
    if perturbation is None:
        return image
    return perturbation(image)


def robustness_transform(kind: str, value: float, image_size: int):
    """Compose a perturbation with the shared model preprocessing.

    Args:
        kind: Registered perturbation family.
        value: Family-specific perturbation setting.
        image_size: Final square tensor size expected by the classifier.

    Returns:
        A torchvision composition that perturbs first, directly resizes to the model
        input, converts to a tensor, and applies ImageNet normalization.

    Raises:
        ValueError: If ``kind`` is not recognized.
    """
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
    """Evaluate a fixed checkpoint under every registered robustness condition.

    This function performs no training or model selection. Every condition uses the
    same test rows, threshold, resize, and ImageNet normalization; only the preceding
    deterministic image-space perturbation changes.

    Args:
        model: Already selected binary classifier.
        test_frame: Test metadata supplied identically to every condition.
        criterion: Binary loss used to report condition-level mean loss.
        device: Device used for inference.
        image_size: Final square model input size.
        batch_size: Images evaluated per batch.
        num_workers: Worker processes used by each condition loader.
        threshold: Fixed fake-class probability threshold.

    Returns:
        One row per entry in :data:`ROBUSTNESS_CONDITIONS`, in registry order. Columns
        include ``condition``, ``value``, loss, accuracy, precision, recall, F1, and
        ROC-AUC.
    """
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
