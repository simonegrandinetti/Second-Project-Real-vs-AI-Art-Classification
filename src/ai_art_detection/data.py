"""Dataset scanning, splitting, and DataLoader helpers.

This module is where most of the experimental protocol is enforced.  It turns
the Kaggle folder layout into tabular metadata, validates the pinned dataset
inventory, samples exact source/style quotas, and builds deterministic loaders.
"""

from __future__ import annotations

import random
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Folder names in Kaggle-style datasets are not always perfectly standardized.
# Aliases make metadata inference explicit and testable instead of relying on
# fragile substring checks.
SOURCE_ALIASES = {
    "Human": {"human", "real", "artbench", "artbench_10", "traditional"},
    "Stable_Diffusion": {
        "ai_sd",
        "stable",
        "stable_diffusion",
        "standard_diffusion",
        "stablediffusion",
        "sd",
    },
    "Latent_Diffusion": {
        "ai_ld",
        "latent",
        "latent_diffusion",
        "latentdiffusion",
        "ld",
    },
    "AI_Unknown": {"ai", "fake", "generated", "synthetic"},
}

STYLE_ALIASES = {
    "Art_Nouveau": {"art_nouveau", "nouveau"},
    "Baroque": {"baroque"},
    "Expressionism": {"expressionism", "expressionist"},
    "Impressionism": {"impressionism", "impressionist"},
    "Post_Impressionism": {
        "post_impressionism",
        "postimpressionism",
        "post_impressionist",
    },
    "Realism": {"realism", "realist"},
    "Renaissance": {"renaissance"},
    "Romanticism": {"romanticism", "romanticist"},
    "Surrealism": {"surrealism", "surrealist"},
    "Ukiyo_e": {"ukiyo_e", "ukiyoe", "ukiyo"},
}

OFFICIAL_SPLIT_ALIASES = {
    "train": {"train", "training"},
    "test": {"test", "testing"},
}

# Quotas are per style.  With ten styles, the train split becomes
# 3,200 Human + 1,600 LD + 1,600 SD = 6,400 images, and similarly for
# validation and test.
COURSEWORK_SPLIT_QUOTAS: dict[str, dict[str, int]] = {
    "train": {"Human": 320, "Latent_Diffusion": 160, "Stable_Diffusion": 160},
    "val": {"Human": 80, "Latent_Diffusion": 40, "Stable_Diffusion": 40},
    "test": {"Human": 100, "Latent_Diffusion": 50, "Stable_Diffusion": 50},
}

REPLICATION_TEST_QUOTAS: dict[str, int] = {
    "Human": 100,
    "Latent_Diffusion": 50,
    "Stable_Diffusion": 50,
}

STANDARD_EVAL_TRANSFORM_ID = "standard_resize_224_imagenet"

EXPECTED_INVENTORY = {
    ("train", "Human"): 50_000,
    ("train", "Latent_Diffusion"): 52_092,
    ("train", "Stable_Diffusion"): 52_923,
    ("test", "Human"): 10_000,
    ("test", "Latent_Diffusion"): 10_000,
    ("test", "Stable_Diffusion"): 10_000,
}


def seed_everything(seed: int = 42) -> None:
    """Seed the random-number generators used by the experiment pipeline.

    Args:
        seed: Shared seed for Python, NumPy, CPU PyTorch, and all CUDA devices.

    Note:
        cuDNN benchmarking is disabled and deterministic kernels are requested. This
        improves repeatability but can reduce training speed, and exact reproducibility
        can still depend on hardware and PyTorch versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_component(value: str) -> str:
    """Convert a path component to the canonical alias-matching form.

    Args:
        value: Directory name, filename, or other path-like component.

    Returns:
        The lowercase stem with punctuation collapsed to underscores and leading or
        trailing separators removed.
    """
    value = Path(value).stem.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _matches(component: str, alias: str) -> bool:
    """Match a complete alias without treating ``surrealism`` as ``real``."""
    return component == alias or component.startswith(f"{alias}_")


def infer_metadata(path: Path, data_root: Path) -> tuple[str, int, str]:
    """Infer an image's source, binary target, and artistic style.

    Specific generator aliases are checked before generic AI aliases. This ordering
    preserves the Latent Diffusion and Stable Diffusion subgroups used in sampling.

    Args:
        path: Image path located below ``data_root``.
        data_root: Root against which directory components are interpreted.

    Returns:
        A ``(source_label, binary_label, style_label)`` tuple. Human images receive
        binary label ``0``, known or generic AI sources receive ``1``, and an
        unrecognized source receives ``-1``. Unknown text labels are returned as
        ``"Unknown"``.

    Raises:
        ValueError: If ``path`` is not contained below ``data_root``.
    """
    relative = path.relative_to(data_root)
    components = [normalize_component(part) for part in relative.parts]

    source = "Unknown"
    # Specific generators take precedence over generic real/fake folders.
    source_order = ("Stable_Diffusion", "Latent_Diffusion", "Human", "AI_Unknown")
    for candidate in source_order:
        if any(
            _matches(component, alias)
            for component in components
            for alias in SOURCE_ALIASES[candidate]
        ):
            source = candidate
            break

    if source == "Human":
        binary_label = 0
    elif source in {"Stable_Diffusion", "Latent_Diffusion", "AI_Unknown"}:
        binary_label = 1
    else:
        binary_label = -1

    style = "Unknown"
    style_components = []
    for component in components:
        for source_prefix in ("ai_ld_", "ai_sd_"):
            if component.startswith(source_prefix):
                component = component.removeprefix(source_prefix)
                break
        style_components.append(component)
    # Longer aliases first to keep post-impressionism distinct.
    for candidate, aliases in STYLE_ALIASES.items():
        if any(
            _matches(component, alias)
            for component in style_components
            for alias in sorted(aliases, key=len, reverse=True)
        ):
            style = candidate
            break

    return source, binary_label, style


def infer_official_split(path: Path, data_root: Path) -> str:
    """Infer the official AI-ArtBench partition encoded in a path.

    Args:
        path: Image path located below ``data_root``.
        data_root: Root against which directory components are interpreted.

    Returns:
        ``"train"`` or ``"test"`` when exactly one split alias is found; otherwise
        ``"Unknown"`` so ambiguous layouts fail later validation.

    Raises:
        ValueError: If ``path`` is not contained below ``data_root``.
    """
    relative = path.relative_to(data_root)
    components = [normalize_component(part) for part in relative.parts[:-1]]
    matches = {
        split
        for split, aliases in OFFICIAL_SPLIT_ALIASES.items()
        if any(
            _matches(component, alias)
            for component in components
            for alias in aliases
        )
    }
    if len(matches) == 1:
        return matches.pop()
    return "Unknown"


def scan_dataset(data_root: Path | str) -> pd.DataFrame:
    """Scan an extracted dataset and build the canonical metadata table.

    Args:
        data_root: Directory containing the extracted AI-ArtBench hierarchy.

    Returns:
        One row per supported image, sorted by path. Columns are ``image_path``,
        ``source_label``, ``binary_label``, ``binary_name``, ``style_label``,
        ``official_split``, and ``extension``. The scanner records unknown metadata;
        callers choose the appropriate strictness through the validation helpers.

    Raises:
        FileNotFoundError: If the dataset root does not exist.
    """
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_root}. "
            "See README.md for the expected layout."
        )

    rows = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        source, binary_label, style = infer_metadata(path, data_root)
        official_split = infer_official_split(path, data_root)
        rows.append(
            {
                "image_path": str(path),
                "source_label": source,
                "binary_label": binary_label,
                "binary_name": {0: "Real/Human", 1: "Fake/AI"}.get(
                    binary_label, "Unknown"
                ),
                "style_label": style,
                "official_split": official_split,
                "extension": path.suffix.lower(),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "source_label",
            "binary_label",
            "binary_name",
            "style_label",
            "official_split",
            "extension",
        ],
    )


def validate_labels(frame: pd.DataFrame, max_unknown_fraction: float = 0.01) -> None:
    """Check that a scanned table contains enough recognized binary labels.

    Args:
        frame: Metadata table produced by :func:`scan_dataset`.
        max_unknown_fraction: Largest permitted fraction of rows with label ``-1``.

    Raises:
        ValueError: If the table is empty or its unknown-label fraction exceeds the
            requested limit.
    """
    if frame.empty:
        raise ValueError("No supported images were found under the dataset root.")
    unknown = (frame["binary_label"] == -1).mean()
    if unknown > max_unknown_fraction:
        examples = frame.loc[frame["binary_label"] == -1, "image_path"].head(5).tolist()
        raise ValueError(
            f"{unknown:.1%} of images have unknown labels. Inspect folder names or "
            f"extend SOURCE_ALIASES. Examples: {examples}"
        )


def validate_dataset_inventory(
    frame: pd.DataFrame,
    expected: Mapping[tuple[str, str], int] = EXPECTED_INVENTORY,
) -> None:
    """Verify that metadata matches an expected official-split inventory.

    This is the strict gate used before final experiments. In addition to exact source
    counts, every row must have a recognized style, source, and official partition.

    Args:
        frame: Scanned metadata to validate.
        expected: Exact image count for each ``(official_split, source_label)`` pair.

    Raises:
        ValueError: If labels or path-derived metadata are unknown, or if grouped
            counts differ from ``expected``.
    """
    validate_labels(frame, max_unknown_fraction=0.0)
    unknown_styles = frame["style_label"].eq("Unknown")
    unknown_splits = frame["official_split"].eq("Unknown")
    if unknown_styles.any() or unknown_splits.any():
        raise ValueError(
            "Dataset metadata inference failed: "
            f"{int(unknown_styles.sum())} unknown styles and "
            f"{int(unknown_splits.sum())} unknown official splits."
        )
    actual = frame.groupby(["official_split", "source_label"]).size().to_dict()
    if actual != dict(expected):
        raise ValueError(
            "AI-ArtBench inventory does not match pinned version 5. "
            f"Expected {dict(expected)}, found {actual}."
        )


def validate_image_readability(
    frame: pd.DataFrame,
    *,
    progress_every: int | None = 25_000,
) -> None:
    """Verify that every image path can be decoded by Pillow.

    Args:
        frame: Table containing an ``image_path`` column.
        progress_every: Print progress after this many images. Use ``None`` or ``0``
            to keep validation quiet.

    Raises:
        ValueError: If one or more images cannot be opened and verified. At most ten
            failures are collected before validation stops.
    """
    failures = []
    for index, path in enumerate(frame["image_path"], start=1):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            failures.append(f"{path}: {error}")
            if len(failures) >= 10:
                break
        if progress_every and index % progress_every == 0:
            print(f"Verified {index:,}/{len(frame):,} image files")
    if failures:
        raise ValueError(
            "Unreadable images were found (showing at most 10): "
            + "; ".join(failures)
        )


def sample_source_style_quotas(
    frame: pd.DataFrame,
    official_split: str,
    per_source_style: Mapping[str, int],
    *,
    seed: int = 42,
    exclude_paths: Iterable[str] = (),
) -> pd.DataFrame:
    """Sample exact, deterministic quotas from one official partition.

    Each value in ``per_source_style`` is applied independently to all ten artistic
    styles. For example, ``{"Human": 100}`` selects 100 human images per style,
    not 100 images in total. Excluded paths are removed before availability checks.

    Args:
        frame: Canonical metadata containing source, style, split, and path columns.
        official_split: Published source pool to sample: ``"train"`` or ``"test"``.
        per_source_style: Number of images required for every requested source/style
            pair.
        seed: Base seed used for group sampling and final row shuffling.
        exclude_paths: Image paths that must not appear in the result.

    Returns:
        A shuffled DataFrame containing every requested quota exactly once.

    Raises:
        ValueError: If the split is invalid, a quota is non-positive, no quota is
            supplied, or a source/style group is undersized after exclusions.
    """
    if official_split not in {"train", "test"}:
        raise ValueError("official_split must be 'train' or 'test'.")

    styles = tuple(STYLE_ALIASES)
    excluded = set(exclude_paths)
    candidates = frame[
        (frame["official_split"] == official_split)
        & ~frame["image_path"].isin(excluded)
    ]
    selected = []
    for group_index, (source, count) in enumerate(sorted(per_source_style.items())):
        if count <= 0:
            raise ValueError(f"Quota for {source} must be positive.")

        # Each source quota is applied to every artistic style.  For example,
        # a Human test quota of 100 means 100 images for each of the ten styles.
        for style_index, style in enumerate(styles):
            group = candidates[
                (candidates["source_label"] == source)
                & (candidates["style_label"] == style)
            ]
            if len(group) < count:
                raise ValueError(
                    f"Need {count} {official_split}/{source}/{style} images, "
                    f"but only {len(group)} are available."
                )

            random_state = seed + group_index * len(styles) + style_index
            selected.append(group.sample(count, random_state=random_state))

    if not selected:
        raise ValueError("At least one source quota is required.")

    return (
        pd.concat(selected, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def coursework_split(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    quotas: Mapping[str, Mapping[str, int]] = COURSEWORK_SPLIT_QUOTAS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic train, validation, and test coursework splits.

    Training and validation are sampled without overlap from the official training
    partition. Test data comes only from the official test partition. With the default
    per-style quotas, the returned sizes are 6,400, 1,600, and 2,000 images.

    Args:
        frame: Validated canonical metadata for the complete dataset.
        seed: Base sampling seed. Fixed offsets derive validation and test seeds.
        quotas: Per-style source quotas keyed by ``train``, ``val``, and ``test``.

    Returns:
        ``(train, validation, test)`` DataFrames with reset indices and no shared
        image paths.

    Raises:
        ValueError: If a split quota is missing or a requested group is undersized.
        RuntimeError: If the constructed splits unexpectedly overlap.
    """
    missing = {"train", "val", "test"} - set(quotas)
    if missing:
        raise ValueError(f"Missing split quotas: {sorted(missing)}")

    # Training and validation both come from the official training partition,
    # but validation explicitly excludes every selected training path.
    train = sample_source_style_quotas(
        frame, "train", quotas["train"], seed=seed
    )
    val = sample_source_style_quotas(
        frame,
        "train",
        quotas["val"],
        seed=seed + 10_000,
        exclude_paths=train["image_path"],
    )
    test = sample_source_style_quotas(
        frame, "test", quotas["test"], seed=seed + 20_000
    )

    split_path_sets = {
        "train": set(train["image_path"]),
        "val": set(val["image_path"]),
        "test": set(test["image_path"]),
    }
    split_path_values = list(split_path_sets.values())
    overlaps = [
        first & second
        for index, first in enumerate(split_path_values)
        for second in split_path_values[:index]
    ]
    if any(overlaps):
        raise RuntimeError("Coursework split construction produced overlapping paths.")

    return train, val, test


def replication_test_split(
    frame: pd.DataFrame,
    *,
    exclude_paths: Iterable[str],
    seed: int = 4242,
    per_source_style: Mapping[str, int] = REPLICATION_TEST_QUOTAS,
) -> pd.DataFrame:
    """Build the independent official-test replication holdout.

    Args:
        frame: Canonical metadata for the complete dataset.
        exclude_paths: Paths from all original train, validation, and test splits.
        seed: Sampling seed reserved for the post-training audit.
        per_source_style: Exact per-style quotas for each requested source.

    Returns:
        A deterministic replication DataFrame with no excluded image paths.

    Raises:
        ValueError: If exclusions are empty or a requested quota cannot be filled.
        RuntimeError: If an excluded path appears despite the sampling guard.
    """
    excluded = set(exclude_paths)
    if not excluded:
        raise ValueError("Replication sampling requires non-empty exclusions.")
    replication = sample_source_style_quotas(
        frame,
        "test",
        per_source_style,
        seed=seed,
        exclude_paths=excluded,
    )
    overlap = set(replication["image_path"]) & excluded
    if overlap:
        raise RuntimeError(
            f"Replication split overlaps excluded paths: {sorted(overlap)[:5]}"
        )
    return replication


def balanced_sample(
    frame: pd.DataFrame,
    n_real: int,
    n_fake: int,
    seed: int = 42,
    balance_fake_sources: bool = True,
) -> pd.DataFrame:
    """Legacy helper for smaller binary debugging subsets.

    The final coursework protocol uses `coursework_split`; this function stays
    available for quick experiments and older tests.

    Args:
        frame: Metadata containing binary and source labels.
        n_real: Requested number of human images.
        n_fake: Requested number of AI-generated images.
        seed: Sampling and final-shuffle seed.
        balance_fake_sources: If true, draw fake rows round-robin across generators.

    Returns:
        A shuffled binary sample. If a class is undersized, all available rows from
        that class are returned and a warning is emitted.

    Raises:
        ValueError: If the input does not contain both binary classes.
    """
    valid = frame[frame["binary_label"].isin([0, 1])].copy()
    real = valid[valid["binary_label"] == 0]
    fake = valid[valid["binary_label"] == 1]
    if real.empty or fake.empty:
        raise ValueError("Both real (0) and fake (1) images are required.")

    real_n = min(n_real, len(real))
    fake_n = min(n_fake, len(fake))
    if real_n < n_real or fake_n < n_fake:
        warnings.warn(
            f"Requested {n_real} real/{n_fake} fake; using {real_n}/{fake_n}.",
            stacklevel=2,
        )
    real_sample = real.sample(real_n, random_state=seed)

    if balance_fake_sources and fake["source_label"].nunique() > 1:
        # Round-robin sampling avoids silently overrepresenting one generator.
        shuffled = {
            source: part.sample(frac=1, random_state=seed + index)
            for index, (source, part) in enumerate(fake.groupby("source_label"))
        }
        selected: list[pd.Series] = []
        sources = sorted(shuffled)
        positions = {source: 0 for source in sources}
        while len(selected) < fake_n:
            progressed = False
            for source in sources:
                position = positions[source]
                if position < len(shuffled[source]) and len(selected) < fake_n:
                    selected.append(shuffled[source].iloc[position])
                    positions[source] += 1
                    progressed = True
            if not progressed:
                break
        fake_sample = pd.DataFrame(selected)
    else:
        fake_sample = fake.sample(fake_n, random_state=seed)

    return (
        pd.concat([real_sample, fake_sample], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def _candidate_strata(frame: pd.DataFrame) -> Iterable[pd.Series]:
    """Yield stratification labels from most detailed to most general."""
    yield (
        frame["binary_label"].astype(str)
        + "|"
        + frame["source_label"].astype(str)
        + "|"
        + frame["style_label"].astype(str)
    )
    yield frame["binary_label"].astype(str) + "|" + frame["source_label"].astype(str)
    yield frame["binary_label"].astype(str)


def _choose_strata(frame: pd.DataFrame, held_out_fraction: float) -> pd.Series:
    """Choose the most detailed stratification that supports a valid split."""
    held_out_count = int(np.ceil(len(frame) * held_out_fraction))
    for candidate in _candidate_strata(frame):
        counts = candidate.value_counts()
        if counts.min() >= 2 and held_out_count >= len(counts):
            return candidate
    raise ValueError("Dataset is too small for a stratified binary split.")


def stratified_split(
    frame: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an arbitrary sample while preserving the richest viable strata.

    This legacy helper first tries label/source/style strata, then progressively falls
    back to label/source and label-only strata when groups are too small.

    Args:
        frame: Sample containing binary, source, and style labels.
        val_size: Fraction of all rows assigned to validation.
        test_size: Fraction of all rows assigned to testing.
        seed: Random seed passed to both scikit-learn splits.

    Returns:
        ``(train, validation, test)`` DataFrames with reset indices.

    Raises:
        ValueError: If split fractions are invalid or the sample is too small for a
            stratified binary split.
    """
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    first_strata = _choose_strata(frame, test_size)
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=first_strata,
    )
    relative_val_size = val_size / (1 - test_size)
    second_strata = _choose_strata(train_val, relative_val_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=second_strata,
    )
    return tuple(
        part.reset_index(drop=True) for part in (train, val, test)
    )  # type: ignore[return-value]


def get_transforms(image_size: int = 224, augment: bool = False):
    """Return training and evaluation transforms.

    Evaluation always uses direct resize + ImageNet normalization.  Training can
    optionally add mild augmentation for the E1--E4 experiments.

    Args:
        image_size: Height and width of the returned image tensors.
        augment: Add random crop, flip, mild colour jitter, and autocontrast to the
            training pipeline when true.

    Returns:
        ``(training_transform, evaluation_transform)``. Both produce normalized
        ``float32`` tensors with shape ``(3, image_size, image_size)`` using the
        ImageNet channel mean and standard deviation expected by pretrained models.
    """
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.10,
                            # Hue jitter in torchvision 0.18 overflows with NumPy 2.
                            hue=0.0,
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomAutocontrast(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


class ArtBinaryDataset(Dataset):
    """Load RGB artwork and preserve the metadata needed during evaluation.

    The input frame must contain ``image_path``, ``binary_label``, ``source_label``,
    and ``style_label``. Each item is a dictionary with ``image``, scalar float
    ``label``, ``path``, ``source_label``, and ``style_label`` entries.

    Args:
        frame: Image-level metadata. Its index is reset and its rows are otherwise
            preserved.
        transform: Optional callable applied to the decoded Pillow RGB image.
    """

    def __init__(self, frame: pd.DataFrame, transform=None):
        """Store a reset-index copy of the metadata and its image transform."""
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        """Load one image and return its tensor-ready sample dictionary."""
        row = self.frame.iloc[index]
        path = row["image_path"]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not read image: {path}") from error
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(float(row["binary_label"]), dtype=torch.float32),
            "path": path,
            "source_label": row["source_label"],
            "style_label": row["style_label"],
        }


def build_loaders(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three DataLoaders used by one experiment.

    Args:
        train: Training metadata.
        val: Validation metadata.
        test: Test metadata.
        image_size: Square model input size.
        batch_size: Images per batch in every loader.
        num_workers: Worker processes per loader.
        augment: Enable mild random augmentation only for training images.

    Returns:
        ``(train_loader, validation_loader, test_loader)``. Training rows are
        shuffled; validation and test rows retain their DataFrame order. CUDA hosts
        use pinned memory, and worker processes remain persistent when requested.
    """
    train_transform, eval_transform = get_transforms(image_size, augment)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        ArtBinaryDataset(train, train_transform), shuffle=True, **common
    )
    val_loader = DataLoader(
        ArtBinaryDataset(val, eval_transform), shuffle=False, **common
    )
    test_loader = DataLoader(
        ArtBinaryDataset(test, eval_transform), shuffle=False, **common
    )
    return train_loader, val_loader, test_loader


def build_standard_eval_loader(
    frame: pd.DataFrame,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """Build the clean loader used by the independent replication audit.

    Args:
        frame: Metadata to evaluate in its existing row order.
        image_size: Direct-resize height and width.
        batch_size: Images per evaluation batch.
        num_workers: Worker processes used to decode images.

    Returns:
        A non-shuffled DataLoader applying only direct resize, tensor conversion, and
        ImageNet normalization. No augmentation or common-resampling transform enters
        this pipeline.
    """
    _, eval_transform = get_transforms(image_size, augment=False)
    return DataLoader(
        ArtBinaryDataset(frame, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
