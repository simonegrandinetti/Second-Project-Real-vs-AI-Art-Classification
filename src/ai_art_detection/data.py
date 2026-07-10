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
    """Seed Python, NumPy, PyTorch, and cuDNN for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_component(value: str) -> str:
    """Normalize a path component before comparing it with aliases."""
    value = Path(value).stem.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _matches(component: str, alias: str) -> bool:
    """Match directory-like labels without treating 'surrealism' as 'real'."""
    return component == alias or component.startswith(f"{alias}_")


def infer_metadata(path: Path, data_root: Path) -> tuple[str, int, str]:
    """Infer source, binary target, and style from the relative path."""
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
    """Infer the published AI-ArtBench train/test partition from directory names."""
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
    """Scan all supported image files and return one metadata row per image."""
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
    """Reject scans with too many unknown binary labels."""
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
    """Fail fast if the pinned dataset is incomplete or has an unexpected layout."""
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
    """Verify that every selected image can be decoded by Pillow."""
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
    """Sample an exact quota for every source/style pair in one official split."""
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
    """Build the fixed 6,400/1,600/2,000 official-split coursework protocol."""
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
    """Build a second official-test holdout disjoint from all prior splits."""
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
    """Legacy stratified train/validation/test split for arbitrary samples."""
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
    """PyTorch dataset returning images plus labels and subgroup metadata."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
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
    """Build train/validation/test DataLoaders for one experiment."""
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
    """Build a deterministic clean loader with no augmentation or resampling."""
    _, eval_transform = get_transforms(image_size, augment=False)
    return DataLoader(
        ArtBinaryDataset(frame, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
