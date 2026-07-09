from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torchvision import transforms

from ai_art_detection.data import (
    ArtBinaryDataset,
    balanced_sample,
    build_standard_eval_loader,
    coursework_split,
    get_transforms,
    infer_metadata,
    infer_official_split,
    replication_test_split,
    scan_dataset,
    stratified_split,
    validate_dataset_inventory,
    validate_image_readability,
)


def test_inference_does_not_confuse_surrealism_with_real(tmp_path: Path):
    root = tmp_path / "real-ai-art"
    path = root / "train" / "stable_diffusion" / "surrealism" / "one.jpg"
    path.parent.mkdir(parents=True)
    path.touch()
    assert infer_metadata(path, root) == ("Stable_Diffusion", 1, "Surrealism")


def test_official_split_and_compact_source_aliases(tmp_path: Path):
    root = tmp_path / "real-ai-art"
    train_path = root / "training_data" / "AI_SD_baroque" / "one.jpg"
    test_path = root / "testing" / "AI_LD_ukiyo_e" / "two.jpg"
    assert infer_official_split(train_path, root) == "train"
    assert infer_official_split(test_path, root) == "test"
    assert infer_metadata(train_path, root) == ("Stable_Diffusion", 1, "Baroque")
    assert infer_metadata(test_path, root) == ("Latent_Diffusion", 1, "Ukiyo_e")


def test_scan_and_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    paths = [
        root / "human" / "realism" / "human.jpg",
        root / "latent_diffusion" / "baroque" / "fake.jpg",
    ]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((20, 30, 3), index * 100, dtype=np.uint8)).save(path)
    frame = scan_dataset(root)
    assert frame["binary_label"].tolist() == [0, 1]
    _, transform = get_transforms(32)
    item = ArtBinaryDataset(frame, transform)[0]
    assert item["image"].shape == (3, 32, 32)
    validate_image_readability(frame, progress_every=None)


def test_readability_validation_rejects_corrupt_images(tmp_path: Path):
    path = tmp_path / "broken.jpg"
    path.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Unreadable images"):
        validate_image_readability(
            pd.DataFrame({"image_path": [str(path)]}), progress_every=None
        )


def test_augmented_transform_is_numpy_two_compatible():
    transform, _ = get_transforms(32, augment=True)
    image = Image.fromarray(np.full((48, 64, 3), 127, dtype=np.uint8))
    for _ in range(20):
        assert transform(image).shape == (3, 32, 32)


def test_balanced_stratified_split():
    rows = []
    for label, source in ((0, "Human"), (1, "Stable_Diffusion")):
        for index in range(40):
            rows.append(
                {
                    "image_path": f"{source}/{index}.jpg",
                    "source_label": source,
                    "binary_label": label,
                    "binary_name": "Real/Human" if label == 0 else "Fake/AI",
                    "style_label": "Baroque" if index % 2 else "Realism",
                    "extension": ".jpg",
                }
            )
    frame = pd.DataFrame(rows)
    selected = balanced_sample(frame, 30, 30)
    train, val, test = stratified_split(selected, 0.15, 0.15)
    assert len(train) + len(val) + len(test) == 60
    for part in (train, val, test):
        assert set(part["binary_label"]) == {0, 1}


def test_coursework_split_is_exact_deterministic_and_disjoint():
    styles = (
        "Art_Nouveau",
        "Baroque",
        "Expressionism",
        "Impressionism",
        "Post_Impressionism",
        "Realism",
        "Renaissance",
        "Romanticism",
        "Surrealism",
        "Ukiyo_e",
    )
    sources = {
        "Human": 0,
        "Latent_Diffusion": 1,
        "Stable_Diffusion": 1,
    }
    rows = []
    for official_split in ("train", "test"):
        for source, label in sources.items():
            for style in styles:
                for index in range(5):
                    rows.append(
                        {
                            "image_path": (
                                f"/{official_split}/{source}/{style}/{index}.jpg"
                            ),
                            "source_label": source,
                            "binary_label": label,
                            "binary_name": "Real/Human" if label == 0 else "Fake/AI",
                            "style_label": style,
                            "official_split": official_split,
                            "extension": ".jpg",
                        }
                    )
    frame = pd.DataFrame(rows)
    quotas = {
        "train": {"Human": 2, "Latent_Diffusion": 1, "Stable_Diffusion": 1},
        "val": {"Human": 2, "Latent_Diffusion": 1, "Stable_Diffusion": 1},
        "test": {"Human": 2, "Latent_Diffusion": 1, "Stable_Diffusion": 1},
    }
    first = coursework_split(frame, seed=42, quotas=quotas)
    second = coursework_split(frame, seed=42, quotas=quotas)
    assert [part["image_path"].tolist() for part in first] == [
        part["image_path"].tolist() for part in second
    ]
    assert [len(part) for part in first] == [40, 40, 40]
    path_sets = [set(part["image_path"]) for part in first]
    assert not path_sets[0] & path_sets[1]
    assert not path_sets[0] & path_sets[2]
    assert not path_sets[1] & path_sets[2]
    for part, split_name in zip(first, ("train", "val", "test")):
        expected_pool = "test" if split_name == "test" else "train"
        assert set(part["official_split"]) == {expected_pool}
        counts = part.groupby(["source_label", "style_label"]).size()
        for source, count in quotas[split_name].items():
            assert set(counts[source]) == {count}


def test_replication_split_is_exact_deterministic_and_excluded():
    styles = (
        "Art_Nouveau",
        "Baroque",
        "Expressionism",
        "Impressionism",
        "Post_Impressionism",
        "Realism",
        "Renaissance",
        "Romanticism",
        "Surrealism",
        "Ukiyo_e",
    )
    sources = {"Human": 0, "Latent_Diffusion": 1, "Stable_Diffusion": 1}
    rows = []
    excluded = set()
    for source, label in sources.items():
        for style in styles:
            for index in range(4):
                path = f"/test/{source}/{style}/{index}.jpg"
                rows.append(
                    {
                        "image_path": path,
                        "source_label": source,
                        "binary_label": label,
                        "style_label": style,
                        "official_split": "test",
                    }
                )
                if index == 0:
                    excluded.add(path)
    frame = pd.DataFrame(rows)
    quotas = {"Human": 2, "Latent_Diffusion": 1, "Stable_Diffusion": 1}
    first = replication_test_split(
        frame, exclude_paths=excluded, seed=4242, per_source_style=quotas
    )
    second = replication_test_split(
        frame, exclude_paths=excluded, seed=4242, per_source_style=quotas
    )
    assert first["image_path"].tolist() == second["image_path"].tolist()
    assert len(first) == 40
    assert not set(first["image_path"]) & excluded
    counts = first.groupby(["source_label", "style_label"]).size()
    for source, count in quotas.items():
        assert set(counts[source]) == {count}


def test_standard_eval_loader_has_no_augmentation_or_resampling():
    frame = pd.DataFrame(
        [
            {
                "image_path": "/unused/example.jpg",
                "source_label": "Human",
                "binary_label": 0,
                "style_label": "Baroque",
            }
        ]
    )
    loader = build_standard_eval_loader(
        frame, image_size=32, batch_size=1, num_workers=0
    )
    pipeline = loader.dataset.transform.transforms
    assert [type(step) for step in pipeline] == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert all(type(step).__name__ != "CommonResampleTransform" for step in pipeline)


def test_inventory_validation_accepts_explicit_expected_counts():
    frame = pd.DataFrame(
        [
            {
                "image_path": "/train/human/baroque/one.jpg",
                "source_label": "Human",
                "binary_label": 0,
                "style_label": "Baroque",
                "official_split": "train",
            },
            {
                "image_path": "/test/ai_ld/baroque/two.jpg",
                "source_label": "Latent_Diffusion",
                "binary_label": 1,
                "style_label": "Baroque",
                "official_split": "test",
            },
        ]
    )
    validate_dataset_inventory(
        frame,
        expected={("train", "Human"): 1, ("test", "Latent_Diffusion"): 1},
    )
