"""Run a one-epoch synthetic check across training, persistence, and reloading."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ai_art_detection.config import ProjectConfig
from ai_art_detection.experiments import (
    Experiment,
    load_experiment_checkpoint,
    run_experiment,
)


def make_frame(root: Path, split: str, count: int = 4) -> pd.DataFrame:
    """Create a tiny balanced image split for the end-to-end smoke test.

    Args:
        root: Temporary directory below which images are written.
        split: Split folder name and source of the official-split field.
        count: Number of alternating human and generated examples.

    Returns:
        Canonical image metadata accepted by the experiment runner.
    """
    rows = []
    for index in range(count):
        label = index % 2
        source = "Human" if label == 0 else "Latent_Diffusion"
        path = root / split / source / f"{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.full((64, 64, 3), 40 + index * 40, dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        rows.append(
            {
                "image_path": str(path),
                "source_label": source,
                "binary_label": label,
                "binary_name": "Real/Human" if label == 0 else "Fake/AI",
                "style_label": "Baroque",
                "official_split": "train" if split != "test" else "test",
                "extension": ".png",
            }
        )
    return pd.DataFrame(rows)


def test_one_epoch_end_to_end_smoke(tmp_path: Path):
    config = ProjectConfig(
        data_root=tmp_path / "images",
        output_dir=tmp_path / "outputs",
        image_size=64,
        batch_size=2,
        num_workers=0,
        epochs=1,
        patience=1,
    )
    config.make_output_dirs()
    train = make_frame(config.data_root, "train")
    val = make_frame(config.data_root, "val")
    test = make_frame(config.data_root, "test")
    experiment = Experiment("smoke", "mobilenet_v2", "frozen", False)
    _, result = run_experiment(
        experiment,
        train,
        val,
        test,
        config,
        torch.device("cpu"),
        pretrained=False,
    )
    assert result["best_epoch"] == 1
    assert (config.output_dir / "models/smoke_best.pt").exists()
    assert (config.output_dir / "metrics/smoke_test_predictions.csv").exists()
    assert (config.output_dir / "tables/smoke_source_errors.csv").exists()
    loaded = load_experiment_checkpoint(experiment, config, torch.device("cpu"))
    with torch.inference_mode():
        assert loaded(torch.randn(1, 3, 64, 64)).shape == (1, 1)
