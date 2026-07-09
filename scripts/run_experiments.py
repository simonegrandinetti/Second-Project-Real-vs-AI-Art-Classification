#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from torch import nn

from ai_art_detection.config import ProjectConfig
from ai_art_detection.data import (
    coursework_split,
    scan_dataset,
    seed_everything,
    validate_dataset_inventory,
)
from ai_art_detection.evaluation import (
    evaluate_robustness,
    plot_confusion,
    plot_roc,
    source_error_summary,
    style_metrics,
)
from ai_art_detection.experiments import (
    DEFAULT_EXPERIMENTS,
    load_experiment_checkpoint,
    run_experiment,
)
from ai_art_detection.gradcam import save_gradcam_panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiments",
        nargs="*",
        choices=[experiment.name for experiment in DEFAULT_EXPERIMENTS],
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download/use ImageNet weights (only useful for smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed experiment JSON, predictions, and checkpoints.",
    )
    return parser.parse_args()


def save_manifest(
    output_path: Path,
    config: ProjectConfig,
    device: torch.device,
    scanned: pd.DataFrame,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    split_counts = {}
    for name, frame in zip(("train", "val", "test"), splits):
        split_counts[name] = {
            source: int(count)
            for source, count in frame["source_label"].value_counts().items()
        }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "config": config.as_serializable_dict(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "dataset": {
            "images": len(scanned),
            "inventory": [
                {
                    "official_split": split,
                    "source_label": source,
                    "count": int(count),
                }
                for (split, source), count in scanned.groupby(
                    ["official_split", "source_label"]
                ).size().items()
            ],
        },
        "selected_split_counts": split_counts,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = ProjectConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    config.make_output_dirs()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scanned = scan_dataset(config.data_root)
    validate_dataset_inventory(scanned)
    print("\nScanned binary labels:")
    print(scanned["binary_name"].value_counts(dropna=False))
    print("\nScanned sources:")
    print(scanned["source_label"].value_counts(dropna=False))
    inventory = (
        scanned.groupby(["official_split", "source_label", "style_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    inventory.to_csv(
        config.output_dir / "tables" / "dataset_inventory.csv", index=False
    )
    train, val, test = coursework_split(scanned, seed=config.seed)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_csv(config.output_dir / "tables" / f"{name}_split.csv", index=False)
        print(f"{name}: {len(frame)} images")
    save_manifest(
        config.output_dir / "metrics" / "run_manifest.json",
        config,
        device,
        scanned,
        (train, val, test),
    )

    requested = set(args.experiments or [])
    experiments = [
        experiment
        for experiment in DEFAULT_EXPERIMENTS
        if not requested or experiment.name in requested
    ]
    rows = []
    best_model = None
    best_experiment = None
    best_validation_f1 = -1.0
    for experiment in experiments:
        print(f"\nRunning {experiment.name}")
        result_path = (
            config.output_dir / "metrics" / f"{experiment.name}_result.json"
        )
        prediction_path = (
            config.output_dir
            / "metrics"
            / f"{experiment.name}_test_predictions.csv"
        )
        checkpoint_path = (
            config.output_dir / "models" / f"{experiment.name}_best.pt"
        )
        if (
            args.resume
            and result_path.exists()
            and prediction_path.exists()
            and checkpoint_path.exists()
        ):
            print("Reusing completed artifacts.")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            model = load_experiment_checkpoint(experiment, config, device)
        else:
            model, result = run_experiment(
                experiment,
                train,
                val,
                test,
                config,
                device,
                pretrained=not args.no_pretrained,
            )
        rows.append(result)
        predictions = pd.read_csv(prediction_path)
        style_metrics(predictions).to_csv(
            config.output_dir / "tables" / f"{experiment.name}_style_metrics.csv",
            index=False,
        )
        source_error_summary(predictions).to_csv(
            config.output_dir / "tables" / f"{experiment.name}_source_errors.csv",
            index=False,
        )
        plot_confusion(
            predictions,
            config.output_dir / "figures" / f"{experiment.name}_confusion.png",
        )
        plot_roc(
            predictions,
            config.output_dir / "figures" / f"{experiment.name}_roc.png",
        )
        if result["val_f1"] > best_validation_f1:
            if best_model is not None and device.type == "cuda":
                del best_model
                torch.cuda.empty_cache()
            best_model = model
            best_experiment = experiment
            best_validation_f1 = result["val_f1"]
        elif device.type == "cuda":
            del model
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    results.to_csv(
        config.output_dir / "tables" / "experiment_results.csv", index=False
    )
    print("\nResults (ranked by validation F1):")
    print(results.to_string(index=False))

    assert best_model is not None and best_experiment is not None
    robust = evaluate_robustness(
        best_model,
        test,
        nn.BCEWithLogitsLoss(),
        device,
        config.image_size,
        config.batch_size,
        config.num_workers,
        config.threshold,
    )
    robust.to_csv(
        config.output_dir
        / "tables"
        / f"{best_experiment.name}_robustness.csv",
        index=False,
    )
    labels = [
        (
            "clean"
            if condition == "clean"
            else (
                f"contrast {value:g}x"
                if condition == "contrast"
                else (
                    f"JPEG Q{int(value)}"
                    if condition == "jpeg"
                    else f"resample {int(value)} px"
                )
            )
        )
        for condition, value in zip(robust["condition"], robust["value"])
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(labels, robust["f1"], marker="o")
    axis.set(ylabel="F1", title="Robustness of validation-selected model")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(
        config.output_dir
        / "figures"
        / f"{best_experiment.name}_robustness_f1.png",
        dpi=200,
    )
    plt.close(figure)
    best_predictions = pd.read_csv(
        config.output_dir
        / "metrics"
        / f"{best_experiment.name}_test_predictions.csv"
    )
    save_gradcam_panels(
        best_model,
        best_experiment.model_name,
        best_predictions,
        config.output_dir / "figures",
        device,
        image_size=config.image_size,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
