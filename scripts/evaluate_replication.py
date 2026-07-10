#!/usr/bin/env python3
"""Run the independent holdout overfitting audit.

This command does not train or select models. It reloads the saved checkpoints,
evaluates them on the clean training split and on a second official-test
holdout, then writes an isolated replication result folder.

The preprocessing is deliberately the standard 224x224 evaluation transform;
the robustness resampling transform is not used here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torchvision
from torch import nn

from ai_art_detection.config import ProjectConfig
from ai_art_detection.data import (
    REPLICATION_TEST_QUOTAS,
    STANDARD_EVAL_TRANSFORM_ID,
    build_standard_eval_loader,
    replication_test_split,
    scan_dataset,
    seed_everything,
    validate_dataset_inventory,
)
from ai_art_detection.evaluation import (
    binary_metrics,
    evaluate,
    source_error_summary,
    style_metrics,
)
from ai_art_detection.experiments import (
    DEFAULT_EXPERIMENTS,
    load_experiment_checkpoint,
)
from ai_art_detection.replication import f1_audit_intervals, split_path_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints on a disjoint clean replication holdout."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw/real-ai-art")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--experiments",
        nargs="*",
        choices=[experiment.name for experiment in DEFAULT_EXPERIMENTS],
    )
    return parser.parse_args()


def read_original_splits(
    tables_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load the original split CSVs and reject any path overlap."""
    splits = {
        name: pd.read_csv(tables_dir / f"{name}_split.csv")
        for name in ("train", "val", "test")
    }
    path_sets = {
        name: set(frame["image_path"].astype(str))
        for name, frame in splits.items()
    }
    if (
        path_sets["train"] & path_sets["val"]
        or path_sets["train"] & path_sets["test"]
        or path_sets["val"] & path_sets["test"]
    ):
        raise ValueError("Original split files overlap; replication audit aborted.")
    return splits


def main() -> None:
    # 1) Read audit settings. Defaults match the report protocol.
    args = parse_args()
    config = ProjectConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Preprocessing: {STANDARD_EVAL_TRANSFORM_ID}")

    # 2) Keep every replication artifact under outputs/replication so the
    # original experiment results remain untouched.
    original_tables = config.output_dir / "tables"
    original_metrics = config.output_dir / "metrics"
    replication_root = config.output_dir / "replication"
    replication_tables = replication_root / "tables"
    replication_metrics = replication_root / "metrics"
    replication_tables.mkdir(parents=True, exist_ok=True)
    replication_metrics.mkdir(parents=True, exist_ok=True)

    # 3) Reconstruct the candidate pool and exclude every original train,
    # validation, and test path before sampling the second holdout.
    scanned = scan_dataset(config.data_root)
    validate_dataset_inventory(scanned)
    splits = read_original_splits(original_tables)
    excluded_paths = set().union(
        *(set(frame["image_path"].astype(str)) for frame in splits.values())
    )
    replication = replication_test_split(
        scanned,
        exclude_paths=excluded_paths,
        seed=args.seed,
    )
    replication_path = replication_tables / "replication_test_split.csv"
    replication.to_csv(replication_path, index=False)
    print(f"Replication holdout: {len(replication)} images")

    # 4) Select checkpoints to audit. With no explicit argument, audit E0--E4.
    requested = set(args.experiments or [])
    experiments = [
        experiment
        for experiment in DEFAULT_EXPERIMENTS
        if not requested or experiment.name in requested
    ]
    criterion = nn.BCEWithLogitsLoss()
    rows = []
    checkpoint_names = []
    replication_paths_reference: list[str] | None = None

    for experiment_index, experiment in enumerate(experiments):
        print(f"Evaluating {experiment.name}")
        model = load_experiment_checkpoint(experiment, config, device)
        checkpoint_names.append(f"{experiment.name}_best.pt")

        # Use the same clean, deterministic evaluation transform for every
        # checkpoint. No augmentation or robustness resampling is mixed in.
        train_loader = build_standard_eval_loader(
            splits["train"],
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        replication_loader = build_standard_eval_loader(
            replication,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        train_values, train_predictions = evaluate(
            model,
            train_loader,
            criterion,
            device,
            threshold=config.threshold,
        )
        replication_values, replication_predictions = evaluate(
            model,
            replication_loader,
            criterion,
            device,
            threshold=config.threshold,
        )

        # The original-test predictions are read, not recomputed, so this audit
        # compares the new holdout against the exact reported test outputs.
        original_predictions_path = (
            original_metrics / f"{experiment.name}_test_predictions.csv"
        )
        original_predictions = pd.read_csv(original_predictions_path)
        if set(original_predictions["image_path"]) != set(splits["test"]["image_path"]):
            raise ValueError(
                f"Original predictions do not match test_split.csv: "
                f"{original_predictions_path}"
            )
        original_values = binary_metrics(
            original_predictions["label"],
            original_predictions["logit"],
            threshold=config.threshold,
        )

        # All models must be evaluated on the identical replication path order.
        # This protects the paired comparisons and bootstrap intervals.
        current_replication_paths = replication_predictions["image_path"].tolist()
        if replication_paths_reference is None:
            replication_paths_reference = current_replication_paths
        elif current_replication_paths != replication_paths_reference:
            raise RuntimeError("Models were not evaluated on identical replication paths.")

        # Save per-model predictions and subgroup summaries for inspection.
        train_predictions.to_csv(
            replication_metrics
            / f"{experiment.name}_train_clean_predictions.csv",
            index=False,
        )
        replication_predictions.to_csv(
            replication_metrics
            / f"{experiment.name}_replication_predictions.csv",
            index=False,
        )
        source_error_summary(replication_predictions).to_csv(
            replication_tables
            / f"{experiment.name}_replication_source_errors.csv",
            index=False,
        )
        style_metrics(replication_predictions).to_csv(
            replication_tables
            / f"{experiment.name}_replication_style_metrics.csv",
            index=False,
        )

        intervals = f1_audit_intervals(
            train_predictions,
            original_predictions,
            replication_predictions,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed + experiment_index * 100,
            threshold=config.threshold,
        )
        rows.append(
            {
                "exp_name": experiment.name,
                "model_name": experiment.model_name,
                "train_clean_accuracy": train_values["accuracy"],
                "train_clean_f1": train_values["f1"],
                "train_clean_roc_auc": train_values["roc_auc"],
                "original_test_accuracy": original_values["accuracy"],
                "original_test_f1": original_values["f1"],
                "original_test_roc_auc": original_values["roc_auc"],
                "replication_accuracy": replication_values["accuracy"],
                "replication_precision": replication_values["precision"],
                "replication_recall": replication_values["recall"],
                "replication_f1": replication_values["f1"],
                "replication_roc_auc": replication_values["roc_auc"],
                "replication_loss": replication_values["loss"],
                "train_replication_f1_gap": (
                    train_values["f1"] - replication_values["f1"]
                ),
                "replication_original_f1_delta": (
                    replication_values["f1"] - original_values["f1"]
                ),
                **intervals,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 5) Write the combined audit table and the manifest that documents the
    # seed, quotas, preprocessing identifier, split hashes, and overlap checks.
    results = pd.DataFrame(rows)
    results["replication_rank"] = (
        results["replication_f1"].rank(method="min", ascending=False).astype(int)
    )
    results.to_csv(
        replication_tables / "replication_overfitting_results.csv",
        index=False,
    )

    overlap_counts = {
        f"replication_vs_{name}": len(
            set(replication["image_path"]) & set(frame["image_path"])
        )
        for name, frame in splits.items()
    }
    if any(overlap_counts.values()):
        raise RuntimeError(f"Replication overlap detected: {overlap_counts}")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "dataset_handle": config.dataset_handle,
        "protocol": "official_test_replication_2k",
        "seed": args.seed,
        "quotas_per_source_style": REPLICATION_TEST_QUOTAS,
        "preprocessing": STANDARD_EVAL_TRANSFORM_ID,
        "image_size": config.image_size,
        "threshold": config.threshold,
        "bootstrap_resamples": args.bootstrap_resamples,
        "split_counts": {
            "train": len(splits["train"]),
            "validation": len(splits["val"]),
            "original_test": len(splits["test"]),
            "replication_test": len(replication),
        },
        "split_path_hashes": {
            "train": split_path_hash(splits["train"]),
            "validation": split_path_hash(splits["val"]),
            "original_test": split_path_hash(splits["test"]),
            "replication_test": split_path_hash(replication),
        },
        "overlap_counts": overlap_counts,
        "checkpoints": checkpoint_names,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
        },
    }
    (replication_metrics / "replication_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        results[
            [
                "exp_name",
                "train_clean_f1",
                "original_test_f1",
                "replication_f1",
                "train_replication_f1_gap",
                "replication_original_f1_delta",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
