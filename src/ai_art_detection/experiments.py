"""Define the E0--E4 matrix and run one experiment from end to end.

An experiment joins an architecture, freeze policy, and augmentation choice.
The runner then applies the shared training protocol and writes a predictable
set of artifacts that the notebook, replication audit, and report can reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn

from .config import ProjectConfig
from .data import build_loaders
from .evaluation import evaluate, source_error_summary, style_metrics
from .models import build_model, count_trainable_parameters
from .training import fit, save_result


@dataclass(frozen=True, slots=True)
class Experiment:
    """Describe one controlled model comparison.

    Attributes:
        name: Stable identifier used in checkpoint and result filenames.
        model_name: Architecture identifier accepted by :func:`build_model`.
        mode: Transfer-learning freeze policy.
        augment: Whether the training loader receives mild random augmentation.
    """

    name: str
    model_name: str
    mode: str
    augment: bool


DEFAULT_EXPERIMENTS = (
    Experiment("E0_mobilenetv2_noaug_frozen", "mobilenet_v2", "frozen", False),
    Experiment("E1_mobilenetv2_aug_frozen", "mobilenet_v2", "frozen", True),
    Experiment("E2_convnext_tiny_aug_frozen", "convnext_tiny", "frozen", True),
    Experiment("E3_convnext_tiny_aug_laststage", "convnext_tiny", "last_stage", True),
    Experiment(
        "E4_convnext_tiny_se_aug_laststage",
        "convnext_tiny_se",
        "last_stage",
        True,
    ),
)


def run_experiment(
    experiment: Experiment,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: ProjectConfig,
    device: torch.device,
    pretrained: bool = True,
) -> tuple[nn.Module, dict]:
    """Train one experiment, evaluate its selected state, and write artifacts.

    Args:
        experiment: Architecture, freeze policy, and augmentation choice to run.
        train_frame: Fixed training metadata.
        val_frame: Fixed validation metadata used for early stopping and selection.
        test_frame: Fixed official-test metadata used only after fitting.
        config: Shared paths, hyperparameters, seed, and probability threshold.
        device: Device used for training and inference.
        pretrained: Load torchvision ``DEFAULT`` ImageNet weights when true.

    Returns:
        The validation-selected model and its flat result dictionary. Result keys
        include experiment metadata, best epoch, validation and test metrics,
        trainable parameter count, and elapsed training minutes.

    Note:
        This function writes the best checkpoint, epoch history, validation and test
        predictions, source/style summaries, and result JSON below
        ``config.output_dir``. It does not choose among E0--E4; the outer workflow
        ranks completed experiments by validation F1.
    """
    # 1) Build loaders.  Only the training loader receives augmentation.
    train_loader, val_loader, test_loader = build_loaders(
        train_frame,
        val_frame,
        test_frame,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augment=experiment.augment,
    )
    model = build_model(
        experiment.model_name, experiment.mode, pretrained=pretrained
    ).to(device)

    # 2) Fine-tuned models use the lower learning rate; frozen-head baselines
    # use the head learning rate.
    learning_rate = (
        config.lr_finetune
        if experiment.mode in {"last_stage", "full"}
        else config.lr_head
    )

    # 3) Fit with early stopping on validation F1 and save the best checkpoint.
    train_result = fit(
        model,
        train_loader,
        val_loader,
        device,
        learning_rate,
        config.weight_decay,
        config.epochs,
        config.patience,
        config.output_dir / "models" / f"{experiment.name}_best.pt",
        {
            "experiment": experiment.name,
            "model_name": experiment.model_name,
            "mode": experiment.mode,
            "augment": experiment.augment,
            "config": config.as_serializable_dict(),
        },
        threshold=config.threshold,
    )

    # 4) Re-evaluate the restored best checkpoint on validation and test splits.
    criterion = nn.BCEWithLogitsLoss()
    val_metrics, val_predictions = evaluate(
        model, val_loader, criterion, device, threshold=config.threshold
    )
    test_metrics, test_predictions = evaluate(
        model, test_loader, criterion, device, threshold=config.threshold
    )

    # 5) Persist history, predictions, and subgroup tables for notebook/report
    # use.  The report reads these files instead of hand-entered values.
    train_result.history.to_csv(
        config.output_dir / "metrics" / f"{experiment.name}_history.csv",
        index=False,
    )
    val_predictions.to_csv(
        config.output_dir / "metrics" / f"{experiment.name}_val_predictions.csv",
        index=False,
    )
    test_predictions.to_csv(
        config.output_dir / "metrics" / f"{experiment.name}_test_predictions.csv",
        index=False,
    )
    style_metrics(test_predictions).to_csv(
        config.output_dir / "tables" / f"{experiment.name}_style_metrics.csv",
        index=False,
    )
    source_error_summary(test_predictions).to_csv(
        config.output_dir / "tables" / f"{experiment.name}_source_errors.csv",
        index=False,
    )
    result = {
        "exp_name": experiment.name,
        "model_name": experiment.model_name,
        "mode": experiment.mode,
        "augment": experiment.augment,
        "best_epoch": train_result.best_epoch,
        "best_val_f1": train_result.best_val_f1,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "trainable_params": count_trainable_parameters(model),
        "time_min": train_result.elapsed_seconds / 60,
    }
    save_result(
        result, config.output_dir / "metrics" / f"{experiment.name}_result.json"
    )
    return model, result


def load_experiment_checkpoint(
    experiment: Experiment,
    config: ProjectConfig,
    device: torch.device,
) -> nn.Module:
    """Reconstruct a model and restore its saved best-validation weights.

    Args:
        experiment: Definition needed to rebuild the original architecture and freeze
            policy.
        config: Configuration whose output directory contains the checkpoint.
        device: Destination device and checkpoint map location.

    Returns:
        An evaluation-mode model loaded from ``<experiment.name>_best.pt``. The model
        is created without downloading pretrained weights because the checkpoint
        already contains every parameter.

    Raises:
        FileNotFoundError: If the expected best checkpoint does not exist.
    """
    checkpoint_path = (
        config.output_dir / "models" / f"{experiment.name}_best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model = build_model(
        experiment.model_name, experiment.mode, pretrained=False
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
