"""Train one-logit PyTorch classifiers with validation-based model selection.

The training loop deliberately has no access to the test split. AdamW updates
only parameters marked trainable by the model builder, a plateau scheduler
reacts to validation F1, and early stopping restores the best validation state
before evaluation artifacts are produced.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import binary_metrics, evaluate


@dataclass(slots=True)
class TrainResult:
    """Collect the selected model and its training history.

    Attributes:
        model: Input model restored to its highest-validation-F1 state.
        history: One row per completed epoch with prefixed train/validation metrics
            and the learning rate used after scheduling.
        best_epoch: One-based epoch at which validation F1 was highest.
        best_val_f1: Validation F1 used to select ``model``.
        elapsed_seconds: Wall-clock fitting time measured with a monotonic clock.
    """

    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_val_f1: float
    elapsed_seconds: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Run one optimizer pass over a non-empty training loader.

    Args:
        model: Binary classifier returning one logit per image.
        loader: Training loader whose batches contain ``image`` and float ``label``.
        optimizer: Optimizer configured with the model's trainable parameters.
        criterion: Loss accepting flattened logits and labels.
        device: Training device.
        scaler: Optional CUDA gradient scaler. It is ignored on non-CUDA devices.
        threshold: Fake-class probability threshold used only for reported metrics.

    Returns:
        Accuracy, precision, recall, F1, ROC-AUC, and sample-weighted mean training
        loss. CUDA execution uses automatic mixed precision; CPU execution remains
        full precision.
    """
    model.train()
    loss_total = 0.0
    labels_all: list[float] = []
    logits_all: list[float] = []
    use_amp = device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # CUDA runs use autocast + GradScaler for speed; CPU runs use the same
        # code path with AMP disabled.
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images).flatten()
            loss = criterion(logits, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_total += loss.item() * len(images)
        labels_all.extend(labels.detach().cpu().tolist())
        logits_all.extend(logits.detach().cpu().tolist())

    metrics = binary_metrics(labels_all, logits_all, threshold)
    metrics["loss"] = loss_total / len(loader.dataset)
    return metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict | None = None,
    threshold: float = 0.5,
) -> TrainResult:
    """Fit a model and restore the checkpoint with the best validation F1.

    Args:
        model: Classifier whose trainable flags already encode the transfer policy.
        train_loader: Augmented or clean training loader.
        val_loader: Clean validation loader used for scheduling and model selection.
        device: Device used for training and validation.
        learning_rate: Initial AdamW learning rate.
        weight_decay: Decoupled AdamW weight-decay coefficient.
        epochs: Maximum number of epochs.
        patience: Stop after this many consecutive epochs without an F1 improvement.
        checkpoint_path: Optional destination for the best checkpoint.
        checkpoint_metadata: Context stored alongside checkpoint weights and scores.
        threshold: Fixed probability threshold used for train and validation metrics.

    Returns:
        A :class:`TrainResult` whose model has been restored to the best recorded
        state, even when later epochs performed worse.

    Note:
        A checkpoint contains ``model_state_dict``, ``best_epoch``, ``best_val_f1``,
        and ``metadata``. Progress for every completed epoch is printed to standard
        output. Equal F1 values do not replace an earlier checkpoint.
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history_rows = []
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        # Train once, evaluate on validation, then let the scheduler react to
        # validation F1.  Test data is deliberately absent from this loop.
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, threshold
        )
        val_metrics, _ = evaluate(
            model, val_loader, criterion, device, threshold=threshold
        )
        scheduler.step(val_metrics["f1"])
        history_rows.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Store the best validation-F1 weights.  The final returned model is
        # restored to this state even if later epochs overfit.
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "best_val_f1": best_val_f1,
                        "metadata": checkpoint_metadata or {},
                    },
                    checkpoint_path,
                )
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        history=pd.DataFrame(history_rows),
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        elapsed_seconds=time.monotonic() - started,
    )


def save_result(result: dict, output_path: Path) -> None:
    """Serialize an experiment summary as indented JSON.

    Args:
        result: JSON-serializable metrics and experiment metadata.
        output_path: Destination file. Missing parent directories are created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
