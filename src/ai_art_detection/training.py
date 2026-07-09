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
    model.train()
    loss_total = 0.0
    labels_all: list[float] = []
    logits_all: list[float] = []
    use_amp = device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
