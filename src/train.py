"""
Training and validation loop implementation with checkpoint saving and history logging.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Update once per batch and return the sample-weighted loss and accuracy."""
    model.train()
    weighted_loss_total = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss detected ({loss.item()}). "
                "Consider reducing the learning rate or inspecting input normalization."
            )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        weighted_loss_total += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += int((predictions == labels).sum().item())
        total_samples += batch_size

    epoch_loss = weighted_loss_total / max(1, total_samples)
    epoch_accuracy = correct_predictions / max(1, total_samples)
    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate with frozen training-time behavior and a mean loss per batch."""
    model.eval()
    weighted_loss_total = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        weighted_loss_total += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += int((predictions == labels).sum().item())
        total_samples += batch_size

    epoch_loss = weighted_loss_total / max(1, total_samples)
    epoch_accuracy = correct_predictions / max(1, total_samples)
    return epoch_loss, epoch_accuracy


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_path: str | Path,
    history_csv_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> Dict[str, object]:
    """Run the complete multi-epoch training and validation loop, saving the best checkpoint."""
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}.")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_epoch = 0
    history_records: List[Dict[str, float | int]] = []
    start_time = time.time()

    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, criterion, device
            )
            epoch_duration = time.time() - epoch_start

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "epoch_time_sec": round(epoch_duration, 3),
            }
            history_records.append(epoch_metrics)

            # Strict improvement retains the earlier epoch when validation accuracy ties.
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    checkpoint_file,
                )

            if logger is not None:
                checkpoint_marker = " *BEST*" if is_best else ""
                logger.info(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}% | "
                    f"Time: {epoch_duration:.1f}s{checkpoint_marker}"
                )
    except KeyboardInterrupt:
        if logger is not None:
            logger.warning(
                "Training interrupted by user (KeyboardInterrupt). "
                "Preserving completed epoch logs and latest best checkpoint."
            )

    total_time = time.time() - start_time

    if history_csv_path is not None and history_records:
        history_file = Path(history_csv_path)
        history_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_records).to_csv(history_file, index=False)

    return {
        "best_val_acc": round(best_val_acc, 6),
        "best_epoch": best_epoch,
        "total_training_time_sec": round(total_time, 3),
        "checkpoint_path": str(checkpoint_file),
        "history": history_records,
    }
