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
    """Execute one full training epoch."""
    model.train()
    running_loss = 0.0
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
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += int((predictions == labels).sum().item())
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = correct_predictions / max(1, total_samples)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model performance on the validation set without gradient tracking."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += int((predictions == labels).sum().item())
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    epoch_acc = correct_predictions / max(1, total_samples)
    return epoch_loss, epoch_acc


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

    ckpt_file = Path(checkpoint_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)

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

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "epoch_time_sec": round(epoch_duration, 3),
            }
            history_records.append(record)

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
                    ckpt_file,
                )

            if logger is not None:
                marker = " *BEST*" if is_best else ""
                logger.info(
                    f"Epoch [{epoch:02d}/{epochs:02d}] | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}% | "
                    f"Time: {epoch_duration:.1f}s{marker}"
                )
    except KeyboardInterrupt:
        if logger is not None:
            logger.warning(
                "Training interrupted by user (KeyboardInterrupt). "
                "Preserving completed epoch logs and latest best checkpoint."
            )

    total_time = time.time() - start_time

    if history_csv_path is not None and history_records:
        csv_file = Path(history_csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history_records).to_csv(csv_file, index=False)

    return {
        "best_val_acc": round(best_val_acc, 6),
        "best_epoch": best_epoch,
        "total_training_time_sec": round(total_time, 3),
        "checkpoint_path": str(ckpt_file),
        "history": history_records,
    }
