"""
Model evaluation, classification metrics calculation, and diagnostic visualizations.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import CIFAR10_CLASSES, denormalize_image


def compute_classification_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: List[str] = CIFAR10_CLASSES,
) -> Dict[str, object]:
    """Calculate overall accuracy, per-class precision/recall/F1, and confusion matrix."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    overall_accuracy = float((y_true_arr == y_pred_arr).mean())
    labels_indices = list(range(len(class_names)))

    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_indices)

    return {
        "test_accuracy": round(overall_accuracy, 6),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = CIFAR10_CLASSES,
) -> Dict[str, object]:
    """Run inference on the held-out test set and return comprehensive metrics and sample predictions."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    all_preds: List[int] = []
    all_targets: List[int] = []
    running_loss = 0.0
    total_samples = 0

    correct_examples: List[Dict[str, object]] = []
    incorrect_examples: List[Dict[str, object]] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

        for i in range(batch_size):
            if len(correct_examples) >= 8 and len(incorrect_examples) >= 8:
                break
            true_idx = int(labels[i].item())
            pred_idx = int(preds[i].item())
            conf_val = float(confidences[i].item())
            entry = {
                "image": images[i].detach().cpu(),
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "confidence": conf_val,
            }
            if true_idx == pred_idx and len(correct_examples) < 8:
                correct_examples.append(entry)
            elif true_idx != pred_idx and len(incorrect_examples) < 8:
                incorrect_examples.append(entry)

    avg_test_loss = running_loss / max(1, total_samples)
    metrics = compute_classification_metrics(all_targets, all_preds, class_names)
    metrics["test_loss"] = round(avg_test_loss, 6)
    metrics["correct_examples"] = correct_examples
    metrics["incorrect_examples"] = incorrect_examples

    return metrics


def plot_training_curves(
    history: List[Dict[str, float | int]],
    output_path: str | Path,
    title_prefix: str = "Experiment",
) -> None:
    """Plot and save side-by-side training/validation Loss and Accuracy curves."""
    df = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    ax1.plot(df["epoch"], df["val_loss"], marker="s", label="Validation Loss")
    ax1.set_title(f"{title_prefix} — Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(df["epoch"], df["train_acc"] * 100.0, marker="o", label="Train Accuracy")
    ax2.plot(df["epoch"], df["val_acc"] * 100.0, marker="s", label="Validation Accuracy")
    ax2.set_title(f"{title_prefix} — Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save an annotated confusion matrix heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(conf_mat.shape[1]),
        yticks=np.arange(conf_mat.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = conf_mat.max() / 2.0 if conf_mat.size > 0 else 0.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_prediction_examples(
    correct_examples: List[Dict[str, object]],
    incorrect_examples: List[Dict[str, object]],
    output_path: str | Path,
) -> None:
    """Save a visual grid of correctly classified (top row) and misclassified (bottom row) samples."""
    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(12, 6))

    for idx in range(cols):
        ax_corr = axes[0, idx]
        if idx < len(correct_examples):
            item = correct_examples[idx]
            ax_corr.imshow(denormalize_image(item["image"]))
            ax_corr.set_title(
                f"CORRECT\nTrue: {item['true_label']}\nPred: {item['pred_label']} ({item['confidence']*100:.1f}%)",
                fontsize=9,
                color="darkgreen",
            )
        ax_corr.axis("off")

        ax_inc = axes[1, idx]
        if idx < len(incorrect_examples):
            item = incorrect_examples[idx]
            ax_inc.imshow(denormalize_image(item["image"]))
            ax_inc.set_title(
                f"MISCLASSIFIED\nTrue: {item['true_label']}\nPred: {item['pred_label']} ({item['confidence']*100:.1f}%)",
                fontsize=9,
                color="darkred",
            )
        ax_inc.axis("off")

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
