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
    """Summarize aligned one-dimensional labels in the requested class order."""
    true_labels = np.asarray(y_true, dtype=int)
    predicted_labels = np.asarray(y_pred, dtype=int)

    overall_accuracy = float((true_labels == predicted_labels).mean())
    class_indices = list(range(len(class_names)))

    class_report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    confusion_counts = confusion_matrix(true_labels, predicted_labels, labels=class_indices)

    return {
        "test_accuracy": round(overall_accuracy, 6),
        "classification_report": class_report,
        "confusion_matrix": confusion_counts,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = CIFAR10_CLASSES,
) -> Dict[str, object]:
    """Evaluate the held-out split and collect the first examples of each outcome."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    predicted_labels: List[int] = []
    true_labels: List[int] = []
    weighted_loss_total = 0.0
    total_samples = 0

    correct_examples: List[Dict[str, object]] = []
    incorrect_examples: List[Dict[str, object]] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        class_probabilities = torch.softmax(logits, dim=1)
        confidences, batch_predictions = torch.max(class_probabilities, dim=1)

        batch_size = images.size(0)
        weighted_loss_total += loss.item() * batch_size
        total_samples += batch_size

        predicted_labels.extend(batch_predictions.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

        for i in range(batch_size):
            if len(correct_examples) >= 8 and len(incorrect_examples) >= 8:
                break
            true_class_index = int(labels[i].item())
            predicted_class_index = int(batch_predictions[i].item())
            predicted_probability = float(confidences[i].item())
            prediction_example = {
                "image": images[i].detach().cpu(),
                "true_label": class_names[true_class_index],
                "pred_label": class_names[predicted_class_index],
                "confidence": predicted_probability,
            }
            if true_class_index == predicted_class_index and len(correct_examples) < 8:
                correct_examples.append(prediction_example)
            elif true_class_index != predicted_class_index and len(incorrect_examples) < 8:
                incorrect_examples.append(prediction_example)

    avg_test_loss = weighted_loss_total / max(1, total_samples)
    metrics = compute_classification_metrics(true_labels, predicted_labels, class_names)
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
    epoch_metrics = pd.DataFrame(history)
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))

    loss_axis.plot(epoch_metrics["epoch"], epoch_metrics["train_loss"], marker="o", label="Train Loss")
    loss_axis.plot(epoch_metrics["epoch"], epoch_metrics["val_loss"], marker="s", label="Validation Loss")
    loss_axis.set_title(f"{title_prefix} — Loss Curve")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-Entropy Loss")
    loss_axis.grid(True, linestyle="--", alpha=0.5)
    loss_axis.legend()

    accuracy_axis.plot(epoch_metrics["epoch"], epoch_metrics["train_acc"] * 100.0, marker="o", label="Train Accuracy")
    accuracy_axis.plot(epoch_metrics["epoch"], epoch_metrics["val_acc"] * 100.0, marker="s", label="Validation Accuracy")
    accuracy_axis.set_title(f"{title_prefix} — Accuracy Curve")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.grid(True, linestyle="--", alpha=0.5)
    accuracy_axis.legend()

    plt.tight_layout()
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save an annotated confusion matrix heatmap using Matplotlib."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    heatmap = ax.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(heatmap, ax=ax)

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

    annotation_threshold = conf_mat.max() / 2.0 if conf_mat.size > 0 else 0.0
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                format(conf_mat[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_mat[i, j] > annotation_threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150)
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
        correct_axis = axes[0, idx]
        if idx < len(correct_examples):
            prediction_example = correct_examples[idx]
            correct_axis.imshow(denormalize_image(prediction_example["image"]))
            correct_axis.set_title(
                f"CORRECT\nTrue: {prediction_example['true_label']}\nPred: {prediction_example['pred_label']} ({prediction_example['confidence']*100:.1f}%)",
                fontsize=9,
                color="darkgreen",
            )
        correct_axis.axis("off")

        incorrect_axis = axes[1, idx]
        if idx < len(incorrect_examples):
            prediction_example = incorrect_examples[idx]
            incorrect_axis.imshow(denormalize_image(prediction_example["image"]))
            incorrect_axis.set_title(
                f"MISCLASSIFIED\nTrue: {prediction_example['true_label']}\nPred: {prediction_example['pred_label']} ({prediction_example['confidence']*100:.1f}%)",
                fontsize=9,
                color="darkred",
            )
        incorrect_axis.axis("off")

    plt.tight_layout()
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)
