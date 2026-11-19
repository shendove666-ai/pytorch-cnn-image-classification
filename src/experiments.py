"""Run the three CIFAR-10 experiment conditions and publish their metrics."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from src.dataset import CIFAR10_CLASSES, get_dataloaders, save_sample_grid
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_prediction_examples,
    plot_training_curves,
)
from src.models import build_model, get_model_summary
from src.train import fit_model
from src.utils import collect_environment_info, get_device, save_json, set_seed

EXPERIMENT_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "exp_id": "exp_a_baseline",
        "title": "Experiment A: Baseline CNN (No Augmentation)",
        "model_name": "baseline",
        "use_augmentation": False,
    },
    {
        "exp_id": "exp_b_augmented",
        "title": "Experiment B: Baseline CNN + Data Augmentation",
        "model_name": "baseline",
        "use_augmentation": True,
    },
    {
        "exp_id": "exp_c_improved",
        "title": "Experiment C: Improved CNN + Data Augmentation",
        "model_name": "improved",
        "use_augmentation": True,
    },
]


def run_single_experiment(
    exp_spec: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute one controlled experiment from data loading through final test evaluation."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = get_device(prefer_cuda=True)

    exp_id = exp_spec["exp_id"]
    model_name = exp_spec["model_name"]
    use_augmentation = bool(exp_spec["use_augmentation"])

    data_config = config["data"]
    training_config = config["training"]

    epochs = 2 if smoke_test else int(training_config["epochs"])
    batch_size = 32 if smoke_test else int(data_config["batch_size"])

    output_dir = Path(training_config["save_dir"])
    figures_dir = Path(training_config["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(f"Starting {exp_spec['title']} (smoke_test={smoke_test})")
    logger.info(
        f"Device: {device} | Seed: {seed} | Epochs: {epochs} | "
        f"Batch Size: {batch_size} | Augmentation: {use_augmentation}"
    )

    dataloaders = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=batch_size,
        val_split=float(data_config["val_split"]),
        use_augmentation=use_augmentation,
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        smoke_test=smoke_test,
    )

    if exp_id == "exp_a_baseline":
        # Keep the archived protocol: the preview consumes the training sampler's first batch.
        save_sample_grid(
            dataloaders["train"],
            output_path=figures_dir / "sample_training_images.png",
        )

    model = build_model(model_name, num_classes=10)
    model_summary = get_model_summary(model)
    logger.info(
        f"Model: {model_summary['model_class']} | "
        f"Trainable Parameters: {model_summary['trainable_params']:,}"
    )

    checkpoint_path = output_dir / f"{exp_id}_best.pth"
    history_csv_path = output_dir / f"{exp_id}_history.csv"

    training_result = fit_model(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=epochs,
        learning_rate=float(training_config["learning_rate"]),
        momentum=float(training_config["momentum"]),
        weight_decay=float(training_config["weight_decay"]),
        device=device,
        checkpoint_path=checkpoint_path,
        history_csv_path=history_csv_path,
        logger=logger,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluation_result = evaluate_model(model, dataloaders["test"], device=device)

    class_report = evaluation_result["classification_report"]
    per_class_rows = []
    for class_name in CIFAR10_CLASSES:
        class_metrics = class_report[class_name]
        per_class_rows.append(
            {
                "class": class_name,
                "precision": round(float(class_metrics["precision"]), 4),
                "recall": round(float(class_metrics["recall"]), 4),
                "f1_score": round(float(class_metrics["f1-score"]), 4),
                "support": int(class_metrics["support"]),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / f"{exp_id}_per_class_metrics.csv", index=False
    )

    plot_training_curves(
        training_result["history"],
        output_path=figures_dir / f"{exp_id}_curves.png",
        title_prefix=exp_spec["title"],
    )
    plot_confusion_matrix(
        evaluation_result["confusion_matrix"],
        class_names=CIFAR10_CLASSES,
        output_path=figures_dir / f"{exp_id}_confusion_matrix.png",
        title=f"{exp_spec['title']} — Test Confusion Matrix",
    )
    plot_prediction_examples(
        evaluation_result["correct_examples"],
        evaluation_result["incorrect_examples"],
        output_path=figures_dir / f"{exp_id}_prediction_examples.png",
    )

    macro_metrics = class_report["macro avg"]
    experiment_result = {
        "exp_id": exp_id,
        "title": exp_spec["title"],
        "model_class": model_summary["model_class"],
        "trainable_params": model_summary["trainable_params"],
        "use_augmentation": use_augmentation,
        "seed": seed,
        "device": str(device),
        "epochs_run": len(training_result["history"]),
        "smoke_test": smoke_test,
        "best_epoch": training_result["best_epoch"],
        "best_val_acc": training_result["best_val_acc"],
        "test_acc": evaluation_result["test_accuracy"],
        "test_loss": evaluation_result["test_loss"],
        "macro_precision": round(float(macro_metrics["precision"]), 4),
        "macro_recall": round(float(macro_metrics["recall"]), 4),
        "macro_f1": round(float(macro_metrics["f1-score"]), 4),
        "training_time_sec": training_result["total_training_time_sec"],
    }

    logger.info(
        f"Completed {exp_id} | Best Val Acc: {experiment_result['best_val_acc']*100:.2f}% | "
        f"Final Test Acc: {experiment_result['test_acc']*100:.2f}% | "
        f"Macro F1: {experiment_result['macro_f1']:.4f}"
    )
    return experiment_result


def run_all_experiments(
    config: Dict[str, Any],
    logger: logging.Logger,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Run Experiments A, B, and C sequentially and export comparison tables."""
    output_dir = Path(config["training"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    environment_info = collect_environment_info()
    save_json(
        {"environment": environment_info, "config": config, "smoke_test": smoke_test},
        output_dir / "run_metadata.json",
    )

    results: List[Dict[str, Any]] = []
    for experiment_spec in EXPERIMENT_DEFINITIONS:
        experiment_result = run_single_experiment(experiment_spec, config, logger, smoke_test=smoke_test)
        results.append(experiment_result)

    comparison_table = pd.DataFrame(results)
    summary_csv = output_dir / "experiments_comparison.csv"
    comparison_table.to_csv(summary_csv, index=False)
    logger.info(f"Saved consolidated comparison table to '{summary_csv}'.")
    return comparison_table
