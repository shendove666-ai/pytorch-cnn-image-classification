"""
Controlled experiment runner for comparing Baseline CNN (Exp A),
Baseline CNN + Data Augmentation (Exp B), and Improved CNN + Data Augmentation (Exp C).
"""

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

    data_cfg = config["data"]
    train_cfg = config["training"]

    epochs = 2 if smoke_test else int(train_cfg["epochs"])
    batch_size = 32 if smoke_test else int(data_cfg["batch_size"])

    output_dir = Path(train_cfg["save_dir"])
    figures_dir = Path(train_cfg["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(f"Starting {exp_spec['title']} (smoke_test={smoke_test})")
    logger.info(
        f"Device: {device} | Seed: {seed} | Epochs: {epochs} | "
        f"Batch Size: {batch_size} | Augmentation: {use_augmentation}"
    )

    loaders = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=batch_size,
        val_split=float(data_cfg["val_split"]),
        use_augmentation=use_augmentation,
        num_workers=int(data_cfg["num_workers"]),
        seed=seed,
        smoke_test=smoke_test,
    )

    if exp_id == "exp_a_baseline":
        save_sample_grid(
            loaders["train"],
            output_path=figures_dir / "sample_training_images.png",
        )

    model = build_model(model_name, num_classes=10)
    summary = get_model_summary(model)
    logger.info(
        f"Model: {summary['model_class']} | "
        f"Trainable Parameters: {summary['trainable_params']:,}"
    )

    ckpt_path = output_dir / f"{exp_id}_best.pth"
    history_csv = output_dir / f"{exp_id}_history.csv"

    train_result = fit_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        epochs=epochs,
        learning_rate=float(train_cfg["learning_rate"]),
        momentum=float(train_cfg["momentum"]),
        weight_decay=float(train_cfg["weight_decay"]),
        device=device,
        checkpoint_path=ckpt_path,
        history_csv_path=history_csv,
        logger=logger,
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_result = evaluate_model(model, loaders["test"], device=device)

    cls_report = eval_result["classification_report"]
    per_class_rows = []
    for cls_name in CIFAR10_CLASSES:
        metrics_row = cls_report[cls_name]
        per_class_rows.append(
            {
                "class": cls_name,
                "precision": round(float(metrics_row["precision"]), 4),
                "recall": round(float(metrics_row["recall"]), 4),
                "f1_score": round(float(metrics_row["f1-score"]), 4),
                "support": int(metrics_row["support"]),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        output_dir / f"{exp_id}_per_class_metrics.csv", index=False
    )

    plot_training_curves(
        train_result["history"],
        output_path=figures_dir / f"{exp_id}_curves.png",
        title_prefix=exp_spec["title"],
    )
    plot_confusion_matrix(
        eval_result["confusion_matrix"],
        class_names=CIFAR10_CLASSES,
        output_path=figures_dir / f"{exp_id}_confusion_matrix.png",
        title=f"{exp_spec['title']} — Test Confusion Matrix",
    )
    plot_prediction_examples(
        eval_result["correct_examples"],
        eval_result["incorrect_examples"],
        output_path=figures_dir / f"{exp_id}_prediction_examples.png",
    )

    macro_avg = cls_report["macro avg"]
    summary_record = {
        "exp_id": exp_id,
        "title": exp_spec["title"],
        "model_class": summary["model_class"],
        "trainable_params": summary["trainable_params"],
        "use_augmentation": use_augmentation,
        "seed": seed,
        "device": str(device),
        "epochs_run": len(train_result["history"]),
        "smoke_test": smoke_test,
        "best_epoch": train_result["best_epoch"],
        "best_val_acc": train_result["best_val_acc"],
        "test_acc": eval_result["test_accuracy"],
        "test_loss": eval_result["test_loss"],
        "macro_precision": round(float(macro_avg["precision"]), 4),
        "macro_recall": round(float(macro_avg["recall"]), 4),
        "macro_f1": round(float(macro_avg["f1-score"]), 4),
        "training_time_sec": train_result["total_training_time_sec"],
    }

    logger.info(
        f"Completed {exp_id} | Best Val Acc: {summary_record['best_val_acc']*100:.2f}% | "
        f"Final Test Acc: {summary_record['test_acc']*100:.2f}% | "
        f"Macro F1: {summary_record['macro_f1']:.4f}"
    )
    return summary_record


def run_all_experiments(
    config: Dict[str, Any],
    logger: logging.Logger,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Run Experiments A, B, and C sequentially and export comparison tables."""
    output_dir = Path(config["training"]["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    env_info = collect_environment_info()
    save_json(
        {"environment": env_info, "config": config, "smoke_test": smoke_test},
        output_dir / "run_metadata.json",
    )

    results: List[Dict[str, Any]] = []
    for spec in EXPERIMENT_DEFINITIONS:
        res = run_single_experiment(spec, config, logger, smoke_test=smoke_test)
        results.append(res)

    df_summary = pd.DataFrame(results)
    summary_csv = output_dir / "experiments_comparison.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved consolidated comparison table to '{summary_csv}'.")
    return df_summary
