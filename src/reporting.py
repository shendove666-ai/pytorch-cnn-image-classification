"""Aggregation helpers for repeated experiment runs."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


REPEATED_METRICS = (
    "best_val_acc",
    "test_acc",
    "test_loss",
    "macro_f1",
    "training_time_sec",
)


def summarize_repeated_results(
    results: pd.DataFrame,
    metrics: Iterable[str] = REPEATED_METRICS,
) -> pd.DataFrame:
    """Return one mean/std/count row per experiment across random seeds."""
    metrics = tuple(metrics)
    required_columns = {"exp_id", "seed", *metrics}
    missing_columns = sorted(required_columns.difference(results.columns))
    if missing_columns:
        raise ValueError(f"Missing required result columns: {', '.join(missing_columns)}")

    if results.duplicated(["exp_id", "seed"]).any():
        raise ValueError("Each experiment/seed pair must appear exactly once.")

    results_by_experiment = results.groupby("exp_id", sort=False)
    aggregate_metrics = results_by_experiment[list(metrics)].agg(["mean", "std"])
    aggregate_metrics.columns = [f"{metric}_{stat}" for metric, stat in aggregate_metrics.columns]
    aggregate_metrics.insert(0, "n_seeds", results_by_experiment["seed"].nunique())
    return aggregate_metrics.reset_index()
