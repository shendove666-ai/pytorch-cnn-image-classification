"""Tests for multi-seed result aggregation."""

import pandas as pd
import pytest

from src.reporting import summarize_repeated_results


def sample_results() -> pd.DataFrame:
    rows = []
    for exp_id, base in (("exp_a", 0.70), ("exp_b", 0.60)):
        for seed, offset in ((41, 0.00), (42, 0.02), (43, 0.04)):
            rows.append(
                {
                    "exp_id": exp_id,
                    "seed": seed,
                    "best_val_acc": base + offset,
                    "test_acc": base + offset,
                    "test_loss": 1.0 - base,
                    "macro_f1": base - 0.01 + offset,
                    "training_time_sec": 10.0 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_summary_reports_mean_std_and_seed_count() -> None:
    summary = summarize_repeated_results(sample_results()).set_index("exp_id")
    assert summary.loc["exp_a", "n_seeds"] == 3
    assert summary.loc["exp_a", "test_acc_mean"] == pytest.approx(0.72)
    assert summary.loc["exp_a", "test_acc_std"] == pytest.approx(0.02)


def test_duplicate_experiment_seed_is_rejected() -> None:
    data = sample_results()
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly once"):
        summarize_repeated_results(duplicated)
