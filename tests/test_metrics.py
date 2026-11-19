"""
Unit tests for classification metric calculation and confusion matrix generation.
"""

import numpy as np
from src.evaluate import compute_classification_metrics


def test_compute_classification_metrics_accuracy_and_shape() -> None:
    class_names = ["class_0", "class_1", "class_2"]
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    assert np.isclose(metrics["test_accuracy"], 4.0 / 6.0)

    confusion_counts = metrics["confusion_matrix"]
    assert confusion_counts.shape == (3, 3)
    assert confusion_counts.sum() == 6
    assert confusion_counts[1, 1] == 2
