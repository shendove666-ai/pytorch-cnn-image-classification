"""
Unit tests for classification metric calculation and confusion matrix generation.
"""

import numpy as np
from src.evaluate import compute_classification_metrics


def test_compute_classification_metrics_accuracy_and_shape() -> None:
    class_names = ["class_0", "class_1", "class_2"]
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    res = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    assert np.isclose(res["test_accuracy"], 4.0 / 6.0)

    conf_mat = res["confusion_matrix"]
    assert conf_mat.shape == (3, 3)
    assert conf_mat.sum() == 6
    assert conf_mat[1, 1] == 2
