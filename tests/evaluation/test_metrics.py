import numpy as np

from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.threshold import compute_optimal_threshold


def test_compute_classification_metrics():
    # 8 normal (0), 2 anomalous (1)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    # Perfect scores (high scores for anomalies)
    scores = np.array([0.1, 0.2, 0.1, 0.3, 0.15, 0.25, 0.1, 0.2, 0.9, 0.95])
    predictions = (scores > 0.5).astype(int)

    metrics = compute_classification_metrics(predictions, labels, scores)

    # Verify perfect separation
    assert np.isclose(metrics.auc_roc, 1.0)
    assert np.isclose(metrics.auc_pr, 1.0)


def test_compute_optimal_threshold():
    normal_errors = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = compute_optimal_threshold(
        normal_errors, method="percentile", percentile=95
    )
    assert threshold > 0.35

    threshold_std = compute_optimal_threshold(normal_errors, method="std", n_std=3)
    assert threshold_std > 0.4
