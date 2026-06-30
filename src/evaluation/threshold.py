"""
Threshold Selection
===================

Methods for selecting optimal anomaly detection threshold.
"""

import numpy as np
from typing import Dict, Optional, Literal
from sklearn.metrics import f1_score, roc_curve

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdSelector:
    """
    Select optimal threshold for anomaly detection.

    Supports multiple selection methods.
    """

    def __init__(
        self,
        method: Literal["percentile", "std", "f1", "youden", "fixed"] = "percentile",
        **kwargs,
    ):
        """
        Initialize threshold selector.

        Args:
            method: Selection method
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.threshold_ = None
        self.fit_statistics_ = None

    def fit(
        self,
        normal_errors: np.ndarray,
        anomaly_errors: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> "ThresholdSelector":
        """
        Fit threshold based on reconstruction errors.

        Args:
            normal_errors: Reconstruction errors for normal samples
            anomaly_errors: Reconstruction errors for anomaly samples (for supervised methods)
            labels: Labels for supervised methods

        Returns:
            Self
        """
        # Store statistics
        self.fit_statistics_ = {
            "mean": np.mean(normal_errors),
            "std": np.std(normal_errors),
            "min": np.min(normal_errors),
            "max": np.max(normal_errors),
            "median": np.median(normal_errors),
        }

        if self.method == "percentile":
            percentile = self.kwargs.get("percentile", 95)
            self.threshold_ = np.percentile(normal_errors, percentile)

        elif self.method == "std":
            n_std = self.kwargs.get("n_std", 3)
            self.threshold_ = (
                self.fit_statistics_["mean"] + n_std * self.fit_statistics_["std"]
            )

        elif self.method == "f1":
            if anomaly_errors is None:
                raise ValueError("f1 method requires anomaly_errors")
            self.threshold_ = self._find_optimal_f1(normal_errors, anomaly_errors)

        elif self.method == "youden":
            if anomaly_errors is None:
                raise ValueError("youden method requires anomaly_errors")
            self.threshold_ = self._find_youden_threshold(normal_errors, anomaly_errors)

        elif self.method == "fixed":
            self.threshold_ = self.kwargs.get("value", 0.1)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.info(f"Threshold ({self.method}): {self.threshold_:.6f}")

        return self

    def _find_optimal_f1(
        self,
        normal_errors: np.ndarray,
        anomaly_errors: np.ndarray,
    ) -> float:
        """Find threshold that maximizes F1 score."""
        all_errors = np.concatenate([normal_errors, anomaly_errors])
        labels = np.concatenate(
            [
                np.zeros(len(normal_errors)),
                np.ones(len(anomaly_errors)),
            ]
        )

        # Try different thresholds
        thresholds = np.percentile(all_errors, np.arange(50, 100, 1))

        best_f1 = 0
        best_threshold = thresholds[0]

        for thresh in thresholds:
            predictions = (all_errors > thresh).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        return best_threshold

    def _find_youden_threshold(
        self,
        normal_errors: np.ndarray,
        anomaly_errors: np.ndarray,
    ) -> float:
        """Find threshold using Youden's J statistic."""
        all_errors = np.concatenate([normal_errors, anomaly_errors])
        labels = np.concatenate(
            [
                np.zeros(len(normal_errors)),
                np.ones(len(anomaly_errors)),
            ]
        )

        fpr, tpr, thresholds = roc_curve(labels, all_errors)

        # Youden's J = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)

        return thresholds[best_idx]

    @property
    def threshold(self) -> float:
        """Get fitted threshold."""
        if self.threshold_ is None:
            raise RuntimeError("Threshold not fitted. Call fit() first.")
        return self.threshold_

    def predict(self, errors: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on threshold.

        Args:
            errors: Reconstruction errors

        Returns:
            Binary predictions (0=normal, 1=anomaly)
        """
        return (errors > self.threshold).astype(int)


def compute_optimal_threshold(
    normal_errors: np.ndarray,
    method: str = "percentile",
    percentile: float = 95,
    n_std: float = 3,
) -> float:
    """
    Convenience function to compute optimal threshold.

    Args:
        normal_errors: Reconstruction errors for normal samples
        method: Threshold method ("percentile" or "std")
        percentile: Percentile for percentile method
        n_std: Number of standard deviations for std method

    Returns:
        Computed threshold
    """
    if method == "percentile":
        return np.percentile(normal_errors, percentile)
    elif method == "std":
        return np.mean(normal_errors) + n_std * np.std(normal_errors)
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_threshold_sensitivity(
    errors: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Analyze sensitivity to threshold selection.

    Args:
        errors: Reconstruction errors
        labels: True labels
        thresholds: Thresholds to evaluate (auto if None)

    Returns:
        Dictionary with precision, recall, f1 at each threshold
    """
    if thresholds is None:
        thresholds = np.percentile(errors, np.arange(50, 100, 2))

    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        predictions = (errors > thresh).astype(int)

        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "thresholds": thresholds,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1_scores),
    }
