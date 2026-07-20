"""
Evaluation Metrics
==================

Metrics for anomaly detection evaluation.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyMetrics:
    """Container for anomaly detection metrics."""

    # Classification metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0

    # ROC metrics
    auc_roc: float = 0.0
    auc_pr: float = 0.0

    # Threshold-independent
    average_precision: float = 0.0

    # Confusion matrix elements
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Reconstruction metrics
    mean_normal_error: float = 0.0
    mean_anomaly_error: float = 0.0
    std_normal_error: float = 0.0
    std_anomaly_error: float = 0.0
    separability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "accuracy": float(self.accuracy),
            "auc_roc": float(self.auc_roc),
            "auc_pr": float(self.auc_pr),
            "average_precision": float(self.average_precision),
            "true_positives": int(self.true_positives),
            "true_negatives": int(self.true_negatives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
            "mean_normal_error": float(self.mean_normal_error),
            "mean_anomaly_error": float(self.mean_anomaly_error),
            "std_normal_error": float(self.std_normal_error),
            "std_anomaly_error": float(self.std_anomaly_error),
            "separability": float(self.separability),
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"AnomalyMetrics(\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall: {self.recall:.4f}\n"
            f"  F1: {self.f1:.4f}\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  AUC-ROC: {self.auc_roc:.4f}\n"
            f"  AUC-PR: {self.auc_pr:.4f}\n"
            f"  Mean Normal Error: {self.mean_normal_error:.6f}\n"
            f"  Mean Anomaly Error: {self.mean_anomaly_error:.6f}\n"
            f")"
        )


def compute_reconstruction_metrics(
    reconstruction_errors: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute reconstruction error statistics.

    Args:
        reconstruction_errors: Reconstruction errors per sample
        labels: True labels (0=normal, 1=anomaly)

    Returns:
        Dictionary with error statistics
    """
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    normal_errors = reconstruction_errors[normal_mask]
    anomaly_errors = reconstruction_errors[anomaly_mask]

    metrics = {
        "mean_normal_error": np.mean(normal_errors) if len(normal_errors) > 0 else 0,
        "std_normal_error": np.std(normal_errors) if len(normal_errors) > 0 else 0,
        "mean_anomaly_error": np.mean(anomaly_errors) if len(anomaly_errors) > 0 else 0,
        "std_anomaly_error": np.std(anomaly_errors) if len(anomaly_errors) > 0 else 0,
        "min_normal_error": np.min(normal_errors) if len(normal_errors) > 0 else 0,
        "max_normal_error": np.max(normal_errors) if len(normal_errors) > 0 else 0,
        "min_anomaly_error": np.min(anomaly_errors) if len(anomaly_errors) > 0 else 0,
        "max_anomaly_error": np.max(anomaly_errors) if len(anomaly_errors) > 0 else 0,
    }

    # Separability ratio
    if metrics["std_normal_error"] > 0:
        raw_separability = (
            metrics["mean_anomaly_error"] - metrics["mean_normal_error"]
        ) / metrics["std_normal_error"]
        # Normalize to [0, 1] using S / (1 + S)
        metrics["separability"] = raw_separability / (1 + raw_separability)
    else:
        metrics["separability"] = 0

    return metrics


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> AnomalyMetrics:
    """
    Compute classification metrics for anomaly detection.

    Args:
        predictions: Binary predictions (0=normal, 1=anomaly)
        labels: True labels
        scores: Anomaly scores (for ROC/PR curves)

    Returns:
        AnomalyMetrics object
    """
    metrics = AnomalyMetrics()

    # Basic classification metrics
    metrics.precision = precision_score(labels, predictions, zero_division=0)
    metrics.recall = recall_score(labels, predictions, zero_division=0)
    metrics.f1 = f1_score(labels, predictions, zero_division=0)
    metrics.accuracy = np.mean(predictions == labels)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        metrics.true_negatives = cm[0, 0]
        metrics.false_positives = cm[0, 1]
        metrics.false_negatives = cm[1, 0]
        metrics.true_positives = cm[1, 1]

    # ROC and PR metrics (require scores)
    if scores is not None:
        try:
            if len(np.unique(labels)) > 1:
                metrics.auc_roc = roc_auc_score(labels, scores)
                metrics.auc_pr = average_precision_score(labels, scores)
                metrics.average_precision = average_precision_score(labels, scores)
            else:
                logger.warning("Labels only contain one class. AUC-ROC/PR metrics cannot be computed.")
                metrics.auc_roc = 0.0
                metrics.auc_pr = 0.0
                metrics.average_precision = 0.0
        except ValueError as e:
            logger.warning(f"Could not compute AUC metrics: {e}")

    return metrics


def evaluate_model(
    model,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> AnomalyMetrics:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained autoencoder model
        test_data: Test data
        test_labels: Test labels
        threshold: Anomaly threshold

    Returns:
        Complete AnomalyMetrics
    """
    # Get reconstruction errors
    errors = model.compute_reconstruction_error(test_data)

    # Get predictions
    predictions = (errors > threshold).astype(int)

    # Compute metrics
    metrics = compute_classification_metrics(predictions, test_labels, errors)

    # Add reconstruction metrics
    recon_metrics = compute_reconstruction_metrics(errors, test_labels)
    metrics.mean_normal_error = recon_metrics["mean_normal_error"]
    metrics.mean_anomaly_error = recon_metrics["mean_anomaly_error"]
    metrics.std_normal_error = recon_metrics["std_normal_error"]
    metrics.std_anomaly_error = recon_metrics["std_anomaly_error"]
    metrics.separability = recon_metrics["separability"]

    return metrics


def get_roc_data(
    reconstruction_errors: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get ROC curve data."""
    fpr, tpr, thresholds = roc_curve(labels, reconstruction_errors)
    return fpr, tpr, thresholds


def get_pr_data(
    reconstruction_errors: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Precision-Recall curve data."""
    precision, recall, thresholds = precision_recall_curve(
        labels, reconstruction_errors
    )
    return precision, recall, thresholds
