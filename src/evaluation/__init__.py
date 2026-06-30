"""Evaluation and metrics for anomaly detection."""

from .metrics import (
    AnomalyMetrics,
    compute_reconstruction_metrics,
    compute_classification_metrics,
)
from .threshold import (
    ThresholdSelector,
    compute_optimal_threshold,
)
from .visualization import (
    plot_reconstruction_error_distribution,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_reconstructions,
    plot_latent_space,
    plot_training_history,
)

__all__ = [
    # Metrics
    "AnomalyMetrics",
    "compute_reconstruction_metrics",
    "compute_classification_metrics",
    # Threshold
    "ThresholdSelector",
    "compute_optimal_threshold",
    # Visualization
    "plot_reconstruction_error_distribution",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_confusion_matrix",
    "plot_reconstructions",
    "plot_latent_space",
    "plot_training_history",
]
