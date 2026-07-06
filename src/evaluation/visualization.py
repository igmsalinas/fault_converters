"""
Visualization
=============

Plotting utilities for anomaly detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, Dict, List
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns

from ..utils.logger import get_logger
from ..visualization.deck_style import (
    apply_deck_style,
    style_axis,
    text_on,
    INK,
    ACCENT,
    MUTED,
    CATEGORICAL,
    MONO_CMAP,
)

logger = get_logger(__name__)

apply_deck_style()


def set_style():
    """Apply the deck's editorial plotting style."""
    apply_deck_style()
    plt.rcParams.update({"figure.figsize": (10, 6)})


def plot_reconstruction_error_distribution(
    normal_errors: np.ndarray,
    anomaly_errors: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Reconstruction Error Distribution",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot reconstruction error distributions.

    Args:
        normal_errors: Errors for normal samples
        anomaly_errors: Errors for anomaly samples
        threshold: Threshold line to plot
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms
    ax.hist(
        normal_errors, bins=50, alpha=0.65, label="Normal", color=INK, density=True
    )
    ax.hist(
        anomaly_errors, bins=50, alpha=0.65, label="Anomaly", color=ACCENT, density=True
    )

    # Plot threshold
    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="#3f5a52",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {threshold:.4f}",
        )

    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")

    return fig


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot ROC curve.

    Args:
        labels: True labels
        scores: Anomaly scores
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    fpr, tpr, _ = roc_curve(labels, scores)
    from sklearn.metrics import auc

    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, linestyle="--")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot Precision-Recall curve.

    Args:
        labels: True labels
        scores: Anomaly scores
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    precision, recall, _ = precision_recall_curve(labels, scores)
    from sklearn.metrics import average_precision_score

    avg_precision = average_precision_score(labels, scores)

    ax.plot(
        recall,
        precision,
        color=ACCENT,
        lw=2,
        label=f"PR curve (AP = {avg_precision:.4f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot confusion matrix.

    Args:
        labels: True labels
        predictions: Predicted labels
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = confusion_matrix(labels, predictions)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=MONO_CMAP,
        cbar=False,
        linewidths=1.4,
        linecolor="#fbf9f5",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
    )

    vmax = cm.max() if cm.max() > 0 else 1
    for txt, val in zip(ax.texts, cm.flatten()):
        txt.set_color(text_on(MONO_CMAP(val / vmax)))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.tick_params(length=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reconstructions(
    original: np.ndarray,
    reconstructed: np.ndarray,
    num_samples: int = 5,
    feature_names: Optional[List[str]] = None,
    title: str = "Reconstruction Comparison",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot original vs reconstructed signals.

    Args:
        original: Original data
        reconstructed: Reconstructed data
        num_samples: Number of samples to plot
        feature_names: Names of features
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()

    n_features = original.shape[-1]
    feature_names = feature_names or [f"Feature {i + 1}" for i in range(n_features)]

    fig, axes = plt.subplots(
        num_samples, n_features, figsize=(5 * n_features, 3 * num_samples)
    )

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if n_features == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        for j in range(n_features):
            ax = axes[i, j]
            ax.plot(original[i, :, j], label="Original", color=INK, alpha=0.8)
            ax.plot(
                reconstructed[i, :, j],
                label="Reconstructed",
                color=ACCENT,
                alpha=0.85,
                linestyle="--",
            )

            if i == 0:
                ax.set_title(feature_names[j])
            if j == 0:
                ax.set_ylabel(f"Sample {i + 1}")
            if i == num_samples - 1:
                ax.set_xlabel("Time Step")
            if i == 0 and j == 0:
                ax.legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latent_space(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    title: str = "Latent Space Visualization",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot 2D visualization of latent space.

    Args:
        latent_vectors: Latent representations
        labels: Labels (0=normal, 1=anomaly)
        method: Dimensionality reduction method (pca, tsne)
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reduce to 2D
    if latent_vectors.shape[1] > 2:
        if method == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=2)
        elif method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        reduced = reducer.fit_transform(latent_vectors)
    else:
        reduced = latent_vectors

    # Plot
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    ax.scatter(
        reduced[normal_mask, 0],
        reduced[normal_mask, 1],
        c="#4a6670",
        alpha=0.55,
        label="Normal",
        s=20,
        edgecolors="none",
    )
    ax.scatter(
        reduced[anomaly_mask, 0],
        reduced[anomaly_mask, 1],
        c=ACCENT,
        alpha=0.55,
        label="Anomaly",
        s=20,
        edgecolors="none",
    )

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot training history.

    Args:
        history: Training history dictionary
        metrics: Metrics to plot (None for all)
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    set_style()

    metrics = metrics or list(history.keys())
    n_metrics = len([m for m in metrics if not m.startswith("val_")])

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    plot_idx = 0
    for metric in metrics:
        if metric.startswith("val_"):
            continue

        ax = axes[plot_idx]

        # Training metric
        ax.plot(history[metric], label=f"Train {metric}", color=INK)

        # Validation metric
        val_metric = f"val_{metric}"
        if val_metric in history:
            ax.plot(history[val_metric], label=f"Val {metric}", color=ACCENT)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()

        plot_idx += 1

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_evaluation_report(
    model,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    output_dir: str,
    model_name: str = "model",
) -> None:
    """
    Create comprehensive evaluation report with all visualizations.

    Args:
        model: Trained model
        test_data: Test data
        test_labels: Test labels
        threshold: Anomaly threshold
        output_dir: Output directory
        model_name: Model name for file naming
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions and errors
    errors = model.compute_reconstruction_error(test_data)
    predictions = (errors > threshold).astype(int)

    # Separate normal and anomaly errors
    normal_errors = errors[test_labels == 0]
    anomaly_errors = errors[test_labels == 1]

    # 1. Error distribution
    plot_reconstruction_error_distribution(
        normal_errors,
        anomaly_errors,
        threshold,
        title=f"{model_name} - Reconstruction Error Distribution",
        save_path=str(output_dir / f"{model_name}_error_distribution.png"),
    )
    plt.close()

    # 2. ROC curve
    plot_roc_curve(
        test_labels,
        errors,
        title=f"{model_name} - ROC Curve",
        save_path=str(output_dir / f"{model_name}_roc_curve.png"),
    )
    plt.close()

    # 3. PR curve
    plot_precision_recall_curve(
        test_labels,
        errors,
        title=f"{model_name} - Precision-Recall Curve",
        save_path=str(output_dir / f"{model_name}_pr_curve.png"),
    )
    plt.close()

    # 4. Confusion matrix
    plot_confusion_matrix(
        test_labels,
        predictions,
        title=f"{model_name} - Confusion Matrix",
        save_path=str(output_dir / f"{model_name}_confusion_matrix.png"),
    )
    plt.close()

    # 5. Reconstructions
    reconstructed = model.predict(test_data[:5])
    plot_reconstructions(
        test_data[:5],
        reconstructed,
        num_samples=5,
        feature_names=["Amplitude", "Phase"],
        title=f"{model_name} - Reconstruction Comparison",
        save_path=str(output_dir / f"{model_name}_reconstructions.png"),
    )
    plt.close()

    # 6. Latent space
    latent = model.encode(test_data)
    plot_latent_space(
        latent,
        test_labels,
        method="pca",
        title=f"{model_name} - Latent Space (PCA)",
        save_path=str(output_dir / f"{model_name}_latent_space.png"),
    )
    plt.close()

    logger.info(f"Evaluation report saved to {output_dir}")
