"""
Transfer Function Visualization
==============================

Visualization utilities for comparing normal and anomalous transfer functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Union
import warnings

from ..utils.logger import get_logger
from .deck_style import apply_deck_style, INK, ACCENT

logger = get_logger(__name__)

apply_deck_style()


def load_transfer_function(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load transfer function data from a simulation file.

    Args:
        filepath: Path to the .txt file containing transfer function data

    Returns:
        Tuple of (frequency, amplitude, phase) arrays
        - frequency: in Hz
        - amplitude: in dB
        - phase: in degrees
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Load data - skip header row, whitespace-separated
        data = np.loadtxt(filepath, skiprows=1)

        if data.shape[1] < 3:
            raise ValueError(
                f"File {filepath} must have at least 3 columns (freq, amp, phase)"
            )

        frequency = data[:, 0]
        amplitude = data[:, 1]
        phase = data[:, 2]

        logger.debug(
            f"Loaded transfer function from {filepath}: {len(frequency)} points"
        )

        return frequency, amplitude, phase

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise


def plot_amplitude(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    label: str = "",
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    **kwargs,
) -> plt.Axes:
    """
    Plot amplitude vs frequency.

    Args:
        frequency: Frequency array in Hz
        amplitude: Amplitude array in dB
        label: Plot label
        ax: Matplotlib axes to plot on (creates new if None)
        color: Line color
        linestyle: Line style
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(
        frequency, amplitude, label=label, color=color, linestyle=linestyle, **kwargs
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title("Transfer Function Amplitude")
    ax.grid(True, axis="both")

    if label:
        ax.legend()

    return ax


def plot_phase(
    frequency: np.ndarray,
    phase: np.ndarray,
    label: str = "",
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    linestyle: str = "-",
    **kwargs,
) -> plt.Axes:
    """
    Plot phase vs frequency.

    Args:
        frequency: Frequency array in Hz
        phase: Phase array in degrees
        label: Plot label
        ax: Matplotlib axes to plot on (creates new if None)
        color: Line color
        linestyle: Line style
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogx(
        frequency, phase, label=label, color=color, linestyle=linestyle, **kwargs
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (degrees)")
    ax.set_title("Transfer Function Phase")
    ax.grid(True, axis="both")

    if label:
        ax.legend()

    return ax


def plot_transfer_function(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    label: str = "",
    figsize: Tuple[int, int] = (12, 5),
    **kwargs,
) -> plt.Figure:
    """
    Plot both amplitude and phase vs frequency in subplots.

    Args:
        frequency: Frequency array in Hz
        amplitude: Amplitude array in dB
        phase: Phase array in degrees
        label: Plot label
        figsize: Figure size
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Amplitude plot
    plot_amplitude(frequency, amplitude, label, ax=ax1, **kwargs)
    ax1.set_title(f"Amplitude - {label}" if label else "Amplitude")

    # Phase plot
    plot_phase(frequency, phase, label, ax=ax2, **kwargs)
    ax2.set_title(f"Phase - {label}" if label else "Phase")

    plt.tight_layout()
    return fig


def compare_transfer_functions(
    normal_files: List[Union[str, Path]],
    anomaly_files: List[Union[str, Path]],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    max_files: Optional[int] = None,
    stack_normals: bool = True,
    stack_anomalies: bool = True,
) -> plt.Figure:
    """
    Compare normal vs anomalous transfer functions with stacking.

    Args:
        normal_files: List of paths to normal transfer function files
        anomaly_files: List of paths to anomalous transfer function files
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        max_files: Maximum number of files to plot from each category
        stack_normals: Whether to stack all normal functions on same plot
        stack_anomalies: Whether to stack all anomaly functions on same plot

    Returns:
        Matplotlib figure object
    """
    if max_files:
        normal_files = normal_files[:max_files]
        anomaly_files = anomaly_files[:max_files]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Colors
    normal_color = INK
    anomaly_color = ACCENT
    normal_alpha = 0.3 if stack_normals and len(normal_files) > 1 else 1.0
    anomaly_alpha = 0.3 if stack_anomalies and len(anomaly_files) > 1 else 1.0

    # Plot normal functions
    normal_count = 0
    for i, filepath in enumerate(normal_files):
        try:
            freq, amp, phase = load_transfer_function(filepath)

            if stack_normals:
                label = "Normal" if i == 0 else ""
            else:
                label = f"Normal {i + 1}"

            plot_amplitude(
                freq, amp, label, ax=ax1, color=normal_color, alpha=normal_alpha
            )
            plot_phase(
                freq, phase, label, ax=ax2, color=normal_color, alpha=normal_alpha
            )
            normal_count += 1

        except Exception as e:
            logger.warning(f"Failed to load normal file {filepath}: {e}")
            continue

    # Plot anomalous functions
    anomaly_count = 0
    for i, filepath in enumerate(anomaly_files):
        try:
            freq, amp, phase = load_transfer_function(filepath)

            if stack_anomalies:
                label = "Anomaly" if i == 0 else ""
            else:
                label = f"Anomaly {i + 1}"

            plot_amplitude(
                freq, amp, label, ax=ax3, color=anomaly_color, alpha=anomaly_alpha
            )
            plot_phase(
                freq, phase, label, ax=ax4, color=anomaly_color, alpha=anomaly_alpha
            )
            anomaly_count += 1

        except Exception as e:
            logger.warning(f"Failed to load anomaly file {filepath}: {e}")
            continue

    # Set titles with counts
    ax1.set_title(f"Normal Functions (n={normal_count}) - Amplitude")
    ax2.set_title(f"Normal Functions (n={normal_count}) - Phase")
    ax3.set_title(f"Anomalous Functions (n={anomaly_count}) - Amplitude")
    ax4.set_title(f"Anomalous Functions (n={anomaly_count}) - Phase")

    # Adjust legends
    for ax in [ax1, ax2]:
        if ax.get_legend():
            ax.legend(loc="best", fontsize="small")

    for ax in [ax3, ax4]:
        if ax.get_legend():
            ax.legend(loc="best", fontsize="small")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def create_comparison_plot(
    data_dir: Union[str, Path],
    normal_pattern: str = "*_+0__*_+0__*_+0__*_+0__*_+0.txt",
    anomaly_pattern: str = "*.txt",
    exclude_normal: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    max_files: int = 5,
    stack_normals: bool = True,
    stack_anomalies: bool = True,
) -> plt.Figure:
    """
    Create a comparison plot by automatically finding normal and anomalous files.

    Normal files are identified as those with 0% parameter variation (all +0 values).
    Anomalous files are all other .txt files with parameter variations.

    Args:
        data_dir: Directory containing transfer function files
        normal_pattern: Glob pattern for normal files (0% variation)
        anomaly_pattern: Glob pattern for anomaly files
        exclude_normal: Whether to exclude normal files from anomaly search
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        max_files: Maximum files to plot from each category
        stack_normals: Whether to stack normal functions
        stack_anomalies: Whether to stack anomalous functions

    Returns:
        Matplotlib figure object
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Resolve version directory and find the raw simulation files directory
    from ..data.loader import resolve_data_dir
    resolved_dir = resolve_data_dir(data_dir)
    txt_dir = resolved_dir / "txts" if (resolved_dir / "txts").is_dir() else resolved_dir

    # Find normal files (0% variation - all parameters at +0)
    normal_files = list(txt_dir.glob(normal_pattern))
    logger.info(f"Found {len(normal_files)} normal files (0% variation)")

    # Find anomaly files (parameter variations)
    all_files = list(txt_dir.glob(anomaly_pattern))
    if exclude_normal:
        anomaly_files = [f for f in all_files if f not in normal_files]
    else:
        anomaly_files = all_files

    logger.info(f"Found {len(anomaly_files)} anomaly files (with variations)")

    if not normal_files:
        warnings.warn("No normal files found (0% variation)")
    if not anomaly_files:
        warnings.warn("No anomaly files found")

    return compare_transfer_functions(
        normal_files=normal_files,
        anomaly_files=anomaly_files,
        save_path=save_path,
        show_plot=show_plot,
        max_files=max_files,
        stack_normals=stack_normals,
        stack_anomalies=stack_anomalies,
    )


# Example usage
if __name__ == "__main__":
    # Example: Compare normal (0% variation) vs anomalous transfer functions
    data_dir = Path("../../data/simulation_results")

    if data_dir.exists():
        fig = create_comparison_plot(
            data_dir=data_dir,
            save_path="transfer_function_comparison.png",
            max_files=5,
            stack_normals=True,
            stack_anomalies=True,
        )
        print("Stacked comparison plot created and saved.")
    else:
        print(f"Data directory not found: {data_dir}")
