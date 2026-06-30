"""
Visualization utilities for power converter transfer function data.

This module provides functions to visualize and compare transfer functions
from simulation data, including amplitude and phase plots for normal vs
anomalous conditions. Functions support stacking multiple transfer functions
to visualize parameter variations and identify anomaly patterns.
"""

from .plotter import (
    load_transfer_function,
    plot_amplitude,
    plot_phase,
    plot_transfer_function,
    compare_transfer_functions,
    create_comparison_plot,
)

__all__ = [
    "load_transfer_function",
    "plot_amplitude",
    "plot_phase",
    "plot_transfer_function",
    "compare_transfer_functions",
    "create_comparison_plot",
]
