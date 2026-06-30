"""
Logging Utilities
=================

Centralized logging configuration for the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global logger registry
_loggers = {}


def setup_logger(
    name: str = "power_converter_anomaly",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file

    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "power_converter_anomaly") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class TrainingLogger:
    """Logger for training metrics and progress."""

    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.logger = setup_logger(
            f"training_{experiment_name}",
            str(self.log_dir),
        )
        self.metrics_history = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        """Log metrics for an epoch."""
        # Convert tensors to python floats if necessary
        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "lr": float(lr) if lr is not None else None,
        }
        if extra_metrics:
            converted_extra = {
                k: float(v) if hasattr(v, "__float__") else v
                for k, v in extra_metrics.items()
            }
            metrics.update(converted_extra)

        self.metrics_history.append(metrics)

        msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.6f}"
        if lr is not None:
            msg += f" | LR: {lr:.2e}"
        if extra_metrics:
            for k, v in extra_metrics.items():
                if isinstance(v, float):
                    msg += f" | {k}: {v:.4f}"

        self.logger.info(msg)

    def log_message(self, message: str, level: str = "info") -> None:
        """Log a custom message."""
        getattr(self.logger, level)(message)

    def save_metrics(self, path: Optional[str] = None) -> None:
        """Save metrics history to JSON."""
        import json
        import numpy as np

        if path is None:
            path = self.log_dir / f"{self.experiment_name}_metrics.json"

        def convert_to_serializable(obj):
            if hasattr(obj, "numpy"):  # tf tensors
                obj = obj.numpy()
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable_history = [
            convert_to_serializable(m) for m in self.metrics_history
        ]

        with open(path, "w") as f:
            json.dump(serializable_history, f, indent=2)
