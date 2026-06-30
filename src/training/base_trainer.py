"""
Base Trainer Class
==================

Provides common training orchestration functionality like GPU setup,
checkpoint management, and logging for all autoencoder trainers.
"""

import keras
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from ..utils.logger import get_logger, TrainingLogger

logger = get_logger(__name__)


@dataclass
class BaseTrainingConfig:
    """Base configuration for training."""

    # Training params
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "adam"

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-5

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # Logging
    log_dir: str = "logs"

    # Learning rate schedule
    reduce_lr: bool = True
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    # GPU
    use_gpu: bool = True
    mixed_precision: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "checkpoint_dir": self.checkpoint_dir,
            "save_best_only": self.save_best_only,
            "log_dir": self.log_dir,
            "reduce_lr": self.reduce_lr,
            "lr_patience": self.lr_patience,
            "lr_factor": self.lr_factor,
            "min_lr": self.min_lr,
            "use_gpu": self.use_gpu,
            "mixed_precision": self.mixed_precision,
        }


class BaseTrainer:
    """
    Base trainer providing common GPU, directory, and checkpoint management.
    """

    def __init__(
        self,
        config: BaseTrainingConfig,
        experiment_name: str = "experiment",
    ):
        """
        Initialize base trainer.

        Args:
            config: Base training configuration
            experiment_name: Name for this experiment
        """
        self.config = config
        self.experiment_name = experiment_name

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir) / experiment_name
        self.log_dir = Path(config.log_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training logger
        self.training_logger = TrainingLogger(str(self.log_dir), experiment_name)

        # History
        self.history: Optional[
            Union[keras.callbacks.History, Dict[str, List[float]]]
        ] = None

        # Setup GPU
        self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Configure GPU and mixed precision."""
        import tensorflow as tf

        if self.config.use_gpu:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning("No GPU found. Using CPU.")
        else:
            tf.config.set_visible_devices([], "GPU")
            logger.info("GPU disabled. Using CPU.")

        if self.config.mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("Mixed precision enabled.")
