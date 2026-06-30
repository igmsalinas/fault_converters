"""
Keras Callbacks
===============

Custom and configured callbacks for training.
"""

from keras import callbacks
import numpy as np
from pathlib import Path
from typing import List

from ..utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStoppingWithRestore(callbacks.Callback):
    """Early stopping that restores best weights."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0,
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_weights = None
        self.best_value = np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best_value - self.min_delta:
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        logger.info(
                            f"Restoring best weights from epoch {epoch - self.wait}"
                        )
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            logger.info(f"Early stopping at epoch {self.stopped_epoch + 1}")


class MetricsLogger(callbacks.Callback):
    """Log metrics to file during training."""

    def __init__(self, log_file: str):
        super().__init__()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, "a") as f:
            metrics = ", ".join([f"{k}: {v:.6f}" for k, v in logs.items()])
            f.write(f"Epoch {epoch + 1}: {metrics}\n")


class ReconstructionVisualizer(callbacks.Callback):
    """Periodically visualize reconstructions during training."""

    def __init__(
        self,
        validation_data: np.ndarray,
        output_dir: str,
        frequency: int = 10,
        num_samples: int = 5,
    ):
        super().__init__()
        self.validation_data = validation_data
        self.output_dir = Path(output_dir)
        self.frequency = frequency
        self.num_samples = num_samples
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.frequency != 0:
            return

        try:
            import matplotlib.pyplot as plt

            # Get reconstructions
            samples = self.validation_data[: self.num_samples]
            reconstructed = self.model.predict(samples, verbose=0)

            # Plot
            fig, axes = plt.subplots(
                self.num_samples, 2, figsize=(12, 3 * self.num_samples)
            )

            for i in range(self.num_samples):
                # Original
                axes[i, 0].plot(samples[i])
                axes[i, 0].set_title(f"Sample {i + 1} - Original")

                # Reconstruction
                axes[i, 1].plot(reconstructed[i])
                axes[i, 1].set_title(f"Sample {i + 1} - Reconstructed")

            plt.tight_layout()
            plt.savefig(
                self.output_dir / f"reconstruction_epoch_{epoch + 1}.png", dpi=100
            )
            plt.close()

        except ImportError:
            pass


def get_callbacks(
    checkpoint_dir: str,
    log_dir: str,
    early_stopping: bool = True,
    patience: int = 15,
    min_delta: float = 1e-5,
    reduce_lr: bool = True,
    lr_patience: int = 10,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    tensorboard: bool = True,
) -> List[callbacks.Callback]:
    """
    Get configured callbacks for training.

    Args:
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        early_stopping: Enable early stopping
        patience: Early stopping patience
        min_delta: Minimum improvement
        reduce_lr: Enable learning rate reduction
        lr_patience: LR reduction patience
        lr_factor: LR reduction factor
        min_lr: Minimum learning rate
        tensorboard: Enable TensorBoard logging

    Returns:
        List of configured callbacks
    """
    callback_list = []

    # Model checkpoint - save best
    checkpoint_path = Path(checkpoint_dir) / "best"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callback_list.append(
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "model.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    # Early stopping
    if early_stopping:
        callback_list.append(
            EarlyStoppingWithRestore(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1,
            )
        )

    # Learning rate reduction
    if reduce_lr:
        callback_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=1,
            )
        )

    # TensorBoard
    if tensorboard:
        callback_list.append(
            callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
            )
        )

    # Metrics logger
    callback_list.append(MetricsLogger(str(Path(log_dir) / "metrics.log")))

    return callback_list
