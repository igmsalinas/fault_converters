"""
Trainer Class
=============

Main training orchestration for autoencoder models.
"""

import keras
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json
import time

from .base_trainer import BaseTrainer, BaseTrainingConfig

from ..models.base import BaseAutoencoder
from ..models.conv1d_ae import Conv1DAutoencoder
from ..models.lstm_ae import LSTMAutoencoder
from ..models.gru_ae import GRUAutoencoder
from ..models.mlp_ae import MLPAutoencoder
from ..models.vae import VariationalAutoencoder
from ..models.transformer_ae import TransformerAutoencoder
from ..data.dataset import PowerConverterDataset
from ..utils.logger import get_logger
from .callbacks import get_callbacks

logger = get_logger(__name__)


@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training configuration."""

    loss: str = "mse"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = super().to_dict()
        d["loss"] = self.loss
        return d


MODEL_REGISTRY = {
    "conv1d_ae": Conv1DAutoencoder,
    "lstm_ae": LSTMAutoencoder,
    "gru_ae": GRUAutoencoder,
    "mlp_ae": MLPAutoencoder,
    "vae": VariationalAutoencoder,
    "transformer_ae": TransformerAutoencoder,
}


class Trainer(BaseTrainer):
    """
    Training orchestrator for autoencoder models.

    Handles model creation, training, and evaluation with GPU support.
    """

    def __init__(
        self,
        config: TrainingConfig,
        experiment_name: str = "experiment",
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            experiment_name: Name for this experiment
        """
        super().__init__(config, experiment_name)

        # Model instance
        self.model: Optional[BaseAutoencoder] = None

    def create_model(
        self,
        model_type: str,
        input_shape: tuple,
        **model_kwargs,
    ) -> BaseAutoencoder:
        """
        Create autoencoder model.

        Args:
            model_type: Type of model (conv1d_ae, lstm_ae, vae, transformer_ae)
            input_shape: Input shape (seq_len, n_features)
            **model_kwargs: Model-specific arguments

        Returns:
            Built model instance
        """
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = MODEL_REGISTRY[model_type]
        self.model = model_class(input_shape=input_shape, **model_kwargs)
        self.model.build()

        logger.info(f"Created {model_type} model")

        return self.model

    def compile_model(self) -> None:
        """Compile model with configured optimizer and loss."""
        if self.model is None:
            raise RuntimeError("No model created. Call create_model() first.")

        self.model.compile(
            optimizer=self.config.optimizer,
            learning_rate=self.config.learning_rate,
            loss=self.config.loss,
        )

        logger.info(
            f"Model compiled with {self.config.optimizer} optimizer, "
            f"lr={self.config.learning_rate}, loss={self.config.loss}"
        )

    def train(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """
        Train model.

        Args:
            train_data: Training data
            val_data: Validation data
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            raise RuntimeError("No model created.")

        # Get callbacks
        callbacks = get_callbacks(
            checkpoint_dir=str(self.checkpoint_dir),
            log_dir=str(self.log_dir),
            early_stopping=self.config.early_stopping,
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            reduce_lr=self.config.reduce_lr,
            lr_patience=self.config.lr_patience,
            lr_factor=self.config.lr_factor,
            min_lr=self.config.min_lr,
        )

        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()

        self.history = self.model.fit(
            train_data=train_data,
            val_data=val_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s")

        # Log final metrics
        self._log_final_metrics()

        return self.history

    def _log_final_metrics(self) -> None:
        """Log final training metrics."""
        if self.history is None:
            return

        final_loss = (
            self.history.history.get("loss", [])[-1]
            if "loss" in self.history.history
            else None
        )
        final_val_loss = (
            self.history.history.get("val_loss", [])[-1]
            if "val_loss" in self.history.history
            else None
        )

        logger.info(f"Final loss: {final_loss:.6f}")
        if final_val_loss is not None:
            logger.info(f"Final val_loss: {final_val_loss:.6f}")

        # Save history
        history_path = self.log_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(
                {
                    k: [float(v) for v in vals]
                    for k, vals in self.history.history.items()
                },
                f,
            )

    def save_experiment(self) -> None:
        """Save complete experiment state."""
        # Save model
        if self.model is not None:
            self.model.save(str(self.checkpoint_dir / "final"))

        # Save config
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model config
        if self.model is not None:
            model_config_path = self.log_dir / "model_config.json"
            with open(model_config_path, "w") as f:
                json.dump(self.model.get_config(), f, indent=2)

        logger.info(f"Experiment saved to {self.log_dir}")

    def load_best_model(self) -> None:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best"
        if best_path.exists():
            self.model.load(str(best_path))
            logger.info("Loaded best model checkpoint")
        else:
            logger.warning("No best checkpoint found")


def train_model(
    model_type: str,
    dataset: PowerConverterDataset,
    config: TrainingConfig,
    experiment_name: str,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a model.

    Args:
        model_type: Type of model
        dataset: Prepared dataset
        config: Training config
        experiment_name: Experiment name
        **model_kwargs: Model arguments

    Returns:
        Dictionary with training results
    """
    trainer = Trainer(config, experiment_name)

    # Create and compile model
    trainer.create_model(
        model_type=model_type,
        input_shape=dataset.input_shape,
        **model_kwargs,
    )
    trainer.compile_model()

    # Train
    history = trainer.train(
        train_data=dataset.train_data,
        val_data=dataset.val_data,
    )

    # Save
    trainer.save_experiment()

    return {
        "trainer": trainer,
        "history": history,
        "model": trainer.model,
    }
