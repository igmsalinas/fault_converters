"""
Base Autoencoder Class
======================

Abstract base class for all autoencoder models.
"""

import keras
from keras import Model
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseAutoencoder(ABC):
    """
    Abstract base class for autoencoder models.

    Provides common interface for training, evaluation, and anomaly detection.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        name: str = "autoencoder",
    ):
        """
        Initialize base autoencoder.

        Args:
            input_shape: Input shape (sequence_length, n_features)
            latent_dim: Dimension of latent space
            name: Model name
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.name = name

        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None

        self._is_built = False

    @abstractmethod
    def _build_encoder(self) -> Model:
        """Build encoder model."""
        pass

    @abstractmethod
    def _build_decoder(self) -> Model:
        """Build decoder model."""
        pass

    def build(self) -> "BaseAutoencoder":
        """Build full autoencoder model."""
        logger.info(f"Building {self.name} with input shape {self.input_shape}")

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Create full autoencoder
        inputs = keras.Input(shape=self.input_shape)
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)

        self.autoencoder = Model(inputs, decoded, name=self.name)
        self._is_built = True

        logger.info(f"Model built. Parameters: {self.autoencoder.count_params():,}")

        return self

    def compile(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        loss: str = "mse",
        **kwargs,
    ) -> "BaseAutoencoder":
        """
        Compile autoencoder model.

        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            **kwargs: Additional compile arguments
        """
        if not self._is_built:
            raise RuntimeError("Model not built. Call build() first.")

        if optimizer == "adam":
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "adamw":
            opt = keras.optimizers.AdamW(learning_rate=learning_rate)
        elif optimizer == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer

        self.autoencoder.compile(optimizer=opt, loss=loss, **kwargs)

        return self

    def fit(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[list] = None,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """
        Train autoencoder.

        Args:
            train_data: Training data
            val_data: Validation data
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: Keras callbacks
            verbose: Verbosity level

        Returns:
            Training history
        """
        if not self._is_built:
            raise RuntimeError("Model not built. Call build() first.")

        validation_data = (val_data, val_data) if val_data is not None else None

        history = self.autoencoder.fit(
            train_data,
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        return history

    def predict(self, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get reconstructions."""
        return self.autoencoder.predict(data, batch_size=batch_size, verbose=0)

    def encode(self, data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Encode data to latent space."""
        return self.encoder.predict(data, batch_size=batch_size, verbose=0)

    def decode(self, latent: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Decode from latent space."""
        return self.decoder.predict(latent, batch_size=batch_size, verbose=0)

    def compute_reconstruction_error(
        self,
        data: np.ndarray,
        reduction: str = "mean",
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.

        Args:
            data: Input data
            reduction: How to reduce error ("mean", "sum", "none")
            batch_size: Batch size for inference

        Returns:
            Reconstruction errors per sample
        """
        reconstructed = self.predict(data, batch_size=batch_size)
        error = np.square(data - reconstructed)

        if reduction == "none":
            return error
        elif reduction == "sum":
            return np.sum(error, axis=(1, 2))
        else:  # mean
            return np.mean(error, axis=(1, 2))

    def detect_anomalies(
        self,
        data: np.ndarray,
        threshold: float,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on reconstruction error.

        Args:
            data: Input data
            threshold: Anomaly threshold
            batch_size: Batch size for inference

        Returns:
            Tuple of (predictions, reconstruction_errors)
        """
        errors = self.compute_reconstruction_error(data, batch_size=batch_size)
        predictions = (errors > threshold).astype(int)
        return predictions, errors

    def summary(self) -> None:
        """Print model summary."""
        if self._is_built:
            self.autoencoder.summary()

    def save(self, path: str) -> None:
        """Save model weights."""
        if not self._is_built:
            raise RuntimeError("Model not built.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.autoencoder.save_weights(str(path / "autoencoder.weights.h5"))
        self.encoder.save_weights(str(path / "encoder.weights.h5"))
        self.decoder.save_weights(str(path / "decoder.weights.h5"))

        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "BaseAutoencoder":
        """Load model weights."""
        if not self._is_built:
            self.build()

        path = Path(path)
        self.autoencoder.load_weights(str(path / "autoencoder.weights.h5"))
        self.encoder.load_weights(str(path / "encoder.weights.h5"))
        self.decoder.load_weights(str(path / "decoder.weights.h5"))

        logger.info(f"Model loaded from {path}")
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "name": self.name,
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
        }
