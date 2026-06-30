"""
GRU Autoencoder
===============

GRU-based Autoencoder for transfer function anomaly detection.
Uses reset_after=True for CuDNN compatibility.
"""

import keras
from keras import layers, Model
from typing import Tuple, List

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GRUAutoencoder(BaseAutoencoder):
    """
    GRU Autoencoder.

    Architecture:
    - Encoder: Stacked GRU layers
    - Bottleneck: Dense latent representation
    - Decoder: Stacked GRU layers with RepeatVector
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        gru_units: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        bidirectional: bool = False,
        activation: str = "relu",
        name: str = "gru_ae",
    ):
        """
        Initialize GRU Autoencoder.

        Args:
            input_shape: Input shape (seq_len, n_features)
            latent_dim: Latent space dimension
            gru_units: List of GRU units for each layer
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional GRUs
            activation: Activation function for dense layers
            name: Model name
        """
        super().__init__(input_shape, latent_dim, name)

        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.activation = activation

    def _gru_layer(
        self,
        units: int,
        return_sequences: bool,
        name: str,
    ):
        """Create GRU layer with optional bidirectional wrapper and separate dropout."""
        # Use simple GRU instantiation with reset_after=True
        # to ensure Keras uses the optimized CuDNN implementation on GPU.
        gru = layers.GRU(
            units,
            return_sequences=return_sequences,
            reset_after=True,
            name=name,
        )

        if self.bidirectional:
            gru = layers.Bidirectional(gru, name=f"bi_{name}")

        if self.dropout_rate > 0.0:
            return keras.Sequential(
                [gru, layers.Dropout(self.dropout_rate, name=f"drop_{name}")],
                name=f"seq_{name}",
            )
        return gru

    def _build_encoder(self) -> Model:
        """Build encoder network."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")
        x = inputs

        # Stacked GRU layers
        for i, units in enumerate(self.gru_units[:-1]):
            x = self._gru_layer(units, return_sequences=True, name=f"enc_gru{i}")(x)

        # Final GRU - no sequence output
        x = self._gru_layer(
            self.gru_units[-1],
            return_sequences=False,
            name=f"enc_gru{len(self.gru_units) - 1}",
        )(x)

        # Project to latent space with activation
        x = layers.Dense(self.latent_dim * 2, name="latent_pre")(x)
        x = layers.Activation(self.activation, name="latent_act")(x)
        latent = layers.Dense(self.latent_dim, name="latent")(x)

        encoder = Model(inputs, latent, name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """Build decoder network."""
        seq_len = self.input_shape[0]
        n_features = self.input_shape[1]

        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        # Project and repeat with activation
        x = layers.Dense(self.gru_units[-1], name="dec_dense")(latent_inputs)
        x = layers.Activation(self.activation, name="dec_act")(x)
        x = layers.RepeatVector(seq_len, name="dec_repeat")(x)

        # Stacked GRU layers (reversed order)
        for i, units in enumerate(reversed(self.gru_units)):
            x = self._gru_layer(units, return_sequences=True, name=f"dec_gru{i}")(x)

        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(n_features),
            name="decoder_output",
        )(x)

        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "gru_units": self.gru_units,
                "dropout_rate": self.dropout_rate,
                "bidirectional": self.bidirectional,
                "activation": self.activation,
            }
        )
        return config
