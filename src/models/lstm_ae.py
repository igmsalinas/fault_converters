"""
LSTM Autoencoder
================

LSTM-based Autoencoder for transfer function anomaly detection.
"""

import keras
from keras import layers, Model
from typing import Tuple, List

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


class LSTMAutoencoder(BaseAutoencoder):
    """
    LSTM Autoencoder.

    Architecture:
    - Encoder: Stacked LSTM layers
    - Bottleneck: Dense latent representation
    - Decoder: Stacked LSTM layers with RepeatVector
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.1,
        recurrent_dropout: float = 0.0,
        bidirectional: bool = False,
        activation: str = "relu",
        name: str = "lstm_ae",
    ):
        """
        Initialize LSTM Autoencoder.

        Args:
            input_shape: Input shape (seq_len, n_features)
            latent_dim: Latent space dimension
            lstm_units: List of LSTM units for each layer
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            bidirectional: Whether to use bidirectional LSTMs
            activation: Activation function for dense layers (relu, gelu, swish, mish, elu, leaky_relu)
            name: Model name
        """
        super().__init__(input_shape, latent_dim, name)

        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        self.activation = activation

    def _lstm_layer(
        self,
        units: int,
        return_sequences: bool,
        name: str,
    ):
        """Create LSTM layer with optional bidirectional wrapper and separate dropout."""
        # Use simple LSTM instantiation without inner dropout/recurrent_dropout
        # to ensure Keras uses the optimized CuDNN implementation on GPU.
        lstm = layers.LSTM(
            units,
            return_sequences=return_sequences,
            name=name,
        )

        if self.bidirectional:
            lstm = layers.Bidirectional(lstm, name=f"bi_{name}")

        if self.dropout_rate > 0.0:
            return keras.Sequential(
                [lstm, layers.Dropout(self.dropout_rate, name=f"drop_{name}")],
                name=f"seq_{name}",
            )
        return lstm

    def _build_encoder(self) -> Model:
        """Build encoder network."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")
        x = inputs

        # Stacked LSTM layers
        for i, units in enumerate(self.lstm_units[:-1]):
            x = self._lstm_layer(units, return_sequences=True, name=f"enc_lstm{i}")(x)

        # Final LSTM - no sequence output
        x = self._lstm_layer(
            self.lstm_units[-1],
            return_sequences=False,
            name=f"enc_lstm{len(self.lstm_units) - 1}",
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
        x = layers.Dense(self.lstm_units[-1], name="dec_dense")(latent_inputs)
        x = layers.Activation(self.activation, name="dec_act")(x)
        x = layers.RepeatVector(seq_len, name="dec_repeat")(x)

        # Stacked LSTM layers (reversed order)
        for i, units in enumerate(reversed(self.lstm_units)):
            x = self._lstm_layer(units, return_sequences=True, name=f"dec_lstm{i}")(x)

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
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "recurrent_dropout": self.recurrent_dropout,
                "bidirectional": self.bidirectional,
                "activation": self.activation,
            }
        )
        return config
