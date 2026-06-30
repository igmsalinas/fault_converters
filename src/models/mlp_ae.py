"""
MLP Autoencoder
===============

Multi-Layer Perceptron (Dense) Autoencoder for transfer function anomaly
detection.  The simplest architecture in the model zoo — input is flattened
and processed through fully-connected layers, making it a strong
baseline for comparison against convolutional, recurrent, and
attention-based autoencoders.
"""

import keras
from keras import layers, Model
import numpy as np
from typing import Tuple, List

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MLPAutoencoder(BaseAutoencoder):
    """
    Multi-Layer Perceptron Autoencoder.

    Architecture
    ------------
    - Encoder: Flatten → Dense blocks with optional BatchNorm + Dropout
    - Bottleneck: Dense latent representation
    - Decoder: Mirror Dense blocks → Reshape to original input shape
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        encoder_units: List[int] = [256, 128, 64],
        decoder_units: List[int] = [64, 128, 256],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        name: str = "mlp_ae",
    ):
        """
        Initialize MLP Autoencoder.

        Args:
            input_shape: Input shape (seq_len, n_features)
            latent_dim: Latent space dimension
            encoder_units: List of dense layer sizes for the encoder
            decoder_units: List of dense layer sizes for the decoder
                           (should mirror encoder_units in reverse)
            dropout_rate: Dropout rate after each dense block
            use_batch_norm: Whether to use batch normalization
            activation: Activation function (relu, gelu, swish, mish, elu, …)
            name: Model name
        """
        super().__init__(input_shape, latent_dim, name)

        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _dense_block(
        self,
        x,
        units: int,
        name_prefix: str,
    ):
        """Create a dense block: Dense → (BatchNorm) → Activation → (Dropout)."""
        x = layers.Dense(units, name=f"{name_prefix}_dense")(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

        x = layers.Activation(self.activation, name=f"{name_prefix}_act")(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(x)

        return x

    # -----------------------------------------------------------------
    # Encoder / Decoder
    # -----------------------------------------------------------------

    def _build_encoder(self) -> Model:
        """Build encoder network."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")

        # Flatten the (seq_len, n_features) input to a 1-D vector
        x = layers.Flatten(name="enc_flatten")(inputs)

        # Dense blocks
        for i, units in enumerate(self.encoder_units):
            x = self._dense_block(x, units, name_prefix=f"enc_block{i}")

        # Project to latent space
        latent = layers.Dense(self.latent_dim, name="latent")(x)

        encoder = Model(inputs, latent, name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """Build decoder network."""
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")
        x = latent_inputs

        # Dense blocks
        for i, units in enumerate(self.decoder_units):
            x = self._dense_block(x, units, name_prefix=f"dec_block{i}")

        # Project back to original flattened size and reshape
        flat_size = int(np.prod(self.input_shape))
        x = layers.Dense(flat_size, activation="linear", name="dec_output_dense")(x)
        outputs = layers.Reshape(self.input_shape, name="decoder_output")(x)

        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    # -----------------------------------------------------------------
    # Config serialisation
    # -----------------------------------------------------------------

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "encoder_units": self.encoder_units,
                "decoder_units": self.decoder_units,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation,
            }
        )
        return config
