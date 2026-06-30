"""
Conv1D Autoencoder
==================

1D Convolutional Autoencoder for transfer function anomaly detection.
"""

import keras
from keras import layers, Model
from typing import Tuple, List

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Conv1DAutoencoder(BaseAutoencoder):
    """
    1D Convolutional Autoencoder.

    Architecture:
    - Encoder: Conv1D blocks with increasing filters
    - Bottleneck: Dense latent representation
    - Decoder: Conv1DTranspose blocks with decreasing filters
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        filters: List[int] = [32, 64, 128],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        name: str = "conv1d_ae",
    ):
        """
        Initialize Conv1D Autoencoder.

        Args:
            input_shape: Input shape (seq_len, n_features)
            latent_dim: Latent space dimension
            filters: List of filter sizes for each conv block
            kernel_size: Convolution kernel size
            pool_size: Pooling size
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function
            name: Model name
        """
        super().__init__(input_shape, latent_dim, name)

        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        # Calculate output shape after encoding
        self._encoder_output_shape = None

    def _conv_block(
        self,
        x,
        filters: int,
        kernel_size: int,
        name_prefix: str,
    ):
        """Create a convolutional block."""
        x = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            name=f"{name_prefix}_conv",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

        x = layers.Activation(self.activation, name=f"{name_prefix}_act")(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(x)

        x = layers.MaxPooling1D(self.pool_size, name=f"{name_prefix}_pool")(x)

        return x

    def _deconv_block(
        self,
        x,
        filters: int,
        kernel_size: int,
        name_prefix: str,
    ):
        """Create a deconvolutional (transpose conv) block."""
        x = layers.UpSampling1D(self.pool_size, name=f"{name_prefix}_upsample")(x)

        x = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            name=f"{name_prefix}_conv",
        )(x)

        if self.use_batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

        x = layers.Activation(self.activation, name=f"{name_prefix}_act")(x)

        if self.dropout_rate > 0:
            x = layers.Dropout(self.dropout_rate, name=f"{name_prefix}_drop")(x)

        return x

    def _build_encoder(self) -> Model:
        """Build encoder network."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")
        x = inputs

        # Convolutional blocks
        for i, filters in enumerate(self.filters):
            x = self._conv_block(x, filters, self.kernel_size, f"enc_block{i}")

        # Store shape before flattening
        self._encoder_output_shape = x.shape[1:]

        # Flatten and project to latent space
        x = layers.Flatten(name="enc_flatten")(x)
        latent = layers.Dense(self.latent_dim, name="latent")(x)

        encoder = Model(inputs, latent, name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """Build decoder network."""
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        # Calculate flattened size
        flat_size = int(self._encoder_output_shape[0] * self._encoder_output_shape[1])

        # Project and reshape
        x = layers.Dense(flat_size, name="dec_dense")(latent_inputs)
        x = layers.Reshape(self._encoder_output_shape, name="dec_reshape")(x)

        # Deconvolutional blocks (reverse order)
        for i, filters in enumerate(reversed(self.filters[:-1])):
            x = self._deconv_block(x, filters, self.kernel_size, f"dec_block{i}")

        # Final upsampling to match input size
        x = layers.UpSampling1D(self.pool_size, name="dec_final_upsample")(x)

        # Adjust sequence length if needed
        target_len = self.input_shape[0]
        current_len = x.shape[1]

        if current_len != target_len:
            # Use cropping or padding
            if current_len > target_len:
                crop = (current_len - target_len) // 2
                x = layers.Cropping1D((crop, current_len - target_len - crop))(x)
            else:
                pad = target_len - current_len
                x = layers.ZeroPadding1D((pad // 2, pad - pad // 2))(x)

        # Output layer
        outputs = layers.Conv1D(
            self.input_shape[1],
            self.kernel_size,
            padding="same",
            activation="linear",
            name="decoder_output",
        )(x)

        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
                "activation": self.activation,
            }
        )
        return config
