"""
Transformer Autoencoder
=======================

Transformer-based Autoencoder with self-attention for transfer function anomaly detection.
"""

import keras
from keras import layers, Model
import numpy as np
from typing import Tuple

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


@keras.saving.register_keras_serializable(package="Transformer")
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding."""

    def __init__(self, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        d_model = input_shape[-1]

        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = np.zeros((self.max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = np.cos(position * div_term[: d_model // 2])

        self.pe = self.add_weight(
            name="pe",
            shape=(self.max_len, d_model),
            initializer=keras.initializers.Constant(pe),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        seq_len = keras.ops.shape(x)[1]
        return x + self.pe[:seq_len, :]


@keras.saving.register_keras_serializable(package="Transformer")
class TransformerBlock(layers.Layer):
    """Transformer encoder block with multi-head attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        ff_activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.ff_activation = ff_activation

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim),
                layers.Activation(ff_activation),
                layers.Dense(d_model),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerAutoencoder(BaseAutoencoder):
    """
    Transformer-based Autoencoder.

    Uses self-attention mechanism for capturing long-range dependencies.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        ff_activation: str = "gelu",
        name: str = "transformer_ae",
    ):
        """
        Initialize Transformer Autoencoder.

        Args:
            input_shape: Input shape (seq_len, n_features)
            latent_dim: Latent space dimension
            d_model: Transformer model dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of transformer blocks
            dropout_rate: Dropout rate
            ff_activation: Activation function for FFN (gelu, relu, swish, mish)
            name: Model name
        """
        super().__init__(input_shape, latent_dim, name)

        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.ff_activation = ff_activation

    def _build_encoder(self) -> Model:
        """Build transformer encoder."""
        seq_len, n_features = self.input_shape

        inputs = keras.Input(shape=self.input_shape, name="encoder_input")

        # Project to d_model
        x = layers.Dense(self.d_model)(inputs)
        x = PositionalEncoding(max_len=seq_len)(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.ff_dim,
                self.dropout_rate,
                ff_activation=self.ff_activation,
                name=f"enc_transformer_{i}",
            )(x)

        # Global pooling and latent projection
        x = layers.GlobalAveragePooling1D()(x)
        latent = layers.Dense(self.latent_dim, name="latent")(x)

        encoder = Model(inputs, latent, name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """Build transformer decoder."""
        seq_len, n_features = self.input_shape

        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        # Expand to sequence
        x = layers.Dense(seq_len * self.d_model)(latent_inputs)
        x = layers.Reshape((seq_len, self.d_model))(x)
        x = PositionalEncoding(max_len=seq_len)(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                self.d_model,
                self.num_heads,
                self.ff_dim,
                self.dropout_rate,
                ff_activation=self.ff_activation,
                name=f"dec_transformer_{i}",
            )(x)

        # Project to output
        outputs = layers.Dense(n_features, name="decoder_output")(x)

        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "ff_activation": self.ff_activation,
            }
        )
        return config
