"""
Variational Autoencoder
=======================

VAE for transfer function anomaly detection with KL divergence regularization.
"""

import keras
from keras import layers, Model
import tensorflow as tf
import numpy as np
from typing import Tuple, List

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


@keras.saving.register_keras_serializable(package="VAE")
class Sampling(layers.Layer):
    """Reparameterization trick sampling layer."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(BaseAutoencoder):
    """
    Variational Autoencoder (VAE).

    Uses reparameterization trick and KL divergence regularization.
    Anomaly score combines reconstruction error and latent probability.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        encoder_units: List[int] = [128, 64],
        decoder_units: List[int] = [64, 128],
        dropout_rate: float = 0.1,
        kl_weight: float = 1.0,
        use_conv: bool = True,
        filters: List[int] = [32, 64],
        kernel_size: int = 3,
        activation: str = "relu",
        name: str = "vae",
    ):
        """
        Initialize VAE.

        Args:
            input_shape: Input shape (seq_len, n_features)
            latent_dim: Latent space dimension
            encoder_units: Dense units for encoder
            decoder_units: Dense units for decoder
            dropout_rate: Dropout rate
            kl_weight: Weight for KL divergence loss
            use_conv: Whether to use conv layers
            filters: Conv filter sizes
            kernel_size: Conv kernel size
            activation: Activation function (relu, gelu, swish, mish, elu, leaky_relu)
            name: Model name
        """
        super().__init__(input_shape, latent_dim, name)

        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.dropout_rate = dropout_rate
        self.kl_weight = kl_weight
        self.use_conv = use_conv
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation

        self._flat_size = None
        self._conv_shape = None

    def _build_encoder(self) -> Model:
        """Build encoder with mean and log_var outputs."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")
        x = inputs

        if self.use_conv:
            for i, f in enumerate(self.filters):
                x = layers.Conv1D(f, self.kernel_size, padding="same")(x)
                x = layers.Activation(self.activation)(x)
                x = layers.MaxPooling1D(2)(x)
            self._conv_shape = x.shape[1:]

        x = layers.Flatten()(x)
        self._flat_size = x.shape[1]

        for units in self.encoder_units:
            x = layers.Dense(units)(x)
            x = layers.Activation(self.activation)(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)

        # Mean and log variance
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Sample
        z = Sampling()([z_mean, z_log_var])

        encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """Build decoder network."""
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")
        x = latent_inputs

        for units in self.decoder_units:
            x = layers.Dense(units)(x)
            x = layers.Activation(self.activation)(x)
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)

        if self.use_conv:
            x = layers.Dense(int(np.prod(self._conv_shape)))(x)
            x = layers.Activation(self.activation)(x)
            x = layers.Reshape(self._conv_shape)(x)

            for i, f in enumerate(reversed(self.filters)):
                x = layers.UpSampling1D(2)(x)
                x = layers.Conv1D(f, self.kernel_size, padding="same")(x)
                x = layers.Activation(self.activation)(x)

            # Adjust length
            target_len = self.input_shape[0]
            current_len = x.shape[1]
            if current_len != target_len:
                if current_len > target_len:
                    x = layers.Cropping1D((0, current_len - target_len))(x)
                else:
                    x = layers.ZeroPadding1D((0, target_len - current_len))(x)

            outputs = layers.Conv1D(self.input_shape[1], 1, padding="same")(x)
        else:
            x = layers.Dense(int(np.prod(self.input_shape)), activation="linear")(x)
            outputs = layers.Reshape(self.input_shape)(x)

        decoder = Model(latent_inputs, outputs, name="decoder")
        return decoder

    def build(self) -> "VariationalAutoencoder":
        """Build VAE with custom training step."""
        logger.info(f"Building {self.name} with input shape {self.input_shape}")

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Build full VAE
        inputs = keras.Input(shape=self.input_shape)
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)

        self.autoencoder = VAEModel(
            inputs,
            outputs,
            encoder=self.encoder,
            decoder=self.decoder,
            z_mean=z_mean,
            z_log_var=z_log_var,
            kl_weight=self.kl_weight,
            name=self.name,
        )

        self._is_built = True
        logger.info(f"VAE built. Parameters: {self.autoencoder.count_params():,}")

        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode to latent mean."""
        z_mean, _, _ = self.encoder.predict(data, verbose=0)
        return z_mean

    def compute_anomaly_score(
        self,
        data: np.ndarray,
        use_reconstruction: bool = True,
        use_kl: bool = True,
    ) -> np.ndarray:
        """
        Compute anomaly score combining reconstruction and KL.

        Args:
            data: Input data
            use_reconstruction: Include reconstruction error
            use_kl: Include KL divergence

        Returns:
            Anomaly scores per sample
        """
        z_mean, z_log_var, z = self.encoder.predict(data, verbose=0)
        reconstructed = self.decoder.predict(z, verbose=0)

        scores = np.zeros(len(data))

        if use_reconstruction:
            recon_error = np.mean(np.square(data - reconstructed), axis=(1, 2))
            scores += recon_error

        if use_kl:
            kl = -0.5 * np.sum(
                1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1
            )
            scores += self.kl_weight * kl

        return scores

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update(
            {
                "encoder_units": self.encoder_units,
                "decoder_units": self.decoder_units,
                "dropout_rate": self.dropout_rate,
                "kl_weight": self.kl_weight,
                "use_conv": self.use_conv,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="VAE")
class VAEModel(Model):
    """Custom VAE model with KL loss."""

    def __init__(
        self,
        *args,
        encoder=None,
        decoder=None,
        z_mean=None,
        z_log_var=None,
        kl_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_model = encoder
        self.decoder_model = decoder
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.kl_weight = kl_weight
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder_model(x)  # encoder
            reconstruction = self.decoder_model(z)  # decoder

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(y, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )
            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": total_loss,
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder_model(x)
        reconstruction = self.decoder_model(z)

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.mean_squared_error(y, reconstruction), axis=1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )

        return {
            "loss": recon_loss + self.kl_weight * kl_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
