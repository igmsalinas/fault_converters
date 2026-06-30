"""
Contrastive Autoencoder Model
=============================

Autoencoder with projection head for contrastive learning.
Implements CARLA-style architecture for anomaly detection.

Reference:
    Darban et al., "CARLA: Self-supervised Contrastive Representation Learning
    for Time Series Anomaly Detection", arXiv:2308.09296
"""

import keras
from keras import layers, Model, ops
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from .base import BaseAutoencoder
from ..utils.logger import get_logger

logger = get_logger(__name__)


@keras.saving.register_keras_serializable(package="CARLA")
class ProjectionHead(layers.Layer):
    """
    Projection head for contrastive learning.

    Maps encoder outputs to a lower-dimensional space for contrastive loss.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        self.layers_list = []

    def build(self, input_shape):
        for i in range(self.num_layers - 1):
            self.layers_list.append(layers.Dense(self.hidden_dim))
            if self.use_batch_norm:
                self.layers_list.append(layers.BatchNormalization())
            self.layers_list.append(layers.Activation(self.activation))

        # Final layer without activation (for contrastive loss)
        self.layers_list.append(layers.Dense(self.output_dim))

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_list:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config


class ContrastiveAutoencoder(BaseAutoencoder):
    """
    Autoencoder with contrastive learning capability.

    Combines reconstruction objective with contrastive loss using
    a projection head. Suitable for CARLA-style anomaly detection.

    Architecture:
        Input -> Encoder -> Latent Space -> Decoder -> Reconstruction
                              |
                              v
                      Projection Head -> Contrastive Space

    Args:
        input_shape: Input shape (sequence_length, n_features)
        latent_dim: Dimension of latent space
        projection_dim: Dimension of contrastive projection space
        encoder_type: Type of encoder ("conv1d", "lstm", "gru", "transformer", "mlp")
        encoder_filters: Filters for Conv1D encoder
        encoder_units: Units for LSTM encoder
        num_heads: Number of attention heads for transformer
        projection_hidden_dim: Hidden dimension in projection head
        projection_layers: Number of layers in projection head
        dropout_rate: Dropout rate
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        latent_dim: int = 32,
        projection_dim: int = 64,
        encoder_type: str = "conv1d",  # "conv1d", "lstm", "gru", "transformer", "mlp"
        encoder_filters: List[int] = [32, 64, 128],
        encoder_units: List[int] = [64, 32],
        num_heads: int = 4,
        projection_hidden_dim: int = 128,
        projection_layers: int = 2,
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        name: str = "contrastive_ae",
    ):
        super().__init__(input_shape, latent_dim, name)

        self.projection_dim = projection_dim
        self.encoder_type = encoder_type
        self.encoder_filters = encoder_filters
        self.encoder_units = encoder_units
        self.num_heads = num_heads
        self.projection_hidden_dim = projection_hidden_dim
        self.projection_layers = projection_layers
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size

        self.projection_head: Optional[Model] = None

    def _build_encoder(self) -> Model:
        """Build encoder based on specified type."""
        inputs = keras.Input(shape=self.input_shape, name="encoder_input")

        if self.encoder_type == "conv1d":
            x = self._build_conv1d_encoder(inputs)
        elif self.encoder_type == "lstm":
            x = self._build_lstm_encoder(inputs)
        elif self.encoder_type == "gru":
            x = self._build_gru_encoder(inputs)
        elif self.encoder_type == "transformer":
            x = self._build_transformer_encoder(inputs)
        elif self.encoder_type == "mlp":
            x = self._build_mlp_encoder(inputs)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        # Flatten and project to latent space
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, name="latent")(x)

        return Model(inputs, latent, name="encoder")

    def _build_conv1d_encoder(self, inputs):
        """Build Conv1D encoder layers."""
        x = inputs

        for i, filters in enumerate(self.encoder_filters):
            x = layers.Conv1D(
                filters,
                kernel_size=self.kernel_size,
                strides=2,
                padding="same",
                name=f"enc_conv_{i}",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        return x

    def _build_lstm_encoder(self, inputs):
        """Build LSTM encoder layers."""
        x = inputs

        for i, units in enumerate(self.encoder_units[:-1]):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f"enc_lstm_{i}",
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Last LSTM without return_sequences
        x = layers.LSTM(
            self.encoder_units[-1],
            return_sequences=False,
            name=f"enc_lstm_{len(self.encoder_units) - 1}",
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Reshape((1, self.encoder_units[-1]))(x)

        return x

    def _build_gru_encoder(self, inputs):
        """Build GRU encoder layers."""
        x = inputs

        for i, units in enumerate(self.encoder_units[:-1]):
            x = layers.GRU(
                units,
                return_sequences=True,
                reset_after=True,
                name=f"enc_gru_{i}",
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Last GRU without return_sequences
        x = layers.GRU(
            self.encoder_units[-1],
            return_sequences=False,
            reset_after=True,
            name=f"enc_gru_{len(self.encoder_units) - 1}",
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Reshape((1, self.encoder_units[-1]))(x)

        return x

    def _build_transformer_encoder(self, inputs):
        """Build Transformer encoder layers."""
        seq_len, n_features = self.input_shape

        # Linear projection
        x = layers.Dense(self.encoder_filters[0])(inputs)

        # Positional encoding (learnable)
        positions = layers.Embedding(seq_len, self.encoder_filters[0])(
            ops.arange(seq_len)
        )
        x = x + positions

        # Transformer blocks
        for i, filters in enumerate(self.encoder_filters):
            if i > 0:
                x = layers.Dense(filters)(x)

            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=filters // self.num_heads,
                name=f"enc_mha_{i}",
            )(x, x)
            attn_output = layers.Dropout(self.dropout_rate)(attn_output)
            x = layers.LayerNormalization()(x + attn_output)

            # Feed-forward
            ff_output = layers.Dense(filters * 4, activation="relu")(x)
            ff_output = layers.Dense(filters)(ff_output)
            ff_output = layers.Dropout(self.dropout_rate)(ff_output)
            x = layers.LayerNormalization()(x + ff_output)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Reshape((1, self.encoder_filters[-1]))(x)

        return x

    def _build_mlp_encoder(self, inputs):
        """Build MLP encoder layers."""
        x = layers.Flatten()(inputs)

        for i, units in enumerate(self.encoder_units):
            x = layers.Dense(units, name=f"enc_mlp_{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        x = layers.Reshape((1, self.encoder_units[-1]))(x)
        return x

    def _build_decoder(self) -> Model:
        """Build decoder matching encoder architecture."""
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="decoder_input")

        if self.encoder_type == "conv1d":
            x = self._build_conv1d_decoder(latent_inputs)
        elif self.encoder_type == "lstm":
            x = self._build_lstm_decoder(latent_inputs)
        elif self.encoder_type == "gru":
            x = self._build_gru_decoder(latent_inputs)
        elif self.encoder_type == "transformer":
            x = self._build_transformer_decoder(latent_inputs)
        elif self.encoder_type == "mlp":
            x = self._build_mlp_decoder(latent_inputs)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")

        return Model(latent_inputs, x, name="decoder")

    def _build_conv1d_decoder(self, latent_inputs):
        """Build Conv1D decoder layers."""
        seq_len, n_features = self.input_shape

        # Calculate compressed size
        compressed_len = seq_len // (2 ** len(self.encoder_filters))
        compressed_len = max(1, compressed_len)

        x = layers.Dense(compressed_len * self.encoder_filters[-1])(latent_inputs)
        x = layers.Reshape((compressed_len, self.encoder_filters[-1]))(x)

        # Reverse encoder filters
        decoder_filters = self.encoder_filters[::-1]

        for i, filters in enumerate(decoder_filters):
            x = layers.Conv1DTranspose(
                filters,
                kernel_size=self.kernel_size,
                strides=2,
                padding="same",
                name=f"dec_conv_{i}",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Adjust to exact output shape
        x = layers.Conv1D(n_features, kernel_size=self.kernel_size, padding="same")(x)

        # Ensure correct sequence length
        current_len = x.shape[1]

        if current_len != seq_len:
            # Use cropping or padding
            if current_len > seq_len:
                crop = (current_len - seq_len) // 2
                x = layers.Cropping1D((crop, current_len - seq_len - crop))(x)
            else:
                pad = seq_len - current_len
                x = layers.ZeroPadding1D((pad // 2, pad - pad // 2))(x)

        return x

    def _build_lstm_decoder(self, latent_inputs):
        """Build LSTM decoder layers."""
        seq_len, n_features = self.input_shape

        # Repeat latent vector for each timestep
        x = layers.RepeatVector(seq_len)(latent_inputs)

        # Reverse encoder units
        decoder_units = self.encoder_units[::-1]

        for i, units in enumerate(decoder_units):
            x = layers.LSTM(
                units,
                return_sequences=True,
                name=f"dec_lstm_{i}",
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        x = layers.TimeDistributed(layers.Dense(n_features))(x)

        return x

    def _build_gru_decoder(self, latent_inputs):
        """Build GRU decoder layers."""
        seq_len, n_features = self.input_shape

        # Repeat latent vector for each timestep
        x = layers.RepeatVector(seq_len)(latent_inputs)

        # Reverse encoder units
        decoder_units = self.encoder_units[::-1]

        for i, units in enumerate(decoder_units):
            x = layers.GRU(
                units,
                return_sequences=True,
                reset_after=True,
                name=f"dec_gru_{i}",
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        x = layers.TimeDistributed(layers.Dense(n_features))(x)

        return x

    def _build_transformer_decoder(self, latent_inputs):
        """Build Transformer decoder layers."""
        seq_len, n_features = self.input_shape

        # Expand latent to sequence
        x = layers.Dense(self.encoder_filters[-1])(latent_inputs)
        x = layers.RepeatVector(seq_len)(x)

        # Positional encoding
        positions = layers.Embedding(seq_len, self.encoder_filters[-1])(
            ops.arange(seq_len)
        )
        x = x + positions

        # Transformer blocks (reverse)
        decoder_filters = self.encoder_filters[::-1]

        for i, filters in enumerate(decoder_filters):
            if i > 0:
                x = layers.Dense(filters)(x)

            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=filters // self.num_heads,
                name=f"dec_mha_{i}",
            )(x, x)
            attn_output = layers.Dropout(self.dropout_rate)(attn_output)
            x = layers.LayerNormalization()(x + attn_output)

            # Feed-forward
            ff_output = layers.Dense(filters * 4, activation="relu")(x)
            ff_output = layers.Dense(filters)(ff_output)
            ff_output = layers.Dropout(self.dropout_rate)(ff_output)
            x = layers.LayerNormalization()(x + ff_output)

        # Output projection
        x = layers.Dense(n_features)(x)

        return x

    def _build_mlp_decoder(self, latent_inputs):
        """Build MLP decoder layers."""
        seq_len, n_features = self.input_shape

        x = latent_inputs

        # Reverse encoder units for decoder
        decoder_units = self.encoder_units[::-1]

        for i, units in enumerate(decoder_units):
            x = layers.Dense(units, name=f"dec_mlp_{i}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Output projection and reshape
        flat_size = seq_len * n_features
        x = layers.Dense(flat_size)(x)
        x = layers.Reshape((seq_len, n_features))(x)

        return x

    def _build_projection_head(self) -> Model:
        """Build projection head for contrastive learning."""
        latent_inputs = keras.Input(shape=(self.latent_dim,), name="projection_input")

        x = ProjectionHead(
            hidden_dim=self.projection_hidden_dim,
            output_dim=self.projection_dim,
            num_layers=self.projection_layers,
        )(latent_inputs)

        return Model(latent_inputs, x, name="projection_head")

    def build(self) -> "ContrastiveAutoencoder":
        """Build full model including projection head."""
        # Build base autoencoder
        super().build()

        # Build projection head
        self.projection_head = self._build_projection_head()

        logger.info(
            f"Projection head parameters: {self.projection_head.count_params():,}"
        )

        return self

    def project(self, data: np.ndarray) -> np.ndarray:
        """
        Project data to contrastive space.

        Args:
            data: Input data or latent representations

        Returns:
            Projections in contrastive space
        """
        # Encode if needed
        if data.shape[-1] != self.latent_dim or len(data.shape) == 3:
            latent = self.encode(data)
        else:
            latent = data

        return self.projection_head.predict(latent, verbose=0)

    def get_embeddings(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get all embeddings for analysis.

        Args:
            data: Input data

        Returns:
            Dictionary with 'latent' and 'projection' embeddings
        """
        latent = self.encode(data)
        projection = self.projection_head.predict(latent, verbose=0)

        return {
            "latent": latent,
            "projection": projection,
        }

    def summary(self):
        """Print model summaries."""
        print("=" * 60)
        print("ENCODER")
        print("=" * 60)
        self.encoder.summary()

        print("\n" + "=" * 60)
        print("DECODER")
        print("=" * 60)
        self.decoder.summary()

        print("\n" + "=" * 60)
        print("PROJECTION HEAD")
        print("=" * 60)
        self.projection_head.summary()

        print("\n" + "=" * 60)
        print("FULL AUTOENCODER")
        print("=" * 60)
        self.autoencoder.summary()

    def save(self, path: str):
        """Save all model components (weights and config)."""
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.encoder.save_weights(path / "encoder.weights.h5")
        self.decoder.save_weights(path / "decoder.weights.h5")
        self.projection_head.save_weights(path / "projection_head.weights.h5")
        self.autoencoder.save_weights(path / "autoencoder.weights.h5")

        # Save config
        import json

        config = self.get_config()
        # Save as both config.json (internal) and model_config.json (standard)
        for fn in ["config.json", "model_config.json"]:
            with open(path / fn, "w") as f:
                json.dump(config, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls, path: str, config: Optional[Dict[str, Any]] = None
    ) -> "ContrastiveAutoencoder":
        """
        Load model weights from path.

        Args:
            path: Directory containing weights
            config: Optional architecture config (bypasses reading model_config.json)
        """
        from pathlib import Path
        import json

        path = Path(path)

        # Load config if not provided
        if config is None:
            config_path = path / "model_config.json"
            if not config_path.exists():
                config_path = path / "config.json"

            with open(config_path, "r") as f:
                config = json.load(f)

        # Reconstruct model
        model = cls(
            input_shape=tuple(config["input_shape"]),
            latent_dim=config["latent_dim"],
            projection_dim=config["projection_dim"],
            encoder_type=config["encoder_type"],
            encoder_filters=config["encoder_filters"],
            encoder_units=config.get("encoder_units", [64, 32]),
            num_heads=config.get("num_heads", 4),
            projection_hidden_dim=config.get("projection_hidden_dim", 128),
            projection_layers=config.get("projection_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.1),
            kernel_size=config.get("kernel_size", 3),
            name=config.get("name", "contrastive_ae"),
        )
        model.build()

        # Load weights
        # Try weights.h5 format first, then fallback to .keras (legacy)
        for name, component in [
            ("encoder", model.encoder),
            ("decoder", model.decoder),
            ("projection_head", model.projection_head),
            ("autoencoder", model.autoencoder),
        ]:
            weight_path = path / f"{name}.weights.h5"
            legacy_path = path / f"{name}.keras"

            if weight_path.exists():
                component.load_weights(weight_path)
            elif legacy_path.exists():
                # If it's a legacy .keras file, it might be a full model save
                try:
                    # Try loading as weights first
                    component.load_weights(legacy_path)
                except:
                    # Fallback to loading as full model and copying weights
                    legacy_model = keras.models.load_model(legacy_path)
                    component.set_weights(legacy_model.get_weights())
            else:
                logger.warning(f"Weight file for {name} not found in {path}")

        model._is_built = True
        logger.info(f"Model loaded from {path}")

        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "projection_dim": self.projection_dim,
            "encoder_type": self.encoder_type,
            "encoder_filters": self.encoder_filters,
            "encoder_units": self.encoder_units,
            "num_heads": self.num_heads,
            "projection_hidden_dim": self.projection_hidden_dim,
            "projection_layers": self.projection_layers,
            "dropout_rate": self.dropout_rate,
            "kernel_size": self.kernel_size,
            "name": self.name,
        }
