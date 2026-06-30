"""
Hyperparameter Search
=====================

Hyperparameter optimization using Keras Tuner.

Supports:
- Multiple optimizer types: Adam, AdamW, SGD, RMSprop, Lion, Nadam
- Learning rate schedules: constant, cosine decay, warmup cosine, exponential, step
- Activation functions: relu, gelu, swish, mish, elu, leaky_relu

Note: Optimizers and LR schedules are implemented in optimizers.py with OOP design.
"""

import keras
import keras_tuner as kt
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..models.conv1d_ae import Conv1DAutoencoder
from ..models.lstm_ae import LSTMAutoencoder
from ..models.mlp_ae import MLPAutoencoder
from ..models.vae import VariationalAutoencoder
from ..models.transformer_ae import TransformerAutoencoder
from ..utils.logger import get_logger

# Import modular optimizers and LR schedules
from .optimizers import (
    # OOP Classes
    create_lr_schedule,
    create_optimizer,
)

logger = get_logger(__name__)


# =============================================================================
# Activation Functions Reference
# =============================================================================

# Supported activation functions:
# - relu: Standard ReLU, fast but can have dead neurons
# - gelu: Gaussian Error Linear Unit, smoother than ReLU, popular in transformers
# - swish/silu: Self-gated activation, x * sigmoid(x), good for deep networks
# - mish: Smoother than swish, x * tanh(softplus(x))
# - elu: Exponential Linear Unit, handles negative values better
# - leaky_relu: Allows small gradient for negative values
# - selu: Scaled ELU, self-normalizing
# - tanh: Hyperbolic tangent, output in [-1, 1]

ACTIVATION_FUNCTIONS = [
    "relu",
    "gelu",
    "swish",  # Also known as silu
    "mish",
    "elu",
    "leaky_relu",
    "selu",
]

OPTIMIZER_TYPES = [
    "adam",
    "adamw",
    "nadam",
    "sgd",
    "rmsprop",
    "lion",
    "adagrad",
]

LR_SCHEDULE_TYPES = [
    "constant",
    "cosine",
    "warmup_cosine",
    "exponential",
    "step",
    "polynomial",
]


@dataclass
class SearchSpace:
    """
    Hyperparameter search space definition.

    Defines ranges and options for all tunable hyperparameters including:
    - Model architecture (latent dim, filters, units, etc.)
    - Optimizer configuration (type, weight decay, momentum)
    - Learning rate schedules (type, warmup, decay rates)
    - Activation functions
    - Regularization (dropout, gradient clipping)
    """

    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-2

    # Optimizer options
    optimizers: list = None
    weight_decay_min: float = 1e-5
    weight_decay_max: float = 0.1
    momentum_options: list = None

    # Learning rate schedule options
    lr_schedules: list = None
    warmup_ratio_options: list = None  # Fraction of total steps for warmup
    decay_rate_options: list = None

    # Activation functions
    activations: list = None

    # Gradient clipping
    use_gradient_clipping: bool = True
    clipnorm_options: list = None

    # Latent dimension
    latent_dims: list = None

    # Conv1D specific
    filter_options: list = None
    kernel_sizes: list = None

    # LSTM specific
    lstm_unit_options: list = None

    # GRU specific
    gru_unit_options: list = None

    # MLP specific
    mlp_encoder_unit_options: list = None
    mlp_decoder_unit_options: list = None

    # Common
    dropout_rates: list = None

    def __post_init__(self):
        # Model architecture
        self.latent_dims = self.latent_dims or [16, 32, 64]
        self.filter_options = self.filter_options or [
            [32, 64],
            [32, 64, 128],
            [64, 128],
        ]
        self.kernel_sizes = self.kernel_sizes or [3, 5, 7]
        self.lstm_unit_options = self.lstm_unit_options or [
            [32, 16],
            [64, 32],
            [128, 64],
        ]
        self.gru_unit_options = self.gru_unit_options or [
            [32, 16],
            [64, 32],
            [128, 64],
        ]
        self.mlp_encoder_unit_options = self.mlp_encoder_unit_options or [
            [128, 64],
            [256, 128],
            [256, 128, 64],
            [512, 256, 128],
        ]
        self.mlp_decoder_unit_options = self.mlp_decoder_unit_options or [
            [64, 128],
            [128, 256],
            [64, 128, 256],
            [128, 256, 512],
        ]
        self.dropout_rates = self.dropout_rates or [0.0, 0.1, 0.2, 0.3]

        # Optimizer options
        self.optimizers = self.optimizers or ["adam", "adamw", "nadam", "lion"]
        self.momentum_options = self.momentum_options or [0.9, 0.95, 0.99]

        # LR schedule options
        self.lr_schedules = self.lr_schedules or ["constant", "cosine", "warmup_cosine"]
        self.warmup_ratio_options = self.warmup_ratio_options or [0.0, 0.05, 0.1]
        self.decay_rate_options = self.decay_rate_options or [0.9, 0.95, 0.99]

        # Activation functions
        self.activations = self.activations or ["relu", "gelu", "swish", "mish"]

        # Gradient clipping
        self.clipnorm_options = self.clipnorm_options or [0.0, 1.0, 5.0]


class Conv1DHyperModel(kt.HyperModel):
    """Hypermodel for Conv1D Autoencoder."""

    def __init__(
        self,
        input_shape: tuple,
        search_space: SearchSpace,
        total_steps: int = 10000,  # Estimated total training steps
    ):
        self.input_shape = input_shape
        self.search_space = search_space
        self.total_steps = total_steps

    def build(self, hp):
        # === Model Architecture Hyperparameters ===
        latent_dim = hp.Choice("latent_dim", self.search_space.latent_dims)
        filters_idx = hp.Choice(
            "filters_idx", list(range(len(self.search_space.filter_options)))
        )
        filters = self.search_space.filter_options[filters_idx]
        kernel_size = hp.Choice("kernel_size", self.search_space.kernel_sizes)
        dropout_rate = hp.Choice("dropout_rate", self.search_space.dropout_rates)

        # Activation function
        activation = hp.Choice("activation", self.search_space.activations)

        # === Optimizer Hyperparameters ===
        optimizer_type = hp.Choice("optimizer", self.search_space.optimizers)
        initial_lr = hp.Float(
            "learning_rate",
            self.search_space.lr_min,
            self.search_space.lr_max,
            sampling="log",
        )

        # Weight decay (for AdamW, Lion)
        weight_decay = hp.Float(
            "weight_decay",
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            sampling="log",
        )

        # === Learning Rate Schedule ===
        lr_schedule_type = hp.Choice("lr_schedule", self.search_space.lr_schedules)
        warmup_ratio = hp.Choice("warmup_ratio", self.search_space.warmup_ratio_options)
        warmup_steps = int(self.total_steps * warmup_ratio)

        # Create learning rate schedule
        learning_rate = create_lr_schedule(
            schedule_type=lr_schedule_type,
            initial_lr=initial_lr,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
        )

        # Gradient clipping
        clipnorm = hp.Choice("clipnorm", self.search_space.clipnorm_options)
        clipnorm = clipnorm if clipnorm > 0 else None  # 0.0 means no clipping

        # Create optimizer
        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

        # Build model
        model = Conv1DAutoencoder(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            filters=filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        model.build()

        # Compile with custom optimizer
        model.autoencoder.compile(optimizer=optimizer, loss="mse")

        return model.autoencoder


class LSTMHyperModel(kt.HyperModel):
    """Hypermodel for LSTM Autoencoder."""

    def __init__(
        self,
        input_shape: tuple,
        search_space: SearchSpace,
        total_steps: int = 10000,
    ):
        self.input_shape = input_shape
        self.search_space = search_space
        self.total_steps = total_steps

    def build(self, hp):
        # === Model Architecture Hyperparameters ===
        latent_dim = hp.Choice("latent_dim", self.search_space.latent_dims)
        units_idx = hp.Choice(
            "units_idx", list(range(len(self.search_space.lstm_unit_options)))
        )
        lstm_units = self.search_space.lstm_unit_options[units_idx]
        dropout_rate = hp.Choice("dropout_rate", self.search_space.dropout_rates)
        bidirectional = hp.Boolean("bidirectional")

        # Activation function for dense layers
        activation = hp.Choice("activation", self.search_space.activations)

        # === Optimizer Hyperparameters ===
        optimizer_type = hp.Choice("optimizer", self.search_space.optimizers)
        initial_lr = hp.Float(
            "learning_rate",
            self.search_space.lr_min,
            self.search_space.lr_max,
            sampling="log",
        )

        weight_decay = hp.Float(
            "weight_decay",
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            sampling="log",
        )

        # === Learning Rate Schedule ===
        lr_schedule_type = hp.Choice("lr_schedule", self.search_space.lr_schedules)
        warmup_ratio = hp.Choice("warmup_ratio", self.search_space.warmup_ratio_options)
        warmup_steps = int(self.total_steps * warmup_ratio)

        learning_rate = create_lr_schedule(
            schedule_type=lr_schedule_type,
            initial_lr=initial_lr,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
        )

        clipnorm = hp.Choice("clipnorm", self.search_space.clipnorm_options)
        clipnorm = clipnorm if clipnorm > 0 else None

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

        model = LSTMAutoencoder(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            activation=activation,
        )
        model.build()
        model.autoencoder.compile(optimizer=optimizer, loss="mse")

        return model.autoencoder


class GRUHyperModel(kt.HyperModel):
    """Hypermodel for GRU Autoencoder."""

    def __init__(
        self,
        input_shape: tuple,
        search_space: SearchSpace,
        total_steps: int = 10000,
    ):
        self.input_shape = input_shape
        self.search_space = search_space
        self.total_steps = total_steps

    def build(self, hp):
        # === Model Architecture Hyperparameters ===
        latent_dim = hp.Choice("latent_dim", self.search_space.latent_dims)
        units_idx = hp.Choice(
            "units_idx", list(range(len(self.search_space.gru_unit_options)))
        )
        gru_units = self.search_space.gru_unit_options[units_idx]
        dropout_rate = hp.Choice("dropout_rate", self.search_space.dropout_rates)
        bidirectional = hp.Boolean("bidirectional")

        # Activation function for dense layers
        activation = hp.Choice("activation", self.search_space.activations)

        # === Optimizer Hyperparameters ===
        optimizer_type = hp.Choice("optimizer", self.search_space.optimizers)
        initial_lr = hp.Float(
            "learning_rate",
            self.search_space.lr_min,
            self.search_space.lr_max,
            sampling="log",
        )

        weight_decay = hp.Float(
            "weight_decay",
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            sampling="log",
        )

        # === Learning Rate Schedule ===
        lr_schedule_type = hp.Choice("lr_schedule", self.search_space.lr_schedules)
        warmup_ratio = hp.Choice("warmup_ratio", self.search_space.warmup_ratio_options)
        warmup_steps = int(self.total_steps * warmup_ratio)

        learning_rate = create_lr_schedule(
            schedule_type=lr_schedule_type,
            initial_lr=initial_lr,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
        )

        clipnorm = hp.Choice("clipnorm", self.search_space.clipnorm_options)
        clipnorm = clipnorm if clipnorm > 0 else None

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

        from ..models.gru_ae import GRUAutoencoder

        model = GRUAutoencoder(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            gru_units=gru_units,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            activation=activation,
        )
        model.build()
        model.autoencoder.compile(optimizer=optimizer, loss="mse")

        return model.autoencoder


class VAEHyperModel(kt.HyperModel):
    """Hypermodel for VAE."""

    def __init__(
        self,
        input_shape: tuple,
        search_space: SearchSpace,
        total_steps: int = 10000,
    ):
        self.input_shape = input_shape
        self.search_space = search_space
        self.total_steps = total_steps

    def build(self, hp):
        # === Model Architecture Hyperparameters ===
        latent_dim = hp.Choice("latent_dim", self.search_space.latent_dims)
        dropout_rate = hp.Choice("dropout_rate", self.search_space.dropout_rates)
        kl_weight = hp.Float("kl_weight", 0.1, 2.0, step=0.1)

        # Activation function
        activation = hp.Choice("activation", self.search_space.activations)

        # === Optimizer Hyperparameters ===
        optimizer_type = hp.Choice("optimizer", self.search_space.optimizers)
        initial_lr = hp.Float(
            "learning_rate",
            self.search_space.lr_min,
            self.search_space.lr_max,
            sampling="log",
        )

        weight_decay = hp.Float(
            "weight_decay",
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            sampling="log",
        )

        # === Learning Rate Schedule ===
        lr_schedule_type = hp.Choice("lr_schedule", self.search_space.lr_schedules)
        warmup_ratio = hp.Choice("warmup_ratio", self.search_space.warmup_ratio_options)
        warmup_steps = int(self.total_steps * warmup_ratio)

        learning_rate = create_lr_schedule(
            schedule_type=lr_schedule_type,
            initial_lr=initial_lr,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
        )

        clipnorm = hp.Choice("clipnorm", self.search_space.clipnorm_options)
        clipnorm = clipnorm if clipnorm > 0 else None

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

        model = VariationalAutoencoder(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            kl_weight=kl_weight,
            activation=activation,
        )
        model.build()
        model.autoencoder.compile(optimizer=optimizer)

        return model.autoencoder


class TransformerHyperModel(kt.HyperModel):
    """Hypermodel for Transformer Autoencoder."""

    def __init__(
        self,
        input_shape: tuple,
        search_space: SearchSpace,
        total_steps: int = 10000,
    ):
        self.input_shape = input_shape
        self.search_space = search_space
        self.total_steps = total_steps

    def build(self, hp):
        # === Model Architecture Hyperparameters ===
        latent_dim = hp.Choice("latent_dim", self.search_space.latent_dims)
        d_model = hp.Choice("d_model", [32, 64, 128])
        num_heads = hp.Choice("num_heads", [2, 4, 8])
        num_layers = hp.Choice("num_layers", [1, 2, 3])
        dropout_rate = hp.Choice("dropout_rate", self.search_space.dropout_rates)

        # Activation function for FFN layers
        activation = hp.Choice("activation", self.search_space.activations)

        # === Optimizer Hyperparameters ===
        # For transformers, AdamW with warmup cosine is typically best
        optimizer_type = hp.Choice("optimizer", self.search_space.optimizers)
        initial_lr = hp.Float(
            "learning_rate",
            self.search_space.lr_min,
            self.search_space.lr_max,
            sampling="log",
        )

        weight_decay = hp.Float(
            "weight_decay",
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            sampling="log",
        )

        # === Learning Rate Schedule ===
        # Warmup is especially important for transformers
        lr_schedule_type = hp.Choice("lr_schedule", self.search_space.lr_schedules)
        warmup_ratio = hp.Choice("warmup_ratio", self.search_space.warmup_ratio_options)
        warmup_steps = int(self.total_steps * warmup_ratio)

        learning_rate = create_lr_schedule(
            schedule_type=lr_schedule_type,
            initial_lr=initial_lr,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
        )

        # Gradient clipping is critical for transformer stability
        clipnorm = hp.Choice("clipnorm", self.search_space.clipnorm_options)
        clipnorm = clipnorm if clipnorm > 0 else None

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

        model = TransformerAutoencoder(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            ff_activation=activation,
        )
        model.build()
        model.autoencoder.compile(optimizer=optimizer, loss="mse")

        return model.autoencoder


class MLPHyperModel(kt.HyperModel):
    """Hypermodel for MLP Autoencoder."""

    def __init__(
        self,
        input_shape: tuple,
        search_space: SearchSpace,
        total_steps: int = 10000,
    ):
        self.input_shape = input_shape
        self.search_space = search_space
        self.total_steps = total_steps

    def build(self, hp):
        # === Model Architecture Hyperparameters ===
        latent_dim = hp.Choice("latent_dim", self.search_space.latent_dims)
        enc_idx = hp.Choice(
            "enc_units_idx",
            list(range(len(self.search_space.mlp_encoder_unit_options))),
        )
        encoder_units = self.search_space.mlp_encoder_unit_options[enc_idx]
        dec_idx = hp.Choice(
            "dec_units_idx",
            list(range(len(self.search_space.mlp_decoder_unit_options))),
        )
        decoder_units = self.search_space.mlp_decoder_unit_options[dec_idx]
        dropout_rate = hp.Choice("dropout_rate", self.search_space.dropout_rates)
        use_batch_norm = hp.Boolean("use_batch_norm")

        # Activation function
        activation = hp.Choice("activation", self.search_space.activations)

        # === Optimizer Hyperparameters ===
        optimizer_type = hp.Choice("optimizer", self.search_space.optimizers)
        initial_lr = hp.Float(
            "learning_rate",
            self.search_space.lr_min,
            self.search_space.lr_max,
            sampling="log",
        )

        weight_decay = hp.Float(
            "weight_decay",
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
            sampling="log",
        )

        # === Learning Rate Schedule ===
        lr_schedule_type = hp.Choice("lr_schedule", self.search_space.lr_schedules)
        warmup_ratio = hp.Choice("warmup_ratio", self.search_space.warmup_ratio_options)
        warmup_steps = int(self.total_steps * warmup_ratio)

        learning_rate = create_lr_schedule(
            schedule_type=lr_schedule_type,
            initial_lr=initial_lr,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
        )

        clipnorm = hp.Choice("clipnorm", self.search_space.clipnorm_options)
        clipnorm = clipnorm if clipnorm > 0 else None

        optimizer = create_optimizer(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

        model = MLPAutoencoder(
            input_shape=self.input_shape,
            latent_dim=latent_dim,
            encoder_units=encoder_units,
            decoder_units=decoder_units,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )
        model.build()
        model.autoencoder.compile(optimizer=optimizer, loss="mse")

        return model.autoencoder


HYPERMODEL_REGISTRY = {
    "conv1d_ae": Conv1DHyperModel,
    "lstm_ae": LSTMHyperModel,
    "gru_ae": GRUHyperModel,
    "mlp_ae": MLPHyperModel,
    "vae": VAEHyperModel,
    "transformer_ae": TransformerHyperModel,
}


class TunerBackupMixin:
    """Injects a trial-specific BackupAndRestore callback to resume interrupted trials."""

    def run_trial(self, trial, *args, **kwargs):
        from pathlib import Path
        import keras

        # `self.project_dir` is the path where tuner saves trial subfolders
        backup_dir = Path(self.project_dir) / f"trial_{trial.trial_id}" / "backup"
        backup_callback = keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir))

        callbacks = kwargs.get("callbacks", [])
        kwargs["callbacks"] = list(callbacks) + [backup_callback]

        return super().run_trial(trial, *args, **kwargs)


class ResumableBayesianOptimization(TunerBackupMixin, kt.BayesianOptimization):
    pass


class ResumableRandomSearch(TunerBackupMixin, kt.RandomSearch):
    pass


class ResumableHyperband(TunerBackupMixin, kt.Hyperband):
    pass


class HyperparameterSearch:
    """
    Hyperparameter search orchestrator.

    Supports Bayesian optimization, random search, and hyperband.

    Features:
    - Multiple optimizer types: Adam, AdamW, SGD, RMSprop, Lion, Nadam
    - Learning rate schedules: constant, cosine decay, warmup cosine, exponential
    - Activation functions: relu, gelu, swish, mish, elu, leaky_relu
    - Gradient clipping for stability
    """

    def __init__(
        self,
        model_type: str,
        input_shape: tuple,
        search_space: Optional[SearchSpace] = None,
        project_name: str = "hp_search",
        directory: str = "hp_tuning",
        estimated_samples: int = 10000,
    ):
        """
        Initialize hyperparameter search.

        Args:
            model_type: Type of model to tune
            input_shape: Input shape for model
            search_space: Search space definition
            project_name: Project name for Keras Tuner
            directory: Directory for search results
            estimated_samples: Estimated number of training samples (for LR schedule)
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.search_space = search_space or SearchSpace()
        self.project_name = project_name
        self.directory = Path(directory)
        self.estimated_samples = estimated_samples

        if model_type not in HYPERMODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")

        # Hypermodel will be created in search() when we know the exact steps
        self._hypermodel_class = HYPERMODEL_REGISTRY[model_type]
        self.hypermodel = None
        self.tuner = None
        self.best_hps = None

    def search(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        method: str = "bayesian",
        max_trials: int = 20,
        epochs: int = 50,
        batch_size: int = 32,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter search.

        Args:
            train_data: Training data
            val_data: Validation data
            method: Search method (bayesian, random, hyperband)
            max_trials: Maximum number of trials
            epochs: Epochs per trial
            batch_size: Batch size

        Returns:
            Dictionary with best hyperparameters
        """
        logger.info(f"Starting {method} search for {self.model_type}")

        # Calculate total training steps for LR schedules
        n_samples = len(train_data)
        steps_per_epoch = max(1, n_samples // batch_size)
        total_steps = steps_per_epoch * epochs

        logger.info(
            f"Estimated total steps: {total_steps} ({steps_per_epoch} steps/epoch × {epochs} epochs)"
        )

        # Create hypermodel with estimated total steps
        self.hypermodel = self._hypermodel_class(
            self.input_shape,
            self.search_space,
            total_steps=total_steps,
        )

        # Create tuner
        if method == "bayesian":
            self.tuner = ResumableBayesianOptimization(
                self.hypermodel,
                objective="val_loss",
                max_trials=max_trials,
                directory=str(self.directory),
                project_name=self.project_name,
            )
        elif method == "random":
            self.tuner = ResumableRandomSearch(
                self.hypermodel,
                objective="val_loss",
                max_trials=max_trials,
                directory=str(self.directory),
                project_name=self.project_name,
            )
        elif method == "hyperband":
            self.tuner = ResumableHyperband(
                self.hypermodel,
                objective="val_loss",
                max_epochs=epochs,
                factor=3,
                directory=str(self.directory),
                project_name=self.project_name,
            )
        else:
            raise ValueError(f"Unknown search method: {method}")

        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        # Run search
        self.tuner.search(
            train_data,
            train_data,
            validation_data=(val_data, val_data),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

        # Get best hyperparameters
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        results = {
            "best_hyperparameters": self.best_hps.values,
            "best_trial": self.tuner.oracle.get_best_trials(1)[0].trial_id,
        }

        logger.info(f"Best hyperparameters: {results['best_hyperparameters']}")

        return results

    def get_best_model(self) -> keras.Model:
        """Get best model from search."""
        if self.tuner is None:
            raise RuntimeError("No search performed yet.")
        return self.tuner.get_best_models(num_models=1)[0]

    def summary(self) -> None:
        """Print search summary."""
        if self.tuner is not None:
            self.tuner.results_summary()
