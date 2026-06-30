"""
Configuration Management
========================

Centralized configuration for the anomaly detection pipeline.
Supports YAML/JSON config files and programmatic configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Data paths
    data_dir: str = "./data/simulation_results"
    cache_dir: str = "./cache"

    # Data characteristics
    sequence_length: int = 101  # Number of frequency points (excluding header)
    num_features: int = 2  # amplitude and phase

    # Preprocessing
    normalize: bool = True
    normalization_method: str = "standard"  # "standard", "minmax", "robust"
    log_transform_amplitude: bool = True  # Log transform for amplitude (dB scale)

    # Train/Val/Test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Normal/Anomaly definition
    # Files with parameter variations within this threshold are considered "normal"
    normal_threshold_percent: float = 5.0  # ±5% variation = normal

    # Data augmentation
    augment: bool = False
    noise_std: float = 0.01
    time_warp: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, (
            "Train/val/test ratios must sum to 1.0"
        )
        assert self.normalization_method in ["standard", "minmax", "robust"], (
            f"Unknown normalization method: {self.normalization_method}"
        )


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Model type
    model_type: str = "conv1d_ae"  # "conv1d_ae", "lstm_ae", "vae", "transformer_ae"

    # Input shape
    sequence_length: int = 101
    num_features: int = 2

    # Encoder architecture (Conv1D)
    encoder_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    encoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2])

    # Latent space
    latent_dim: int = 16

    # Decoder architecture (mirrors encoder by default)
    decoder_filters: Optional[List[int]] = None
    decoder_kernel_sizes: Optional[List[int]] = None
    decoder_strides: Optional[List[int]] = None

    # LSTM-specific
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    bidirectional: bool = True

    # Transformer-specific
    num_heads: int = 4
    ff_dim: int = 64
    num_transformer_blocks: int = 2

    # VAE-specific
    kl_weight: float = 0.001  # Weight for KL divergence loss

    # Regularization
    dropout_rate: float = 0.2
    l2_regularization: float = 1e-5
    use_batch_norm: bool = True

    # Activation
    activation: str = "relu"
    output_activation: str = "linear"

    def __post_init__(self):
        """Set decoder architecture to mirror encoder if not specified."""
        if self.decoder_filters is None:
            self.decoder_filters = self.encoder_filters[::-1]
        if self.decoder_kernel_sizes is None:
            self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]
        if self.decoder_strides is None:
            self.decoder_strides = self.encoder_strides[::-1]


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3

    # Optimizer
    optimizer: str = "adam"  # "adam", "adamw", "sgd", "rmsprop"
    weight_decay: float = 1e-5

    # Learning rate schedule
    lr_scheduler: str = (
        "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "exponential", "none"
    )
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Early stopping
    early_stopping: bool = True
    es_patience: int = 20
    es_min_delta: float = 1e-5
    es_monitor: str = "val_loss"

    # Loss function
    loss: str = "mse"  # "mse", "mae", "huber"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True

    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 10

    # Hardware
    use_gpu: bool = True
    mixed_precision: bool = True

    # Reproducibility
    seed: int = 42


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    # Threshold determination
    threshold_method: str = "percentile"  # "max", "percentile", "std", "dynamic"
    threshold_percentile: float = 95.0  # For percentile method
    threshold_std_multiplier: float = 3.0  # For std method

    # Anomaly score
    anomaly_score: str = (
        "reconstruction_error"  # "reconstruction_error", "latent_distance", "combined"
    )
    reconstruction_metric: str = "mae"  # "mse", "mae"

    # Post-processing
    smoothing_window: int = 1  # Temporal smoothing window
    min_anomaly_length: int = 1  # Minimum consecutive points for anomaly


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)

    # Experiment tracking
    experiment_name: str = "power_converter_anomaly_detection"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        anomaly_config = AnomalyConfig(**config_dict.get("anomaly", {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            anomaly=anomaly_config,
            experiment_name=config_dict.get(
                "experiment_name", "power_converter_anomaly_detection"
            ),
            run_name=config_dict.get("run_name"),
            tags=config_dict.get("tags", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_config_for_model(model_type: str) -> Config:
    """Get configuration optimized for specific model type."""
    config = Config()
    config.model.model_type = model_type

    if model_type == "conv1d_ae":
        config.model.encoder_filters = [32, 64, 128]
        config.model.encoder_kernel_sizes = [7, 5, 3]
        config.model.latent_dim = 16

    elif model_type == "lstm_ae":
        config.model.lstm_units = [64, 32]
        config.model.bidirectional = True
        config.model.latent_dim = 32
        config.training.batch_size = 32  # LSTM benefits from smaller batches

    elif model_type == "vae":
        config.model.encoder_filters = [32, 64]
        config.model.latent_dim = 8
        config.model.kl_weight = 0.001

    elif model_type == "transformer_ae":
        config.model.num_heads = 4
        config.model.ff_dim = 64
        config.model.num_transformer_blocks = 2
        config.model.latent_dim = 32
        config.training.learning_rate = 1e-4  # Transformers need lower LR

    return config
