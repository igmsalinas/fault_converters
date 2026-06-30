"""Training pipeline and utilities."""

from .trainer import Trainer, TrainingConfig
from .callbacks import get_callbacks, EarlyStoppingWithRestore
from .hyperparameter_search import HyperparameterSearch
from .carla_trainer import CARLATrainer, CARLAConfig
from .carla_hyperparameter_search import CARLAHyperparameterSearch, CARLASearchSpace

# OOP Optimizers and LR Schedules
from .optimizers import (
    # Base classes
    BaseLRSchedule,
    BaseOptimizer,
    # LR Schedules
    ConstantLR,
    CosineLR,
    WarmupCosineLR,
    ExponentialLR,
    StepLR,
    PolynomialLR,
    InverseTimeLR,
    OneCycleLR,
    CyclicLR,
    # Optimizers
    AdamOptimizer,
    AdamWOptimizer,
    NadamOptimizer,
    SGDOptimizer,
    RMSpropOptimizer,
    LionOptimizer,
    AdagradOptimizer,
    AdadeltaOptimizer,
    # Registries and factories
    LR_SCHEDULE_REGISTRY,
    OPTIMIZER_REGISTRY,
    get_lr_schedule,
    get_optimizer,
    OptimizerConfig,
    # Backward compatible
    create_lr_schedule,
    create_optimizer,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainingConfig",
    "get_callbacks",
    "EarlyStoppingWithRestore",
    "HyperparameterSearch",
    "CARLATrainer",
    "CARLAConfig",
    "CARLAHyperparameterSearch",
    "CARLASearchSpace",
    # LR Schedules
    "BaseLRSchedule",
    "ConstantLR",
    "CosineLR",
    "WarmupCosineLR",
    "ExponentialLR",
    "StepLR",
    "PolynomialLR",
    "InverseTimeLR",
    "OneCycleLR",
    "CyclicLR",
    "LR_SCHEDULE_REGISTRY",
    "get_lr_schedule",
    # Optimizers
    "BaseOptimizer",
    "AdamOptimizer",
    "AdamWOptimizer",
    "NadamOptimizer",
    "SGDOptimizer",
    "RMSpropOptimizer",
    "LionOptimizer",
    "AdagradOptimizer",
    "AdadeltaOptimizer",
    "OPTIMIZER_REGISTRY",
    "get_optimizer",
    "OptimizerConfig",
    "create_lr_schedule",
    "create_optimizer",
]
