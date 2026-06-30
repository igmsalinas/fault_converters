"""
Optimizers and Learning Rate Schedulers
=======================================

Modular OOP-based implementations for optimizers and learning rate schedules.

Supported Optimizers:
- Adam, AdamW, Nadam, SGD, RMSprop, Lion, Adagrad, Adadelta

Supported LR Schedules:
- Constant, Cosine, Warmup Cosine, Exponential, Step, Polynomial,
  Inverse Time, One Cycle, Cyclic
"""

import keras
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Keras Schedules (Serializable)
# =============================================================================


@keras.saving.register_keras_serializable(package="Training")
class _WarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warmup_steps = keras.ops.cast(self.warmup_steps, "float32")
        total_steps = keras.ops.cast(self.total_steps, "float32")

        # Linear warmup
        warmup_lr = self.initial_lr * (step / keras.ops.maximum(warmup_steps, 1))

        # Cosine decay after warmup
        progress = (step - warmup_steps) / keras.ops.maximum(
            total_steps - warmup_steps, 1
        )
        progress = keras.ops.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1 + keras.ops.cos(np.pi * progress)
        )

        return keras.ops.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


@keras.saving.register_keras_serializable(package="Training")
class _OneCycleSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        max_lr,
        total_steps,
        pct_start,
        div_factor,
        final_div_factor,
        anneal_strategy,
    ):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        total = keras.ops.cast(self.total_steps, "float32")
        pct = step / total

        warmup_pct = self.pct_start

        # Phase 1: Warmup (increase LR)
        warmup_progress = pct / warmup_pct
        warmup_lr = self.initial_lr + (self.max_lr - self.initial_lr) * warmup_progress

        # Phase 2: Anneal (decrease LR)
        anneal_progress = (pct - warmup_pct) / (1.0 - warmup_pct)
        if self.anneal_strategy == "cos":
            anneal_lr = self.final_lr + (self.max_lr - self.final_lr) * (
                0.5 * (1 + keras.ops.cos(np.pi * anneal_progress))
            )
        else:  # linear
            anneal_lr = self.max_lr - (self.max_lr - self.final_lr) * anneal_progress

        return keras.ops.where(pct < warmup_pct, warmup_lr, anneal_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "pct_start": self.pct_start,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
            "anneal_strategy": self.anneal_strategy,
        }


@keras.saving.register_keras_serializable(package="Training")
class _CyclicSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, min_lr, step_size, mode, gamma):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        step_size = keras.ops.cast(self.step_size, "float32")

        cycle = keras.ops.floor(1 + step / (2 * step_size))
        x = keras.ops.abs(step / step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1))
        else:  # exp_range
            scale = self.gamma**step

        lr = (
            self.min_lr
            + (self.max_lr - self.min_lr) * keras.ops.maximum(0.0, 1 - x) * scale
        )

        return lr

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "step_size": self.step_size,
            "mode": self.mode,
            "gamma": self.gamma,
        }


# =============================================================================
# Learning Rate Schedules (OOP)
# =============================================================================


class BaseLRSchedule(ABC):
    """
    Abstract base class for learning rate schedules.

    All schedules must implement the build() method to create the Keras schedule.
    """

    name: str = "base"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
    ):
        """
        Initialize base learning rate schedule.

        Args:
            initial_lr: Initial/maximum learning rate
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.min_lr = min_lr

    @abstractmethod
    def build(self) -> Union[float, keras.optimizers.schedules.LearningRateSchedule]:
        """Build and return the learning rate schedule."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return schedule configuration."""
        return {
            "name": self.name,
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(initial_lr={self.initial_lr}, total_steps={self.total_steps})"


class ConstantLR(BaseLRSchedule):
    """Constant learning rate (no decay)."""

    name = "constant"

    def build(self) -> float:
        return self.initial_lr


class CosineLR(BaseLRSchedule):
    """
    Cosine annealing learning rate schedule.

    Smoothly decreases LR following a cosine curve from initial_lr to min_lr.
    Popular for image classification and general deep learning.
    """

    name = "cosine"

    def build(self) -> keras.optimizers.schedules.CosineDecay:
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.total_steps,
            alpha=self.min_lr / self.initial_lr,
        )


class WarmupCosineLR(BaseLRSchedule):
    """
    Warmup + Cosine decay learning rate schedule.

    Linear warmup followed by cosine annealing. This is the most popular
    schedule for transformer models (BERT, GPT, ViT, etc.).

    Reference: "Attention Is All You Need" (Vaswani et al., 2017)
    """

    name = "warmup_cosine"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.1,
    ):
        """
        Initialize warmup cosine schedule.

        Args:
            initial_lr: Peak learning rate (reached after warmup)
            total_steps: Total training steps
            min_lr: Minimum learning rate at end of training
            warmup_steps: Number of warmup steps (takes precedence)
            warmup_ratio: Fraction of total_steps for warmup (if warmup_steps=0)
        """
        super().__init__(initial_lr, total_steps, min_lr)
        self.warmup_steps = (
            warmup_steps if warmup_steps > 0 else int(total_steps * warmup_ratio)
        )
        self.warmup_ratio = warmup_ratio

    def build(self) -> keras.optimizers.schedules.LearningRateSchedule:
        """Build warmup cosine schedule."""
        return _WarmupCosineSchedule(
            self.initial_lr, self.warmup_steps, self.total_steps, self.min_lr
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "warmup_steps": self.warmup_steps,
                "warmup_ratio": self.warmup_ratio,
            }
        )
        return config


class ExponentialLR(BaseLRSchedule):
    """
    Exponential decay learning rate schedule.

    LR decays by decay_rate every decay_steps steps.
    Classic schedule, works well for most tasks.
    """

    name = "exponential"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        decay_rate: float = 0.96,
        decay_steps: int = 1000,
    ):
        super().__init__(initial_lr, total_steps, min_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def build(self) -> keras.optimizers.schedules.ExponentialDecay:
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=False,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "decay_rate": self.decay_rate,
                "decay_steps": self.decay_steps,
            }
        )
        return config


class StepLR(BaseLRSchedule):
    """
    Step decay learning rate schedule.

    LR drops by decay_rate every decay_steps steps (staircase pattern).
    Simple and effective for many tasks.
    """

    name = "step"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        decay_rate: float = 0.5,
        decay_steps: int = 2000,
    ):
        super().__init__(initial_lr, total_steps, min_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def build(self) -> keras.optimizers.schedules.ExponentialDecay:
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "decay_rate": self.decay_rate,
                "decay_steps": self.decay_steps,
            }
        )
        return config


class PolynomialLR(BaseLRSchedule):
    """
    Polynomial decay learning rate schedule.

    LR decays following a polynomial curve. With power=1.0, this is linear decay.
    Popular for BERT-style fine-tuning.

    Reference: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
    """

    name = "polynomial"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        power: float = 1.0,
    ):
        super().__init__(initial_lr, total_steps, min_lr)
        self.power = power

    def build(self) -> keras.optimizers.schedules.PolynomialDecay:
        return keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.total_steps,
            end_learning_rate=self.min_lr,
            power=self.power,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["power"] = self.power
        return config


class InverseTimeLR(BaseLRSchedule):
    """
    Inverse time decay learning rate schedule.

    LR = initial_lr / (1 + decay_rate * step / decay_steps)
    """

    name = "inverse_time"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        decay_rate: float = 0.5,
        decay_steps: int = 1000,
    ):
        super().__init__(initial_lr, total_steps, min_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def build(self) -> keras.optimizers.schedules.InverseTimeDecay:
        return keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "decay_rate": self.decay_rate,
                "decay_steps": self.decay_steps,
            }
        )
        return config


class OneCycleLR(BaseLRSchedule):
    """
    One Cycle learning rate schedule.

    Implements the 1cycle policy: warmup to max_lr, then decay below initial_lr.
    Known for achieving faster convergence and better generalization.

    Reference: "Super-Convergence" (Smith & Topin, 2018)
    """

    name = "one_cycle"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-7,
        max_lr: float = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ):
        """
        Initialize one cycle schedule.

        Args:
            initial_lr: Base learning rate (used as max_lr if not specified)
            total_steps: Total training steps
            min_lr: Not used directly, computed from div factors
            max_lr: Maximum learning rate (default: initial_lr)
            pct_start: Percentage of cycle spent increasing LR
            anneal_strategy: 'cos' or 'linear'
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = initial_lr / final_div_factor
        """
        super().__init__(initial_lr, total_steps, min_lr)
        self.max_lr = max_lr or initial_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

    def build(self) -> keras.optimizers.schedules.LearningRateSchedule:
        """Build one cycle schedule."""
        return _OneCycleSchedule(
            self.max_lr,
            self.total_steps,
            self.pct_start,
            self.div_factor,
            self.final_div_factor,
            self.anneal_strategy,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "max_lr": self.max_lr,
                "pct_start": self.pct_start,
                "anneal_strategy": self.anneal_strategy,
                "div_factor": self.div_factor,
                "final_div_factor": self.final_div_factor,
            }
        )
        return config


class CyclicLR(BaseLRSchedule):
    """
    Cyclic learning rate schedule.

    LR oscillates between min_lr and max_lr with triangular or triangular2 pattern.

    Reference: "Cyclical Learning Rates for Training Neural Networks" (Smith, 2017)
    """

    name = "cyclic"

    def __init__(
        self,
        initial_lr: float = 1e-3,
        total_steps: int = 10000,
        min_lr: float = 1e-5,
        step_size: int = 2000,
        mode: str = "triangular",
        gamma: float = 0.99,
    ):
        """
        Initialize cyclic schedule.

        Args:
            initial_lr: Maximum learning rate
            min_lr: Minimum learning rate
            step_size: Half-cycle size in steps
            mode: 'triangular', 'triangular2', or 'exp_range'
            gamma: Decay factor for exp_range mode
        """
        super().__init__(initial_lr, total_steps, min_lr)
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def build(self) -> keras.optimizers.schedules.LearningRateSchedule:
        """Build cyclic schedule."""
        return _CyclicSchedule(
            self.initial_lr, self.min_lr, self.step_size, self.mode, self.gamma
        )


# =============================================================================
# LR Schedule Registry
# =============================================================================

LR_SCHEDULE_REGISTRY: Dict[str, type] = {
    "constant": ConstantLR,
    "cosine": CosineLR,
    "warmup_cosine": WarmupCosineLR,
    "exponential": ExponentialLR,
    "step": StepLR,
    "polynomial": PolynomialLR,
    "inverse_time": InverseTimeLR,
    "one_cycle": OneCycleLR,
    "cyclic": CyclicLR,
}


def get_lr_schedule(
    schedule_type: str,
    initial_lr: float = 1e-3,
    total_steps: int = 10000,
    **kwargs,
) -> BaseLRSchedule:
    """
    Factory function to create learning rate schedule.

    Args:
        schedule_type: Type of schedule (constant, cosine, warmup_cosine, etc.)
        initial_lr: Initial learning rate
        total_steps: Total training steps
        **kwargs: Additional schedule-specific arguments

    Returns:
        BaseLRSchedule instance
    """
    if schedule_type not in LR_SCHEDULE_REGISTRY:
        logger.warning(f"Unknown schedule type: {schedule_type}, using constant")
        schedule_type = "constant"

    schedule_class = LR_SCHEDULE_REGISTRY[schedule_type]
    return schedule_class(initial_lr=initial_lr, total_steps=total_steps, **kwargs)


# =============================================================================
# Optimizers (OOP)
# =============================================================================


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers.

    Wraps Keras optimizers with consistent interface and configuration.
    """

    name: str = "base"
    supports_weight_decay: bool = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-3,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        """
        Initialize base optimizer.

        Args:
            learning_rate: Learning rate or schedule
            clipnorm: Global norm for gradient clipping
            clipvalue: Value for gradient clipping
        """
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue

    @abstractmethod
    def build(self) -> keras.optimizers.Optimizer:
        """Build and return the Keras optimizer."""
        pass

    def _get_common_kwargs(self) -> Dict[str, Any]:
        """Get common kwargs for all optimizers."""
        kwargs = {}
        if self.clipnorm is not None:
            kwargs["clipnorm"] = self.clipnorm
        if self.clipvalue is not None:
            kwargs["clipvalue"] = self.clipvalue
        return kwargs

    def get_config(self) -> Dict[str, Any]:
        """Return optimizer configuration."""
        return {
            "name": self.name,
            "learning_rate": self.learning_rate
            if isinstance(self.learning_rate, float)
            else "schedule",
            "clipnorm": self.clipnorm,
            "clipvalue": self.clipvalue,
        }

    def __repr__(self) -> str:
        lr_str = (
            f"{self.learning_rate:.2e}"
            if isinstance(self.learning_rate, float)
            else "schedule"
        )
        return f"{self.__class__.__name__}(lr={lr_str})"


class AdamOptimizer(BaseOptimizer):
    """
    Adam optimizer.

    Adaptive Moment Estimation - most popular optimizer for deep learning.

    Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015)
    """

    name = "adam"
    supports_weight_decay = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self) -> keras.optimizers.Adam:
        return keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            amsgrad=self.amsgrad,
            **self._get_common_kwargs(),
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


class AdamWOptimizer(BaseOptimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Fixes the weight decay implementation in Adam. Preferred for transformers
    and when using weight decay regularization.

    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
    """

    name = "adamw"
    supports_weight_decay = True

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-3,
        weight_decay: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self) -> keras.optimizers.AdamW:
        return keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            amsgrad=self.amsgrad,
            **self._get_common_kwargs(),
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "weight_decay": self.weight_decay,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config


class NadamOptimizer(BaseOptimizer):
    """
    Nadam optimizer - Adam with Nesterov momentum.

    Combines Adam with Nesterov accelerated gradient for potentially faster convergence.

    Reference: "Incorporating Nesterov Momentum into Adam" (Dozat, 2016)
    """

    name = "nadam"
    supports_weight_decay = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self) -> keras.optimizers.Nadam:
        return keras.optimizers.Nadam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            **self._get_common_kwargs(),
        )


class SGDOptimizer(BaseOptimizer):
    """
    Stochastic Gradient Descent with momentum.

    Classic optimizer, still competitive for many tasks especially with
    learning rate schedules. Often used for fine-tuning.
    """

    name = "sgd"
    supports_weight_decay = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.momentum = momentum
        self.nesterov = nesterov

    def build(self) -> keras.optimizers.SGD:
        return keras.optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            **self._get_common_kwargs(),
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
            }
        )
        return config


class RMSpropOptimizer(BaseOptimizer):
    """
    RMSprop optimizer.

    Good for recurrent neural networks and non-stationary objectives.
    """

    name = "rmsprop"
    supports_weight_decay = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-3,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.centered = centered

    def build(self) -> keras.optimizers.RMSprop:
        return keras.optimizers.RMSprop(
            learning_rate=self.learning_rate,
            rho=self.rho,
            momentum=self.momentum,
            epsilon=self.epsilon,
            centered=self.centered,
            **self._get_common_kwargs(),
        )


class LionOptimizer(BaseOptimizer):
    """
    Lion optimizer (Evolved Sign Momentum).

    Google's 2023 optimizer that's more memory efficient than Adam.
    Works particularly well with transformers and large batch sizes.

    Reference: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
    """

    name = "lion"
    supports_weight_decay = True

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-4,
        weight_decay: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def build(self) -> keras.optimizers.Optimizer:
        try:
            return keras.optimizers.Lion(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                **self._get_common_kwargs(),
            )
        except AttributeError:
            logger.warning("Lion optimizer not available, falling back to AdamW")
            return keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                **self._get_common_kwargs(),
            )


class AdagradOptimizer(BaseOptimizer):
    """
    Adagrad optimizer.

    Adapts learning rate per-parameter. Good for sparse gradients.
    """

    name = "adagrad"
    supports_weight_decay = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1e-2,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-7,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.initial_accumulator_value = initial_accumulator_value
        self.epsilon = epsilon

    def build(self) -> keras.optimizers.Adagrad:
        return keras.optimizers.Adagrad(
            learning_rate=self.learning_rate,
            initial_accumulator_value=self.initial_accumulator_value,
            epsilon=self.epsilon,
            **self._get_common_kwargs(),
        )


class AdadeltaOptimizer(BaseOptimizer):
    """
    Adadelta optimizer.

    Extension of Adagrad with adaptive learning rate. No need to set initial LR.
    """

    name = "adadelta"
    supports_weight_decay = False

    def __init__(
        self,
        learning_rate: Union[
            float, keras.optimizers.schedules.LearningRateSchedule
        ] = 1.0,
        rho: float = 0.95,
        epsilon: float = 1e-7,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate, clipnorm, clipvalue)
        self.rho = rho
        self.epsilon = epsilon

    def build(self) -> keras.optimizers.Adadelta:
        return keras.optimizers.Adadelta(
            learning_rate=self.learning_rate,
            rho=self.rho,
            epsilon=self.epsilon,
            **self._get_common_kwargs(),
        )


# =============================================================================
# Optimizer Registry
# =============================================================================

OPTIMIZER_REGISTRY: Dict[str, type] = {
    "adam": AdamOptimizer,
    "adamw": AdamWOptimizer,
    "nadam": NadamOptimizer,
    "sgd": SGDOptimizer,
    "rmsprop": RMSpropOptimizer,
    "lion": LionOptimizer,
    "adagrad": AdagradOptimizer,
    "adadelta": AdadeltaOptimizer,
}


def get_optimizer(
    optimizer_type: str,
    learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule] = 1e-3,
    **kwargs,
) -> BaseOptimizer:
    """
    Factory function to create optimizer.

    Args:
        optimizer_type: Type of optimizer (adam, adamw, sgd, etc.)
        learning_rate: Learning rate or schedule
        **kwargs: Additional optimizer-specific arguments

    Returns:
        BaseOptimizer instance
    """
    if optimizer_type not in OPTIMIZER_REGISTRY:
        logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam")
        optimizer_type = "adam"

    optimizer_class = OPTIMIZER_REGISTRY[optimizer_type]
    return optimizer_class(learning_rate=learning_rate, **kwargs)


# =============================================================================
# Convenience Builder
# =============================================================================


@dataclass
class OptimizerConfig:
    """
    Complete optimizer configuration including LR schedule.

    Example usage:
        config = OptimizerConfig(
            optimizer_type="adamw",
            lr_schedule_type="warmup_cosine",
            initial_lr=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.1,
        )
        optimizer = config.build(total_steps=10000)
    """

    # Optimizer settings
    optimizer_type: str = "adamw"
    initial_lr: float = 1e-3
    weight_decay: float = 0.01
    momentum: float = 0.9
    beta_1: float = 0.9
    beta_2: float = 0.999

    # LR schedule settings
    lr_schedule_type: str = "warmup_cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    min_lr: float = 1e-7
    decay_rate: float = 0.96
    decay_steps: int = 1000

    # Gradient clipping
    clipnorm: Optional[float] = 1.0
    clipvalue: Optional[float] = None

    def build(self, total_steps: int) -> keras.optimizers.Optimizer:
        """
        Build configured optimizer with LR schedule.

        Args:
            total_steps: Total training steps

        Returns:
            Configured Keras optimizer
        """
        # Build LR schedule
        warmup_steps = (
            self.warmup_steps
            if self.warmup_steps > 0
            else int(total_steps * self.warmup_ratio)
        )

        schedule_kwargs = {
            "initial_lr": self.initial_lr,
            "total_steps": total_steps,
            "min_lr": self.min_lr,
        }

        if self.lr_schedule_type in ["warmup_cosine"]:
            schedule_kwargs["warmup_steps"] = warmup_steps
        elif self.lr_schedule_type in ["exponential", "step", "inverse_time"]:
            schedule_kwargs["decay_rate"] = self.decay_rate
            schedule_kwargs["decay_steps"] = self.decay_steps

        lr_schedule = get_lr_schedule(self.lr_schedule_type, **schedule_kwargs)
        learning_rate = lr_schedule.build()

        # Build optimizer
        optimizer_kwargs = {
            "learning_rate": learning_rate,
            "clipnorm": self.clipnorm,
            "clipvalue": self.clipvalue,
        }

        if self.optimizer_type in ["adamw", "lion"]:
            optimizer_kwargs["weight_decay"] = self.weight_decay
        if self.optimizer_type == "sgd":
            optimizer_kwargs["momentum"] = self.momentum
        if self.optimizer_type in ["adam", "adamw", "nadam"]:
            optimizer_kwargs["beta_1"] = self.beta_1
            optimizer_kwargs["beta_2"] = self.beta_2

        optimizer = get_optimizer(self.optimizer_type, **optimizer_kwargs)
        return optimizer.build()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimizer_type": self.optimizer_type,
            "initial_lr": self.initial_lr,
            "weight_decay": self.weight_decay,
            "lr_schedule_type": self.lr_schedule_type,
            "warmup_ratio": self.warmup_ratio,
            "min_lr": self.min_lr,
            "clipnorm": self.clipnorm,
        }


# =============================================================================
# Backward Compatibility Functions
# =============================================================================


def create_lr_schedule(
    schedule_type: str,
    initial_lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    decay_rate: float = 0.96,
    decay_steps: int = 1000,
    min_lr: float = 1e-7,
) -> Union[float, keras.optimizers.schedules.LearningRateSchedule]:
    """
    Create learning rate schedule (backward compatible function).

    Use get_lr_schedule() for OOP interface.
    """
    kwargs = {
        "initial_lr": initial_lr,
        "total_steps": total_steps,
        "min_lr": min_lr,
    }

    if schedule_type == "warmup_cosine":
        kwargs["warmup_steps"] = warmup_steps
    elif schedule_type in ["exponential", "step", "inverse_time"]:
        kwargs["decay_rate"] = decay_rate
        kwargs["decay_steps"] = decay_steps

    schedule = get_lr_schedule(schedule_type, **kwargs)
    return schedule.build()


def create_optimizer(
    optimizer_type: str,
    learning_rate: Union[float, keras.optimizers.schedules.LearningRateSchedule],
    weight_decay: float = 0.01,
    momentum: float = 0.9,
    clipnorm: Optional[float] = None,
) -> keras.optimizers.Optimizer:
    """
    Create optimizer (backward compatible function).

    Use get_optimizer() for OOP interface.
    """
    kwargs = {
        "learning_rate": learning_rate,
        "clipnorm": clipnorm,
    }

    if optimizer_type in ["adamw", "lion"]:
        kwargs["weight_decay"] = weight_decay
    if optimizer_type == "sgd":
        kwargs["momentum"] = momentum

    optimizer = get_optimizer(optimizer_type, **kwargs)
    return optimizer.build()
