"""
Anomaly Injection Module
========================

Synthetic anomaly generators for transfer function data.
Based on CARLA approach: inject anomalies as negative samples for contrastive learning.

Reference:
    Darban et al., "CARLA: Self-supervised Contrastive Representation Learning
    for Time Series Anomaly Detection", arXiv:2308.09296
"""

import numpy as np
from typing import Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyType(Enum):
    """Types of synthetic anomalies for transfer functions."""

    # Point anomalies (local)
    SPIKE = "spike"  # Sudden spike in amplitude/phase
    DROPOUT = "dropout"  # Value drops to zero or baseline
    NOISE = "noise"  # High-frequency noise injection

    # Contextual anomalies (shape distortion)
    AMPLITUDE_SHIFT = "amplitude_shift"  # DC offset in amplitude
    PHASE_SHIFT = "phase_shift"  # DC offset in phase
    GAIN_CHANGE = "gain_change"  # Overall gain increase/decrease
    SLOPE_CHANGE = "slope_change"  # Change in roll-off rate

    # Collective anomalies (structural)
    RESONANCE_SHIFT = "resonance_shift"  # Shift resonance frequency
    RESONANCE_ADD = "resonance_add"  # Add spurious resonance
    BANDWIDTH_CHANGE = "bandwidth_change"  # Change bandwidth
    OSCILLATION = "oscillation"  # Add oscillatory behavior

    # Physics-based anomalies (simulating component faults)
    CAPACITOR_DEGRADATION = "capacitor_degradation"  # ESR increase effect
    INDUCTOR_SATURATION = "inductor_saturation"  # Reduced inductance
    SWITCH_DEGRADATION = "switch_degradation"  # Increased Rds effect


@dataclass
class AnomalyConfig:
    """Configuration for anomaly injection."""

    # Probability of applying each anomaly type
    anomaly_prob: float = 0.5

    # Point anomaly parameters
    spike_magnitude: Tuple[float, float] = (1.2, 2.5)  # Reduced from (2.0, 5.0)
    spike_width: Tuple[int, int] = (1, 3)  # Reduced from (1, 5)
    dropout_prob: float = 0.05  # Reduced from 0.1
    noise_std: Tuple[float, float] = (0.05, 0.2)  # Reduced from (0.1, 0.5)

    # Contextual anomaly parameters
    amplitude_shift_range: Tuple[float, float] = (-3.0, 3.0)  # Reduced from (-10, 10)
    phase_shift_range: Tuple[float, float] = (-10.0, 10.0)  # Reduced from (-30, 30)
    gain_change_range: Tuple[float, float] = (0.8, 1.2)  # Reduced from (0.5, 2.0)
    slope_change_range: Tuple[float, float] = (0.8, 1.2)  # Reduced from (0.5, 2.0)

    # Collective anomaly parameters
    resonance_shift_range: Tuple[float, float] = (-0.1, 0.1)  # Reduced from (-0.3, 0.3)
    resonance_q_range: Tuple[float, float] = (1.0, 5.0)  # Reduced from (1.0, 10.0)
    bandwidth_change_range: Tuple[float, float] = (0.8, 1.2)  # Reduced from (0.5, 2.0)
    oscillation_freq_range: Tuple[int, int] = (2, 6)  # Reduced from (2, 10)
    oscillation_amp_range: Tuple[float, float] = (0.2, 1.0)  # Reduced from (0.5, 3.0)


class BaseAnomalyGenerator(ABC):
    """Base class for anomaly generators."""

    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()

    @abstractmethod
    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate anomaly in the data.

        Args:
            data: Input data, shape (seq_len, n_features) or (seq_len,)
            amplitude_idx: Index of amplitude feature
            phase_idx: Index of phase feature

        Returns:
            Tuple of (anomalous_data, anomaly_mask)
            anomaly_mask indicates which points are anomalous
        """
        pass


class SpikeAnomaly(BaseAnomalyGenerator):
    """Generate spike anomalies (sudden increases/decreases)."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]
        mask = np.zeros(seq_len, dtype=bool)

        # Random spike position
        pos = np.random.randint(0, seq_len - self.config.spike_width[1])
        width = np.random.randint(*self.config.spike_width)

        # Random magnitude and direction
        magnitude = np.random.uniform(*self.config.spike_magnitude)
        direction = np.random.choice([-1, 1])

        # Apply spike to amplitude
        if data.ndim == 2:
            # Calculate spike based on local std
            local_std = np.std(data[:, amplitude_idx])
            spike_value = direction * magnitude * local_std
            data[pos : pos + width, amplitude_idx] += spike_value
        else:
            local_std = np.std(data)
            spike_value = direction * magnitude * local_std
            data[pos : pos + width] += spike_value

        mask[pos : pos + width] = True

        return data, mask


class DropoutAnomaly(BaseAnomalyGenerator):
    """Generate dropout anomalies (values drop to baseline)."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]
        mask = np.zeros(seq_len, dtype=bool)

        # Select random points for dropout
        n_dropout = max(1, int(seq_len * self.config.dropout_prob))
        dropout_idx = np.random.choice(seq_len, n_dropout, replace=False)

        if data.ndim == 2:
            # Set to minimum value (simulating sensor failure)
            min_val = np.min(data[:, amplitude_idx])
            data[dropout_idx, amplitude_idx] = min_val
        else:
            data[dropout_idx] = np.min(data)

        mask[dropout_idx] = True

        return data, mask


class NoiseAnomaly(BaseAnomalyGenerator):
    """Inject high-frequency noise into signal."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Noise in a random segment
        start = np.random.randint(0, seq_len // 2)
        end = np.random.randint(start + seq_len // 4, seq_len)

        noise_std = np.random.uniform(*self.config.noise_std)

        if data.ndim == 2:
            signal_std = np.std(data[:, amplitude_idx])
            noise = np.random.normal(0, noise_std * signal_std, end - start)
            data[start:end, amplitude_idx] += noise
        else:
            signal_std = np.std(data)
            noise = np.random.normal(0, noise_std * signal_std, end - start)
            data[start:end] += noise

        mask = np.zeros(seq_len, dtype=bool)
        mask[start:end] = True

        return data, mask


class AmplitudeShiftAnomaly(BaseAnomalyGenerator):
    """Add DC offset to amplitude (gain change)."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        shift = np.random.uniform(*self.config.amplitude_shift_range)

        if data.ndim == 2:
            data[:, amplitude_idx] += shift
        else:
            data += shift

        # Full sequence is anomalous for contextual anomalies
        mask = np.ones(seq_len, dtype=bool)

        return data, mask


class PhaseShiftAnomaly(BaseAnomalyGenerator):
    """Add DC offset to phase."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        shift = np.random.uniform(*self.config.phase_shift_range)

        if data.ndim == 2:
            data[:, phase_idx] += shift
        else:
            pass  # Can't apply phase shift to single channel

        mask = np.ones(seq_len, dtype=bool)

        return data, mask


class ResonanceShiftAnomaly(BaseAnomalyGenerator):
    """Shift the resonance frequency (squeeze/stretch frequency axis)."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Shift factor
        shift = 1.0 + np.random.uniform(*self.config.resonance_shift_range)

        # Resample the data (frequency axis compression/expansion)
        old_idx = np.arange(seq_len)
        new_idx = np.linspace(0, (seq_len - 1) * shift, seq_len)
        new_idx = np.clip(new_idx, 0, seq_len - 1)

        if data.ndim == 2:
            for feat in range(data.shape[1]):
                data[:, feat] = np.interp(old_idx, new_idx, data[:, feat])
        else:
            data = np.interp(old_idx, new_idx, data)

        mask = np.ones(seq_len, dtype=bool)

        return data, mask


class ResonanceAddAnomaly(BaseAnomalyGenerator):
    """Add spurious resonance peak."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Random resonance position
        center = np.random.randint(seq_len // 4, 3 * seq_len // 4)
        q_factor = np.random.uniform(*self.config.resonance_q_range)

        # Create resonance peak (second-order system response)
        x = np.arange(seq_len)
        width = seq_len / (2 * q_factor)
        resonance = 1 / (1 + ((x - center) / width) ** 2)

        # Peak magnitude
        if data.ndim == 2:
            peak_amp = np.std(data[:, amplitude_idx]) * 2
            data[:, amplitude_idx] += resonance * peak_amp
            # Add phase shift around resonance
            phase_effect = -np.arctan2(x - center, width) * 30  # degrees
            data[:, phase_idx] += phase_effect
        else:
            peak_amp = np.std(data) * 2
            data += resonance * peak_amp

        mask = resonance > 0.1  # Mark significant resonance region

        return data, mask


class OscillationAnomaly(BaseAnomalyGenerator):
    """Add oscillatory behavior (ripple)."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Random oscillation parameters
        n_cycles = np.random.randint(*self.config.oscillation_freq_range)
        amplitude = np.random.uniform(*self.config.oscillation_amp_range)

        # Create oscillation
        x = np.linspace(0, n_cycles * 2 * np.pi, seq_len)
        oscillation = amplitude * np.sin(x)

        if data.ndim == 2:
            signal_std = np.std(data[:, amplitude_idx])
            data[:, amplitude_idx] += oscillation * signal_std
        else:
            signal_std = np.std(data)
            data += oscillation * signal_std

        mask = np.ones(seq_len, dtype=bool)

        return data, mask


class SlopeChangeAnomaly(BaseAnomalyGenerator):
    """Change the slope/roll-off rate."""

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Change slope from a random point
        change_point = np.random.randint(seq_len // 4, 3 * seq_len // 4)
        slope_factor = np.random.uniform(*self.config.slope_change_range)

        if data.ndim == 2:
            # Get original slope
            original = data[change_point:, amplitude_idx].copy()
            baseline = original[0]
            # Apply slope change
            data[change_point:, amplitude_idx] = (
                baseline + (original - baseline) * slope_factor
            )
        else:
            original = data[change_point:].copy()
            baseline = original[0]
            data[change_point:] = baseline + (original - baseline) * slope_factor

        mask = np.zeros(seq_len, dtype=bool)
        mask[change_point:] = True

        return data, mask


class CapacitorDegradationAnomaly(BaseAnomalyGenerator):
    """Simulate capacitor degradation (ESR increase).
    Effect: Increased damping, slight shift in resonance/zero.
    """

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Shift factor (subtle)
        degradation_factor = np.random.uniform(1.05, 1.2)

        if data.ndim == 2:
            # Apply to both amplitude and phase for a physical effect
            # We simulate a "dampening" and a slight shift
            data[:, amplitude_idx] -= np.linspace(0, 2.0, seq_len) * (
                degradation_factor - 1
            )
            data[:, phase_idx] -= np.linspace(0, 5.0, seq_len) * (
                degradation_factor - 1
            )
        else:
            data -= np.linspace(0, 2.0, seq_len) * (degradation_factor - 1)

        return data, np.ones(seq_len, dtype=bool)


class InductorSaturationAnomaly(BaseAnomalyGenerator):
    """Simulate inductor saturation (inductance decrease).
    Effect: Resonance frequency shifts higher.
    """

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # In saturation, L decreases, so f_res = 1/(2*pi*sqrt(LC)) increases.
        # We simulate this by stretching the frequency axis slightly.
        shift = 1.0 + np.random.uniform(0.02, 0.08)

        old_idx = np.arange(seq_len)
        new_idx = np.linspace(0, (seq_len - 1) * (1 / shift), seq_len)

        if data.ndim == 2:
            for feat in range(data.shape[1]):
                data[:, feat] = np.interp(old_idx, new_idx, data[:, feat])
        else:
            data = np.interp(old_idx, new_idx, data)

        return data, np.ones(seq_len, dtype=bool)


class SwitchDegradationAnomaly(BaseAnomalyGenerator):
    """Simulate switch degradation (increased Rds_on).
    Effect: Broadband gain reduction, increased losses.
    """

    def __call__(
        self,
        data: np.ndarray,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        data = data.copy()
        seq_len = data.shape[0]

        # Increased resistance drops overall gain
        loss_db = np.random.uniform(0.5, 2.0)

        if data.ndim == 2:
            data[:, amplitude_idx] -= loss_db
        else:
            data -= loss_db

        return data, np.ones(seq_len, dtype=bool)


class AnomalyInjector:
    """
    Main class for injecting synthetic anomalies into transfer function data.

    Supports multiple anomaly types for contrastive learning.
    """

    # Map anomaly types to generators
    GENERATORS = {
        AnomalyType.SPIKE: SpikeAnomaly,
        AnomalyType.DROPOUT: DropoutAnomaly,
        AnomalyType.AMPLITUDE_SHIFT: AmplitudeShiftAnomaly,
        AnomalyType.PHASE_SHIFT: PhaseShiftAnomaly,
        AnomalyType.RESONANCE_SHIFT: ResonanceShiftAnomaly,
        AnomalyType.RESONANCE_ADD: ResonanceAddAnomaly,
        AnomalyType.OSCILLATION: OscillationAnomaly,
        AnomalyType.SLOPE_CHANGE: SlopeChangeAnomaly,
        AnomalyType.CAPACITOR_DEGRADATION: CapacitorDegradationAnomaly,
        AnomalyType.INDUCTOR_SATURATION: InductorSaturationAnomaly,
        AnomalyType.SWITCH_DEGRADATION: SwitchDegradationAnomaly,
    }

    def __init__(
        self,
        config: Optional[AnomalyConfig] = None,
        anomaly_types: Optional[List[AnomalyType]] = None,
    ):
        """
        Initialize anomaly injector.

        Args:
            config: Anomaly configuration
            anomaly_types: List of anomaly types to use (None for all)
        """
        self.config = config or AnomalyConfig()

        if anomaly_types is None:
            # Filter out NOISE if it was in the Enum but we don't want it by default
            anomaly_types = [
                t for t in self.GENERATORS.keys() if t != AnomalyType.NOISE
            ]

        self.anomaly_types = anomaly_types
        self.generators = {
            atype: self.GENERATORS[atype](self.config)
            for atype in anomaly_types
            if atype in self.GENERATORS
        }

        logger.info(
            f"Initialized AnomalyInjector with {len(self.generators)} anomaly types"
        )

    def inject(
        self,
        data: np.ndarray,
        anomaly_type: Optional[AnomalyType] = None,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, AnomalyType]:
        """
        Inject anomaly into data.

        Args:
            data: Input data, shape (seq_len, n_features)
            anomaly_type: Specific anomaly type (None for random)
            amplitude_idx: Index of amplitude feature
            phase_idx: Index of phase feature

        Returns:
            Tuple of (anomalous_data, anomaly_mask, anomaly_type)
        """
        if anomaly_type is None:
            anomaly_type = np.random.choice(list(self.generators.keys()))

        generator = self.generators[anomaly_type]
        anomalous_data, mask = generator(data, amplitude_idx, phase_idx)

        return anomalous_data, mask, anomaly_type

    def inject_batch(
        self,
        data: np.ndarray,
        anomaly_prob: Optional[float] = None,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject anomalies into a batch of data.

        Args:
            data: Input batch, shape (batch_size, seq_len, n_features)
            anomaly_prob: Probability of injecting anomaly (None uses config)
            amplitude_idx: Index of amplitude feature
            phase_idx: Index of phase feature

        Returns:
            Tuple of (anomalous_data, anomaly_masks, is_anomaly)
            is_anomaly: Boolean array indicating which samples have anomalies
        """
        if anomaly_prob is None:
            anomaly_prob = self.config.anomaly_prob

        batch_size = data.shape[0]
        seq_len = data.shape[1]

        anomalous_data = data.copy()
        masks = np.zeros((batch_size, seq_len), dtype=bool)
        is_anomaly = np.random.random(batch_size) < anomaly_prob

        for i in range(batch_size):
            if is_anomaly[i]:
                anomalous_data[i], masks[i], _ = self.inject(
                    data[i],
                    amplitude_idx=amplitude_idx,
                    phase_idx=phase_idx,
                )

        return anomalous_data, masks, is_anomaly

    def generate_contrastive_pairs(
        self,
        data: np.ndarray,
        n_positive: int = 1,
        n_negative: int = 1,
        amplitude_idx: int = 0,
        phase_idx: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate contrastive pairs for a single sample.

        Args:
            data: Input sample, shape (seq_len, n_features)
            n_positive: Number of positive pairs (augmented normal)
            n_negative: Number of negative pairs (with anomalies)
            amplitude_idx: Index of amplitude feature
            phase_idx: Index of phase feature

        Returns:
            Tuple of (anchor, positives, negatives)
            - anchor: Original data
            - positives: Augmented normal samples
            - negatives: Samples with injected anomalies
        """
        positives = []
        negatives = []

        # Generate positive pairs (light augmentation, no anomalies)
        for _ in range(n_positive):
            augmented = self._light_augment(data)
            positives.append(augmented)

        # Generate negative pairs (with anomalies)
        for _ in range(n_negative):
            anomalous, _, _ = self.inject(
                data,
                amplitude_idx=amplitude_idx,
                phase_idx=phase_idx,
            )
            negatives.append(anomalous)

        positives = (
            np.stack(positives, axis=0) if positives else np.empty((0,) + data.shape)
        )
        negatives = (
            np.stack(negatives, axis=0) if negatives else np.empty((0,) + data.shape)
        )

        return data, positives, negatives

    def _light_augment(self, data: np.ndarray) -> np.ndarray:
        """Apply light augmentation for positive pairs."""
        augmented = data.copy()

        # Small Gaussian noise
        noise_std = 0.01 * np.std(data)
        augmented += np.random.normal(0, noise_std, data.shape)

        # Small scaling
        scale = np.random.uniform(0.98, 1.02)
        augmented *= scale

        return augmented


def create_anomaly_injector(
    anomaly_types: Optional[List[str]] = None,
    **config_kwargs,
) -> AnomalyInjector:
    """
    Factory function to create AnomalyInjector.

    Args:
        anomaly_types: List of anomaly type names (strings)
        **config_kwargs: Arguments for AnomalyConfig

    Returns:
        AnomalyInjector instance
    """
    config = AnomalyConfig(**config_kwargs)

    if anomaly_types is not None:
        anomaly_types = [AnomalyType(t) for t in anomaly_types]

    return AnomalyInjector(config=config, anomaly_types=anomaly_types)
