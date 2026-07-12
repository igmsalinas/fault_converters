"""
Physics-Based Anomaly Injection
===============================

Generates CARLA negative (anomalous) samples by perturbing the *components* of a
power converter and rebuilding its small-signal transfer function, rather than by
applying heuristic signal-space distortions.

Rationale
---------
The CARLA encoder consumes the **complex frequency response** ``H(jw)`` of the
converter (the preprocessor emits normalized ``Real``/``Imag`` channels by
default, ``use_real_imag=True``). A component fault is therefore most faithfully
modelled in the ``s``-plane: a capacitor ESR increase drags the ESR zero towards
the origin, a capacitance drop pushes the LC resonance up, etc.

Instead of reconstructing an *absolute* Bode plot from the perturbed model (which
would discard every per-sample characteristic and collapse the contrastive
negatives), we apply the fault as a **multiplicative transfer ratio**::

    G(jw) = H_fault(jw) / H_nominal(jw)
    H_out(jw) = H_sample(jw) * G(jw)

``G`` is ``~1`` where the fault has no effect and warps magnitude *and* phase
coherently around the affected poles/zeros. Each perturbed sample keeps its own
identity (a *hard* negative) while carrying a physically consistent degradation.

Design
------
Converters are modelled as a polymorphic class hierarchy:

- :class:`PowerConverter` (abstract) — holds the component values and declares the
  two responsibilities every converter must define: its **own transfer function**
  (:meth:`~PowerConverter.transfer_function`) and its **degradable components**
  (:attr:`~PowerConverter.fault_modes`). Concrete subclasses
  (:class:`BuckConverter`, :class:`BoostConverter`, :class:`BuckBoostConverter`)
  implement the topology-specific control-to-output model. New converters —
  including higher-order ones (Cuk, SEPIC, flyback, resonant) — are added simply
  by subclassing; each auto-registers by ``name`` (Open/Closed).
- :class:`FaultMode` — a degradable component + the multiplicative range its value
  can drift over. A fault is applied at the **component level**: scale the
  component, ask the converter to rebuild its transfer function, done. This is
  exact physics (the damping / zero shifts fall out of the model) and needs no
  root-identification heuristics, so it works for any converter order.

The second-order PWM converters share the canonical control-to-output form
(Erickson & Maksimovic, *Fundamentals of Power Electronics*, 2nd ed., Table
8.2)::

    Gvd(s) = Gd0 * (1 + s/wz_esr) * (1 - s/wz_rhp)
                 / (1 + s/(Q*w0) + (s/w0)^2)

with topology-specific ``Gd0``, ``w0``, ``Q`` and (for boost / buck-boost) an
extra right-half-plane zero ``wz_rhp``.

Reference:
    Darban et al., "CARLA: Self-supervised Contrastive Representation Learning
    for Time Series Anomaly Detection", arXiv:2308.09296
    Erickson & Maksimovic, "Fundamentals of Power Electronics", 2nd ed., ch. 8.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import scipy.signal as signal

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Transfer Object
# ---------------------------------------------------------------------------
@dataclass
class TransferFunctionModel:
    """Abstract ``s``-plane representation of the plant (zeros, poles, gain)."""

    zeros: np.ndarray
    poles: np.ndarray
    gain: float

    def frequency_response(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate ``H(jw)`` on an angular-frequency grid (rad/s)."""
        _, h = signal.freqs_zpk(self.zeros, self.poles, self.gain, worN=omega)
        return h


# ---------------------------------------------------------------------------
# Fault modes (degradable components)
# ---------------------------------------------------------------------------
class PhysicsAnomalyType(str, Enum):
    """Physically grounded converter degradation modes (topology-agnostic)."""

    CAPACITOR_ESR = "capacitor_esr"  # Output-cap ESR increase (ageing/dry-out)
    CAPACITANCE_DROP = "capacitance_drop"  # Output-cap capacitance loss
    INDUCTOR_SATURATION = "inductor_saturation"  # Effective inductance drop
    SWITCH_DEGRADATION = "switch_degradation"  # Rds_on / conduction loss increase
    LOAD_CHANGE = "load_change"  # Load resistance deviation (Q / RHP-zero shift)


@dataclass(frozen=True)
class FaultMode:
    """
    A degradable component and the multiplicative range its value drifts over.

    A fault is applied by scaling ``component`` by a sampled ``severity`` and
    rebuilding the converter's transfer function, e.g. ``capacitor_esr`` scales
    ``esr_cap`` by 2..8x, ``capacitance_drop`` scales ``capacitance`` by
    0.4..0.85x. Because the model is rebuilt from the perturbed component, the
    resulting pole/zero and damping shifts are exact — no root heuristics.
    """

    name: str  # fault identifier (matches PhysicsAnomalyType)
    component: str  # converter component field to scale
    severity_range: Tuple[float, float]  # (low, high) multiplier on the component
    log_uniform: bool = False  # sample the multiplier log-uniformly (symmetric)

    def sample_severity(self, rng: np.random.Generator) -> float:
        lo, hi = self.severity_range
        if self.log_uniform:
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(rng.uniform(lo, hi))


#: Standard degradation modes shared by the basic PWM converters. Severity bands
#: mirror the anomalous ranges declared per converter in
#: ``data/<converter>/component_ranges.json`` so injected negatives sit *outside*
#: the healthy tolerance envelope (no overlap / gray zone).
DEFAULT_FAULT_MODES: Tuple[FaultMode, ...] = (
    FaultMode(PhysicsAnomalyType.CAPACITOR_ESR.value, "esr_cap", (3.0, 8.0)),
    FaultMode(PhysicsAnomalyType.CAPACITANCE_DROP.value, "capacitance", (0.30, 0.70)),
    FaultMode(PhysicsAnomalyType.INDUCTOR_SATURATION.value, "inductance", (0.40, 0.70)),
    FaultMode(PhysicsAnomalyType.SWITCH_DEGRADATION.value, "rds", (3.0, 15.0)),
    FaultMode(PhysicsAnomalyType.LOAD_CHANGE.value, "load", (0.3, 3.0), log_uniform=True),
)


def available_fault_modes() -> List[str]:
    """Names of the standard fault modes."""
    return [fm.name for fm in DEFAULT_FAULT_MODES]


# ---------------------------------------------------------------------------
# Second-order canonical assembly
# ---------------------------------------------------------------------------
def _assemble_second_order(
    gd0: float,
    w0: float,
    q: float,
    wz_esr: Optional[float],
    wz_rhp: Optional[float],
) -> TransferFunctionModel:
    """
    Assemble a ``zpk`` model from canonical parameters.

    Builds the poles from ``s^2 + (w0/Q) s + w0^2`` (real or complex), adds the
    LHP ESR zero and optional RHP zero, and scales the gain so ``|H(0)| == Gd0``.
    """
    zeros: List[complex] = []
    if wz_esr is not None:
        zeros.append(complex(-wz_esr, 0.0))  # LHP ESR zero
    if wz_rhp is not None:
        zeros.append(complex(+wz_rhp, 0.0))  # RHP zero (boost / buck-boost)

    poles = np.roots([1.0, w0 / q, w0 * w0]).astype(complex)
    zeros_arr = np.asarray(zeros, dtype=complex)

    num_dc = np.prod(-zeros_arr) if zeros_arr.size else (1.0 + 0j)
    den_dc = np.prod(-poles)  # == w0^2
    gain = float(np.real(abs(gd0) * den_dc / num_dc))

    return TransferFunctionModel(zeros=zeros_arr, poles=poles, gain=gain)


# ---------------------------------------------------------------------------
# Power converters (abstract base + concrete topologies)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerConverter(ABC):
    """
    Abstract PWM converter: component values + its own control-to-output model.

    Each concrete converter is responsible for two things:

    - :meth:`transfer_function` — its small-signal control-to-output ``zpk`` model
      (any order), built from the component fields below.
    - :attr:`fault_modes` — the degradable components it exposes (defaults to the
      standard set; override to add / restrict faults for a given converter).

    Subclasses auto-register by their :attr:`name`, so :meth:`from_name` /
    :func:`make_converter` can build any registered converter, and adding a new
    topology is just a subclass definition (Open/Closed).
    """

    vin: float = 50.0  # Input voltage [V]
    inductance: float = 200e-6  # L [H]
    capacitance: float = 100e-6  # C [F]
    load: float = 4.0  # Load resistance R [ohm]
    esr_cap: float = 20e-3  # Output-capacitor ESR [ohm]
    esr_ind: float = 100e-3  # Inductor DCR [ohm]
    rds: float = 10e-3  # Switch on-resistance [ohm]
    duty: float = 0.4  # Nominal duty cycle D

    #: Registry key; each concrete subclass sets this.
    name: ClassVar[str] = ""
    _registry: ClassVar[Dict[str, Type["PowerConverter"]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", ""):
            PowerConverter._registry[cls.name] = cls

    # -- responsibilities every converter defines -----------------------------
    @abstractmethod
    def transfer_function(self) -> TransferFunctionModel:
        """Nominal small-signal control-to-output ``zpk`` model."""

    @property
    def fault_modes(self) -> Tuple[FaultMode, ...]:
        """Degradable-component fault modes exposed by this converter."""
        return DEFAULT_FAULT_MODES

    # -- component perturbation ------------------------------------------------
    def scaled(self, **component_factors: float) -> "PowerConverter":
        """Return a copy with the named component fields multiplied by factors."""
        updates = {}
        for fieldname, factor in component_factors.items():
            if not hasattr(self, fieldname):
                raise ValueError(
                    f"{type(self).__name__} has no component {fieldname!r}"
                )
            updates[fieldname] = getattr(self, fieldname) * factor
        return replace(self, **updates)

    # -- construction / discovery ---------------------------------------------
    @classmethod
    def from_name(cls, name: Union[str, Enum], **components) -> "PowerConverter":
        key = name.value if isinstance(name, Enum) else str(name).strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown converter {name!r}. Available: {cls.available()}. "
                f"Define a PowerConverter subclass with name = {key!r} to add it."
            )
        return cls._registry[key](**components)

    @classmethod
    def available(cls) -> List[str]:
        """Names of all registered converter topologies."""
        return sorted(cls._registry)

    # -- shared canonical assembly (second-order topologies) ------------------
    def _canonical(
        self, m: float, gd0: float, wz_rhp: Optional[float]
    ) -> TransferFunctionModel:
        """
        Build the canonical second-order model from this converter's components.

        ``m`` is the conversion factor (``1`` for buck, ``1 - D`` for boost /
        buck-boost). Parasitics fold into ``Q`` via
        ``1/Q = w0*Le/R + w0*C*Rc + (Rl+Ron)/(w0*Le)`` with ``Le = L / m^2``.
        """
        L = max(self.inductance, 1e-15)
        C = max(self.capacitance, 1e-15)
        R = max(self.load, 1e-9)
        rc = max(self.esr_cap, 0.0)
        series_r = max(self.esr_ind, 0.0) + max(self.rds, 0.0)

        w0 = m / np.sqrt(L * C)
        le = L / (m * m)  # output-referred inductance
        inv_q = (w0 * le / R) + (w0 * C * rc) + (series_r / (w0 * le))
        q = 1.0 / max(inv_q, 1e-12)
        wz_esr = 1.0 / (rc * C) if rc > 0.0 else None
        return _assemble_second_order(abs(gd0), w0, q, wz_esr, wz_rhp)


@dataclass(frozen=True)
class BuckConverter(PowerConverter):
    """Buck (step-down): minimum phase, no RHP zero. ``Gd0 = Vg``."""

    name: ClassVar[str] = "buck"

    def transfer_function(self) -> TransferFunctionModel:
        return self._canonical(m=1.0, gd0=self.vin, wz_rhp=None)


@dataclass(frozen=True)
class BoostConverter(PowerConverter):
    """Boost (step-up): RHP zero ``wz = (1-D)^2 R / L``. ``Gd0 = Vg/(1-D)^2``."""

    name: ClassVar[str] = "boost"

    def transfer_function(self) -> TransferFunctionModel:
        m = 1.0 - float(np.clip(self.duty, 1e-3, 1.0 - 1e-3))
        wz_rhp = (m * m) * max(self.load, 1e-9) / max(self.inductance, 1e-15)
        return self._canonical(m=m, gd0=self.vin / (m * m), wz_rhp=wz_rhp)


@dataclass(frozen=True)
class BuckBoostConverter(PowerConverter):
    """Buck-boost: RHP zero ``wz = (1-D)^2 R / (D L)``. ``Gd0 = Vg/(1-D)^2``."""

    name: ClassVar[str] = "buck_boost"

    def transfer_function(self) -> TransferFunctionModel:
        d = float(np.clip(self.duty, 1e-3, 1.0 - 1e-3))
        m = 1.0 - d
        wz_rhp = (m * m) * max(self.load, 1e-9) / (d * max(self.inductance, 1e-15))
        return self._canonical(m=m, gd0=self.vin / (m * m), wz_rhp=wz_rhp)


def make_converter(name: Union[str, Enum], **components) -> PowerConverter:
    """Instantiate a registered converter by name (e.g. ``make_converter('boost')``)."""
    return PowerConverter.from_name(name, **components)


def available_converters() -> List[str]:
    """Names of all registered converter topologies."""
    return PowerConverter.available()


# ---------------------------------------------------------------------------
# Pipeline Orchestrator (drop-in negative-sample generator for CARLA)
# ---------------------------------------------------------------------------
class PhysicsAnomalyInjector:
    """
    Converts a healthy (normalized) complex-response sample into a physically
    consistent anomalous one by degrading a converter component.

    The injector is a drop-in replacement for :class:`AnomalyInjector`: its
    :meth:`inject` returns ``(anomalous_sample, mask, anomaly_type)``.

    Args:
        frequencies: Frequency grid of the samples, in **Hz** (shape ``(seq_len,)``).
        scaler: Fitted feature scaler exposing ``transform`` / ``inverse_transform``
            (operates on the ``Real``/``Imag`` — or amplitude — feature channels).
        use_real_imag: Whether the two channels are ``(Real, Imag)`` of ``H`` (default)
            or a single amplitude-in-dB channel.
        fault_modes: Names of the fault modes to sample from (default: all the
            converter exposes).
        converter: The :class:`PowerConverter` to model (default: nominal buck).
            Pass any :class:`PowerConverter` subclass instance to model any topology.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        scaler=None,
        use_real_imag: bool = True,
        fault_modes: Optional[List[str]] = None,
        converter: Optional[PowerConverter] = None,
        seed: Optional[int] = None,
    ):
        frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
        if frequencies.size == 0:
            raise ValueError("PhysicsAnomalyInjector requires a non-empty frequency grid")

        self.frequencies = frequencies
        self.omega = 2.0 * np.pi * frequencies  # rad/s
        self.scaler = scaler
        self.use_real_imag = use_real_imag
        self.converter = converter or BuckConverter()
        self.rng = np.random.default_rng(seed)

        available = {fm.name: fm for fm in self.converter.fault_modes}
        if fault_modes:
            selected = []
            for n in fault_modes:
                key = n.value if isinstance(n, Enum) else str(n)
                if key in available:
                    selected.append(available[key])
            self._fault_modes = selected or list(available.values())
        else:
            self._fault_modes = list(available.values())
        self.fault_mode_names = [fm.name for fm in self._fault_modes]

        # Nominal response, evaluated once.
        self.nominal_model = self.converter.transfer_function()
        self._h_nominal = self.nominal_model.frequency_response(self.omega)

        logger.info(
            "Initialized PhysicsAnomalyInjector (%s) with %d fault modes over %d freqs",
            self.converter.name,
            len(self._fault_modes),
            self.frequencies.size,
        )

    # -- normalization helpers ------------------------------------------------
    def _to_features(self, sample: np.ndarray) -> np.ndarray:
        """Un-normalize a sample to physical feature space (Real/Imag or dB)."""
        if self.scaler is not None:
            return self.scaler.inverse_transform(sample[None, ...])[0]
        return sample.copy()

    def _to_normalized(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            return self.scaler.transform(features[None, ...])[0]
        return features

    # -- core -----------------------------------------------------------------
    def _fault_ratio(self, fault: FaultMode, severity: float) -> np.ndarray:
        """Multiplicative transfer ratio ``G(jw) = H_fault / H_nominal``."""
        faulted = self.converter.scaled(**{fault.component: severity})
        h_fault = faulted.transfer_function().frequency_response(self.omega)
        denom = self._h_nominal
        eps = 1e-30
        return h_fault / np.where(np.abs(denom) < eps, eps, denom)

    def _resolve_fault(self, anomaly_type) -> FaultMode:
        if anomaly_type is None:
            return self._fault_modes[int(self.rng.integers(len(self._fault_modes)))]
        key = anomaly_type.value if isinstance(anomaly_type, Enum) else str(anomaly_type)
        for fm in self._fault_modes:
            if fm.name == key:
                return fm
        raise ValueError(
            f"Fault mode {anomaly_type!r} not available for {self.converter.name}. "
            f"Available: {self.fault_mode_names}"
        )

    def inject(
        self,
        data: np.ndarray,
        anomaly_type: Optional[str] = None,
        severity: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Inject a physically consistent fault into a single sample.

        Args:
            data: Normalized sample, shape ``(seq_len, n_features)``.
            anomaly_type: Specific fault mode name (random if ``None``).
            severity: Component multiplier (sampled from the fault range if ``None``).

        Returns:
            ``(anomalous_sample, mask, anomaly_type)`` with an all-``True`` mask
            (the fault is a structural / collective anomaly).
        """
        fault = self._resolve_fault(anomaly_type)
        if severity is None:
            severity = fault.sample_severity(self.rng)

        ratio = self._fault_ratio(fault, severity)
        features = self._to_features(np.asarray(data, dtype=float))

        if self.use_real_imag and features.shape[-1] >= 2:
            h_sample = features[:, 0] + 1j * features[:, 1]
            h_out = h_sample * ratio
            out_features = features.copy()
            out_features[:, 0] = np.real(h_out)
            out_features[:, 1] = np.imag(h_out)
        else:
            # Amplitude-only (dB) channel: apply the magnitude ratio in dB.
            out_features = features.copy()
            out_features[:, 0] = features[:, 0] + 20.0 * np.log10(
                np.clip(np.abs(ratio), 1e-12, None)
            )

        anomalous = self._to_normalized(out_features).astype(data.dtype, copy=False)
        mask = np.ones(anomalous.shape[0], dtype=bool)
        return anomalous, mask, fault.name

    def inject_batch(
        self,
        data: np.ndarray,
        anomaly_prob: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject faults into a batch ``(batch, seq_len, n_features)``."""
        batch_size, seq_len = data.shape[0], data.shape[1]
        anomalous = data.copy()
        masks = np.zeros((batch_size, seq_len), dtype=bool)
        is_anomaly = self.rng.random(batch_size) < anomaly_prob

        for i in range(batch_size):
            if is_anomaly[i]:
                anomalous[i], masks[i], _ = self.inject(data[i])
        return anomalous, masks, is_anomaly
