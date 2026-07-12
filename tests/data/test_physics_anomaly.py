import numpy as np
import pytest
from dataclasses import dataclass
from typing import ClassVar

from src.data.physics_anomaly import (
    BoostConverter,
    BuckBoostConverter,
    BuckConverter,
    DEFAULT_FAULT_MODES,
    FaultMode,
    PhysicsAnomalyInjector,
    PhysicsAnomalyType,
    PowerConverter,
    TransferFunctionModel,
    available_converters,
    available_fault_modes,
    make_converter,
)


@pytest.fixture
def frequencies():
    # Log sweep resembling the buck AC-sweep grid (100 Hz .. 200 kHz).
    return np.logspace(2, np.log10(2e5), 256)


class _IdentityScaler:
    """Per-feature standard scaler stub for (batch, seq, feat) arrays."""

    def __init__(self, features):
        self.mean_ = features.mean(axis=(0, 1), keepdims=True)
        self.std_ = features.std(axis=(0, 1), keepdims=True) + 1e-8

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_


def _scaler_and_sample(converter, frequencies):
    H = converter.transfer_function().frequency_response(2 * np.pi * frequencies)
    feats = np.stack([H.real, H.imag], axis=-1)
    scaler = _IdentityScaler(feats[None])
    sample = scaler.transform(feats[None])[0].astype(np.float32)
    return scaler, sample, H


# ---------------------------------------------------------------------------
# Converter hierarchy + transfer functions
# ---------------------------------------------------------------------------
def test_converters_auto_register():
    assert set(available_converters()) >= {"buck", "boost", "buck_boost"}
    assert isinstance(make_converter("boost"), BoostConverter)
    assert isinstance(make_converter("buck"), BuckConverter)


def test_from_name_unknown_raises():
    with pytest.raises(ValueError, match="Unknown converter"):
        PowerConverter.from_name("sepic")


def test_power_converter_is_abstract():
    with pytest.raises(TypeError):
        PowerConverter()  # abstract transfer_function


def test_buck_transfer_function_physics():
    model = BuckConverter().transfer_function()
    assert model.zeros.shape == (1,)  # only the LHP ESR zero
    assert model.poles.shape == (2,)
    f0 = np.abs(model.poles[0]) / (2 * np.pi)
    assert 900 < f0 < 1400  # LC resonance ~ 1.1 kHz
    fz = np.abs(model.zeros[0]) / (2 * np.pi)
    assert 60e3 < fz < 100e3  # ESR zero ~ 80 kHz
    assert np.all(model.zeros.real < 0)  # minimum phase


def _real_zeros(model):
    z = model.zeros
    return z[np.abs(z.imag) < 1e-6]


@pytest.mark.parametrize("cls", [BoostConverter, BuckBoostConverter])
def test_boost_family_has_rhp_zero(cls):
    model = cls().transfer_function()
    real = _real_zeros(model)
    assert real[real.real > 0].size == 1  # RHP zero
    assert real[real.real < 0].size == 1  # ESR zero


def test_dc_gain_matches_topology_formula():
    vg, d = 50.0, 0.4
    dc_buck = np.abs(BuckConverter(vin=vg, duty=d).transfer_function()
                     .frequency_response(np.array([1.0]))[0])
    assert dc_buck == pytest.approx(vg, rel=0.05)
    for cls in (BoostConverter, BuckBoostConverter):
        dc = np.abs(cls(vin=vg, duty=d).transfer_function()
                    .frequency_response(np.array([1.0]))[0])
        assert dc == pytest.approx(vg / (1 - d) ** 2, rel=0.05)


def test_resonance_scales_with_conversion_ratio():
    d = 0.4
    buck = BuckConverter(duty=d).transfer_function()
    boost = BoostConverter(duty=d).transfer_function()
    ratio = np.abs(boost.poles[0]) / np.abs(buck.poles[0])
    assert ratio == pytest.approx(1 - d, rel=0.05)


def test_custom_parameters_set_resonance_and_gain():
    conv = BuckConverter(vin=12.0, inductance=47e-6, capacitance=220e-6, load=2.0, duty=0.42)
    model = conv.transfer_function()
    f0 = np.abs(model.poles[0]) / (2 * np.pi)
    expected = 1.0 / (2 * np.pi * np.sqrt(47e-6 * 220e-6))
    assert f0 == pytest.approx(expected, rel=0.02)
    dc = np.abs(model.frequency_response(np.array([1.0]))[0])
    assert dc == pytest.approx(12.0, rel=0.05)


def test_scaled_returns_same_type_with_scaled_component():
    conv = BuckConverter()
    faulted = conv.scaled(capacitance=0.5, esr_cap=4.0)
    assert isinstance(faulted, BuckConverter)
    assert faulted.capacitance == pytest.approx(conv.capacitance * 0.5)
    assert faulted.esr_cap == pytest.approx(conv.esr_cap * 4.0)
    assert conv.capacitance == 100e-6  # original unchanged (frozen)


def test_scaled_unknown_component_raises():
    with pytest.raises(ValueError, match="no component"):
        BuckConverter().scaled(nonexistent=2.0)


# ---------------------------------------------------------------------------
# Fault modes
# ---------------------------------------------------------------------------
def test_default_fault_modes_cover_enum():
    assert {fm.name for fm in DEFAULT_FAULT_MODES} == {t.value for t in PhysicsAnomalyType}
    assert available_fault_modes() == [fm.name for fm in DEFAULT_FAULT_MODES]


def test_fault_mode_severity_sampling_in_range():
    rng = np.random.default_rng(0)
    fm = FaultMode("x", "capacitance", (0.4, 0.85))
    vals = [fm.sample_severity(rng) for _ in range(200)]
    assert all(0.4 <= v <= 0.85 for v in vals)

    log_fm = FaultMode("y", "load", (0.3, 3.0), log_uniform=True)
    lv = [log_fm.sample_severity(rng) for _ in range(200)]
    assert all(0.3 <= v <= 3.0 for v in lv)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------
def test_inject_produces_finite_distinct_negative(frequencies):
    scaler, sample, _ = _scaler_and_sample(BuckConverter(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=0)
    for name in available_fault_modes():
        anomalous, mask, atype = inj.inject(sample, anomaly_type=name)
        assert atype == name
        assert anomalous.shape == sample.shape
        assert anomalous.dtype == sample.dtype
        assert np.all(np.isfinite(anomalous))
        assert mask.all()
        assert not np.allclose(anomalous, sample)


def test_inject_defaults_to_buck(frequencies):
    inj = PhysicsAnomalyInjector(frequencies=frequencies, seed=0)
    assert isinstance(inj.converter, BuckConverter)


@pytest.mark.parametrize("cls", [BuckConverter, BoostConverter, BuckBoostConverter])
def test_injector_generalizes_across_topologies(frequencies, cls):
    scaler, sample, _ = _scaler_and_sample(cls(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler,
                                 converter=cls(), seed=0)
    for name in available_fault_modes():
        anomalous, _, _ = inj.inject(sample, anomaly_type=name)
        assert np.all(np.isfinite(anomalous))
        assert not np.allclose(anomalous, sample)


def test_reactive_faults_preserve_dc_gain(frequencies):
    scaler, sample, H = _scaler_and_sample(BuckConverter(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=1)
    # Buck DC gain = Vg, independent of any component -> every fault preserves it.
    for name in available_fault_modes():
        anomalous, _, _ = inj.inject(sample, anomaly_type=name, severity=None)
        faulty = scaler.inverse_transform(anomalous[None])[0]
        dc_faulty = abs(faulty[0, 0] + 1j * faulty[0, 1])
        dc_healthy = abs(H[0])
        assert dc_faulty == pytest.approx(dc_healthy, rel=0.02)


def test_esr_increase_lifts_high_frequency_magnitude(frequencies):
    scaler, sample, H = _scaler_and_sample(BuckConverter(), frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=2)
    anomalous, _, _ = inj.inject(
        sample, anomaly_type=PhysicsAnomalyType.CAPACITOR_ESR, severity=5.0
    )
    faulty = scaler.inverse_transform(anomalous[None])[0]
    mag_healthy = np.abs(H)
    mag_faulty = np.abs(faulty[:, 0] + 1j * faulty[:, 1])
    assert mag_faulty[-1] > mag_healthy[-1]


def test_inject_batch_respects_probability(frequencies):
    scaler, sample, _ = _scaler_and_sample(BuckConverter(), frequencies)
    batch = np.repeat(sample[None], 8, axis=0)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler, seed=3)

    _, _, none = inj.inject_batch(batch, anomaly_prob=0.0)
    assert not none.any()

    anom, masks, flags = inj.inject_batch(batch, anomaly_prob=1.0)
    assert flags.all()
    assert masks.all()
    assert not np.allclose(anom, batch)


def test_unknown_fault_mode_raises(frequencies):
    inj = PhysicsAnomalyInjector(frequencies=frequencies, seed=0)
    with pytest.raises(ValueError, match="not available"):
        inj.inject(np.zeros((frequencies.size, 2), dtype=np.float32),
                   anomaly_type="not_a_fault")


def test_amplitude_only_channel(frequencies):
    H = BuckConverter().transfer_function().frequency_response(2 * np.pi * frequencies)
    db = (20 * np.log10(np.abs(H)))[:, None].astype(np.float32)
    inj = PhysicsAnomalyInjector(
        frequencies=frequencies, scaler=None, use_real_imag=False, seed=5
    )
    anomalous, _, _ = inj.inject(db, anomaly_type=PhysicsAnomalyType.CAPACITANCE_DROP)
    assert anomalous.shape == db.shape
    assert np.all(np.isfinite(anomalous))
    assert not np.allclose(anomalous, db)


# ---------------------------------------------------------------------------
# Extensibility: a custom higher-order converter (subclass auto-registers)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FourthOrderConverter(PowerConverter):
    """Two-resonance converter the canonical second-order form cannot express."""

    name: ClassVar[str] = "fourth_order_test"

    def transfer_function(self) -> TransferFunctionModel:
        poles = np.array(
            [-500 + 6000j, -500 - 6000j, -800 + 30000j, -800 - 30000j], dtype=complex
        )
        # ESR zero depends on esr_cap * capacitance so capacitor faults perturb it.
        zeros = np.array([-1.0 / (self.esr_cap * self.capacitance)], dtype=complex)
        return TransferFunctionModel(zeros=zeros, poles=poles, gain=1e18)


def test_custom_subclass_registers_and_flows_through_injector(frequencies):
    assert "fourth_order_test" in available_converters()
    conv = make_converter("fourth_order_test")
    assert isinstance(conv, _FourthOrderConverter)

    model = conv.transfer_function()
    assert model.poles.shape == (4,)  # canonical template could never do this

    scaler, sample, _ = _scaler_and_sample(conv, frequencies)
    inj = PhysicsAnomalyInjector(frequencies=frequencies, scaler=scaler,
                                 converter=conv, seed=0)
    anomalous, _, _ = inj.inject(sample, anomaly_type=PhysicsAnomalyType.CAPACITOR_ESR)
    assert np.all(np.isfinite(anomalous))
    assert not np.allclose(anomalous, sample)
