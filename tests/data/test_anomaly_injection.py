import numpy as np

from src.data.anomaly_injection import AnomalyInjector, AnomalyConfig, AnomalyType


def test_anomaly_injector_initialization():
    config = AnomalyConfig(anomaly_prob=1.0)
    injector = AnomalyInjector(config)

    assert injector.config.anomaly_prob == 1.0
    assert len(injector.anomaly_types) > 0  # should load all default types


def test_anomaly_injection_generates_anomalies(mock_data):
    config = AnomalyConfig(anomaly_prob=1.0)  # Always inject

    # Restrict to a known anomaly to make asserts simpler
    injector = AnomalyInjector(config, anomaly_types=[AnomalyType.SPIKE])

    sample = mock_data[0]  # (seq_len, features)
    anomalous, is_anomalous, info = injector.inject(sample)

    # It should have modified the sample somewhere
    assert is_anomalous is not None
    assert not np.array_equal(sample, anomalous)
    assert anomalous.shape == sample.shape
    assert info == AnomalyType.SPIKE


def test_anomaly_injection_probability_respect(mock_data):
    config = AnomalyConfig(anomaly_prob=0.0)  # NEVER inject
    injector = AnomalyInjector(config)

    # inject_batch is what respects anomaly_prob internally
    anomalous_batch, masks, is_anomaly = injector.inject_batch(mock_data)

    assert not np.any(is_anomaly)
    assert np.array_equal(mock_data, anomalous_batch)
