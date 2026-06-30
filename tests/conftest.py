import pytest
import numpy as np


@pytest.fixture
def sequence_length():
    return 24


@pytest.fixture
def n_features():
    return 5


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def mock_data(batch_size, sequence_length, n_features):
    """Generate synthetic normal-like data for testing."""
    return np.random.normal(
        0, 1, size=(batch_size, sequence_length, n_features)
    ).astype(np.float32)


@pytest.fixture
def mock_positive_pairs(mock_data):
    """Generate mock positive pairs (slightly jittered data)."""
    noise = np.random.normal(0, 0.05, size=mock_data.shape).astype(np.float32)
    return mock_data + noise


@pytest.fixture
def mock_negative_pairs(batch_size, sequence_length, n_features):
    """Generate mock negative pairs (anomalous-looking data)."""
    # Create 2 negatives per sample
    n_neg = 2
    negatives = np.random.normal(
        5, 2, size=(batch_size, n_neg, sequence_length, n_features)
    ).astype(np.float32)
    return negatives
