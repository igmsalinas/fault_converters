import pytest

from src.models.contrastive_ae import ContrastiveAutoencoder


@pytest.fixture
def contrastive_params(sequence_length, n_features):
    return {
        "input_shape": (sequence_length, n_features),
        "latent_dim": 8,
        "projection_dim": 16,
        "encoder_type": "conv1d",
        "encoder_filters": [16, 32],
    }


def test_contrastive_ae_build_and_forward(contrastive_params, mock_data):
    model = ContrastiveAutoencoder(**contrastive_params).build()

    latent = model.encode(mock_data)
    assert latent.shape == (mock_data.shape[0], contrastive_params["latent_dim"])

    decoded = model.decode(latent)
    assert decoded.shape == mock_data.shape

    projected = model.project(mock_data)
    assert projected.shape == (mock_data.shape[0], contrastive_params["projection_dim"])

    embeddings = model.get_embeddings(mock_data)
    assert "latent" in embeddings
    assert "projection" in embeddings


@pytest.mark.parametrize("encoder_type", ["lstm", "gru", "transformer", "mlp"])
def test_contrastive_ae_other_encoders(encoder_type, sequence_length, n_features, mock_data):
    params = {
        "input_shape": (sequence_length, n_features),
        "latent_dim": 8,
        "projection_dim": 16,
        "encoder_type": encoder_type,
        "encoder_filters": [16, 32],
        "encoder_units": [32, 16],
        "num_heads": 2,
    }
    model = ContrastiveAutoencoder(**params).build()

    latent = model.encode(mock_data)
    assert latent.shape == (mock_data.shape[0], params["latent_dim"])

    decoded = model.decode(latent)
    assert decoded.shape == mock_data.shape

    projected = model.project(mock_data)
    assert projected.shape == (mock_data.shape[0], params["projection_dim"])

    embeddings = model.get_embeddings(mock_data)
    assert "latent" in embeddings
    assert "projection" in embeddings
