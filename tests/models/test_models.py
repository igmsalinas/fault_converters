import pytest

from src.models.conv1d_ae import Conv1DAutoencoder
from src.models.lstm_ae import LSTMAutoencoder
from src.models.transformer_ae import TransformerAutoencoder
from src.models.vae import VariationalAutoencoder


@pytest.fixture
def ae_params(sequence_length, n_features):
    return {
        "input_shape": (sequence_length, n_features),
        "latent_dim": 8,
    }


def test_conv1d_autoencoder_build_and_forward(ae_params, mock_data):
    model = Conv1DAutoencoder(**ae_params, filters=[16, 32]).build()

    latent = model.encode(mock_data)
    assert latent.shape == (mock_data.shape[0], ae_params["latent_dim"])

    decoded = model.decode(latent)
    assert decoded.shape == mock_data.shape

    predicted = model.predict(mock_data)
    assert predicted.shape == mock_data.shape


def test_lstm_autoencoder_build_and_forward(ae_params, mock_data):
    model = LSTMAutoencoder(**ae_params, lstm_units=[16]).build()

    latent = model.encode(mock_data)
    assert latent.shape == (mock_data.shape[0], ae_params["latent_dim"])

    decoded = model.decode(latent)
    assert decoded.shape == mock_data.shape

    predicted = model.predict(mock_data)
    assert predicted.shape == mock_data.shape


def test_transformer_autoencoder_build_and_forward(ae_params, mock_data):
    model = TransformerAutoencoder(
        **ae_params, d_model=16, num_heads=2, ff_dim=32, num_layers=1
    ).build()

    latent = model.encode(mock_data)
    assert latent.shape == (mock_data.shape[0], ae_params["latent_dim"])

    decoded = model.decode(latent)
    assert decoded.shape == mock_data.shape

    predicted = model.predict(mock_data)
    assert predicted.shape == mock_data.shape


def test_variational_autoencoder_build_and_train_step(ae_params, mock_data):
    model = VariationalAutoencoder(
        **ae_params,
        encoder_units=[16],
        decoder_units=[16],
        filters=[16],
    ).build()

    model.compile(optimizer="adam", loss="mse")

    # Check encode shape
    latent = model.encode(mock_data)
    assert latent.shape == (mock_data.shape[0], ae_params["latent_dim"])

    # Check decode shape
    decoded = model.decode(latent)
    assert decoded.shape == mock_data.shape

    # Check train_step works and tracks losses
    metrics = model.autoencoder.train_step((mock_data, mock_data))
    assert float(metrics["loss"]) > 0
    assert float(metrics["recon_loss"]) > 0

    # Check test_step works
    val_metrics = model.autoencoder.test_step((mock_data, mock_data))
    assert float(val_metrics["loss"]) > 0
