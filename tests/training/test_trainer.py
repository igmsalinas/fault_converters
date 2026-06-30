import pytest
import numpy as np

from unittest.mock import patch, MagicMock
from src.training.trainer import Trainer, TrainingConfig
from src.training.carla_trainer import CARLATrainer, CARLAConfig


@pytest.fixture
def mock_trainer_data():
    seq_len = 24
    features = 5
    # Use 16 samples minimum for validation split testing
    n_samples = 32

    data = np.random.normal(0, 1, size=(n_samples, seq_len, features)).astype(
        np.float32
    )
    return data, data


def test_trainer_initialization_and_fit(mock_trainer_data, tmp_path):
    train_data, val_data = mock_trainer_data
    seq_len, features = train_data.shape[1:]

    config = TrainingConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    trainer = Trainer(config, experiment_name="test_exp")

    # Test model creation
    model = trainer.create_model(
        model_type="conv1d_ae",
        input_shape=(seq_len, features),
        filters=[16],
        latent_dim=8,
    )
    assert model is not None
    trainer.compile_model()

    # Test minimal training execution doesn't crash
    history = trainer.train(train_data, val_data, verbose=0)
    assert history is not None
    assert "loss" in history.history


def test_trainer_gpu_initialization(mock_trainer_data, tmp_path):
    train_data, val_data = mock_trainer_data
    seq_len, features = train_data.shape[1:]

    config = TrainingConfig(
        epochs=1,
        batch_size=8,
        use_gpu=True,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    # Mock tensorflow to simulate finding a GPU
    with patch("tensorflow.config.list_physical_devices") as mock_list_devices:
        # Create a mock device
        mock_device = MagicMock()
        mock_list_devices.return_value = [mock_device]

        with patch("src.training.base_trainer.logger.info") as mock_logger_info:
            with patch(
                "tensorflow.config.experimental.set_memory_growth"
            ) as mock_set_memory:
                trainer = Trainer(config, experiment_name="test_gpu_exp")

                # Check that list_physical_devices was called for GPU
                mock_list_devices.assert_called_with("GPU")

                # Check that set_memory_growth was called
                mock_set_memory.assert_called_with(mock_device, True)

                # Check logger
                mock_logger_info.assert_any_call(
                    "Found 1 GPU(s): [<MagicMock id='" + str(id(mock_device)) + "'>]"
                    if not mock_logger_info.call_args_list
                    else mock_logger_info.call_args_list[0][0][0]
                )

                # Quick check that it can still create a model
                model = trainer.create_model(
                    model_type="conv1d_ae",
                    input_shape=(seq_len, features),
                    filters=[16],
                    latent_dim=8,
                )
                assert model is not None


def test_carla_trainer_synthetic_injection_and_fit(mock_trainer_data, tmp_path):
    train_data, val_data = mock_trainer_data
    seq_len, features = train_data.shape[1:]

    config = CARLAConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        n_negative_per_sample=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    trainer = CARLATrainer(config, experiment_name="test_carla_exp")

    trainer.create_model(
        input_shape=(seq_len, features),
        encoder_type="conv1d",
        encoder_filters=[16],
        latent_dim=8,
        projection_dim=16,
    )
    trainer.setup_training()

    # Test batch preparation logic with neg samples
    anchors, positives, negatives, is_normal = trainer._prepare_batch(train_data[:4])
    assert anchors.shape == (4, seq_len, features)
    assert positives.shape == (4, seq_len, features)
    # negatives should be (batch_size, n_neg, seq, feature)
    assert negatives.shape == (4, 1, seq_len, features)

    # Test minimal training loop
    history = trainer.train(train_data, val_data, verbose=0)
    assert history is not None
    assert len(trainer.history["loss"]) == 1
    assert trainer.history["loss"][0] > 0

    # Test anomaly scoring function (kNN)
    scores = trainer.compute_anomaly_scores(val_data[:4], method="knn", k=2)
    assert scores.shape == (4,)
