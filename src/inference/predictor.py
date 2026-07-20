"""
Anomaly Predictor
=================

Inference pipeline for anomaly detection on new samples.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json

from ..models.base import BaseAutoencoder
from ..models.conv1d_ae import Conv1DAutoencoder
from ..models.lstm_ae import LSTMAutoencoder
from ..models.mlp_ae import MLPAutoencoder
from ..models.vae import VariationalAutoencoder
from ..models.transformer_ae import TransformerAutoencoder
from ..models.gru_ae import GRUAutoencoder
from ..data.preprocessor import DataPreprocessor
from ..data.loader import load_simulation_file
from ..utils.logger import get_logger

logger = get_logger(__name__)


MODEL_CLASSES = {
    "conv1d_ae": Conv1DAutoencoder,
    "lstm_ae": LSTMAutoencoder,
    "mlp_ae": MLPAutoencoder,
    "vae": VariationalAutoencoder,
    "transformer_ae": TransformerAutoencoder,
    "gru_ae": GRUAutoencoder,
}


class AnomalyPredictor:
    """
    Production-ready anomaly predictor.

    Loads trained model and preprocessor for inference on new samples.
    """

    def __init__(
        self,
        model_dir: str,
        threshold: Optional[float] = None,
    ):
        """
        Initialize predictor.

        Args:
            model_dir: Directory containing saved model and config
            threshold: Anomaly threshold (loaded from config if None)
        """
        self.model_dir = Path(model_dir)
        self._threshold = threshold

        self.model: Optional[BaseAutoencoder] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.config: Dict[str, Any] = {}

        self._load()

    def _load(self) -> None:
        """Load model, preprocessor, and config."""
        # Load model config
        model_config_path = self.model_dir / "model_config.json"

        # Fallback for CARLA legacy or if not renamed yet
        if not model_config_path.exists():
            for fn in ["best_config.json", "config.json", "best_hyperparameters.json"]:
                if (self.model_dir / fn).exists():
                    model_config_path = self.model_dir / fn
                    break

        if model_config_path.exists():
            with open(model_config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from {model_config_path}")

        # Load threshold if not provided
        threshold_path = self.model_dir / "threshold.json"
        if self._threshold is None:
            if threshold_path.exists():
                with open(threshold_path, "r") as f:
                    threshold_data = json.load(f)
                    self._threshold = threshold_data.get("threshold")
                    if self._threshold is not None:
                        logger.info(f"Loaded threshold: {self._threshold:.6f}")
            else:
                logger.warning(
                    f"No threshold.json found in {self.model_dir}. "
                    "Predictions will not be possible until a threshold is set (run evaluate.py)."
                )

        # Load preprocessor
        preprocessor_path = self.model_dir / "preprocessor"
        if preprocessor_path.exists():
            self.preprocessor = DataPreprocessor.load(str(preprocessor_path))
            logger.info("Loaded preprocessor")

        # Load model
        model_type = self.config.get("type", self.config.get("name", "conv1d_ae"))
        input_shape = tuple(self.config.get("input_shape", (101, 2)))

        # Get model class
        if model_type == "carla" or "carla" in self.config.get("name", ""):
            from ..models.contrastive_ae import ContrastiveAutoencoder

            model_class = ContrastiveAutoencoder
        else:
            model_class = MODEL_CLASSES.get(
                self.config.get("name", "conv1d_ae"), Conv1DAutoencoder
            )

        # Create model with config
        model_kwargs = {
            k: v
            for k, v in self.config.items()
            if k not in ["name", "type", "input_shape"]
        }

        # Mapping for hyperparameter search indices
        if "filters_idx" in model_kwargs and "filters" not in model_kwargs:
            from ..training.hyperparameter_search import SearchSpace

            ss = SearchSpace()
            idx = model_kwargs.pop("filters_idx")
            model_kwargs["filters"] = ss.filter_options[idx]

        if "units_idx" in model_kwargs:
            from ..training.hyperparameter_search import SearchSpace

            ss = SearchSpace()
            idx = model_kwargs.pop("units_idx")
            if "lstm" in self.config.get("name", ""):
                model_kwargs["lstm_units"] = ss.lstm_unit_options[idx]
            elif "gru" in self.config.get("name", ""):
                model_kwargs["gru_units"] = ss.gru_unit_options[idx]

        if "enc_units_idx" in model_kwargs and "encoder_units" not in model_kwargs:
            from ..training.hyperparameter_search import SearchSpace

            ss = SearchSpace()
            idx = model_kwargs.pop("enc_units_idx")
            model_kwargs["encoder_units"] = ss.mlp_encoder_unit_options[idx]

        if "dec_units_idx" in model_kwargs and "decoder_units" not in model_kwargs:
            from ..training.hyperparameter_search import SearchSpace

            ss = SearchSpace()
            idx = model_kwargs.pop("dec_units_idx")
            model_kwargs["decoder_units"] = ss.mlp_decoder_unit_options[idx]

        # Filter kwargs to only include what the model constructor accepts
        import inspect

        sig = inspect.signature(model_class.__init__)
        valid_params = sig.parameters.keys()
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in valid_params}

        # Build and load
        if model_type == "carla":
            from ..models.contrastive_ae import ContrastiveAutoencoder

            self.model = ContrastiveAutoencoder.load(str(self.model_dir))
        else:
            self.model = model_class(input_shape=input_shape, **model_kwargs)
            self.model.build()

            # Load weights from the flat directory
            # Try new naming first, then fallbacks
            weight_files = [
                "model.weights.h5",
                "best_model.weights.h5",
                "autoencoder.weights.h5",
            ]
            loaded = False

            # If autoencoder.weights.h5 exists, use self.model.load(self.model_dir)
            if (self.model_dir / "autoencoder.weights.h5").exists():
                try:
                    self.model.load(str(self.model_dir))
                    logger.info(f"Loaded model weights from {self.model_dir}")
                    loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load weights from {self.model_dir}: {e}")

            if not loaded:
                for wf in weight_files:
                    if (self.model_dir / wf).exists():
                        try:
                            self.model.autoencoder.load_weights(
                                str(self.model_dir / wf)
                            )
                            logger.info(f"Loaded model weights from {wf}")
                            loaded = True
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load weights from {wf}: {e}")

            if not loaded and (self.model_dir / "model.keras").exists():
                try:
                    import keras

                    loaded_model = keras.models.load_model(
                        self.model_dir / "model.keras"
                    )
                    self.model.autoencoder = loaded_model
                    logger.info("Loaded full model from model.keras")
                    loaded = True
                except Exception as e:
                    logger.error(f"Failed to load model.keras: {e}")

            if not loaded:
                logger.warning(f"No weights found in {self.model_dir}")

    @property
    def threshold(self) -> Optional[float]:
        """Get anomaly threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set anomaly threshold."""
        self._threshold = value

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess raw data.

        Args:
            data: Raw data, shape (n_samples, seq_len, 2) or (seq_len, 2)

        Returns:
            Preprocessed data
        """
        if self.preprocessor is None:
            raise RuntimeError("No preprocessor loaded.")

        # Handle single sample
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        return self.preprocessor.transform(data)

    def predict(
        self,
        data: np.ndarray,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict anomaly status for samples.

        Args:
            data: Input data (raw or preprocessed)
            preprocess: Whether to preprocess data

        Returns:
            Dictionary with predictions, scores, and details
        """
        # Handle single sample
        single_sample = data.ndim == 2
        if single_sample:
            data = data[np.newaxis, ...]

        # Preprocess if needed
        if preprocess and self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        # Get reconstruction errors
        reconstruction_errors = self.model.compute_reconstruction_error(data)

        # Get predictions if threshold is available
        has_threshold = self.threshold is not None
        if has_threshold:
            predictions = (reconstruction_errors > self.threshold).astype(int)
            is_anomaly = (
                predictions.tolist() if not single_sample else bool(predictions[0])
            )
            confidence = self._compute_confidence(reconstruction_errors)
        else:
            is_anomaly = None
            confidence = None

        result = {
            "is_anomaly": is_anomaly,
            "reconstruction_error": reconstruction_errors.tolist()
            if not single_sample
            else float(reconstruction_errors[0]),
            "threshold": self.threshold,
            "confidence": confidence,
        }

        if not has_threshold:
            result["warning"] = (
                "No threshold set. Predictions (is_anomaly) and confidence are None. Run evaluate.py to establish a threshold."
            )

        return result

    def _compute_confidence(
        self, errors: np.ndarray
    ) -> Optional[Union[float, List[float]]]:
        """
        Compute confidence score for predictions.

        Confidence is based on distance from threshold.
        """
        if self.threshold is None:
            return None

        # Normalize by threshold
        normalized = errors / self.threshold

        # Confidence: how far from threshold boundary
        confidence = np.abs(normalized - 1.0)
        confidence = np.clip(confidence, 0, 1)

        if len(confidence) == 1:
            return float(confidence[0])
        return confidence.tolist()

    def predict_file(self, filepath: str) -> Dict[str, Any]:
        """
        Predict anomaly status from file.

        Args:
            filepath: Path to simulation file

        Returns:
            Prediction results with metadata
        """
        # Load file
        data, metadata = load_simulation_file(filepath)

        # Predict
        result = self.predict(data, preprocess=True)

        # Add metadata
        result["metadata"] = metadata.to_dict()
        result["filepath"] = str(filepath)

        return result

    def predict_batch(
        self,
        filepaths: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Predict anomaly status for multiple files.

        Args:
            filepaths: List of file paths
            batch_size: Batch size for prediction

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(filepaths), batch_size):
            batch_paths = filepaths[i : i + batch_size]

            # Load batch
            batch_data = []
            batch_metadata = []
            for fp in batch_paths:
                try:
                    data, metadata = load_simulation_file(fp)
                    batch_data.append(data)
                    batch_metadata.append(metadata)
                except Exception as e:
                    logger.warning(f"Error loading {fp}: {e}")
                    continue

            if not batch_data:
                continue

            # Stack and predict
            batch_array = np.stack(batch_data, axis=0)
            predictions = self.predict(batch_array, preprocess=True)

            # Create individual results
            for j, (meta, fp) in enumerate(zip(batch_metadata, batch_paths)):
                result = {
                    "is_anomaly": predictions["is_anomaly"][j]
                    if isinstance(predictions["is_anomaly"], list)
                    else predictions["is_anomaly"],
                    "reconstruction_error": predictions["reconstruction_error"][j]
                    if isinstance(predictions["reconstruction_error"], list)
                    else predictions["reconstruction_error"],
                    "threshold": self.threshold,
                    "metadata": meta.to_dict(),
                    "filepath": str(fp),
                }
                results.append(result)

        return results

    def get_reconstruction(
        self, data: np.ndarray, preprocess: bool = True
    ) -> np.ndarray:
        """
        Get reconstruction for visualization.

        Args:
            data: Input data
            preprocess: Whether to preprocess

        Returns:
            Reconstructed data
        """
        if preprocess and self.preprocessor is not None:
            data = self.preprocessor.transform(data)

        return self.model.predict(data)

    def explain(
        self,
        data: np.ndarray,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """
        Provide explanation for prediction.

        Args:
            data: Input data
            preprocess: Whether to preprocess

        Returns:
            Detailed explanation
        """
        # Handle single sample
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        if preprocess and self.preprocessor is not None:
            processed = self.preprocessor.transform(data)
        else:
            processed = data

        # Get reconstruction
        reconstructed = self.model.predict(processed)

        # Compute per-timestep and per-feature errors
        error_per_timestep = np.mean(np.square(processed - reconstructed), axis=-1)
        error_per_feature = np.mean(np.square(processed - reconstructed), axis=1)

        # Find most anomalous regions
        total_error = self.model.compute_reconstruction_error(processed)[0]

        # Top anomalous timesteps
        top_timesteps = np.argsort(error_per_timestep[0])[-5:][::-1]

        return {
            "total_error": float(total_error),
            "threshold": self.threshold,
            "is_anomaly": bool(total_error > self.threshold),
            "error_per_timestep": error_per_timestep[0].tolist(),
            "error_per_feature": error_per_feature[0].tolist(),
            "most_anomalous_timesteps": top_timesteps.tolist(),
            "mean_feature_errors": np.mean(error_per_feature[0], axis=0).tolist(),
        }

    def save_threshold(self, threshold: float) -> None:
        """Save threshold to model directory."""
        threshold_path = self.model_dir / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold}, f)
        self._threshold = threshold
        logger.info(f"Saved threshold {threshold} to {threshold_path}")


def load_predictor(
    model_dir: str, threshold: Optional[float] = None
) -> AnomalyPredictor:
    """
    Convenience function to load predictor.

    Args:
        model_dir: Directory containing model
        threshold: Anomaly threshold

    Returns:
        AnomalyPredictor instance
    """
    return AnomalyPredictor(model_dir=model_dir, threshold=threshold)
