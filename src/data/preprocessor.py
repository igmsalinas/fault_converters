"""
Data Preprocessing
==================

Preprocessing utilities for power converter transfer function data.
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseScaler(ABC):
    """Abstract base class for scalers."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> "BaseScaler":
        """Fit scaler to data."""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)

    @abstractmethod
    def save(self, path: str) -> None:
        """Save scaler parameters."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseScaler":
        """Load scaler parameters."""
        pass


class StandardScaler(BaseScaler):
    """
    Standard scaler (z-score normalization).

    Transforms data to have zero mean and unit variance.
    """

    def __init__(self, per_feature: bool = True, epsilon: float = 1e-8):
        """
        Initialize standard scaler.

        Args:
            per_feature: Whether to compute statistics per feature
            epsilon: Small value to prevent division by zero
        """
        self.per_feature = per_feature
        self.epsilon = epsilon
        self.mean_ = None
        self.std_ = None
        self._is_fitted = False

    def fit(self, data: np.ndarray) -> "StandardScaler":
        """
        Fit scaler to data.

        Args:
            data: Input data, shape (n_samples, seq_len, n_features)
                  or (n_samples, seq_len)

        Returns:
            Self
        """
        if self.per_feature and data.ndim == 3:
            # Compute per feature
            self.mean_ = np.mean(data, axis=(0, 1), keepdims=True)
            self.std_ = np.std(data, axis=(0, 1), keepdims=True)
        else:
            # Global statistics
            self.mean_ = np.mean(data)
            self.std_ = np.std(data)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted statistics."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (data - self.mean_) / (self.std_ + self.epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return data * (self.std_ + self.epsilon) + self.mean_

    def save(self, path: str) -> None:
        """Save scaler parameters to JSON."""
        params = {
            "type": "StandardScaler",
            "per_feature": self.per_feature,
            "epsilon": self.epsilon,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
        }
        with open(path, "w") as f:
            json.dump(params, f)

    @classmethod
    def load(cls, path: str) -> "StandardScaler":
        """Load scaler from JSON."""
        with open(path, "r") as f:
            params = json.load(f)

        scaler = cls(
            per_feature=params["per_feature"],
            epsilon=params["epsilon"],
        )
        if params["mean"] is not None:
            scaler.mean_ = np.array(params["mean"])
            scaler.std_ = np.array(params["std"])
            scaler._is_fitted = True
        return scaler


class MinMaxScaler(BaseScaler):
    """
    Min-max scaler.

    Scales data to [0, 1] range.
    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        per_feature: bool = True,
        epsilon: float = 1e-8,
    ):
        self.feature_range = feature_range
        self.per_feature = per_feature
        self.epsilon = epsilon
        self.min_ = None
        self.max_ = None
        self._is_fitted = False

    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        """Fit scaler to data."""
        if self.per_feature and data.ndim == 3:
            self.min_ = np.min(data, axis=(0, 1), keepdims=True)
            self.max_ = np.max(data, axis=(0, 1), keepdims=True)
        else:
            self.min_ = np.min(data)
            self.max_ = np.max(data)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to feature range."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        data_range = self.max_ - self.min_
        scaled = (data - self.min_) / (data_range + self.epsilon)

        # Scale to feature range
        min_val, max_val = self.feature_range
        return scaled * (max_val - min_val) + min_val

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        min_val, max_val = self.feature_range
        scaled = (data - min_val) / (max_val - min_val)

        data_range = self.max_ - self.min_
        return scaled * (data_range + self.epsilon) + self.min_

    def save(self, path: str) -> None:
        """Save scaler parameters."""
        params = {
            "type": "MinMaxScaler",
            "feature_range": self.feature_range,
            "per_feature": self.per_feature,
            "epsilon": self.epsilon,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
        }
        with open(path, "w") as f:
            json.dump(params, f)

    @classmethod
    def load(cls, path: str) -> "MinMaxScaler":
        """Load scaler from JSON."""
        with open(path, "r") as f:
            params = json.load(f)

        scaler = cls(
            feature_range=tuple(params["feature_range"]),
            per_feature=params["per_feature"],
            epsilon=params["epsilon"],
        )
        if params["min"] is not None:
            scaler.min_ = np.array(params["min"])
            scaler.max_ = np.array(params["max"])
            scaler._is_fitted = True
        return scaler


class RobustScaler(BaseScaler):
    """
    Robust scaler using median and IQR.

    More robust to outliers than standard scaler.
    """

    def __init__(self, per_feature: bool = True, epsilon: float = 1e-8):
        self.per_feature = per_feature
        self.epsilon = epsilon
        self.median_ = None
        self.iqr_ = None
        self._is_fitted = False

    def fit(self, data: np.ndarray) -> "RobustScaler":
        """Fit scaler to data."""
        if self.per_feature and data.ndim == 3:
            # Reshape for percentile computation
            n_samples, seq_len, n_features = data.shape
            flat = data.reshape(-1, n_features)

            self.median_ = np.median(flat, axis=0, keepdims=True).reshape(
                1, 1, n_features
            )
            q75 = np.percentile(flat, 75, axis=0, keepdims=True).reshape(
                1, 1, n_features
            )
            q25 = np.percentile(flat, 25, axis=0, keepdims=True).reshape(
                1, 1, n_features
            )
            self.iqr_ = q75 - q25
        else:
            self.median_ = np.median(data)
            q75 = np.percentile(data, 75)
            q25 = np.percentile(data, 25)
            self.iqr_ = q75 - q25

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (data - self.median_) / (self.iqr_ + self.epsilon)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform data."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return data * (self.iqr_ + self.epsilon) + self.median_

    def save(self, path: str) -> None:
        """Save scaler parameters."""
        params = {
            "type": "RobustScaler",
            "per_feature": self.per_feature,
            "epsilon": self.epsilon,
            "median": self.median_.tolist() if self.median_ is not None else None,
            "iqr": self.iqr_.tolist() if self.iqr_ is not None else None,
        }
        with open(path, "w") as f:
            json.dump(params, f)

    @classmethod
    def load(cls, path: str) -> "RobustScaler":
        """Load scaler from JSON."""
        with open(path, "r") as f:
            params = json.load(f)

        scaler = cls(
            per_feature=params["per_feature"],
            epsilon=params["epsilon"],
        )
        if params["median"] is not None:
            scaler.median_ = np.array(params["median"])
            scaler.iqr_ = np.array(params["iqr"])
            scaler._is_fitted = True
        return scaler


def get_scaler(method: str, **kwargs) -> BaseScaler:
    """
    Factory function to get scaler by name.

    Args:
        method: Scaler method ("standard", "minmax", "robust")
        **kwargs: Arguments to pass to scaler

    Returns:
        Scaler instance
    """
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    if method not in scalers:
        raise ValueError(f"Unknown scaler method: {method}")

    return scalers[method](**kwargs)


class DataPreprocessor:
    """
    Complete data preprocessing pipeline.

    Handles feature extraction, normalization, and data formatting
    for power converter transfer function data.
    """

    def __init__(
        self,
        normalization_method: str = "standard",
        use_real_imag: bool = True,
        amplitude_col: int = 0,
        phase_col: int = 1,
    ):
        """
        Initialize preprocessor.

        Args:
            normalization_method: Normalization method to use
            use_real_imag: Whether to use real/imaginary representation (True) or amplitude (False)
            amplitude_col: Column index for amplitude (0-indexed)
            phase_col: Column index for phase (0-indexed)
        """
        self.normalization_method = normalization_method
        self.use_real_imag = use_real_imag
        self.amplitude_col = amplitude_col
        self.phase_col = phase_col

        self.scaler = None
        self._is_fitted = False

    def fit(self, data: np.ndarray) -> "DataPreprocessor":
        """
        Fit preprocessor to data.

        Args:
            data: Raw data array, shape (n_samples, seq_len, 2)
                  Columns: [amplitude, phase]

        Returns:
            Self
        """
        logger.info(f"Fitting preprocessor on data shape: {data.shape}")

        # Extract and process features
        processed = self._extract_features(data)

        # Fit scaler
        self.scaler = get_scaler(self.normalization_method, per_feature=True)
        self.scaler.fit(processed)

        self._is_fitted = True
        logger.info(f"Preprocessor fitted. Output features: {processed.shape[-1]}")

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data.

        Args:
            data: Raw data array, shape (n_samples, seq_len, 2)

        Returns:
            Processed data array
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        processed = self._extract_features(data)
        return self.scaler.transform(processed)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform data to unscaled feature space.

        Args:
            data: Processed data array

        Returns:
            Data in unscaled feature space (real/imaginary or amplitude)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        return self.scaler.inverse_transform(data)

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract and process features from raw data.

        Args:
            data: Raw data, shape (n_samples, seq_len, 2)

        Returns:
            Processed features, shape (n_samples, seq_len, n_features)
        """
        # Extract amplitude and phase
        amplitude = data[:, :, self.amplitude_col].copy()
        phase = data[:, :, self.phase_col].copy()

        # Phase always comes in degrees, convert to radians
        phase_rad = np.deg2rad(phase)

        if self.use_real_imag:
            # Convert dB amplitude to linear magnitude
            mag = 10 ** (amplitude / 20)
            real = mag * np.cos(phase_rad)
            imag = mag * np.sin(phase_rad)
            features = np.stack([real, imag], axis=-1)
        else:
            # Use amplitude
            features = amplitude[:, :, np.newaxis]

        return features

    def save(self, path: str) -> None:
        """Save preprocessor configuration and parameters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "normalization_method": self.normalization_method,
            "use_real_imag": self.use_real_imag,
            "amplitude_col": self.amplitude_col,
            "phase_col": self.phase_col,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        # Save scaler
        if self.scaler is not None:
            self.scaler.save(str(path / "scaler.json"))

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from saved files."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        preprocessor = cls(**config)

        # Load scaler
        scaler_path = path / "scaler.json"
        if scaler_path.exists():
            with open(scaler_path, "r") as f:
                scaler_config = json.load(f)

            scaler_type = scaler_config["type"]
            if scaler_type == "StandardScaler":
                preprocessor.scaler = StandardScaler.load(str(scaler_path))
            elif scaler_type == "MinMaxScaler":
                preprocessor.scaler = MinMaxScaler.load(str(scaler_path))
            elif scaler_type == "RobustScaler":
                preprocessor.scaler = RobustScaler.load(str(scaler_path))

            preprocessor._is_fitted = True

        return preprocessor
