"""
Dataset Classes
===============

PyTorch/Keras-compatible dataset classes for power converter data.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from pathlib import Path

from .loader import DataLoader, SimulationMetadata
from .preprocessor import DataPreprocessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TransferFunctionData:
    """
    Container for a single transfer function sample.

    Attributes:
        frequency: Frequency values (Hz)
        amplitude: Amplitude values (dB)
        phase: Phase values (degrees or radians)
        metadata: Optional simulation metadata
    """

    frequency: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    metadata: Optional[SimulationMetadata] = None

    @property
    def num_points(self) -> int:
        """Number of frequency points."""
        return len(self.frequency)

    def to_array(self, include_frequency: bool = False) -> np.ndarray:
        """
        Convert to numpy array.

        Args:
            include_frequency: Whether to include frequency as feature

        Returns:
            Array of shape (num_points, n_features)
        """
        if include_frequency:
            return np.stack([self.frequency, self.amplitude, self.phase], axis=-1)
        return np.stack([self.amplitude, self.phase], axis=-1)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        frequency: Optional[np.ndarray] = None,
        metadata: Optional[SimulationMetadata] = None,
    ) -> "TransferFunctionData":
        """
        Create from numpy array.

        Args:
            data: Array of shape (num_points, 2 or 3)
            frequency: Optional frequency values
            metadata: Optional metadata

        Returns:
            TransferFunctionData instance
        """
        if data.shape[-1] == 3:
            return cls(
                frequency=data[:, 0],
                amplitude=data[:, 1],
                phase=data[:, 2],
                metadata=metadata,
            )
        elif data.shape[-1] == 2:
            if frequency is None:
                frequency = np.arange(data.shape[0])
            return cls(
                frequency=frequency,
                amplitude=data[:, 0],
                phase=data[:, 1],
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")


class PowerConverterDataset:
    """
    Dataset class for power converter transfer function data.

    Provides data loading, preprocessing, and batching functionality
    compatible with Keras training loops.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        preprocessor: Optional[DataPreprocessor] = None,
        normal_threshold: float = 5.0,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing simulation files
            cache_dir: Directory for caching
            preprocessor: Optional preprocessor instance
            normal_threshold: Threshold for normal/anomaly classification
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normal_threshold = normal_threshold

        # Data loader
        self.loader = DataLoader(
            data_dir=data_dir,
            cache_dir=cache_dir,
            normal_threshold=normal_threshold,
        )

        # Preprocessor
        self.preprocessor = preprocessor or DataPreprocessor()

        # Data storage
        self._raw_data = None
        self._processed_data = None
        self._metadata = None
        self._splits = None

        self._is_loaded = False
        self._is_processed = False

    def load(
        self,
        max_files: Optional[int] = None,
        use_cache: bool = True,
    ) -> "PowerConverterDataset":
        """
        Load raw data from files.

        Args:
            max_files: Maximum number of files to load
            use_cache: Whether to use cache

        Returns:
            Self
        """
        self.loader.load(max_files=max_files, use_cache=use_cache)
        self._raw_data = self.loader.data
        self._metadata = self.loader.metadata
        self._is_loaded = True

        logger.info(f"Loaded {len(self._raw_data)} samples")
        return self

    def preprocess(self, fit: bool = True) -> "PowerConverterDataset":
        """
        Preprocess loaded data.

        Args:
            fit: Whether to fit preprocessor (True for training data)

        Returns:
            Self
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        if fit:
            # Fit on normal data only
            normal_idx = self.loader.get_normal_indices()
            if len(normal_idx) > 0:
                normal_data = self._raw_data[normal_idx]
                self.preprocessor.fit(normal_data)
            else:
                # If no normal data, fit on all
                logger.warning("No normal samples found. Fitting on all data.")
                self.preprocessor.fit(self._raw_data)

        self._processed_data = self.preprocessor.transform(self._raw_data)
        self._is_processed = True

        logger.info(f"Preprocessed data shape: {self._processed_data.shape}")
        return self

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        include_anomalies_in_test: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Split raw data into train/val/test sets.

        Delegates to loader's method.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self.loader.get_train_val_test_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            include_anomalies_in_test=include_anomalies_in_test,
        )

    def prepare_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare train/val/test splits.

        Training and validation use only normal data.
        Test includes both normal and anomalous data.

        Args:
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            seed: Random seed

        Returns:
            Dictionary with split arrays and labels
        """
        if not self._is_processed:
            raise RuntimeError("Data not processed. Call preprocess() first.")

        np.random.seed(seed)

        # Get indices
        normal_idx = self.loader.get_normal_indices()
        anomaly_idx = self.loader.get_anomaly_indices()

        logger.info(
            f"Normal samples: {len(normal_idx)}, Anomaly samples: {len(anomaly_idx)}"
        )

        # Shuffle normal indices
        np.random.shuffle(normal_idx)

        # Split normal data
        n_normal = len(normal_idx)
        n_train = int(n_normal * train_ratio)
        n_val = int(n_normal * val_ratio)

        train_idx = normal_idx[:n_train]
        val_idx = normal_idx[n_train : n_train + n_val]
        test_normal_idx = normal_idx[n_train + n_val :]

        # Create fit_val set with anomalies
        np.random.shuffle(anomaly_idx)
        n_fit_anom = min(len(val_idx), len(anomaly_idx))
        fit_val_anom_idx = anomaly_idx[:n_fit_anom]
        test_anomaly_idx = anomaly_idx[n_fit_anom:]

        fit_val_idx = np.concatenate([val_idx, fit_val_anom_idx])
        fit_val_labels = np.concatenate(
            [np.zeros(len(val_idx)), np.ones(len(fit_val_anom_idx))]
        )

        fit_val_perm = np.random.permutation(len(fit_val_idx))
        fit_val_idx = fit_val_idx[fit_val_perm]
        fit_val_labels = fit_val_labels[fit_val_perm]

        # Create test set with remaining anomalies
        test_idx = np.concatenate([test_normal_idx, test_anomaly_idx])
        test_labels = np.concatenate(
            [
                np.zeros(len(test_normal_idx)),
                np.ones(len(test_anomaly_idx)),
            ]
        )

        # Shuffle test set
        test_perm = np.random.permutation(len(test_idx))
        test_idx = test_idx[test_perm]
        test_labels = test_labels[test_perm]

        self._splits = {
            "train": self._processed_data[train_idx],
            "val": self._processed_data[val_idx],
            "test": self._processed_data[test_idx],
            "test_labels": test_labels,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "fit_val_data": self._processed_data[fit_val_idx],
            "fit_val_labels": fit_val_labels,
            "fit_val_idx": fit_val_idx,
        }

        logger.info(
            f"Splits - Train: {len(train_idx)}, Val: {len(val_idx)}, "
            f"Fit Val: {len(fit_val_idx)} (Normal: {len(val_idx)}, Anomaly: {len(fit_val_anom_idx)}), "
            f"Test: {len(test_idx)} (Normal: {len(test_normal_idx)}, Anomaly: {len(test_anomaly_idx)})"
        )

        return self._splits

    @property
    def train_data(self) -> np.ndarray:
        """Get training data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["train"]

    @property
    def val_data(self) -> np.ndarray:
        """Get validation data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["val"]

    @property
    def fit_val_data(self) -> np.ndarray:
        """Get fit validation data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits.get("fit_val_data", self._splits["val"])

    @property
    def fit_val_labels(self) -> np.ndarray:
        """Get fit validation labels."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits.get("fit_val_labels", np.zeros(len(self._splits["val"])))

    @property
    def test_data(self) -> np.ndarray:
        """Get test data."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["test"]

    @property
    def test_labels(self) -> np.ndarray:
        """Get test labels."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")
        return self._splits["test_labels"]

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        return len(self._raw_data) if self._is_loaded else 0

    @property
    def sequence_length(self) -> int:
        """Sequence length (number of frequency points)."""
        if self._processed_data is not None:
            return self._processed_data.shape[1]
        return 0

    @property
    def num_features(self) -> int:
        """Number of features per time step."""
        if self._processed_data is not None:
            return self._processed_data.shape[2]
        return 0

    @property
    def input_shape(self) -> Tuple[int, int]:
        """Input shape for model (seq_len, n_features)."""
        return (self.sequence_length, self.num_features)

    def get_metadata_for_split(self, split: str) -> List[SimulationMetadata]:
        """Get metadata for a specific split."""
        if self._splits is None:
            raise RuntimeError("Splits not prepared. Call prepare_splits() first.")

        idx_key = f"{split}_idx"
        if idx_key not in self._splits:
            raise ValueError(f"Unknown split: {split}")

        indices = self._splits[idx_key]
        return [self._metadata[i] for i in indices]

    def save(self, path: str) -> None:
        """Save dataset state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save preprocessor
        self.preprocessor.save(str(path / "preprocessor"))

        # Save processed data and splits
        if self._processed_data is not None:
            np.save(path / "processed_data.npy", self._processed_data)

        if self._splits is not None:
            for key, value in self._splits.items():
                np.save(path / f"split_{key}.npy", value)

    @classmethod
    def load_processed(cls, path: str) -> "PowerConverterDataset":
        """Load preprocessed dataset."""
        path = Path(path)

        # Create instance with minimal setup
        dataset = cls.__new__(cls)
        dataset.preprocessor = DataPreprocessor.load(str(path / "preprocessor"))

        # Load processed data
        dataset._processed_data = np.load(path / "processed_data.npy")
        dataset._is_processed = True

        # Load splits
        dataset._splits = {}
        for split_file in path.glob("split_*.npy"):
            key = split_file.stem.replace("split_", "")
            dataset._splits[key] = np.load(split_file)

        return dataset


# For backward compatibility
BuckConverterDataset = PowerConverterDataset
