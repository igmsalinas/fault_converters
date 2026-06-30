"""
Data Loading Utilities
======================

Functions and classes for loading power converter simulation data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationMetadata:
    """Metadata extracted from simulation filename."""

    filename: str
    variations: Dict[str, float]

    @property
    def total_deviation(self) -> float:
        """Calculate total absolute deviation from nominal."""
        if not self.variations:
            return 0.0
        return sum(abs(v) for v in self.variations.values())

    @property
    def max_deviation(self) -> float:
        """Get maximum absolute deviation from any parameter."""
        if not self.variations:
            return 0.0
        return max(abs(v) for v in self.variations.values())

    @property
    def deviation_vector(self) -> np.ndarray:
        """Get deviation as numpy array consistently ordered."""
        if not self.variations:
            return np.array([])
        sorted_keys = sorted(self.variations.keys())
        return np.array([self.variations[k] for k in sorted_keys])

    def is_normal(self, threshold: float = 5.0) -> bool:
        """Check if this simulation represents normal operation."""
        return self.max_deviation <= threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = {"filename": self.filename}
        for k, v in self.variations.items():
            d[f"{k}_pct"] = v
        d["total_deviation"] = self.total_deviation
        d["max_deviation"] = self.max_deviation
        return d


def parse_filename(filename: str) -> SimulationMetadata:
    """
    Parse simulation filename to extract parameter variations.

    Filename format: parameter_value__parameter2_value.txt
    Example: Cout_-20__Rds_1_-5.txt

    Args:
        filename: Simulation filename

    Returns:
        SimulationMetadata object
    """
    # Remove .txt extension
    name = filename.replace(".txt", "")

    parts = name.split("__")

    variations = {}
    for part in parts:
        if not part:
            continue
        idx = part.rfind("_")
        if idx == -1:
            raise ValueError(f"Could not parse part '{part}' in '{filename}'")

        comp_name = part[:idx]
        val_str = part[idx + 1 :]
        try:
            variations[comp_name] = float(val_str)
        except ValueError:
            raise ValueError(
                f"Could not parse value '{val_str}' for component '{comp_name}' in '{filename}'"
            )

    return SimulationMetadata(filename=filename, variations=variations)


def load_simulation_file(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, SimulationMetadata]:
    """
    Load a single simulation file.

    Args:
        filepath: Path to simulation file

    Returns:
        Tuple of (data array, metadata)
        Data array shape: (num_points, 2) - [amplitude, phase]
    """
    filepath = Path(filepath)

    # Parse metadata from filename
    metadata = parse_filename(filepath.name)

    # Load data - skip header row, whitespace-separated
    try:
        data = np.loadtxt(filepath, skiprows=1)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise

    # Return only amplitude and phase columns (exclude frequency)
    return data[:, 1:], metadata


def load_all_simulations(
    data_dir: Union[str, Path],
    max_files: Optional[int] = None,
    parallel: bool = True,
    num_workers: int = 8,
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[SimulationMetadata]]:
    """
    Load all simulation files from a directory.

    Args:
        data_dir: Directory containing simulation files
        max_files: Maximum number of files to load (None for all)
        parallel: Whether to use parallel loading
        num_workers: Number of parallel workers
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (data array, metadata list)
        Data array shape: (num_samples, num_points, 3)
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.txt"))

    if max_files is not None:
        files = files[:max_files]

    logger.info(f"Loading {len(files)} simulation files from {data_dir}")

    data_list = []
    metadata_list = []

    if parallel and len(files) > 100:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(load_simulation_file, files),
                    total=len(files),
                    desc="Loading simulations",
                    disable=not show_progress,
                )
            )

        for data, metadata in results:
            data_list.append(data)
            metadata_list.append(metadata)
    else:
        iterator = tqdm(files, desc="Loading simulations", disable=not show_progress)
        for filepath in iterator:
            try:
                data, metadata = load_simulation_file(filepath)
                data_list.append(data)
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Skipping {filepath.name}: {e}")

    # Stack into single array
    data_array = np.stack(data_list, axis=0)

    logger.info(f"Loaded {len(data_list)} simulations, shape: {data_array.shape}")

    return data_array, metadata_list


class DataLoader:
    """
    Data loader for power converter simulations.

    Handles loading, caching, and splitting of simulation data.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        normal_threshold: float = 5.0,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: Directory containing simulation files
            cache_dir: Directory for caching processed data
            normal_threshold: Maximum deviation percentage for normal samples
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normal_threshold = normal_threshold

        self._data = None
        self._metadata = None
        self._is_loaded = False

    def load(
        self,
        max_files: Optional[int] = None,
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> None:
        """
        Load simulation data.

        Args:
            max_files: Maximum number of files to load
            use_cache: Whether to use cached data if available
            force_reload: Force reload even if cache exists
        """
        cache_file = None
        if self.cache_dir and use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"simulations_n{max_files}.npz"

            if cache_file.exists() and not force_reload:
                logger.info(f"Loading from cache: {cache_file}")
                cached = np.load(cache_file, allow_pickle=True)
                self._data = cached["data"]
                self._metadata = cached["metadata"].tolist()
                self._is_loaded = True
                return

        # Load from files
        self._data, self._metadata = load_all_simulations(
            self.data_dir,
            max_files=max_files,
        )

        # Save to cache
        if cache_file:
            np.savez(
                cache_file,
                data=self._data,
                metadata=np.array(self._metadata, dtype=object),
            )
            logger.info(f"Saved to cache: {cache_file}")

        self._is_loaded = True

    @property
    def data(self) -> np.ndarray:
        """Get loaded data array."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._data

    @property
    def metadata(self) -> List[SimulationMetadata]:
        """Get metadata list."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")
        return self._metadata

    @property
    def num_samples(self) -> int:
        """Get number of loaded samples."""
        return len(self._data) if self._is_loaded else 0

    def get_normal_indices(self) -> np.ndarray:
        """Get indices of normal (non-anomalous) samples."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        indices = [
            i
            for i, meta in enumerate(self._metadata)
            if meta.is_normal(self.normal_threshold)
        ]
        return np.array(indices, dtype=np.int64)

    def get_anomaly_indices(self) -> np.ndarray:
        """Get indices of anomalous samples."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        indices = [
            i
            for i, meta in enumerate(self._metadata)
            if not meta.is_normal(self.normal_threshold)
        ]
        return np.array(indices, dtype=np.int64)

    def split_normal_anomaly(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into normal and anomalous sets.

        Returns:
            Tuple of (normal_data, anomaly_data)
        """
        normal_idx = self.get_normal_indices()
        anomaly_idx = self.get_anomaly_indices()

        normal_data = self._data[normal_idx]
        anomaly_data = self._data[anomaly_idx]

        logger.info(
            f"Normal samples: {len(normal_data)}, Anomaly samples: {len(anomaly_data)}"
        )

        return normal_data, anomaly_data

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        include_anomalies_in_test: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train/val/test sets.

        Training and validation use only normal data.
        Test set includes both normal and anomalous data.

        Args:
            train_ratio: Ratio of normal data for training
            val_ratio: Ratio of normal data for validation
            test_ratio: Ratio of normal data for testing
            seed: Random seed for reproducibility
            include_anomalies_in_test: Whether to include anomalies in test set

        Returns:
            Dictionary with train, val, test, and test_labels arrays
        """
        np.random.seed(seed)

        normal_data, anomaly_data = self.split_normal_anomaly()

        # Shuffle normal data
        n_normal = len(normal_data)
        perm = np.random.permutation(n_normal)
        normal_data = normal_data[perm]

        # Split normal data
        n_train = int(n_normal * train_ratio)
        n_val = int(n_normal * val_ratio)

        train_data = normal_data[:n_train]
        val_data = normal_data[n_train : n_train + n_val]
        test_normal = normal_data[n_train + n_val :]

        # Create test set
        if include_anomalies_in_test:
            test_data = np.concatenate([test_normal, anomaly_data], axis=0)
            test_labels = np.concatenate(
                [
                    np.zeros(len(test_normal)),
                    np.ones(len(anomaly_data)),
                ]
            )

            # Shuffle test set
            test_perm = np.random.permutation(len(test_data))
            test_data = test_data[test_perm]
            test_labels = test_labels[test_perm]
        else:
            test_data = test_normal
            test_labels = np.zeros(len(test_normal))

        logger.info(
            f"Split sizes - Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "test_labels": test_labels,
        }

    def get_metadata_dataframe(self) -> pd.DataFrame:
        """Get metadata as pandas DataFrame."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        records = [meta.to_dict() for meta in self._metadata]
        return pd.DataFrame.from_records(records)
