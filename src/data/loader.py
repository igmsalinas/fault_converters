"""
Data Loading Utilities
======================

Functions and classes for loading power converter simulation data.
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from ..utils.logger import get_logger
from .component_ranges import (
    ANOMALOUS,
    NORMAL,
    classify_variations,
    load_ranges_for,
    mult_to_pct,
)
from .manifest import load_manifest_index

logger = get_logger(__name__)

# Opaque generated identifiers, e.g. ``grid_000001`` / ``lhs_000042``.
_OPAQUE_ID_RE = re.compile(r"^(?:grid|lhs|random)_\d+$")


def resolve_data_dir(data_dir: Union[str, Path]) -> Path:
    """Resolve data directory to the actual folder containing the data (txts & manifest).
    
    Checks two levels of structure:
    1. If the dir contains a 'txts' subdirectory, it is a dataset folder. Returns it.
    2. If the dir contains subdirectories that themselves contain 'txts', it is a base converter folder.
       Returns the alphabetical last dataset folder.
    3. Otherwise, falls back to returning the original directory path (legacy structure).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return data_dir

    # 1. Check if it's already a dataset folder (contains 'txts' subdirectory)
    if (data_dir / "txts").is_dir():
        return data_dir

    # 2. Check if it contains dataset subfolders which contain 'txts'
    dataset_dirs = []
    try:
        for p in data_dir.iterdir():
            if p.is_dir() and (p / "txts").is_dir():
                dataset_dirs.append(p)
    except OSError:
        pass

    if dataset_dirs:
        # Return alphabetically last dataset directory (e.g. dataset_00)
        dataset_dirs.sort(key=lambda p: p.name)
        return dataset_dirs[-1]

    return data_dir


def _dirs_cache_key(dirs: List[Union[str, Path]], max_files: Optional[int]) -> str:
    """Stable short cache key for a (multi-)directory + max_files combination."""
    import hashlib

    s = "|".join(sorted(str(Path(d).resolve()) for d in dirs)) + f"|n{max_files}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


@dataclass
class SimulationMetadata:
    """Metadata for a simulation (from the manifest, or parsed from filename)."""

    filename: str
    variations: Dict[str, float]
    label: Optional[str] = None

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
        """Check if this simulation represents normal operation (legacy flat rule)."""
        return self.max_deviation <= threshold

    def classify(self, ranges=None, fallback_threshold: float = 5.0) -> str:
        """Classify as ``normal`` / ``anomalous`` / ``unknown`` using the
        per-component tolerance and degradation bands (``ranges``); components
        without a range fall back to ``|deviation| <= fallback_threshold``.
        """
        return classify_variations(self.variations, ranges, fallback_threshold)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = {"filename": self.filename}
        if self.label is not None:
            d["label"] = self.label
        for k, v in self.variations.items():
            d[f"{k}_pct"] = v
        d["total_deviation"] = self.total_deviation
        d["max_deviation"] = self.max_deviation
        return d


def _metadata_from_manifest(filename: str, entry: Dict) -> SimulationMetadata:
    """Build metadata from a manifest entry (multipliers -> pct variations)."""
    mults = entry.get("multipliers", {}) or {}
    variations = {c: mult_to_pct(float(m)) for c, m in mults.items()}
    return SimulationMetadata(
        filename=filename, variations=variations, label=entry.get("label")
    )


def parse_filename(filename: str) -> SimulationMetadata:
    """
    Parse a **legacy** pct-encoded simulation filename into variations.

    Legacy format: ``parameter_value__parameter2_value.txt`` (e.g.
    ``Cout_-20__Rds_1_-5.txt``). Opaque identifiers from the manifest pipeline
    (``lhs_000042.txt``) return empty variations — their metadata comes from the
    manifest, not the filename.

    Args:
        filename: Simulation filename

    Returns:
        SimulationMetadata object
    """
    # Remove .txt extension
    name = filename.replace(".txt", "")

    # Opaque identifier (grid_000001 / lhs_000042 / random_...): no variations
    # encoded in the name — metadata comes from the manifest instead.
    if _OPAQUE_ID_RE.match(name):
        return SimulationMetadata(filename=filename, variations={})

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
    metadata: Optional[SimulationMetadata] = None,
) -> Tuple[np.ndarray, SimulationMetadata]:
    """
    Load a single simulation file.

    Args:
        filepath: Path to simulation file
        metadata: Pre-built metadata (e.g. from the manifest). If ``None``, it is
            parsed from the (legacy pct-encoded) filename, tolerating opaque names.

    Returns:
        Tuple of (data array, metadata)
        Data array shape: (num_points, 2) - [amplitude, phase]
    """
    filepath = Path(filepath)

    if metadata is None:
        try:
            metadata = parse_filename(filepath.name)
        except Exception:
            metadata = SimulationMetadata(filename=filepath.name, variations={})

    # Load data - skip header row, whitespace-separated
    try:
        data = np.loadtxt(filepath, skiprows=1)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise

    # Return only amplitude and phase columns (exclude frequency)
    return data[:, 1:], metadata


def _load_task(
    task: Tuple[Union[str, Path], Optional[Dict]],
) -> Optional[Tuple[np.ndarray, SimulationMetadata]]:
    """Executor helper: load a file, using a manifest entry for metadata if given."""
    filepath, entry = task
    metadata = (
        _metadata_from_manifest(Path(filepath).name, entry)
        if entry is not None
        else None
    )
    try:
        return load_simulation_file(filepath, metadata=metadata)
    except Exception as e:
        logger.warning(f"Skipping {Path(filepath).name}: {e}")
        return None


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
    txt_dir = data_dir / "txts" if (data_dir / "txts").is_dir() else data_dir
    files = sorted(txt_dir.glob("*.txt"))
    files = [f for f in files if f.name != "parameters.txt"]

    # Manifest (opaque-id datasets) -> per-file metadata; empty for legacy datasets.
    manifest_index = load_manifest_index(data_dir)

    if max_files is not None:
        if manifest_index:
            healthy_files = []
            faulty_files = []
            for f in files:
                entry = manifest_index.get(f.name)
                if entry:
                    if entry.get("set") == "healthy" or entry.get("label") == "normal":
                        healthy_files.append(f)
                    else:
                        faulty_files.append(f)
                else:
                    healthy_files.append(f)
            
            if healthy_files and faulty_files:
                n_healthy = max_files // 2
                n_faulty = max_files - n_healthy
                # Slice and combine
                selected_files = healthy_files[:n_healthy] + faulty_files[:n_faulty]
                files = sorted(selected_files)
            else:
                files = files[:max_files]
        else:
            files = files[:max_files]

    logger.info(f"Loading {len(files)} simulation files from {data_dir}")
    if manifest_index:
        logger.info(f"Using manifest metadata for {len(manifest_index)} samples")

    data_list = []
    metadata_list = []

    tasks = [(f, manifest_index.get(f.name)) for f in files]

    if parallel and len(files) > 100:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_load_task, tasks),
                    total=len(files),
                    desc="Loading simulations",
                    disable=not show_progress,
                )
            )

        for res in results:
            if res is not None:
                data, metadata = res
                data_list.append(data)
                metadata_list.append(metadata)
    else:
        iterator = tqdm(tasks, desc="Loading simulations", disable=not show_progress)
        for task in iterator:
            res = _load_task(task)
            if res is not None:
                data, metadata = res
                data_list.append(data)
                metadata_list.append(metadata)

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
        data_dir: Union[str, Path, List[Union[str, Path]]],
        cache_dir: Optional[Union[str, Path]] = None,
        normal_threshold: float = 5.0,
        use_component_ranges: bool = True,
        component_ranges=None,
    ):
        """
        Initialize data loader.

        Args:
            data_dir: A directory, or a list of directories, containing
                simulation files. Multiple directories are concatenated (e.g.
                combine grid + lhs runs, or several output folders).
            cache_dir: Directory for caching processed data
            normal_threshold: Fallback flat threshold (%) for components without
                a declared range, and the labelling rule when
                ``use_component_ranges`` is False.
            use_component_ranges: If True (default), label samples with the
                per-component tolerance/degradation bands; otherwise use the
                legacy flat ``max|deviation| <= normal_threshold`` rule.
            component_ranges: Optional ranges spec (``{name: ComponentRange}``).
                If None, it is auto-discovered from ``data/<converter>/
                component_ranges.json`` next to the (first) data dir; components
                without an entry fall back to ``normal_threshold``.
        """
        dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        
        # Discover ranges using the original first directory path
        self.ranges = (
            component_ranges
            if component_ranges is not None
            else load_ranges_for(dirs[0])
        )

        # Resolve directories
        self.data_dirs = [resolve_data_dir(d) for d in dirs]
        self.data_dir = self.data_dirs[0]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.normal_threshold = normal_threshold
        self.use_component_ranges = use_component_ranges

        self._data = None
        self._metadata = None
        self._frequencies = None
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
            cache_file = self.cache_dir / f"simulations_{_dirs_cache_key(self.data_dirs, max_files)}.npz"

            if cache_file.exists() and not force_reload:
                logger.info(f"Loading from cache: {cache_file}")
                cached = np.load(cache_file, allow_pickle=True)
                self._data = cached["data"]
                self._metadata = cached["metadata"].tolist()
                self._is_loaded = True
                return

        # Load from files (one or more directories, concatenated)
        data_parts, meta_parts = [], []
        for d in self.data_dirs:
            part_data, part_meta = load_all_simulations(d, max_files=max_files)
            data_parts.append(part_data)
            meta_parts.extend(part_meta)
        self._data = (
            data_parts[0] if len(data_parts) == 1
            else np.concatenate(data_parts, axis=0)
        )
        self._metadata = meta_parts

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
    def frequencies(self) -> Optional[np.ndarray]:
        """
        Frequency grid (Hz) shared by every simulation.

        Lazily read from the first column of the first simulation file (the
        frequency column is dropped from the sample arrays at load time).
        """
        if self._frequencies is not None:
            return self._frequencies

        txt_dir = self.data_dir / "txts" if (self.data_dir / "txts").is_dir() else self.data_dir
        files = sorted(txt_dir.glob("*.txt"))
        if not files:
            return None
        try:
            first = np.loadtxt(files[0], skiprows=1)
            self._frequencies = np.asarray(first[:, 0], dtype=float)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not read frequency grid from {files[0]}: {e}")
            return None
        return self._frequencies

    @property
    def num_samples(self) -> int:
        """Get number of loaded samples."""
        return len(self._data) if self._is_loaded else 0

    def _label(self, meta: "SimulationMetadata") -> str:
        """Return the normal/anomalous/unknown label for a sample."""
        if not self.use_component_ranges:
            return NORMAL if meta.is_normal(self.normal_threshold) else ANOMALOUS
        # Trust the manifest label when present (computed at generation time).
        if getattr(meta, "label", None):
            return meta.label
        return meta.classify(self.ranges, fallback_threshold=self.normal_threshold)

    def get_normal_indices(self) -> np.ndarray:
        """Get indices of normal (healthy) samples."""
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        indices = [
            i for i, meta in enumerate(self._metadata) if self._label(meta) == NORMAL
        ]
        return np.array(indices, dtype=np.int64)

    def get_anomaly_indices(self) -> np.ndarray:
        """Get indices of anomalous (faulty) samples.

        Samples that fall in the gray zone (``unknown`` — outside the healthy
        band but not a modelled fault) are excluded from BOTH sets.
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load() first.")

        indices = [
            i for i, meta in enumerate(self._metadata) if self._label(meta) == ANOMALOUS
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
