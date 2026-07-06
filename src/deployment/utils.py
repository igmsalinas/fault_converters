"""
Deployment Utilities
====================

Helper functions to prepare calibration and benchmarking data.
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from ..data.dataset import PowerConverterDataset
from ..data.preprocessor import DataPreprocessor
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Uniform latency-measurement protocol
# ---------------------------------------------------------------------------
# Every backend (Keras / TFLite / ONNX / TensorRT, CPU or GPU) is timed under
# identical conditions so per-sample latencies are directly comparable:
#   * single-sample batch (n = 1)
#   * 15 discarded warm-up iterations
#   * 150 timed runs (report mean / std / min / max)
TIMING_BATCH_SIZE = 1
TIMING_WARMUP = 15
TIMING_RUNS = 150


def get_file_size_mb(path: str) -> float:
    """Get size of a file or directory in megabytes (MB).

    Args:
        path: Absolute or relative path to a file or directory.

    Returns:
        Size in megabytes, or 0.0 if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    elif p.is_dir():
        total_size = 0
        for dirpath, _, filenames in os.walk(p):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    return 0.0


def load_deployment_datasets(
    data_dir: str,
    cache_dir: Optional[str] = "cache",
    normal_threshold: float = 5.0,
    max_files: Optional[int] = None,
    preprocessor: Optional[DataPreprocessor] = None,
) -> Tuple[PowerConverterDataset, Dict[str, np.ndarray]]:
    """
    Load dataset and prepare splits for deployment calibration and benchmarking.

    Args:
        data_dir: Directory containing simulation files.
        cache_dir: Directory for caching loaded data.
        normal_threshold: Threshold for normal/anomaly classification.
        max_files: Maximum number of files to load.
        preprocessor: Optional pre-fitted preprocessor. If provided, fitting is skipped.

    Returns:
        A tuple of (PowerConverterDataset, splits_dict)
    """
    logger.info(f"Loading dataset from {data_dir} for deployment...")
    dataset = PowerConverterDataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        preprocessor=preprocessor,
        normal_threshold=normal_threshold,
    )
    dataset.load(max_files=max_files)
    dataset.preprocess(fit=(preprocessor is None))
    splits = dataset.prepare_splits()
    return dataset, splits


def get_calibration_dataset(
    data_dir: str,
    cache_dir: Optional[str] = "cache",
    normal_threshold: float = 5.0,
    num_samples: int = 100,
    max_files: Optional[int] = None,
    preprocessor: Optional[DataPreprocessor] = None,
) -> np.ndarray:
    """
    Get a representative subset of normal training data for quantization calibration.

    Args:
        data_dir: Directory containing simulation files.
        cache_dir: Directory for caching.
        normal_threshold: Threshold for classification.
        num_samples: Number of calibration samples to return.
        max_files: Maximum files to load.
        preprocessor: Optional pre-fitted preprocessor.

    Returns:
        Numpy array of shape (num_samples, seq_len, num_features)
    """
    _, splits = load_deployment_datasets(
        data_dir=data_dir,
        cache_dir=cache_dir,
        normal_threshold=normal_threshold,
        max_files=max_files,
        preprocessor=preprocessor,
    )
    train_data = splits["train"]
    
    # Take a random or structured slice of training data
    n_available = len(train_data)
    if n_available == 0:
        raise ValueError("No training data available for calibration.")
        
    num_samples = min(num_samples, n_available)
    indices = np.random.choice(n_available, size=num_samples, replace=False)
    calibration_data = train_data[indices]
    
    logger.info(f"Generated calibration dataset with {len(calibration_data)} samples.")
    return calibration_data.astype(np.float32)
