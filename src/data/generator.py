"""
Data Generators
===============

Keras-compatible data generators for training and evaluation.
"""

import numpy as np
from typing import Optional, Tuple
import keras


class SequenceGenerator(keras.utils.Sequence):
    """
    Keras Sequence generator for autoencoder training.

    For autoencoders, input equals target (reconstruction task).
    """

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize generator.

        Args:
            data: Input data array, shape (n_samples, seq_len, n_features)
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for reproducibility
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.n_samples = len(data)
        self.indices = np.arange(self.n_samples)

        if seed is not None:
            np.random.seed(seed)

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch at index.

        Returns:
            Tuple of (input, target) where input == target for autoencoders
        """
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_samples)

        batch_indices = self.indices[start_idx:end_idx]
        batch_data = self.data[batch_indices]

        # For autoencoders, input == target
        return batch_data, batch_data

    def on_epoch_end(self) -> None:
        """Shuffle indices at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class AugmentedGenerator(keras.utils.Sequence):
    """
    Data generator with online augmentation.

    Supports various augmentation techniques for time series:
    - Gaussian noise injection
    - Time warping
    - Magnitude scaling
    - Feature dropout
    """

    def __init__(
        self,
        data: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        # Augmentation parameters
        noise_std: float = 0.0,
        magnitude_scale_range: Tuple[float, float] = (1.0, 1.0),
        feature_dropout_rate: float = 0.0,
        time_mask_ratio: float = 0.0,
    ):
        """
        Initialize augmented generator.

        Args:
            data: Input data array
            batch_size: Batch size
            shuffle: Whether to shuffle
            seed: Random seed
            noise_std: Standard deviation of Gaussian noise
            magnitude_scale_range: Range for random magnitude scaling
            feature_dropout_rate: Rate of random feature dropout
            time_mask_ratio: Ratio of time steps to mask
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Augmentation parameters
        self.noise_std = noise_std
        self.magnitude_scale_range = magnitude_scale_range
        self.feature_dropout_rate = feature_dropout_rate
        self.time_mask_ratio = time_mask_ratio

        self.n_samples = len(data)
        self.indices = np.arange(self.n_samples)

        if seed is not None:
            np.random.seed(seed)

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get augmented batch."""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_samples)

        batch_indices = self.indices[start_idx:end_idx]
        batch_data = self.data[batch_indices].copy()

        # Apply augmentations
        batch_data = self._augment(batch_data)

        # For autoencoders, use original (non-augmented) as target
        target_data = self.data[batch_indices]

        return batch_data, target_data

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline."""
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, data.shape)
            data = data + noise

        # Magnitude scaling
        min_scale, max_scale = self.magnitude_scale_range
        if min_scale != max_scale:
            scale = np.random.uniform(min_scale, max_scale, (len(data), 1, 1))
            data = data * scale

        # Feature dropout
        if self.feature_dropout_rate > 0:
            mask = np.random.random(data.shape) > self.feature_dropout_rate
            data = data * mask

        # Time masking
        if self.time_mask_ratio > 0:
            seq_len = data.shape[1]
            mask_len = int(seq_len * self.time_mask_ratio)

            for i in range(len(data)):
                start = np.random.randint(0, seq_len - mask_len + 1)
                data[i, start : start + mask_len, :] = 0

        return data

    def on_epoch_end(self) -> None:
        """Shuffle at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_dataset(
    data: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    buffer_size: int = 1000,
    prefetch: bool = True,
):
    """
    Create TensorFlow dataset from numpy array.

    For use with tf.data pipeline when not using Keras Sequence.

    Args:
        data: Input data array
        batch_size: Batch size
        shuffle: Whether to shuffle
        buffer_size: Shuffle buffer size
        prefetch: Whether to prefetch batches

    Returns:
        tf.data.Dataset
    """
    import tensorflow as tf

    # Create dataset - for autoencoders, input == target
    dataset = tf.data.Dataset.from_tensor_slices((data, data))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size)

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
