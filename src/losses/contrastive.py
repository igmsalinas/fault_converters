"""
Contrastive Loss Functions
==========================

Implementation of contrastive losses for CARLA-style anomaly detection.

Reference:
    Darban et al., "CARLA: Self-supervised Contrastive Representation Learning
    for Time Series Anomaly Detection", arXiv:2308.09296
"""

import keras
from keras import ops
from typing import Optional, Tuple
import numpy as np


def reconstruction_loss(y_true, y_pred, loss_type: str = "mse"):
    """
    Compute reconstruction loss.

    Args:
        y_true: Ground truth, shape (batch_size, seq_len, n_features)
        y_pred: Predictions, shape (batch_size, seq_len, n_features)
        loss_type: Type of loss ("mse", "mae", "huber")

    Returns:
        Reconstruction loss value
    """
    if loss_type == "mse":
        return ops.mean(ops.square(y_true - y_pred))
    elif loss_type == "mae":
        return ops.mean(ops.abs(y_true - y_pred))
    elif loss_type == "huber":
        delta = 1.0
        error = y_true - y_pred
        abs_error = ops.abs(error)
        quadratic = ops.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return ops.mean(0.5 * ops.square(quadratic) + delta * linear)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class NTXentLoss(keras.layers.Layer):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Also known as InfoNCE loss, used in SimCLR and related methods.

    Args:
        temperature: Temperature parameter for scaling

    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of
        Visual Representations", ICML 2020
    """

    def __init__(self, temperature: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(
        self,
        anchor: keras.KerasTensor,
        positive: keras.KerasTensor,
        negatives: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """
        Compute NT-Xent loss.

        Args:
            anchor: Anchor embeddings, shape (batch_size, embed_dim)
            positive: Positive embeddings, shape (batch_size, embed_dim)
            negatives: Negative embeddings, shape (batch_size, n_neg, embed_dim)
                      If None, uses other samples in batch as negatives

        Returns:
            NT-Xent loss value
        """
        # Normalize embeddings
        anchor = ops.normalize(anchor, axis=-1)
        positive = ops.normalize(positive, axis=-1)

        batch_size = ops.shape(anchor)[0]

        if negatives is None:
            # Use in-batch negatives
            # Similarity matrix between all pairs
            sim_matrix = ops.matmul(anchor, ops.transpose(positive)) / self.temperature

            # Labels are diagonal (positive pairs)
            labels = ops.arange(batch_size)

            # Cross-entropy loss
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, sim_matrix, from_logits=True
            )
        else:
            # Use explicit negatives
            negatives = ops.normalize(negatives, axis=-1)

            # Positive similarity
            pos_sim = (
                ops.sum(anchor * positive, axis=-1, keepdims=True) / self.temperature
            )

            # Negative similarities
            # anchor: (batch, embed_dim) -> (batch, 1, embed_dim)
            # negatives: (batch, n_neg, embed_dim)
            anchor_expanded = ops.expand_dims(anchor, axis=1)
            neg_sim = ops.sum(anchor_expanded * negatives, axis=-1) / self.temperature

            # Concatenate: positive first, then negatives
            logits = ops.concatenate([pos_sim, neg_sim], axis=-1)

            # Labels: positive is always at index 0
            labels = ops.zeros(batch_size, dtype="int32")

            loss = keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )

        return ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


class ContrastiveLoss(keras.layers.Layer):
    """
    Margin-based contrastive loss.

    Pushes similar pairs together and dissimilar pairs apart.

    Args:
        margin: Margin for dissimilar pairs
    """

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(
        self,
        embeddings1: keras.KerasTensor,
        embeddings2: keras.KerasTensor,
        labels: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Compute contrastive loss.

        Args:
            embeddings1: First set of embeddings, shape (batch_size, embed_dim)
            embeddings2: Second set of embeddings, shape (batch_size, embed_dim)
            labels: Binary labels, 1 for similar pairs, 0 for dissimilar

        Returns:
            Contrastive loss value
        """
        # Euclidean distance
        distances = ops.sqrt(
            ops.sum(ops.square(embeddings1 - embeddings2), axis=-1) + 1e-8
        )

        # Contrastive loss
        # Similar pairs: minimize distance
        # Dissimilar pairs: maximize distance up to margin
        labels = ops.cast(labels, dtype=distances.dtype)

        pos_loss = labels * ops.square(distances)
        neg_loss = (1 - labels) * ops.square(ops.maximum(self.margin - distances, 0))

        loss = 0.5 * (pos_loss + neg_loss)

        return ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


class CARLALoss(keras.layers.Layer):
    """
    Combined loss for CARLA-style contrastive anomaly detection.

    Combines:
    1. Reconstruction loss (autoencoder objective)
    2. Contrastive loss (pull normal together, push anomalies away)

    Args:
        reconstruction_weight: Weight for reconstruction loss
        contrastive_weight: Weight for contrastive loss
        temperature: Temperature for NT-Xent loss
        reconstruction_type: Type of reconstruction loss ("mse", "mae", "huber")
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        temperature: float = 0.1,
        reconstruction_type: str = "mse",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.reconstruction_type = reconstruction_type

        self.ntxent_loss = NTXentLoss(temperature=temperature)

    def call(
        self,
        x_original: keras.KerasTensor,
        x_reconstructed: keras.KerasTensor,
        z_anchor: keras.KerasTensor,
        z_positive: keras.KerasTensor,
        z_negative: Optional[keras.KerasTensor] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Compute combined CARLA loss.

        Args:
            x_original: Original input, shape (batch_size, seq_len, n_features)
            x_reconstructed: Reconstructed input, same shape
            z_anchor: Anchor latent representations, shape (batch_size, embed_dim)
            z_positive: Positive latent representations, same shape
            z_negative: Negative latent representations, shape (batch_size, n_neg, embed_dim)

        Returns:
            Tuple of (total_loss, reconstruction_loss, contrastive_loss)
        """
        # Reconstruction loss
        recon_loss = reconstruction_loss(
            x_original, x_reconstructed, self.reconstruction_type
        )

        # Contrastive loss
        contrast_loss = self.ntxent_loss(z_anchor, z_positive, z_negative)

        # Combined loss
        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.contrastive_weight * contrast_loss
        )

        return total_loss, recon_loss, contrast_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "reconstruction_weight": self.reconstruction_weight,
                "contrastive_weight": self.contrastive_weight,
                "temperature": self.temperature,
                "reconstruction_type": self.reconstruction_type,
            }
        )
        return config


class TripletLoss(keras.layers.Layer):
    """
    Triplet loss for contrastive learning.

    Uses anchor, positive, and negative samples.

    Args:
        margin: Margin for triplet loss
    """

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(
        self,
        anchor: keras.KerasTensor,
        positive: keras.KerasTensor,
        negative: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings, shape (batch_size, embed_dim)
            positive: Positive embeddings, shape (batch_size, embed_dim)
            negative: Negative embeddings, shape (batch_size, embed_dim)

        Returns:
            Triplet loss value
        """
        # Distances
        pos_dist = ops.sqrt(ops.sum(ops.square(anchor - positive), axis=-1) + 1e-8)
        neg_dist = ops.sqrt(ops.sum(ops.square(anchor - negative), axis=-1) + 1e-8)

        # Triplet loss
        loss = ops.maximum(pos_dist - neg_dist + self.margin, 0)

        return ops.mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config


class CenterLoss(keras.layers.Layer):
    """
    Center loss to minimize intra-class variations.

    Learns a center for normal samples and penalizes distance from center.

    Args:
        center_dim: Dimension of the center vector
        alpha: Learning rate for center update
    """

    def __init__(self, center_dim: int, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.center_dim = center_dim
        self.alpha = alpha

    def build(self, input_shape):
        self.center = self.add_weight(
            name="center",
            shape=(self.center_dim,),
            initializer="zeros",
            trainable=False,  # Updated via EMA
        )
        super().build(input_shape)

    def call(
        self,
        embeddings: keras.KerasTensor,
        training: bool = False,
    ) -> keras.KerasTensor:
        """
        Compute center loss.

        Args:
            embeddings: Embeddings, shape (batch_size, embed_dim)
            training: Whether in training mode

        Returns:
            Center loss value
        """
        # Compute loss as distance from center
        diff = embeddings - self.center
        loss = ops.mean(ops.sum(ops.square(diff), axis=-1))

        # Update center with EMA during training
        if training:
            new_center = ops.mean(embeddings, axis=0)
            updated_center = self.alpha * new_center + (1 - self.alpha) * self.center
            self.center.assign(updated_center)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "center_dim": self.center_dim,
                "alpha": self.alpha,
            }
        )
        return config


def anomaly_score_from_embeddings(
    embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    method: str = "knn",
    k: int = 5,
    normalize: bool = True,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Compute anomaly scores from embeddings using distance-based methods.

    Args:
        embeddings: Test embeddings, shape (n_samples, embed_dim)
        reference_embeddings: Reference (normal) embeddings, shape (n_ref, embed_dim)
        method: Scoring method ("knn", "centroid", "mahalanobis", "cosine")
        k: Number of neighbors for kNN
        normalize: Whether to l2-normalize embeddings before scoring
        batch_size: Batch size for distance computation

    Returns:
        Anomaly scores, shape (n_samples,)
    """
    if normalize:
        # L2 normalize embeddings
        embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        )
        reference_embeddings = reference_embeddings / (
            np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8
        )

    n_samples = embeddings.shape[0]
    scores = np.zeros(n_samples, dtype=np.float32)

    if method == "knn":
        # k-NN distance scoring
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(reference_embeddings)

        for i in range(0, n_samples, batch_size):
            chunk = embeddings[i:i + batch_size]
            distances, _ = nn.kneighbors(chunk)
            scores[i:i + batch_size] = np.mean(distances, axis=1)

    elif method == "cosine":
        # Cosine distance = 1 - cosine similarity
        for i in range(0, n_samples, batch_size):
            chunk = embeddings[i:i + batch_size]
            similarities = chunk @ reference_embeddings.T

            if k > 0:
                top_k_sims = np.sort(similarities, axis=1)[:, -k:]
                mean_sim = np.mean(top_k_sims, axis=1)
            else:
                mean_sim = np.mean(similarities, axis=1)

            scores[i:i + batch_size] = 1.0 - mean_sim

    elif method == "centroid":
        # Distance from centroid
        centroid = np.mean(reference_embeddings, axis=0, keepdims=True)
        for i in range(0, n_samples, batch_size):
            chunk = embeddings[i:i + batch_size]
            scores[i:i + batch_size] = np.linalg.norm(chunk - centroid, axis=1)

    elif method == "mahalanobis":
        # Mahalanobis distance
        mean = np.mean(reference_embeddings, axis=0)
        cov = np.cov(reference_embeddings, rowvar=False)

        # Add small regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        for i in range(0, n_samples, batch_size):
            chunk = embeddings[i:i + batch_size]
            diff = chunk - mean
            scores[i:i + batch_size] = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    else:
        raise ValueError(f"Unknown method: {method}")

    return scores
