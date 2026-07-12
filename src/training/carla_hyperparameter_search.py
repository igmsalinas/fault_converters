"""
CARLA Hyperparameter Search
============================

Hyperparameter optimization for CARLA contrastive anomaly detection.
Uses custom training loop, so requires specialized search implementation.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import json
import itertools
import time

from ..training.carla_trainer import CARLATrainer, CARLAConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CARLASearchSpace:
    """Search space for CARLA hyperparameters."""

    # Model architecture
    encoder_types: List[str] = field(
        default_factory=lambda: ["conv1d", "lstm", "gru", "transformer", "mlp"]
    )
    latent_dims: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    projection_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    encoder_filters: List[List[int]] = field(
        default_factory=lambda: [[32, 64], [32, 64, 128], [64, 128]]
    )
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5])
    encoder_unit_options: List[List[int]] = field(
        default_factory=lambda: [[32, 16], [64, 32], [128, 64], [256, 128]]
    )
    num_heads: List[int] = field(default_factory=lambda: [2, 4, 8])

    # Training
    learning_rates: List[float] = field(
        default_factory=lambda: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    )

    # CARLA-specific
    reconstruction_weights: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
    )
    contrastive_weights: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
    )
    temperatures: List[float] = field(
        default_factory=lambda: [0.07, 0.1, 0.15, 0.2, 0.3]
    )
    anomaly_ratios: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])

    # Anomaly scoring
    scoring_methods: List[str] = field(
        default_factory=lambda: ["knn", "centroid", "cosine", "mahalanobis"]
    )
    k_neighbors: List[int] = field(default_factory=lambda: [3, 5, 10])

    # Dropout
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2]
    )

    def get_random_config(self) -> Dict[str, Any]:
        """Sample random configuration from search space."""
        return {
            "encoder_type": str(np.random.choice(self.encoder_types)),
            "latent_dim": int(np.random.choice(self.latent_dims)),
            "projection_dim": int(np.random.choice(self.projection_dims)),
            "encoder_filters": self.encoder_filters[
                np.random.choice(len(self.encoder_filters))
            ],
            "encoder_units": self.encoder_unit_options[
                np.random.choice(len(self.encoder_unit_options))
            ],
            "kernel_size": int(np.random.choice(self.kernel_sizes)),
            "num_heads": int(np.random.choice(self.num_heads)),
            "learning_rate": float(np.random.choice(self.learning_rates)),
            "reconstruction_weight": float(
                np.random.choice(self.reconstruction_weights)
            ),
            "contrastive_weight": float(np.random.choice(self.contrastive_weights)),
            "temperature": float(np.random.choice(self.temperatures)),
            "anomaly_ratio": float(np.random.choice(self.anomaly_ratios)),
            "scoring_method": str(np.random.choice(self.scoring_methods)),
            "k_neighbors": int(np.random.choice(self.k_neighbors)),
            "dropout_rate": float(np.random.choice(self.dropout_rates)),
        }

    def get_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate all grid search configurations."""
        keys = [
            "encoder_type",
            "latent_dim",
            "projection_dim",
            "encoder_filters",
            "encoder_units",
            "kernel_size",
            "num_heads",
            "learning_rate",
            "reconstruction_weight",
            "contrastive_weight",
            "temperature",
            "anomaly_ratio",
            "scoring_method",
            "k_neighbors",
            "dropout_rate",
        ]
        values = [
            self.encoder_types,
            self.latent_dims,
            self.projection_dims,
            self.encoder_filters,
            self.encoder_unit_options,
            self.kernel_sizes,
            self.num_heads,
            self.learning_rates,
            self.reconstruction_weights,
            self.contrastive_weights,
            self.temperatures,
            self.anomaly_ratios,
            self.scoring_methods,
            self.k_neighbors,
            self.dropout_rates,
        ]

        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))

        return configs

    def total_combinations(self) -> int:
        """Calculate total number of combinations."""
        return (
            len(self.encoder_types)
            * len(self.latent_dims)
            * len(self.projection_dims)
            * len(self.encoder_filters)
            * len(self.encoder_unit_options)
            * len(self.kernel_sizes)
            * len(self.num_heads)
            * len(self.learning_rates)
            * len(self.reconstruction_weights)
            * len(self.contrastive_weights)
            * len(self.temperatures)
            * len(self.anomaly_ratios)
            * len(self.scoring_methods)
            * len(self.k_neighbors)
            * len(self.dropout_rates)
        )


@dataclass
class TrialResult:
    """Result from a single hyperparameter trial."""

    trial_id: int
    config: Dict[str, Any]
    val_loss: float
    val_recon_loss: float
    val_contrast_loss: float
    roc_auc: float
    pr_auc: float
    f1_score: float
    training_time: float
    epochs_trained: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "val_loss": self.val_loss,
            "val_recon_loss": self.val_recon_loss,
            "val_contrast_loss": self.val_contrast_loss,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1_score": self.f1_score,
            "training_time": self.training_time,
            "epochs_trained": self.epochs_trained,
        }


class CARLAHyperparameterSearch:
    """
    Hyperparameter search for CARLA models.

    Supports random search and grid search with custom CARLA training loop.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        search_space: Optional[CARLASearchSpace] = None,
        project_name: str = "carla_hp_search",
        directory: str = "hp_tuning/carla",
        objective: str = "roc_auc",  # "roc_auc", "f1_score", "val_loss"
        frequencies: Optional[np.ndarray] = None,
        scaler=None,
        use_real_imag: bool = True,
        converter_spec=None,
    ):
        """
        Initialize CARLA hyperparameter search.

        Args:
            input_shape: Input shape (seq_len, n_features)
            search_space: Search space definition
            project_name: Project name for organizing results
            directory: Directory for search results
            objective: Metric to optimize ("roc_auc", "f1_score", "val_loss")
            frequencies: Frequency grid (Hz) enabling physics-based anomaly injection.
            scaler: Fitted feature scaler for physics-based injection.
            use_real_imag: Whether channels are (Real, Imag) of the response.
        """
        self.input_shape = input_shape
        self.search_space = search_space or CARLASearchSpace()
        self.project_name = project_name
        self.directory = Path(directory) / project_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.objective = objective
        self.frequencies = frequencies
        self.scaler = scaler
        self.use_real_imag = use_real_imag
        self.converter_spec = converter_spec

        self.trials: List[TrialResult] = []
        self.best_trial: Optional[TrialResult] = None
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_trainer: Optional[CARLATrainer] = None

        self._load_progress()

    def _load_progress(self) -> None:
        """Load search progress from disk if it exists."""
        progress_path = self.directory / "search_progress.json"
        if progress_path.exists():
            try:
                with open(progress_path, "r") as f:
                    results = json.load(f)

                for trial_dict in results.get("trials", []):
                    self.trials.append(TrialResult(**trial_dict))

                best_trial_dict = results.get("best_trial")
                if best_trial_dict:
                    self.best_trial = TrialResult(**best_trial_dict)
                    self.best_config = self.best_trial.config

                logger.info(
                    f"Loaded {len(self.trials)} existing trials from {progress_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load search progress from {progress_path}: {e}"
                )

    def _get_or_create_config(
        self, trial_id: int, generate_fn=None, predefined_config=None
    ) -> Dict[str, Any]:
        """Get existing config for an interrupted trial, or save a new one."""
        trial_dir = self.directory / f"trial_{trial_id}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "trial_config.json"

        if config_path.exists():
            logger.info(
                f"Loaded existing configuration for trial {trial_id} to continue mid-experiment"
            )
            with open(config_path, "r") as f:
                return json.load(f)

        config = predefined_config if predefined_config is not None else generate_fn()
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return config

    def _run_trial(
        self,
        trial_id: int,
        config: Dict[str, Any],
        train_data: np.ndarray,
        val_data: np.ndarray,
        monitor_data: np.ndarray,
        monitor_labels: np.ndarray,
        epochs: int,
        batch_size: int,
        verbose: int = 0,
    ) -> TrialResult:
        """Run a single trial with given configuration."""

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Trial {trial_id}: {config}")
        logger.info(f"{'=' * 60}")

        start_time = time.time()

        # Create CARLA config
        carla_config = CARLAConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=float(config["learning_rate"]),
            reconstruction_weight=float(config["reconstruction_weight"]),
            contrastive_weight=float(config["contrastive_weight"]),
            temperature=float(config["temperature"]),
            anomaly_ratio=float(config["anomaly_ratio"]),
            scoring_method=str(config["scoring_method"]),
            k_neighbors=int(config["k_neighbors"]),
            checkpoint_dir=str(self.directory / f"trial_{trial_id}" / "checkpoints"),
            log_dir=str(self.directory / f"trial_{trial_id}" / "logs"),
            early_stopping=True,
            patience=10,
            save_best_only=True,
        )

        # Create trainer
        trainer = CARLATrainer(
            config=carla_config,
            experiment_name=f"trial_{trial_id}",
        )

        # Create model
        trainer.create_model(
            input_shape=self.input_shape,
            latent_dim=int(config["latent_dim"]),
            projection_dim=int(config["projection_dim"]),
            encoder_type=str(config["encoder_type"]),
            encoder_filters=config.get("encoder_filters", [32, 64]),
            encoder_units=config.get("encoder_units", [64, 32]),
            kernel_size=int(config["kernel_size"]),
            num_heads=int(config["num_heads"]),
            dropout_rate=float(config["dropout_rate"]),
        )

        # Setup and train
        trainer.set_physics_context(
            frequencies=self.frequencies,
            scaler=self.scaler,
            use_real_imag=self.use_real_imag,
            converter_spec=self.converter_spec,
        )
        trainer.setup_training()

        try:
            history = trainer.train(
                train_data=train_data,
                val_data=val_data,
                verbose=verbose,
            )

            # Get validation metrics
            val_loss = (
                history["val_loss"][-1] if history["val_loss"] else history["loss"][-1]
            )
            val_recon = (
                history["val_reconstruction_loss"][-1]
                if history["val_reconstruction_loss"]
                else history["reconstruction_loss"][-1]
            )
            val_contrast = (
                history["val_contrastive_loss"][-1]
                if history["val_contrastive_loss"]
                else history["contrastive_loss"][-1]
            )
            epochs_trained = len(history["loss"])

            # Evaluate on monitor data (validation set with anomalies)
            from sklearn.metrics import (
                roc_auc_score,
                f1_score,
                precision_recall_curve,
                auc,
            )

            predictions, scores, threshold = trainer.detect_anomalies(
                monitor_data, percentile=95.0, batch_size=batch_size
            )

            roc_auc = roc_auc_score(monitor_labels, scores)
            f1 = f1_score(monitor_labels, predictions)

            precision, recall, _ = precision_recall_curve(monitor_labels, scores)
            pr_auc = auc(recall, precision)

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            val_loss = float("inf")
            val_recon = float("inf")
            val_contrast = float("inf")
            roc_auc = 0.0
            pr_auc = 0.0
            f1 = 0.0
            epochs_trained = 0
            trainer = None

        training_time = time.time() - start_time

        result = TrialResult(
            trial_id=trial_id,
            config=config,
            val_loss=val_loss,
            val_recon_loss=val_recon,
            val_contrast_loss=val_contrast,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            f1_score=f1,
            training_time=training_time,
            epochs_trained=epochs_trained,
        )

        logger.info(
            f"Trial {trial_id} complete: ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, "
            f"val_loss={val_loss:.4f}, time={training_time:.1f}s"
        )

        # Update best if this is better
        if self._is_better(result, self.best_trial):
            self.best_trial = result
            self.best_config = config
            self.best_trainer = trainer
            logger.info(f"*** New best trial: {trial_id} ***")

        return result

    def _is_better(self, new: TrialResult, current: Optional[TrialResult]) -> bool:
        """Check if new result is better than current best."""
        if current is None:
            return True

        if self.objective == "roc_auc":
            return new.roc_auc > current.roc_auc
        elif self.objective == "f1_score":
            return new.f1_score > current.f1_score
        elif self.objective == "val_loss":
            return new.val_loss < current.val_loss
        else:
            return new.roc_auc > current.roc_auc

    def search_random(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        monitor_data: np.ndarray,
        monitor_labels: np.ndarray,
        n_trials: int = 20,
        epochs_per_trial: int = 30,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Run random search.

        Args:
            train_data: Training data (normal samples only)
            val_data: Validation data (normal samples only)
            monitor_data: Data for metric monitoring (mixed normal and anomaly)
            monitor_labels: Labels for monitor data
            n_trials: Number of random trials
            epochs_per_trial: Max epochs per trial
            verbose: Verbosity level

        Returns:
            Dictionary with best configuration and results
        """
        logger.info(f"Starting random search with {n_trials} trials")

        start_trial_id = len(self.trials)
        if start_trial_id >= n_trials:
            logger.info("Search already completed the requested number of trials.")
            return self._get_results()

        for trial_id in range(start_trial_id, n_trials):
            config = self._get_or_create_config(
                trial_id, generate_fn=lambda: self.search_space.get_random_config()
            )

            result = self._run_trial(
                trial_id=trial_id,
                config=config,
                train_data=train_data,
                val_data=val_data,
                monitor_data=monitor_data,
                monitor_labels=monitor_labels,
                epochs=epochs_per_trial,
                batch_size=batch_size,
                verbose=verbose,
            )

            self.trials.append(result)
            self._save_progress()

        return self._get_results()

    def search_grid(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        monitor_data: np.ndarray,
        monitor_labels: np.ndarray,
        max_trials: Optional[int] = None,
        epochs_per_trial: int = 30,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Run grid search.

        Args:
            train_data: Training data (normal samples only)
            val_data: Validation data (normal samples only)
            monitor_data: Data for metric monitoring
            monitor_labels: Labels for monitor data
            max_trials: Maximum trials (None for full grid)
            epochs_per_trial: Max epochs per trial
            verbose: Verbosity level

        Returns:
            Dictionary with best configuration and results
        """
        configs = self.search_space.get_grid_configs()

        if max_trials is not None and max_trials < len(configs):
            # Random sample from grid
            np.random.shuffle(configs)
            configs = configs[:max_trials]

        logger.info(f"Starting grid search with {len(configs)} configurations")

        for trial_id, config in enumerate(configs):
            if trial_id < len(self.trials):
                continue

            config = self._get_or_create_config(trial_id, predefined_config=config)

            result = self._run_trial(
                trial_id=trial_id,
                config=config,
                train_data=train_data,
                val_data=val_data,
                monitor_data=monitor_data,
                monitor_labels=monitor_labels,
                epochs=epochs_per_trial,
                batch_size=batch_size,
                verbose=verbose,
            )

            self.trials.append(result)
            self._save_progress()

        return self._get_results()

    def search_bayesian(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        monitor_data: np.ndarray,
        monitor_labels: np.ndarray,
        n_trials: int = 20,
        n_initial: int = 5,
        epochs_per_trial: int = 30,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Run Bayesian-inspired search (random initial + exploitation).

        Uses random search initially, then samples near best configurations.

        Args:
            train_data: Training data
            val_data: Validation data
            monitor_data: Monitor data for trials
            monitor_labels: Labels for monitor data
            n_trials: Total number of trials
            n_initial: Number of initial random trials
            epochs_per_trial: Epochs per trial
            verbose: Verbosity level

        Returns:
            Dictionary with best configuration and results
        """
        logger.info(
            f"Starting Bayesian-inspired search: {n_initial} initial + "
            f"{n_trials - n_initial} exploitation trials"
        )

        start_trial_id = len(self.trials)
        if start_trial_id >= n_trials:
            logger.info("Search already completed the requested number of trials.")
            return self._get_results()

        # Initial random exploration
        for trial_id in range(max(0, start_trial_id), n_initial):
            config = self._get_or_create_config(
                trial_id, generate_fn=lambda: self.search_space.get_random_config()
            )

            result = self._run_trial(
                trial_id=trial_id,
                config=config,
                train_data=train_data,
                val_data=val_data,
                monitor_data=monitor_data,
                monitor_labels=monitor_labels,
                epochs=epochs_per_trial,
                batch_size=batch_size,
                verbose=verbose,
            )

            self.trials.append(result)
            self._save_progress()

        # Exploitation: sample near best configs
        for trial_id in range(max(n_initial, start_trial_id), n_trials):

            def generate_exploitation_config():
                sorted_trials = sorted(
                    self.trials,
                    key=lambda t: (
                        t.roc_auc if self.objective == "roc_auc" else -t.val_loss
                    ),
                    reverse=True,
                )
                base_config = sorted_trials[
                    np.random.randint(0, min(3, len(sorted_trials)))
                ].config.copy()
                return self._perturb_config(base_config)

            config = self._get_or_create_config(
                trial_id, generate_fn=generate_exploitation_config
            )

            result = self._run_trial(
                trial_id=trial_id,
                config=config,
                train_data=train_data,
                val_data=val_data,
                monitor_data=monitor_data,
                monitor_labels=monitor_labels,
                epochs=epochs_per_trial,
                batch_size=batch_size,
                verbose=verbose,
            )

            self.trials.append(result)
            self._save_progress()

        return self._get_results()

    def _perturb_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb a configuration slightly."""
        new_config = config.copy()

        # Randomly change 1-3 parameters
        n_changes = np.random.randint(1, 4)
        params_to_change = np.random.choice(
            list(config.keys()), n_changes, replace=False
        )

        for param in params_to_change:
            if param == "encoder_type":
                new_config[param] = str(
                    np.random.choice(self.search_space.encoder_types)
                )
            elif param == "latent_dim":
                new_config[param] = int(np.random.choice(self.search_space.latent_dims))
            elif param == "projection_dim":
                new_config[param] = int(
                    np.random.choice(self.search_space.projection_dims)
                )
            elif param == "encoder_filters":
                new_config[param] = self.search_space.encoder_filters[
                    np.random.choice(len(self.search_space.encoder_filters))
                ]
            elif param == "encoder_units":
                new_config[param] = self.search_space.encoder_unit_options[
                    np.random.choice(len(self.search_space.encoder_unit_options))
                ]
            elif param == "kernel_size":
                new_config[param] = int(
                    np.random.choice(self.search_space.kernel_sizes)
                )
            elif param == "num_heads":
                new_config[param] = int(np.random.choice(self.search_space.num_heads))
            elif param == "learning_rate":
                new_config[param] = float(
                    np.random.choice(self.search_space.learning_rates)
                )
            elif param == "reconstruction_weight":
                new_config[param] = float(
                    np.random.choice(self.search_space.reconstruction_weights)
                )
            elif param == "contrastive_weight":
                new_config[param] = float(
                    np.random.choice(self.search_space.contrastive_weights)
                )
            elif param == "temperature":
                new_config[param] = float(
                    np.random.choice(self.search_space.temperatures)
                )
            elif param == "anomaly_ratio":
                new_config[param] = float(
                    np.random.choice(self.search_space.anomaly_ratios)
                )
            elif param == "scoring_method":
                new_config[param] = str(
                    np.random.choice(self.search_space.scoring_methods)
                )
            elif param == "k_neighbors":
                new_config[param] = int(np.random.choice(self.search_space.k_neighbors))
            elif param == "dropout_rate":
                new_config[param] = float(
                    np.random.choice(self.search_space.dropout_rates)
                )

        return new_config

    def _save_progress(self) -> None:
        """Save search progress to disk."""
        results = {
            "trials": [t.to_dict() for t in self.trials],
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "objective": self.objective,
        }

        with open(self.directory / "search_progress.json", "w") as f:
            json.dump(results, f, indent=2)

    def _get_results(self) -> Dict[str, Any]:
        """Get final search results."""
        return {
            "best_config": self.best_config,
            "best_metrics": {
                "roc_auc": self.best_trial.roc_auc,
                "pr_auc": self.best_trial.pr_auc,
                "f1_score": self.best_trial.f1_score,
                "val_loss": self.best_trial.val_loss,
            }
            if self.best_trial
            else None,
            "n_trials": len(self.trials),
            "all_trials": [t.to_dict() for t in self.trials],
        }

    def get_best_trainer(self) -> Optional[CARLATrainer]:
        """Get the trainer with best configuration."""
        return self.best_trainer

    def summary(self) -> None:
        """Print search summary."""
        print("\n" + "=" * 60)
        print("CARLA HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 60)
        print(f"Total trials: {len(self.trials)}")
        print(f"Objective: {self.objective}")

        if self.best_trial:
            print(f"\nBest Trial: {self.best_trial.trial_id}")
            print(f"  ROC-AUC: {self.best_trial.roc_auc:.4f}")
            print(f"  PR-AUC: {self.best_trial.pr_auc:.4f}")
            print(f"  F1 Score: {self.best_trial.f1_score:.4f}")
            print(f"  Val Loss: {self.best_trial.val_loss:.4f}")
            print(f"  Training Time: {self.best_trial.training_time:.1f}s")
            print("\nBest Configuration:")
            for k, v in self.best_config.items():
                print(f"    {k}: {v}")

        # Top 5 trials
        if len(self.trials) > 1:
            sorted_trials = sorted(
                self.trials,
                key=lambda t: t.roc_auc if self.objective == "roc_auc" else -t.val_loss,
                reverse=True,
            )

            print("\nTop 5 Trials:")
            for i, trial in enumerate(sorted_trials[:5]):
                print(
                    f"  {i + 1}. Trial {trial.trial_id}: "
                    f"ROC-AUC={trial.roc_auc:.4f}, F1={trial.f1_score:.4f}"
                )

        print("=" * 60)
