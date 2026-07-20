"""
Train Anomaly Detection Model with Hyperparameter Search
=========================================================

Runs hyperparameter optimisation for standard autoencoder architectures
(Conv1D-AE, LSTM-AE, VAE, Transformer-AE) and the CARLA contrastive
autoencoder.

Usage::

    # Standard model with HP search
    python -m src.train --model conv1d_ae --n-trials 15 --final-epochs 100

    # CARLA contrastive autoencoder
    python -m src.train --model carla --n-trials 10

    # Quick debug run
    python -m src.train --model conv1d_ae --debug
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import keras
import numpy as np

from .data.dataset import PowerConverterDataset
from .training.carla_hyperparameter_search import (
    CARLAHyperparameterSearch,
    CARLASearchSpace,
)
from .training.carla_trainer import CARLAConfig, CARLATrainer
from .training.hyperparameter_search import HyperparameterSearch, SearchSpace
from .utils.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)

# ---- All supported model names ----
STANDARD_MODELS = [
    "conv1d_ae",
    "lstm_ae",
    "gru_ae",
    "mlp_ae",
    "vae",
    "transformer_ae",
]
CARLA_MODELS = ["carla_conv1d", "carla_lstm", "carla_gru", "carla_transformer", "carla_mlp"]
ALL_MODELS = STANDARD_MODELS + CARLA_MODELS

# ---- Default search spaces ----
STANDARD_SEARCH_SPACES = {
    "conv1d_ae": SearchSpace(
        lr_min=1e-5,
        lr_max=2e-3,
        latent_dims=[16, 32, 64],
        filter_options=[[32, 64], [32, 64, 128], [64, 128]],
        kernel_sizes=[3, 5],
        dropout_rates=[0.0, 0.1, 0.2],
    ),
    "lstm_ae": SearchSpace(
        lr_min=1e-5,
        lr_max=1e-3,
        latent_dims=[16, 32, 64],
        lstm_unit_options=[[32, 16], [64, 32], [128, 64]],
        dropout_rates=[0.0, 0.1, 0.2],
    ),
    "gru_ae": SearchSpace(
        lr_min=1e-5,
        lr_max=1e-3,
        latent_dims=[16, 32, 64],
        gru_unit_options=[[32, 16], [64, 32], [128, 64]],
        dropout_rates=[0.0, 0.1, 0.2],
    ),
    "vae": SearchSpace(
        lr_min=1e-5,
        lr_max=1e-3,
        latent_dims=[16, 32, 64],
        dropout_rates=[0.0, 0.1, 0.2],
    ),
    "transformer_ae": SearchSpace(
        lr_min=1e-5,
        lr_max=1e-3,
        latent_dims=[16, 32, 64],
        dropout_rates=[0.0, 0.1, 0.2],
    ),
    "mlp_ae": SearchSpace(
        lr_min=1e-5,
        lr_max=1e-3,
        latent_dims=[16, 32, 64, 128],
        mlp_encoder_unit_options=[[128, 64], [256, 128], [256, 128, 64]],
        mlp_decoder_unit_options=[[64, 128], [128, 256], [64, 128, 256]],
        dropout_rates=[0.0, 0.1, 0.2],
    ),
}

CARLA_SEARCH_SPACE = CARLASearchSpace(
    encoder_types=["conv1d", "lstm", "gru", "transformer", "mlp"],  # Overridden per-variant in _train_carla()
    latent_dims=[32, 64, 128, 256],
    projection_dims=[64, 128, 256, 512],
    learning_rates=[5e-5, 1e-4, 3e-4, 5e-4],
    reconstruction_weights=[5.0, 10.0, 20.0],
    contrastive_weights=[0.5, 1.0],
    temperatures=[0.05, 0.07, 0.1],
    anomaly_ratios=[0.3, 0.5],
    dropout_rates=[0.0, 0.1],
    encoder_filters=[
        [64, 128], [128, 256],
        [64, 128, 256], [128, 256, 512]
    ],
    encoder_unit_options=[
        [128, 64], [256, 128], [512, 256],
        [256, 128, 64], [512, 256, 128]
    ],
    kernel_sizes=[3, 5],
    num_heads=[2, 4, 8],
    scoring_methods=["knn", "centroid"],
    k_neighbors=[3, 5, 10],
)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train anomaly-detection models with hyperparameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----
    parser.add_argument(
        "--data-dir",
        type=str,
        nargs="+",
        default=["data/buck/buck_data"],
        help="One or more directories with simulation .txt files (concatenated)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for caching loaded data",
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=5.0,
        help="Max deviation %% for labelling samples as normal",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max simulation files to load (None = all)",
    )

    # ---- Model ----
    parser.add_argument(
        "--model",
        type=str,
        default="conv1d_ae",
        choices=ALL_MODELS,
        help="Model architecture to train",
    )

    # ---- HP search ----
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of HP search trials",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=30,
        help="Max epochs during each HP search trial",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=100,
        help="Epochs for final retraining with best hyperparameters",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (used for standard models; CARLA batch size comes from HP search)",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default="bayesian",
        choices=["bayesian", "random", "grid", "hyperband"],
        help="HP search strategy (hyperband is only available for standard models)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "f1_score", "val_loss"],
        help="Optimisation objective for CARLA HP search",
    )

    # ---- CARLA search-space customisation ----
    parser.add_argument(
        "--encoder-types",
        type=str,
        nargs="+",
        default=None,
        help="Encoder types to search (CARLA only, default: derived from model name)",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Latent dimensions to search (CARLA only)",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[5e-5, 1e-4, 3e-4, 5e-4],
        help="Learning rates to search (CARLA only)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Root output directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional subdirectory name for isolating experiment runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # ---- Debug mode ----
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trained models and hyperparameter searches",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: fewer samples, trials, and epochs",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=500,
        help="Max samples per split in debug mode",
    )
    parser.add_argument(
        "--debug-trials",
        type=int,
        default=3,
        help="HP search trials in debug mode",
    )
    parser.add_argument(
        "--debug-epochs",
        type=int,
        default=3,
        help="Epochs per trial in debug mode",
    )

    return parser.parse_args(argv)


# =====================================================================
# Standard AE training pipeline
# =====================================================================


def _train_standard(
    model_name: str,
    dataset: PowerConverterDataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """HP search + final training for standard autoencoder models."""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Hyperparameter Search: {model_name}")
    logger.info("=" * 60)

    search_space = STANDARD_SEARCH_SPACES[model_name]

    hp_search = HyperparameterSearch(
        model_type=model_name,
        input_shape=dataset.input_shape,
        search_space=search_space,
        project_name=f"{model_name}_hp_search",
        directory=str(output_dir / "hp_tuning"),
    )

    # ---- 1. Search ----
    logger.info("\n[1/3] Running hyperparameter search …")
    start_time = time.time()
    search_results = hp_search.search(
        train_data=dataset.train_data,
        val_data=dataset.val_data,
        method=args.search_method,
        max_trials=args.n_trials,
        epochs=args.epochs_per_trial,
        batch_size=args.batch_size,
    )
    search_time = time.time() - start_time

    best_hps = search_results["best_hyperparameters"]
    logger.info(f"Best hyperparameters: {json.dumps(best_hps, indent=2)}")

    # ---- 2. Retrain best model ----
    logger.info(f"\n[2/3] Retraining best model for {args.final_epochs} epochs …")
    best_model = hp_search.get_best_model()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    model_save_dir = output_dir
    model_save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(model_save_dir / "best_model.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    backup = keras.callbacks.BackupAndRestore(backup_dir=str(model_save_dir / "backup"))

    start_time = time.time()
    history = best_model.fit(
        dataset.train_data,
        dataset.train_data,
        validation_data=(dataset.val_data, dataset.val_data),
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        callbacks=[early_stop, checkpoint, backup],
        verbose=1,
    )
    training_time = time.time() - start_time

    # Save artefacts
    # Unified model config
    model_config = {
        "name": model_name,
        "type": "standard",
        "input_shape": dataset.input_shape,
        **best_hps,
    }
    with open(model_save_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    with open(model_save_dir / "training_history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()},
            f,
        )

    # Save preprocessor
    dataset.preprocessor.save(str(model_save_dir / "preprocessor"))

    # Cleanup HP tuning directory
    hp_tuning_dir = output_dir / "hp_tuning"
    if hp_tuning_dir.exists():
        logger.info(f"Cleaning up hyperparameter tuning directory: {hp_tuning_dir}")
        shutil.rmtree(hp_tuning_dir)

    return {
        "model_name": model_name,
        "type": "standard",
        "best_hyperparameters": best_hps,
        "search_time": search_time,
        "training_time": training_time,
    }


# =====================================================================
# CARLA training pipeline
# =====================================================================


def _train_carla(
    model_name: str,
    dataset: PowerConverterDataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """HP search + final training for the CARLA contrastive autoencoder."""

    # Extract encoder type from model name (e.g. "carla_conv1d" -> "conv1d")
    encoder_type = model_name.replace("carla_", "")
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Hyperparameter Search: CARLA Contrastive AE ({encoder_type} encoder)")
    logger.info("=" * 60)

    # CARLA trains only on normal data
    train_normal = dataset.train_data
    val_normal = dataset.val_data

    # Build search space from base search space and apply CLI overrides
    search_space = CARLA_SEARCH_SPACE
    # Pin encoder type from model name, unless explicitly overridden via CLI
    search_space.encoder_types = args.encoder_types or [encoder_type]
    search_space.latent_dims = args.latent_dims
    search_space.learning_rates = args.learning_rates

    logger.info(f"Search space combinations: {search_space.total_combinations()}")

    hp_search = CARLAHyperparameterSearch(
        input_shape=dataset.input_shape,
        search_space=search_space,
        project_name=f"{model_name}_hp_search",
        directory=str(output_dir / "hp_tuning"),
        objective=args.objective,
        frequencies=dataset.frequencies,
        scaler=dataset.preprocessor.scaler,
        use_real_imag=dataset.preprocessor.use_real_imag,
    )

    # ---- 1. Search ----
    logger.info(f"\n[1/3] Running CARLA {args.search_method} search …")
    start_time = time.time()

    search_kwargs = dict(
        train_data=train_normal,
        val_data=val_normal,
        monitor_data=dataset.fit_val_data,
        monitor_labels=dataset.fit_val_labels,
        epochs_per_trial=args.epochs_per_trial,
        batch_size=args.batch_size,
        verbose=1,
    )

    if args.search_method == "random":
        search_results = hp_search.search_random(
            n_trials=args.n_trials,
            **search_kwargs,
        )
    elif args.search_method == "grid":
        search_results = hp_search.search_grid(
            max_trials=args.n_trials,
            **search_kwargs,
        )
    else:  # bayesian (default)
        search_results = hp_search.search_bayesian(
            n_trials=args.n_trials,
            n_initial=min(5, max(1, args.n_trials // 3)),
            **search_kwargs,
        )

    search_time = time.time() - start_time

    # Print search summary
    hp_search.summary()

    best_config = search_results["best_config"]
    logger.info(f"Best configuration: {json.dumps(best_config, indent=2, default=str)}")

    # Save search results
    with open(output_dir / "search_results.json", "w") as f:
        json.dump(search_results, f, indent=2, default=str)

    # ---- 2. Retrain best config ----
    logger.info(f"\n[2/3] Retraining best CARLA model for {args.final_epochs} epochs …")

    model_save_dir = output_dir
    model_save_dir.mkdir(parents=True, exist_ok=True)

    config = CARLAConfig(
        epochs=args.final_epochs,
        batch_size=args.batch_size,
        learning_rate=float(best_config["learning_rate"]),
        reconstruction_weight=float(best_config["reconstruction_weight"]),
        contrastive_weight=float(best_config["contrastive_weight"]),
        temperature=float(best_config["temperature"]),
        anomaly_ratio=float(best_config["anomaly_ratio"]),
        scoring_method=str(best_config["scoring_method"]),
        k_neighbors=int(best_config["k_neighbors"]),
        checkpoint_dir=str(model_save_dir / "checkpoints"),
        log_dir=str(model_save_dir / "logs"),
        early_stopping=True,
        patience=20,
        save_best_only=True,
    )

    trainer = CARLATrainer(config=config, experiment_name=f"{model_name}_best")
    trainer.create_model(
        input_shape=dataset.input_shape,
        latent_dim=int(best_config["latent_dim"]),
        projection_dim=int(best_config["projection_dim"]),
        encoder_type=str(best_config["encoder_type"]),
        encoder_filters=best_config["encoder_filters"],
        encoder_units=best_config.get("encoder_units", [64, 32]),
        kernel_size=int(best_config["kernel_size"]),
        num_heads=int(best_config["num_heads"]),
        dropout_rate=float(best_config["dropout_rate"]),
    )
    trainer.set_physics_context(
        frequencies=dataset.frequencies,
        scaler=dataset.preprocessor.scaler,
        use_real_imag=dataset.preprocessor.use_real_imag,
    )
    trainer.setup_training()

    start_time = time.time()
    history = trainer.train(
        train_data=train_normal,
        val_data=val_normal,
        verbose=1,
    )
    training_time = time.time() - start_time

    # Save artefacts
    # Unified model config
    model_config = {
        "name": model_name,
        "type": "carla",
        "input_shape": dataset.input_shape,
        **best_config,
    }
    with open(model_save_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2, default=str)

    with open(model_save_dir / "training_history.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.items()},
            f,
        )

    # Save preprocessor
    dataset.preprocessor.save(str(model_save_dir / "preprocessor"))

    # Cleanup HP tuning directory
    hp_tuning_dir = output_dir / "hp_tuning"
    if hp_tuning_dir.exists():
        logger.info(f"Cleaning up hyperparameter tuning directory: {hp_tuning_dir}")
        shutil.rmtree(hp_tuning_dir)

    return {
        "model_name": model_name,
        "type": "carla",
        "best_hyperparameters": best_config,
        "search_time": search_time,
        "training_time": training_time,
    }


# =====================================================================
# Debug helpers
# =====================================================================


def _apply_debug_limits(dataset: PowerConverterDataset, max_samples: int) -> None:
    """Truncate dataset splits for debug runs."""
    train_limit = min(max_samples, len(dataset.train_data))
    val_limit = min(max_samples // 4, len(dataset.val_data))
    test_limit = min(max_samples // 4, len(dataset.test_data))

    dataset._splits["train"] = dataset._splits["train"][:train_limit]
    dataset._splits["val"] = dataset._splits["val"][:val_limit]
    dataset._splits["test"] = dataset._splits["test"][:test_limit]
    dataset._splits["test_labels"] = dataset._splits["test_labels"][:test_limit]

    logger.info(
        f"Debug: limited to {train_limit}/{val_limit}/{test_limit} "
        "train/val/test samples"
    )


# =====================================================================
# Main
# =====================================================================


def main(argv=None) -> None:
    """Run the training pipeline."""
    args = parse_args(argv)

    # ---- Seed ----
    import tensorflow as tf

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---- Debug overrides ----
    if args.debug:
        logger.info("\n" + "!" * 60)
        logger.info("DEBUG MODE ENABLED — reduced trials, epochs, and data")
        logger.info("!" * 60)
        # Use a temporary directory for debug results
        args.output_dir = "tmp/experiments"
        args.n_trials = args.debug_trials
        args.epochs_per_trial = args.debug_epochs
        args.final_epochs = args.debug_epochs * 2
        args.max_files = min(args.max_files or 50, 50)

    if args.experiment_name:
        output_dir = Path(args.output_dir) / args.experiment_name / args.model
    else:
        output_dir = Path(args.output_dir) / args.model

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Power Converter Anomaly Detection — Training with HP Search")
    logger.info("=" * 60)
    logger.info(f"Model:             {args.model}")
    logger.info(f"Data directory:    {args.data_dir}")
    logger.info(f"Trials:            {args.n_trials}")
    logger.info(f"Epochs per trial:  {args.epochs_per_trial}")
    logger.info(f"Final epochs:      {args.final_epochs}")

    # ---- Load data ----
    logger.info("\nLoading data …")
    dataset = PowerConverterDataset(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        normal_threshold=args.normal_threshold,
    )
    dataset.load(max_files=args.max_files)
    dataset.preprocess(fit=True)
    dataset.prepare_splits(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=args.seed
    )

    if args.debug:
        _apply_debug_limits(dataset, args.debug_samples)

    logger.info(f"Input shape:   {dataset.input_shape}")
    logger.info(f"Train:         {len(dataset.train_data)}")
    logger.info(f"Val:           {len(dataset.val_data)}")
    logger.info(f"Test:          {len(dataset.test_data)}")

    # ---- Train ----
    if args.model in CARLA_MODELS:
        result = _train_carla(args.model, dataset, output_dir, args)
    else:
        result = _train_standard(args.model, dataset, output_dir, args)

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Search time:    {result['search_time'] / 60:.1f} min")
    logger.info(f"Training time:  {result['training_time'] / 60:.1f} min")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
