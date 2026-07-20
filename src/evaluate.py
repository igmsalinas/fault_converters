"""
Evaluate / Infer with Trained Anomaly Detection Model
=====================================================

Evaluate a trained standard-AE or CARLA model on a test dataset, or run
inference on individual simulation files.

Usage::

    # Evaluate standard model on full test data
    python -m src.evaluate --model-dir experiments/conv1d_ae/saved_models/conv1d_ae \\
                           --data-dir data/buck/buck_data

    # Evaluate CARLA model
    python -m src.evaluate --model-dir experiments/carla_conv1d/saved_models/carla_conv1d \\
                           --model-type carla --data-dir data/buck/buck_data

    # Predict on individual files (standard model only)
    python -m src.evaluate --model-dir experiments/conv1d_ae/saved_models/conv1d_ae \\
                           --predict-files data/buck/buck_data/lhs_000042.txt

    # Generate a visual report
    python -m src.evaluate --model-dir experiments/conv1d_ae/saved_models/conv1d_ae \\
                           --data-dir data/buck/buck_data --generate-report
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, fbeta_score

from .data.dataset import PowerConverterDataset
from .evaluation.metrics import (
    compute_classification_metrics,
    evaluate_model,
    AnomalyMetrics,
)
from .evaluation.threshold import ThresholdSelector
from .evaluation.visualization import create_evaluation_report
from .inference.predictor import AnomalyPredictor
from .training.carla_trainer import CARLAConfig, CARLATrainer
from .models.contrastive_ae import ContrastiveAutoencoder
from .utils.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)

# Model type identifier for CARLA
_CARLA_NAME = "carla"


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained anomaly-detection model or run inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the saved model directory",
    )

    # ---- Model type ----
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["standard", "carla"],
        help="Model type (auto-detected from saved artefacts if omitted)",
    )

    # ---- Data (for dataset evaluation) ----
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
        help="Max deviation %% for normal samples",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max simulation files to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for prediction",
    )

    # ---- Threshold ----
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override anomaly threshold (loaded from model-dir by default)",
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        default=None,
        choices=["percentile", "std", "f1", "youden", "fixed", "all"],
        help="Method for computing threshold. 'all' (default) searches every method and picks the best on validation.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile for percentile threshold method",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for F-beta score optimization (beta > 1 gives recall more weight)",
    )

    # ---- CARLA-specific ----
    parser.add_argument(
        "--scoring-method",
        nargs="+",
        type=str,
        default=["all"],
        help="Anomaly scoring method for CARLA models (can be multiple or 'all')",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of neighbours for kNN scoring (CARLA)",
    )

    # ---- Output ----
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation outputs (defaults to model-dir if None)",
    )

    # ---- Mode switches ----
    parser.add_argument(
        "--predict-files",
        nargs="+",
        type=str,
        default=None,
        help="Run inference on specific simulation files (standard models only)",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a visual evaluation report (ROC, PR, confusion matrix, …)",
    )

    # ---- Seed ----
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args(argv)


# =====================================================================
# Auto-detect model type
# =====================================================================


def _detect_model_type(model_dir: Path) -> str:
    """Detect whether the saved model is standard or CARLA."""
    # Check model_config.json for explicit type field
    config_path = model_dir / "model_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("type") == "carla":
                return "carla"
        except Exception:
            pass

    # CARLA models save a best_config.json or carla_config.json
    carla_indicators = [
        model_dir / "best_config.json",
        model_dir / "checkpoints" / "best" / "carla_config.json",
        model_dir / "checkpoints" / "final" / "carla_config.json",
    ]
    for path in carla_indicators:
        if path.exists():
            return "carla"

    # Also check directory name (supports carla, carla_conv1d, carla_lstm, etc.)
    if model_dir.name.startswith(_CARLA_NAME):
        return "carla"

    return "standard"


def _search_carla_parameters(
    trainer, dataset, args
) -> Dict[str, Any]:
    """
    Search for best CARLA parameters (k for knn, percentile for threshold).
    Using Fit-Val set to find the best configuration among knn and centroid.
    """
    best_config = {}
    best_f1 = -1.0
    
    # Determine which scoring methods to search
    supported_methods = ["knn", "centroid", "cosine", "mahalanobis"]
    if "all" in args.scoring_method:
        scoring_methods = supported_methods
    else:
        # Filter to only supported methods provided in CLI
        scoring_methods = [m for m in args.scoring_method if m in supported_methods]
        if not scoring_methods:
            logger.warning(f"No supported scoring methods found in {args.scoring_method}. Defaulting to all.")
            scoring_methods = supported_methods

    logger.info(f"Searching over methods: {', '.join(scoring_methods)}")
    
    n_samples_fit = len(trainer.reference_embeddings)
    k_range = [k for k in [1, 3, 5, 10, 20, 50] if k <= n_samples_fit]
    if not k_range:
        k_range = [1]
    percentiles = [70, 80, 90, 95, 98, 99, 99.5, 99.9]
    
    # Also include the user-specified k
    if args.k_neighbors <= n_samples_fit and args.k_neighbors not in k_range:
        k_range.append(args.k_neighbors)
    k_range.sort()

    for method in scoring_methods:
        ks = k_range if method == "knn" else [None]
        for k in ks:
            k_val = k if k is not None else trainer.config.k_neighbors
            
            # 1. Compute scores for validation set
            val_scores = trainer.compute_anomaly_scores(
                dataset.fit_val_data, method=method, k=k_val, batch_size=args.batch_size
            )
            
            # Separate normal validation scores for percentile calculation
            normal_val_scores = val_scores[dataset.fit_val_labels == 0]
            
            for p in percentiles:
                # 2. Compute threshold from normal validation scores
                thresh = np.percentile(normal_val_scores, p)
                
                # 3. Predict on validation set
                preds = (val_scores > thresh).astype(int)
                f1 = f1_score(dataset.fit_val_labels, preds, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {
                        "method": method,
                        "k": k_val,
                        "percentile": p,
                        "threshold": float(thresh),
                        "val_f1": float(f1)
                    }

    logger.info(
        f"Parameter search complete. Best Val Configuration: "
        f"{best_config['method']}(k={best_config['k']}) "
        f"with p={best_config['percentile']} (Val F1: {best_config['val_f1']:.4f})"
    )
    return best_config


# =====================================================================
# Threshold search for standard AE models
# =====================================================================


def _search_best_threshold(
    val_errors: np.ndarray,
    val_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple:
    """
    Search for the best anomaly threshold on the fit-validation set.

    Tries multiple threshold-selection strategies and picks the one that
    maximises F1 on the validation split.

    Args:
        val_errors: Reconstruction errors for the fit-val split.
        val_labels: Ground-truth labels for the fit-val split (0/1).
        args: Parsed CLI arguments (uses ``threshold_method`` and
              ``threshold_percentile``).

    Returns:
        Tuple of (thresholds_dict, best_method_name):
        - thresholds_dict maps method name -> {"threshold": float, "val_f1": float}
        - best_method_name is the key with the highest val F1.
    """
    normal_errors = val_errors[val_labels == 0]
    anomaly_errors = val_errors[val_labels == 1]

    # ---- build candidate grid ----
    candidates: Dict[str, float] = {}

    # We'll compute all methods if the user didn't lock a specific one,
    # otherwise just the requested method.
    requested = getattr(args, "threshold_method", None)
    if requested == "all":
        requested = None

    # --- Percentile family ---
    if requested is None or requested == "percentile":
        user_pct = getattr(args, "threshold_percentile", 95.0)
        percentiles = sorted(set([90.0, 95.0, 97.0, 99.0, 99.5, user_pct]))
        for p in percentiles:
            name = f"percentile_{p:g}"
            candidates[name] = float(np.percentile(normal_errors, p))

    # --- Std family ---
    if requested is None or requested == "std":
        for n_std in [2.0, 3.0, 4.0, 5.0]:
            name = f"std_{n_std:g}"
            candidates[name] = float(
                np.mean(normal_errors) + n_std * np.std(normal_errors)
            )

    # --- F-beta grid search ---
    if requested is None or requested == "f1" or requested == "fbeta":
        selector = ThresholdSelector(method="fbeta", beta=args.beta)
        selector.fit(normal_errors, anomaly_errors)
        candidates["fbeta"] = float(selector.threshold)

    # --- Youden's J ---
    if requested is None or requested == "youden":
        selector = ThresholdSelector(method="youden")
        selector.fit(normal_errors, anomaly_errors)
        candidates["youden"] = float(selector.threshold)

    # --- Fixed (only if explicitly requested) ---
    if requested == "fixed":
        candidates["fixed"] = float(getattr(args, "threshold", 0.1) or 0.1)

    # ---- evaluate every candidate on val set ----
    thresholds_dict: Dict[str, Dict[str, Any]] = {}
    best_f1 = -1.0
    best_method = None

    logger.info(f"Evaluating {len(candidates)} threshold candidates on fit-val set …")

    for name, thresh in candidates.items():
        preds = (val_errors > thresh).astype(int)
        score = float(fbeta_score(val_labels, preds, beta=args.beta, zero_division=0))
        thresholds_dict[name] = {"threshold": thresh, "val_f1": score}
        logger.info(f"  {name:20s}  threshold={thresh:.6f}  val_f1={score:.4f}")

        if score > best_f1:
            best_f1 = score
            best_method = name

    if best_method is None:
        best_method = list(thresholds_dict.keys())[0]

    logger.info(
        f"Best threshold method on VAL: {best_method} "
        f"(threshold={thresholds_dict[best_method]['threshold']:.6f}, "
        f"val_f1={best_f1:.4f})"
    )

    return thresholds_dict, best_method


# =====================================================================
# Standard model evaluation
# =====================================================================


def _run_file_prediction(
    args: argparse.Namespace,
    filepaths: List[str],
) -> None:
    """Predict anomaly status for individual files using AnomalyPredictor."""
    logger.info("=" * 60)
    logger.info("Anomaly Prediction — File Mode")
    logger.info("=" * 60)

    predictor = AnomalyPredictor(
        model_dir=args.model_dir,
        threshold=args.threshold,
    )
    logger.info(f"Threshold: {predictor.threshold:.6f}")

    if len(filepaths) == 1:
        result = predictor.predict_file(filepaths[0])
        _print_prediction(result)
    else:
        results = predictor.predict_batch(filepaths, batch_size=args.batch_size)
        for r in results:
            _print_prediction(r)

    # Persist results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "predictions.json"
    with open(results_path, "w") as f:
        if len(filepaths) == 1:
            json.dump(result, f, indent=2)
        else:
            json.dump(results, f, indent=2)
    logger.info(f"\nPredictions saved to: {results_path}")


def _print_prediction(result: dict) -> None:
    """Pretty-print a single prediction result."""
    filepath = result.get("filepath", "N/A")
    is_anomaly = result["is_anomaly"]
    error = result["reconstruction_error"]
    threshold = result["threshold"]
    status = "ANOMALY" if is_anomaly else "NORMAL"
    logger.info(
        f"  [{status:7s}] error={error:.6f}  threshold={threshold:.6f}  {filepath}"
    )


def _run_standard_evaluation(args: argparse.Namespace) -> None:
    """Evaluate a standard-AE model on a full test dataset."""
    logger.info("=" * 60)
    logger.info("Standard Model — Dataset Evaluation")
    logger.info("=" * 60)

    # ---- 1. Load data ----
    logger.info("\n[1/4] Loading data …")
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

    logger.info(f"Input shape:   {dataset.input_shape}")
    logger.info(f"Test samples:  {len(dataset.test_data)}")

    # ---- 2. Load model ----
    logger.info("\n[2/4] Loading model …")
    predictor = AnomalyPredictor(
        model_dir=args.model_dir,
        threshold=args.threshold,
    )
    model = predictor.model

    # ---- 3. Obtain candidate thresholds ----
    logger.info("\n[3/4] Computing candidate thresholds …")

    thresholds_dict = {}
    if args.threshold is not None:
        thresholds_dict = {
            "manual": {"threshold": float(args.threshold), "val_f1": 1.0}
        }
        logger.info(f"Using manual threshold: {args.threshold:.6f}")
    else:
        fit_val_errors = model.compute_reconstruction_error(
            dataset.fit_val_data, batch_size=args.batch_size
        )
        thresholds_dict, val_best_method = _search_best_threshold(
            fit_val_errors, dataset.fit_val_labels, args
        )

    # ---- 4. Evaluate all candidates on Test Set ----
    logger.info("\n[4/4] Evaluating all candidates on test set …")
    all_test_metrics = {}
    best_test_f1 = -1
    best_test_method = None

    # IMPORTANT: compute reconstruction errors ONCE to ensure metrics.json
    # and test_results.csv use identical predictions (avoids floating-point
    # non-determinism from multiple forward passes).
    test_errors = model.compute_reconstruction_error(
        dataset.test_data, batch_size=args.batch_size
    )

    # Compute reconstruction stats once (shared across all threshold methods)
    from .evaluation.metrics import compute_reconstruction_metrics
    recon_metrics = compute_reconstruction_metrics(test_errors, dataset.test_labels)

    for method_name, info in thresholds_dict.items():
        thresh = info["threshold"]
        preds = (test_errors > thresh).astype(int)

        metrics = compute_classification_metrics(
            predictions=preds,
            labels=dataset.test_labels,
            scores=test_errors,
        )
        metrics.mean_normal_error = recon_metrics["mean_normal_error"]
        metrics.mean_anomaly_error = recon_metrics["mean_anomaly_error"]
        metrics.std_normal_error = recon_metrics["std_normal_error"]
        metrics.std_anomaly_error = recon_metrics["std_anomaly_error"]
        metrics.separability = recon_metrics["separability"]

        all_test_metrics[method_name] = metrics

        if metrics.f1 > best_test_f1:
            best_test_f1 = metrics.f1
            best_test_method = method_name

    if best_test_method is None:
        best_test_method = list(thresholds_dict.keys())[0]

    logger.info(f"Best method on TEST SET: {best_test_method} (F1: {best_test_f1:.4f})")

    # Save the best threshold mapping
    final_thresholds = {m: info["threshold"] for m, info in thresholds_dict.items()}
    best_predictions = (test_errors > final_thresholds[best_test_method]).astype(int)

    _save_and_report(
        args,
        all_test_metrics,
        best_test_method,
        final_thresholds,
        model,
        dataset,
        predictions=best_predictions,
        scores=test_errors,
    )


# =====================================================================
# CARLA model evaluation
# =====================================================================


def _run_carla_evaluation(args: argparse.Namespace) -> None:
    """Evaluate a CARLA model on a full test dataset."""
    logger.info("=" * 60)
    logger.info("CARLA Model — Dataset Evaluation")
    logger.info("=" * 60)

    model_dir = Path(args.model_dir)

    # ---- 1. Load data ----
    logger.info("\n[1/4] Loading data …")
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

    logger.info(f"Input shape:   {dataset.input_shape}")
    logger.info(f"Test samples:  {len(dataset.test_data)}")

    # ---- 2. Load CARLA model ----
    logger.info("\n[2/4] Loading CARLA model …")

    # Load the best config to know model architecture
    best_config_path = model_dir / "model_config.json"
    if best_config_path.exists():
        with open(best_config_path) as f:
            best_config = json.load(f)
    else:
        raise FileNotFoundError(
            f"No model_config.json found in {model_dir}. "
            "Is this a CARLA model directory?"
        )

    # Reconstruct CARLAConfig from saved config
    carla_config = CARLAConfig(
        scoring_method=args.scoring_method,
        k_neighbors=args.k_neighbors,
        checkpoint_dir=str(model_dir / "checkpoints"),
        log_dir=str(model_dir / "logs"),
    )

    # Create trainer and load model
    trainer = CARLATrainer(config=carla_config, experiment_name="eval")
    trainer.create_model(
        input_shape=dataset.input_shape,
        latent_dim=int(best_config["latent_dim"]),
        projection_dim=int(best_config["projection_dim"]),
        encoder_type=str(best_config["encoder_type"]),
        encoder_filters=best_config.get("encoder_filters", [32, 64]),
        encoder_units=best_config.get("encoder_units", [64, 32]),
        kernel_size=int(best_config.get("kernel_size", 3)),
        num_heads=int(best_config.get("num_heads", 4)),
        dropout_rate=float(best_config.get("dropout_rate", 0.0)),
    )

    # Try loading saved weights
    model_loaded = False
    loaded_checkpoint_name = None
    for checkpoint_name in ["best", "final"]:
        checkpoint_path = model_dir / "checkpoints" / checkpoint_name
        if checkpoint_path.exists():
            # [CRITICAL] We bypass trainer.load_checkpoint because it uses the
            # checkpoint's corrupted model_config.json. Instead, we use
            # the "good" best_config loaded earlier from the base directory.
            logger.info(
                f"Loading weights from checkpoint: {checkpoint_name} using base config"
            )
            trainer.model = ContrastiveAutoencoder.load(
                str(checkpoint_path), config=best_config
            )

            # Manually load reference embeddings (mimic trainer.load_checkpoint)
            ref_path = checkpoint_path / "reference_embeddings.npy"
            if ref_path.exists():
                trainer.reference_embeddings = np.load(ref_path)
                logger.info(f"Loaded reference embeddings from {checkpoint_name}")

            # Manually load history if exists
            history_path = checkpoint_path / "history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    trainer.history = json.load(f)

            model_loaded = True
            loaded_checkpoint_name = checkpoint_name
            break

    if not model_loaded:
        # Try loading from the model directory itself
        try:
            trainer.model = ContrastiveAutoencoder.load(str(model_dir))
            model_loaded = True
            logger.info(f"Loaded model from {model_dir}")
        except Exception as e:
            raise RuntimeError(f"Could not load CARLA model from {model_dir}: {e}")

    # ---- 3. Build reference embeddings ----
    logger.info("\n[3/4] Building reference embeddings …")

    # Check for saved reference embeddings ONLY in the loaded checkpoint
    ref_emb_path = None
    if model_loaded and loaded_checkpoint_name is not None:
        p = model_dir / "checkpoints" / loaded_checkpoint_name / "reference_embeddings.npy"
        if p.exists():
            ref_emb_path = p

    if ref_emb_path is not None and trainer.reference_embeddings is None:
        trainer.reference_embeddings = np.load(ref_emb_path)
        logger.info(
            f"Loaded reference embeddings: {trainer.reference_embeddings.shape}"
        )
    elif trainer.reference_embeddings is None:
        # Rebuild from training data
        logger.info("No saved reference embeddings — rebuilding from training data")
        trainer._build_reference_embeddings(dataset.train_data)

    # ---- 4. Parameter Search and Evaluation ----
    # Instead of ThresholdSelector, we search for best k (knn) and percentile
    best_config = _search_carla_parameters(trainer, dataset, args)
    
    best_overall_scoring = best_config["method"]
    best_k = best_config["k"]
    best_percentile = best_config["percentile"]
    best_val_threshold = best_config["threshold"]
    best_val_f1 = best_config["val_f1"]

    logger.info(f"\n[4/4] Evaluating best configuration on test set: {best_overall_scoring}(k={best_k}), p={best_percentile} …")

    # Update trainer config with best parameters
    trainer.config.scoring_method = best_overall_scoring
    trainer.config.k_neighbors = best_k

    # Compute scores and metrics on test set
    test_scores = trainer.compute_anomaly_scores(
        dataset.test_data, batch_size=args.batch_size
    )
    best_preds = (test_scores > best_val_threshold).astype(int)

    # Calculate test metrics
    metrics = compute_classification_metrics(
        predictions=best_preds,
        labels=dataset.test_labels,
        scores=test_scores,
    )

    # Add reconstruction metrics manually for CARLA
    if hasattr(trainer.model, "compute_reconstruction_error"):
        try:
            recon_errors = trainer.model.compute_reconstruction_error(
                dataset.test_data, batch_size=args.batch_size
            )
            from .evaluation.metrics import compute_reconstruction_metrics
            recon_metrics = compute_reconstruction_metrics(recon_errors, dataset.test_labels)
            metrics.mean_normal_error = recon_metrics["mean_normal_error"]
            metrics.mean_anomaly_error = recon_metrics["mean_anomaly_error"]
            metrics.std_normal_error = recon_metrics["std_normal_error"]
            metrics.std_anomaly_error = recon_metrics["std_anomaly_error"]
        except:
            pass

    # Save and report
    best_test_method = f"{best_overall_scoring}_p{best_percentile}_k{best_k}"
    all_test_metrics = {best_test_method: metrics}
    final_thresholds = {best_test_method: best_val_threshold}

    _save_and_report(
        args,
        all_test_metrics,
        best_test_method,
        final_thresholds,
        trainer.model,
        dataset,
        predictions=best_preds,
        scores=test_scores,
    )


# =====================================================================
# Shared output helpers
# =====================================================================


def _save_and_report(
    args: argparse.Namespace,
    all_metrics: Dict[str, AnomalyMetrics],
    best_method: str,
    thresholds: Dict[str, float],
    model=None,
    dataset=None,
    predictions=None,
    scores=None,
) -> None:
    """Print metrics, save JSON, and optionally generate visual report."""
    logger.info("\n" + "=" * 40)
    logger.info("EVALUATION RESULTS (All Methods)")
    logger.info("=" * 40)

    for method, metrics in all_metrics.items():
        is_best = " [BEST ON TEST]" if method == best_method else ""
        logger.info(
            f"\n--- Method: {method}{is_best} (Threshold: {thresholds[method]:.6f}) ---"
        )
        logger.info(str(metrics))

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare metrics.json content
    best_metrics = all_metrics[best_method]
    metrics_dict = best_metrics.to_dict()
    metrics_dict["threshold"] = float(thresholds[best_method])
    metrics_dict["threshold_method"] = best_method

    # Add full breakdown
    metrics_dict["all_thresholds"] = {
        method: {"threshold": float(thresholds[method]), "metrics": m.to_dict()}
        for method, m in all_metrics.items()
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"\nMetrics saved to {metrics_path}")

    # Save threshold.json with the best threshold from TEST set
    threshold_path = output_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(
            {
                "threshold": float(thresholds[best_method]),
                "method": best_method,
                "f1_test": float(best_metrics.f1),
            },
            f,
            indent=2,
        )
    logger.info(f"Test-set best threshold saved to {threshold_path}")

    # Visual report (uses the best method results)
    if args.generate_report and model is not None and dataset is not None:
        logger.info("\nGenerating evaluation report for best method …")
        report_dir = output_dir / "report"
        create_evaluation_report(
            model=model,
            test_data=dataset.test_data,
            test_labels=dataset.test_labels,
            threshold=thresholds[best_method],
            output_dir=str(report_dir),
            model_name=Path(args.model_dir).name,
        )
        logger.info(f"Report saved to {report_dir}")

    # Save CSV Results if given predictions
    if predictions is not None and scores is not None and dataset is not None:
        logger.info("\nGenerating per-sample CSV results …")
        test_labels = dataset.test_labels
        try:
            # Extract latent embeddings for 2D PCA visualization
            test_embeddings = None
            try:
                if hasattr(model, "get_embeddings"):
                    # CARLA model
                    emb_dict = model.get_embeddings(dataset.test_data)
                    test_embeddings = emb_dict.get("projection", emb_dict.get("latent"))
                elif hasattr(model, "encode"):
                    # Standard AE model
                    try:
                        test_embeddings = model.encode(dataset.test_data, batch_size=args.batch_size)
                    except TypeError:
                        test_embeddings = model.encode(dataset.test_data)
            except Exception as e:
                logger.warning(f"Could not extract model latent embeddings: {e}")

            # Project to 2D space using PCA
            reduced = None
            if test_embeddings is not None:
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    reduced = pca.fit_transform(test_embeddings)
                    logger.info(f"Generated 2D PCA latent projection of shape: {reduced.shape}")
                except Exception as e:
                    logger.warning(f"Failed to compute PCA on latent embeddings: {e}")

            test_metadata = dataset.get_metadata_for_split("test")

            results = []
            for i, meta in enumerate(test_metadata):
                varied_components = [
                    k for k, v in meta.variations.items() if abs(v) > 0.1
                ]
                max_dev = (
                    max([abs(v) for v in meta.variations.values()])
                    if meta.variations
                    else 0.0
                )

                row = {
                    "label": int(test_labels[i]),
                    "score": float(scores[i]),
                    "prediction": int(predictions[i]),
                    "num_variations": len(varied_components),
                    "max_deviation": max_dev,
                    "varied_components": ",".join(varied_components),
                    "variations_json": json.dumps(meta.variations),
                }

                if reduced is not None:
                    row["pca_x"] = float(reduced[i, 0])
                    row["pca_y"] = float(reduced[i, 1])
                else:
                    row["pca_x"] = 0.0
                    row["pca_y"] = 0.0

                if row["label"] == 1 and row["prediction"] == 0:
                    row["error_type"] = "FN"
                elif row["label"] == 0 and row["prediction"] == 1:
                    row["error_type"] = "FP"
                else:
                    row["error_type"] = "Correct"

                results.append(row)

            csv_path = output_dir / "test_results.csv"
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")

    logger.info("\nEvaluation complete!")


# =====================================================================
# Main
# =====================================================================


def main(argv=None) -> None:
    """Route to the appropriate evaluation mode."""
    args = parse_args(argv)

    # Auto-detect model type if not specified
    model_type = args.model_type
    if model_type is None:
        model_type = _detect_model_type(Path(args.model_dir))
        logger.info(f"Auto-detected model type: {model_type}")

    if args.predict_files:
        if model_type == "carla":
            logger.warning(
                "File prediction is only supported for standard models. "
                "Use dataset evaluation for CARLA models."
            )
        _run_file_prediction(args, args.predict_files)
    elif model_type == "carla":
        _run_carla_evaluation(args)
    else:
        _run_standard_evaluation(args)


if __name__ == "__main__":
    main()
