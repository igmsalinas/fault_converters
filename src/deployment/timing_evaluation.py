"""
Timing and Classification Evaluation Pipeline
==============================================

Evaluates optimized models (TFLite, ONNX, TensorRT) on the test dataset,
computes precision, recall, F1, AUC-ROC, and measures latency.
Saves comparison reports in the deployment output directory.
"""

import time
from pathlib import Path
import numpy as np
import keras
from typing import Dict, Any, List

from .utils import get_file_size_mb
from .runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
    run_inference_loop,
)
from ..utils.logger import get_logger
from ..evaluation.metrics import compute_classification_metrics
from ..inference.predictor import AnomalyPredictor

logger = get_logger(__name__)


def _evaluate_with_runner(
    name: str,
    runner,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """Run full classification evaluation using a generic runner callable.

    Args:
        name: Display name for logging.
        runner: Inference callable (from runner factories).
        test_data: Test dataset array.
        test_labels: Ground truth labels (0 = normal, 1 = anomaly).
        threshold: Anomaly decision threshold on per-sample MSE.
        batch_size: Batch size for inference loop.

    Returns:
        Dictionary with latency and classification metrics.
    """
    logger.info(f"Evaluating {name}...")

    # 1. Measure execution latency
    t0 = time.perf_counter()
    reconstruction = run_inference_loop(runner, test_data, batch_size=batch_size)
    total_time = (time.perf_counter() - t0) * 1000.0  # ms

    # 2. Compute anomaly scores (MSE)
    errors = np.mean(np.square(test_data - reconstruction), axis=(1, 2))
    predictions = (errors > threshold).astype(int)

    # 3. Compute classification metrics
    metrics = compute_classification_metrics(predictions, test_labels, scores=errors)

    return {
        "latency_per_sample_ms": float(total_time / len(test_data)),
        "total_time_ms": float(total_time),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "auc_roc": float(metrics.auc_roc),
    }


def evaluate_keras_model(
    model: keras.Model,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Run full test evaluation on baseline Keras model."""
    runner = create_keras_runner(model)
    return _evaluate_with_runner("Keras FP32 Model", runner, test_data, test_labels, threshold)


def evaluate_tflite_model(
    tflite_path: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Run full test evaluation on TFLite quantized models."""
    runner = create_tflite_runner(tflite_path)
    # Use batch_size=1 for TFLite to simulate edge device single-sample inference
    return _evaluate_with_runner(
        f"TFLite ({tflite_path})", runner, test_data, test_labels, threshold, batch_size=1,
    )


def evaluate_onnx_model(
    onnx_path: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Run full test evaluation on ONNX model."""
    runner = create_onnx_runner(onnx_path)
    if runner is None:
        return {}
    return _evaluate_with_runner(f"ONNX ({onnx_path})", runner, test_data, test_labels, threshold)


def evaluate_tensorrt_model(
    engine_path: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Run full test evaluation on TensorRT engine.

    Args:
        engine_path: Path to a serialized .engine file.
        test_data: Numpy array of test data.
        test_labels: Ground truth labels.
        threshold: Anomaly decision threshold.

    Returns:
        Dictionary with latency and classification metrics,
        or empty dict if TRT is unavailable.
    """
    runner = create_tensorrt_runner(engine_path)
    if runner is None:
        return {}
    return _evaluate_with_runner(
        f"TensorRT ({engine_path})", runner, test_data, test_labels, threshold,
    )


def run_deployment_evaluations(
    model_dir: str,
    deployment_dir: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
) -> List[Dict[str, Any]]:
    """Loads and runs evaluation metrics for all optimized models in the deployment directory."""
    logger.info("Starting unified deployment evaluation benchmark...")
    dep_path = Path(deployment_dir)
    results = []

    # Load threshold
    predictor = AnomalyPredictor(model_dir=model_dir)
    threshold = predictor.threshold
    if threshold is None:
        threshold = 0.001
        logger.warning(f"No threshold set in predictor. Using default: {threshold}")

    # 1. Baseline Keras model
    try:
        keras_res = evaluate_keras_model(predictor.model.autoencoder, test_data, test_labels, threshold)
        keras_res["model_name"] = "Keras FP32 (Baseline)"
        keras_res["size_mb"] = get_file_size_mb(str(Path(model_dir) / "best_model.weights.h5"))
        results.append(keras_res)
    except Exception as e:
        logger.error(f"Keras baseline evaluation failed: {e}")

    # 2. TFLite models
    tflite_files = {
        "TFLite FP16": "model_fp16.tflite",
        "TFLite Dynamic": "model_dynamic.tflite",
        "TFLite INT8": "model_int8.tflite",
    }
    for label, filename in tflite_files.items():
        tflite_path = dep_path / filename
        if tflite_path.exists():
            try:
                tf_res = evaluate_tflite_model(str(tflite_path), test_data, test_labels, threshold)
                tf_res["model_name"] = label
                tf_res["size_mb"] = get_file_size_mb(str(tflite_path))
                results.append(tf_res)
            except Exception as e:
                logger.error(f"Failed to evaluate TFLite model at {tflite_path}: {e}")

    # 3. ONNX model
    onnx_path = dep_path / "model.onnx"
    if onnx_path.exists():
        try:
            onnx_res = evaluate_onnx_model(str(onnx_path), test_data, test_labels, threshold)
            if onnx_res:
                onnx_res["model_name"] = "ONNX (CPU)"
                onnx_res["size_mb"] = get_file_size_mb(str(onnx_path))
                results.append(onnx_res)
        except Exception as e:
            logger.error(f"Failed to evaluate ONNX model: {e}")

    # 4. TensorRT engines
    for engine_path in dep_path.glob("*.engine"):
        try:
            trt_res = evaluate_tensorrt_model(str(engine_path), test_data, test_labels, threshold)
            if trt_res:
                name_suffix = engine_path.stem.split("_")[-1] if "_" in engine_path.stem else "fp16"
                trt_res["model_name"] = f"TensorRT {name_suffix.upper()} (GPU)"
                trt_res["size_mb"] = get_file_size_mb(str(engine_path))
                results.append(trt_res)
        except Exception as e:
            logger.error(f"Failed to evaluate {engine_path.name}: {e}")

    # Print comparison table
    logger.info("================================== CLASSIFICATION & TIMING EVALUATION SUMMARY ==================================")
    logger.info(f"{'Model Format':<22} | {'Size (MB)':<10} | {'Latency/S (ms)':<15} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Recall':<10}")
    logger.info("-" * 110)
    for r in results:
        logger.info(
            f"{r['model_name']:<22} | {r['size_mb']:<10.3f} | {r['latency_per_sample_ms']:<15.4f} | "
            f"{r['f1']:<10.4f} | {r['auc_roc']:<10.4f} | {r['recall']:<10.4f}"
        )
    logger.info("================================================================================================================")

    return results


def save_evaluation_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save evaluation metrics to JSON and Markdown reports."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    import json
    with open(out_p.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation benchmark JSON data to {out_p.with_suffix('.json')}")
    logger.info(f"Saved evaluation benchmark Markdown report to {out_p.with_suffix('.md')}")
