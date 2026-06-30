"""
Unified Edge Compute Benchmarking Suite
========================================

Measures model file size, inference latency (batch sizes 1 & 32),
and reconstruction fidelity (MSE) across all deployment formats.
"""

import time
from pathlib import Path
import numpy as np
import keras
from typing import Dict, Any, Optional, Callable

from .utils import get_file_size_mb
from .runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _benchmark_runner(
    name: str,
    runner: Callable[[np.ndarray], np.ndarray],
    test_data: np.ndarray,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Run latency and reconstruction benchmarks using a generic runner callable.

    Args:
        name: Display name for logging.
        runner: Inference callable (from runner factories).
        test_data: Test dataset array.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.

    Returns:
        Dictionary with latency stats and MSE.
    """
    logger.info(f"Benchmarking {name}...")

    x_single = np.expand_dims(test_data[0], axis=0).astype(np.float32)
    x_batch = test_data[:min(32, len(test_data))].astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        _ = runner(x_single)

    # Latency: Batch size 1
    latencies_1 = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = runner(x_single)
        latencies_1.append((time.perf_counter() - t0) * 1000.0)

    # Latency: Batch size 32
    latencies_32 = []
    if len(test_data) >= 32:
        # Warmup batch
        _ = runner(x_batch)
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = runner(x_batch)
            latencies_32.append((time.perf_counter() - t0) * 1000.0)

    # Compute reconstruction on the first 100 test samples
    eval_subset = test_data[:min(100, len(test_data))].astype(np.float32)
    reconstruction = runner(eval_subset)
    mse = float(np.mean(np.square(eval_subset - reconstruction)))

    return {
        "latency_bs1_mean": float(np.mean(latencies_1)),
        "latency_bs1_std": float(np.std(latencies_1)),
        "latency_bs32_mean": float(np.mean(latencies_32)) if latencies_32 else 0.0,
        "latency_bs32_std": float(np.std(latencies_32)) if latencies_32 else 0.0,
        "mse": mse,
        "reconstructions": reconstruction,
    }


def benchmark_keras_model(
    model: keras.Model,
    test_data: np.ndarray,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark baseline Keras model latency and reconstruction."""
    runner = create_keras_runner(model)
    return _benchmark_runner("Keras FP32 Model", runner, test_data, num_warmup, num_runs)


def benchmark_tflite_model(
    tflite_path: str,
    test_data: np.ndarray,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark TFLite model latency and reconstruction."""
    runner = create_tflite_runner(tflite_path)
    return _benchmark_runner(f"TFLite ({tflite_path})", runner, test_data, num_warmup, num_runs)


def benchmark_onnx_model(
    onnx_path: str,
    test_data: np.ndarray,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark ONNX model latency and reconstruction using ONNX Runtime."""
    runner = create_onnx_runner(onnx_path)
    if runner is None:
        return {}
    return _benchmark_runner(f"ONNX ({onnx_path})", runner, test_data, num_warmup, num_runs)


def benchmark_tensorrt_model(
    engine_path: str,
    test_data: np.ndarray,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """Benchmark TensorRT engine latency and reconstruction.

    Args:
        engine_path: Path to a serialized .engine file.
        test_data: Numpy array of test data.
        num_warmup: Number of warmup iterations.
        num_runs: Number of timed iterations.

    Returns:
        Dictionary with latency stats and MSE, or empty dict if TRT is unavailable.
    """
    runner = create_tensorrt_runner(engine_path)
    if runner is None:
        return {}
    return _benchmark_runner(f"TensorRT ({engine_path})", runner, test_data, num_warmup, num_runs)


def run_deployment_benchmarks(
    model_dir: str,
    test_data: np.ndarray,
    converted_models: Dict[str, str],
) -> Dict[str, Any]:
    """
    Run benchmarking across the baseline model and all converted pipelines.

    Args:
        model_dir: Directory containing baseline model (config and weights).
        test_data: Numpy array of test data.
        converted_models: Dictionary mapping format names ("tflite_fp16", etc.) to paths on disk.

    Returns:
        Dictionary of results for all benchmarks.
    """
    logger.info("Starting unified deployment benchmarks...")

    # 1. Benchmark baseline Keras model
    from ..inference.predictor import AnomalyPredictor

    predictor = AnomalyPredictor(model_dir=model_dir)
    baseline_results = benchmark_keras_model(predictor.model.autoencoder, test_data)

    results = {
        "baseline": {
            "name": "Keras FP32 (Baseline)",
            "size_mb": get_file_size_mb(str(Path(model_dir) / "best_model.weights.h5")),
            "latency_bs1_mean": baseline_results["latency_bs1_mean"],
            "latency_bs1_std": baseline_results["latency_bs1_std"],
            "latency_bs32_mean": baseline_results["latency_bs32_mean"],
            "latency_bs32_std": baseline_results["latency_bs32_std"],
            "mse": baseline_results["mse"],
            "mse_diff_from_baseline": 0.0,
        }
    }

    # 2. Benchmark TFLite models
    for key, path in converted_models.items():
        if "tflite" in key and Path(path).exists():
            try:
                tf_res = benchmark_tflite_model(path, test_data)
                mse_diff = abs(tf_res["mse"] - baseline_results["mse"])
                results[key] = {
                    "name": f"TFLite {key.replace('tflite_', '').upper()}",
                    "size_mb": get_file_size_mb(path),
                    "latency_bs1_mean": tf_res["latency_bs1_mean"],
                    "latency_bs1_std": tf_res["latency_bs1_std"],
                    "latency_bs32_mean": tf_res["latency_bs32_mean"],
                    "latency_bs32_std": tf_res["latency_bs32_std"],
                    "mse": tf_res["mse"],
                    "mse_diff_from_baseline": float(mse_diff),
                }
            except Exception as e:
                logger.error(f"Failed to benchmark TFLite model at {path}: {e}")

    # 3. Benchmark ONNX models
    for key, path in converted_models.items():
        if "onnx" in key and Path(path).exists():
            try:
                onnx_res = benchmark_onnx_model(path, test_data)
                if onnx_res:
                    mse_diff = abs(onnx_res["mse"] - baseline_results["mse"])
                    results[key] = {
                        "name": "ONNX (CPU)",
                        "size_mb": get_file_size_mb(path),
                        "latency_bs1_mean": onnx_res["latency_bs1_mean"],
                        "latency_bs1_std": onnx_res["latency_bs1_std"],
                        "latency_bs32_mean": onnx_res["latency_bs32_mean"],
                        "latency_bs32_std": onnx_res["latency_bs32_std"],
                        "mse": onnx_res["mse"],
                        "mse_diff_from_baseline": float(mse_diff),
                    }
            except Exception as e:
                logger.error(f"Failed to benchmark ONNX model at {path}: {e}")

    # 4. Benchmark TensorRT engines
    for key, path in converted_models.items():
        if "tensorrt" in key and Path(path).exists():
            try:
                trt_res = benchmark_tensorrt_model(path, test_data)
                if trt_res:
                    mse_diff = abs(trt_res["mse"] - baseline_results["mse"])
                    # Extract suffix e.g., fp32, int8
                    name_suffix = key.split("_")[-1] if "_" in key else "fp16"
                    results[key] = {
                        "name": f"TensorRT {name_suffix.upper()} (GPU)",
                        "size_mb": get_file_size_mb(path),
                        "latency_bs1_mean": trt_res["latency_bs1_mean"],
                        "latency_bs1_std": trt_res["latency_bs1_std"],
                        "latency_bs32_mean": trt_res["latency_bs32_mean"],
                        "latency_bs32_std": trt_res["latency_bs32_std"],
                        "mse": trt_res["mse"],
                        "mse_diff_from_baseline": float(mse_diff),
                    }
            except Exception as e:
                logger.error(f"Failed to benchmark TensorRT engine at {path}: {e}")

    # Log and print summary table
    logger.info("=================================== BENCHMARK RESULTS ===================================")
    logger.info(f"{'Format':<25} | {'Size (MB)':<10} | {'BS1 Latency (ms)':<18} | {'BS32 Latency (ms)':<18} | {'Fidelity (MSE)':<15}")
    logger.info("-" * 95)
    for k, v in results.items():
        logger.info(
            f"{v['name']:<25} | {v['size_mb']:<10.3f} | {v['latency_bs1_mean']:<7.3f} ± {v['latency_bs1_std']:<6.3f} | "
            f"{v['latency_bs32_mean']:<7.3f} ± {v['latency_bs32_std']:<6.3f} | {v['mse']:<15.6e}"
        )
    logger.info("=========================================================================================")

    return results


def save_benchmark_report(results: Dict[str, Any], output_path: str) -> None:
    """Save benchmark results to markdown and JSON reports."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    import json
    with open(out_p.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark JSON data to {out_p.with_suffix('.json')}")
