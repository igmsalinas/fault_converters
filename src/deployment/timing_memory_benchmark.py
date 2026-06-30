"""
Timing and Memory Benchmarking Script
======================================

Measures CPU/GPU latency and memory footprint (RAM and VRAM)
across all optimized and quantized model formats.
"""

import os
import gc
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List

# Try importing psutil for CPU RAM monitoring
try:
    import psutil
except ImportError:
    psutil = None

from .runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _get_cpu_ram_usage() -> float:
    """Get resident memory (RSS) of current process in MB."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def _get_gpu_vram_usage() -> float:
    """Get active VRAM usage of GPU 0 in MB using nvidia-smi CLI fallback."""
    import subprocess

    # Try TensorFlow GPU memory info first
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            return mem_info['current'] / (1024 * 1024)
    except Exception:
        pass

    # Fallback to nvidia-smi command line query
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stderr=subprocess.DEVNULL
        )
        return float(result.decode().strip())
    except Exception:
        return 0.0


def benchmark_timing_memory(
    model_name: str,
    runner_fn: Any,
    input_data: np.ndarray,
    num_warmup: int = 15,
    num_runs: int = 150,
) -> Dict[str, Any]:
    """
    Run execution timing loops and memory checks.

    Args:
        model_name: Printable name of model.
        runner_fn: Callable taking input data and returning output.
        input_data: Sample input matching model's expected shape.
        num_warmup: Warmup loops.
        num_runs: Timing loops.

    Returns:
        Dictionary of results.
    """
    logger.info(f"Measuring Timing & Memory for: {model_name}...")

    # 1. Warmup
    for _ in range(num_warmup):
        _ = runner_fn(input_data)

    # Force Garbage Collection before checking base memory
    gc.collect()

    base_ram = _get_cpu_ram_usage()
    base_vram = _get_gpu_vram_usage()

    # 2. Timing loops
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = runner_fn(input_data)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # in ms

    # 3. Peak memory check during a continuous inference stress run
    peak_ram = base_ram
    peak_vram = base_vram

    # Run a quick loop to sample memory levels while executing
    for _ in range(30):
        _ = runner_fn(input_data)
        curr_ram = _get_cpu_ram_usage()
        curr_vram = _get_gpu_vram_usage()
        if curr_ram > peak_ram:
            peak_ram = curr_ram
        if curr_vram > peak_vram:
            peak_vram = curr_vram

    # Calculate net consumption
    net_ram = max(0.0, peak_ram - base_ram)
    net_vram = max(0.0, peak_vram - base_vram)

    return {
        "model_name": model_name,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "net_ram_mb": net_ram,
        "net_vram_mb": net_vram,
        "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram,
    }


def run_full_performance_suite(
    model_dir: str,
    deployment_dir: str,
    test_data: np.ndarray,
) -> List[Dict[str, Any]]:
    """Loads all generated models in the deployment directory and benchmarks them."""
    dep_path = Path(deployment_dir)
    results = []

    # Ensure shape fits sequence and feature sizes
    x_single = np.expand_dims(test_data[0], axis=0).astype(np.float32)

    # 1. Baseline Keras model
    try:
        from ..inference.predictor import AnomalyPredictor
        predictor = AnomalyPredictor(model_dir=model_dir)
        keras_runner = create_keras_runner(predictor.model.autoencoder)
        res = benchmark_timing_memory("Keras FP32 (Baseline)", keras_runner, x_single)
        results.append(res)
    except Exception as e:
        logger.error(f"Failed to benchmark Keras Baseline: {e}")

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
                tflite_runner = create_tflite_runner(str(tflite_path))
                res = benchmark_timing_memory(label, tflite_runner, x_single)
                results.append(res)
            except Exception as e:
                logger.error(f"Failed to benchmark {label}: {e}")

    # 3. ONNX Model
    onnx_path = dep_path / "model.onnx"
    if onnx_path.exists():
        try:
            onnx_runner = create_onnx_runner(str(onnx_path))
            if onnx_runner is not None:
                res = benchmark_timing_memory("ONNX (CPU)", onnx_runner, x_single)
                results.append(res)
        except Exception as e:
            logger.error(f"Failed to benchmark ONNX: {e}")

    # 4. TensorRT Engines
    for engine_path in dep_path.glob("*.engine"):
        try:
            trt_runner = create_tensorrt_runner(str(engine_path))
            if trt_runner is not None:
                # E.g. model_fp16.engine -> TensorRT fp16 (GPU)
                name_suffix = engine_path.stem.split("_")[-1] if "_" in engine_path.stem else "fp16"
                res = benchmark_timing_memory(f"TensorRT {name_suffix.upper()} (GPU)", trt_runner, x_single)
                results.append(res)
        except Exception as e:
            logger.error(f"Failed to benchmark {engine_path.name}: {e}")

    # Print results summary table
    logger.info("================================= PERFORMANCE SUITE SUMMARY =================================")
    logger.info(f"{'Model Format':<22} | {'Mean Latency':<15} | {'Min / Max (ms)':<18} | {'Net RAM (MB)':<12} | {'Net VRAM (MB)':<12}")
    logger.info("-" * 92)
    for r in results:
        logger.info(
            f"{r['model_name']:<22} | {r['latency_mean_ms']:<6.3f}±{r['latency_std_ms']:<6.3f} ms | "
            f"{r['latency_min_ms']:<5.2f}/{r['latency_max_ms']:<5.2f} ms | "
            f"{r['net_ram_mb']:<12.3f} | {r['net_vram_mb']:<12.3f}"
        )
    logger.info("=============================================================================================")
    return results


def save_performance_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save performance timing and memory results to JSON and Markdown files."""
    import json
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    with open(out_p.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved performance benchmark JSON data to {out_p.with_suffix('.json')}")


def run_batch_size_study(
    model_dir: str,
    deployment_dir: str,
    test_data: np.ndarray,
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64, 128],
) -> Dict[str, Any]:
    """
    Run execution timing and memory checks across a range of batch sizes.
    """
    logger.info(f"Initiating Batch Size Scaling Study for batch sizes: {batch_sizes}...")
    dep_path = Path(deployment_dir)
    study_results = {}

    # Load baseline model
    from ..inference.predictor import AnomalyPredictor
    predictor = AnomalyPredictor(model_dir=model_dir)

    def _prepare_batch(bs: int) -> np.ndarray:
        """Prepare a batch of the given size, tiling if necessary."""
        x_batch = test_data[:bs].astype(np.float32)
        if len(x_batch) < bs:
            repeats = (bs // len(test_data)) + 1
            x_batch = np.tile(test_data, (repeats, 1, 1))[:bs].astype(np.float32)
        return x_batch

    # 1. Keras FP32
    keras_runner = create_keras_runner(predictor.model.autoencoder)
    keras_data = {}
    for bs in batch_sizes:
        try:
            x_batch = _prepare_batch(bs)
            res = benchmark_timing_memory(f"Keras FP32 (BS={bs})", keras_runner, x_batch, num_warmup=5, num_runs=30)
            keras_data[bs] = {
                "latency_batch_ms": res["latency_mean_ms"],
                "latency_sample_ms": res["latency_mean_ms"] / bs,
                "net_ram_mb": res["net_ram_mb"],
                "net_vram_mb": res["net_vram_mb"],
            }
        except Exception as e:
            logger.error(f"Keras scaling benchmark failed for batch size {bs}: {e}")

    if keras_data:
        study_results["Keras FP32 (Baseline)"] = keras_data

    # 2. TFLite Models
    tflite_files = {
        "TFLite FP16": "model_fp16.tflite",
        "TFLite Dynamic": "model_dynamic.tflite",
        "TFLite INT8": "model_int8.tflite",
    }

    for label, filename in tflite_files.items():
        tflite_path = dep_path / filename
        if tflite_path.exists():
            tf_data = {}
            try:
                tflite_runner = create_tflite_runner(str(tflite_path))
                for bs in batch_sizes:
                    try:
                        x_batch = _prepare_batch(bs)
                        res = benchmark_timing_memory(f"{label} (BS={bs})", tflite_runner, x_batch, num_warmup=5, num_runs=30)
                        tf_data[bs] = {
                            "latency_batch_ms": res["latency_mean_ms"],
                            "latency_sample_ms": res["latency_mean_ms"] / bs,
                            "net_ram_mb": res["net_ram_mb"],
                            "net_vram_mb": res["net_vram_mb"],
                        }
                    except Exception as e:
                        logger.error(f"{label} scaling failed for batch size {bs}: {e}")
            except Exception as e:
                logger.error(f"Failed to setup scaling runner for {label}: {e}")

            if tf_data:
                study_results[label] = tf_data

    # 3. ONNX Model
    onnx_path = dep_path / "model.onnx"
    if onnx_path.exists():
        onnx_data = {}
        try:
            onnx_runner = create_onnx_runner(str(onnx_path))
            if onnx_runner is not None:
                for bs in batch_sizes:
                    try:
                        x_batch = _prepare_batch(bs)
                        res = benchmark_timing_memory(f"ONNX (BS={bs})", onnx_runner, x_batch, num_warmup=5, num_runs=30)
                        onnx_data[bs] = {
                            "latency_batch_ms": res["latency_mean_ms"],
                            "latency_sample_ms": res["latency_mean_ms"] / bs,
                            "net_ram_mb": res["net_ram_mb"],
                            "net_vram_mb": res["net_vram_mb"],
                        }
                    except Exception as e:
                        logger.error(f"ONNX scaling failed for batch size {bs}: {e}")
        except Exception as e:
            logger.error(f"Failed to setup scaling ONNX session: {e}")

        if onnx_data:
            study_results["ONNX (CPU)"] = onnx_data

    # 4. TensorRT Engine
    engine_path = dep_path / "model.engine"
    if engine_path.exists():
        trt_data = {}
        try:
            trt_runner = create_tensorrt_runner(str(engine_path))
            if trt_runner is not None:
                for bs in batch_sizes:
                    try:
                        x_batch = _prepare_batch(bs)
                        res = benchmark_timing_memory(f"TensorRT (BS={bs})", trt_runner, x_batch, num_warmup=5, num_runs=30)
                        trt_data[bs] = {
                            "latency_batch_ms": res["latency_mean_ms"],
                            "latency_sample_ms": res["latency_mean_ms"] / bs,
                            "net_ram_mb": res["net_ram_mb"],
                            "net_vram_mb": res["net_vram_mb"],
                        }
                    except Exception as e:
                        logger.error(f"TensorRT scaling failed for batch size {bs}: {e}")
        except Exception as e:
            logger.error(f"Failed to setup scaling TensorRT runner: {e}")

        if trt_data:
            study_results["TensorRT (GPU)"] = trt_data

    # Print results summary table
    logger.info("=================================== BATCH SIZE SCALING STUDY ===================================")
    for model_name, data in study_results.items():
        logger.info(f"Model Format: {model_name}")
        logger.info(f"  {'Batch Size':<12} | {'Batch Latency (ms)':<20} | {'Sample Latency (ms)':<20} | {'Net RAM (MB)':<15}")
        logger.info("-" * 80)
        for bs, metrics in data.items():
            logger.info(
                f"  {bs:<12} | {metrics['latency_batch_ms']:<20.3f} | {metrics['latency_sample_ms']:<20.3f} | {metrics['net_ram_mb']:<15.3f}"
            )
        logger.info("=" * 80)

    return study_results


def save_batch_size_report(study_results: Dict[str, Any], output_path: str) -> None:
    """Save batch size scaling study results to JSON and Markdown reports."""
    import json
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    with open(out_p.with_suffix(".json"), "w") as f:
        json.dump(study_results, f, indent=2)
    logger.info(f"Saved batch size study JSON data to {out_p.with_suffix('.json')}")
