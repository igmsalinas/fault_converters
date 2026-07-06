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
from typing import Dict, Any, List, Optional

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
from .utils import TIMING_BATCH_SIZE, TIMING_WARMUP, TIMING_RUNS
from ..utils.logger import get_logger

logger = get_logger(__name__)


def _get_cpu_ram_usage() -> float:
    """Get resident memory (RSS) of the current process in MB.

    RSS is the physical RAM actually held by the process, so the delta
    around loading and running a model is the specific CPU memory it uses.
    """
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


# Lazily-initialised NVML handle so we can query real device memory without
# re-initialising the library on every sample.
_NVML_STATE: Any = None  # None = untried, False = unavailable, tuple = (pynvml, handle)


def _get_nvml() -> Any:
    """Return (pynvml, handle) for GPU 0, or None if NVML is unavailable."""
    global _NVML_STATE
    if _NVML_STATE is not None:
        return _NVML_STATE or None
    try:
        import pynvml  # provided by the nvidia-ml-py / pynvml package
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _NVML_STATE = (pynvml, handle)
    except Exception:
        _NVML_STATE = False
    return _NVML_STATE or None


def _get_gpu_vram_usage() -> float:
    """Get currently used VRAM on GPU 0 in MB.

    Prefers NVML (CUDA management library) which reports the real device
    memory in use, so measuring it before and after execution yields the
    true GPU footprint. Falls back to TensorFlow's allocator info, then to
    the nvidia-smi CLI.
    """
    # 1. NVML — real device memory usage (what the user asked for)
    nvml = _get_nvml()
    if nvml is not None:
        try:
            pynvml, handle = nvml
            return float(pynvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024 * 1024)
        except Exception:
            pass

    # 2. TensorFlow allocator info
    try:
        if tf.config.list_physical_devices('GPU'):
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            return mem_info['current'] / (1024 * 1024)
    except Exception:
        pass

    # 3. Fallback to nvidia-smi command line query
    import subprocess
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stderr=subprocess.DEVNULL
        )
        return float(result.decode().strip().splitlines()[0])
    except Exception:
        return 0.0


def benchmark_timing_memory(
    model_name: str,
    runner_factory: Any,
    input_data: np.ndarray,
    num_warmup: int = TIMING_WARMUP,
    num_runs: int = TIMING_RUNS,
) -> Optional[Dict[str, Any]]:
    """
    Run execution timing loops and measure the real memory footprint.

    The runner is built *inside* this function via ``runner_factory`` so the
    memory baseline is captured before the model is loaded. The delta between
    the baseline and the peak observed during execution is therefore the
    specific memory that loading and running this model consumes:

    * CPU RAM is read from the process RSS (via psutil).
    * GPU VRAM is read from NVML before/after execution to obtain the real number.

    Args:
        model_name: Printable name of model.
        runner_factory: Zero-argument callable that builds and returns the
            inference runner (``Callable[[np.ndarray], np.ndarray]``). It may
            return ``None`` if the backend is unavailable, in which case this
            function returns ``None``.
        input_data: Sample input matching model's expected shape.
        num_warmup: Warmup loops.
        num_runs: Timing loops.

    Returns:
        Dictionary of results, or ``None`` if the runner could not be built.
    """
    logger.info(f"Measuring Timing & Memory for: {model_name}...")

    # 1. Baseline memory BEFORE the model is loaded, so the delta reflects the
    #    real footprint of loading + running this specific model.
    gc.collect()
    base_ram = _get_cpu_ram_usage()
    base_vram = _get_gpu_vram_usage()

    # 2. Build the runner (this is what loads the weights / allocates buffers).
    runner_fn = runner_factory()
    if runner_fn is None:
        logger.warning(f"Runner unavailable for {model_name}; skipping.")
        return None

    # 3. Warmup
    for _ in range(num_warmup):
        _ = runner_fn(input_data)

    # 4. Timing loops (kept free of memory probing for clean latency numbers)
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = runner_fn(input_data)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # in ms

    # 5. Peak memory sampled during a short sustained inference burst
    peak_ram = _get_cpu_ram_usage()
    peak_vram = _get_gpu_vram_usage()
    for _ in range(30):
        _ = runner_fn(input_data)
        peak_ram = max(peak_ram, _get_cpu_ram_usage())
        peak_vram = max(peak_vram, _get_gpu_vram_usage())

    # Net consumption attributable to this model (load + inference).
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

    # Uniform single-sample (n=1) latency so every backend is directly
    # comparable under the same protocol (matches benchmark_cpu.py).
    x_perf = test_data[:TIMING_BATCH_SIZE].astype(np.float32)

    # 1. Baseline Keras model. Loaded lazily inside the factory so the memory
    #    baseline is captured before the weights are allocated.
    def _keras_factory():
        from ..inference.predictor import AnomalyPredictor
        predictor = AnomalyPredictor(model_dir=model_dir)
        return create_keras_runner(predictor.model.autoencoder)

    try:
        res = benchmark_timing_memory("Keras FP32 (Baseline)", _keras_factory, x_perf)
        if res is not None:
            results.append(res)
    except Exception as e:
        logger.error(f"Failed to benchmark Keras Baseline: {e}")

    # 2. TFLite models
    tflite_files = {
        "TFLite FP16": "model_float16.tflite",
        "TFLite Dynamic": "model_dynamic.tflite",
        "TFLite INT8": "model_int8.tflite",
    }

    for label, filename in tflite_files.items():
        tflite_path = dep_path / filename
        if tflite_path.exists():
            try:
                res = benchmark_timing_memory(
                    label, lambda p=str(tflite_path): create_tflite_runner(p), x_perf
                )
                if res is not None:
                    results.append(res)
            except Exception as e:
                logger.error(f"Failed to benchmark {label}: {e}")

    # 3. ONNX Model
    onnx_path = dep_path / "model.onnx"
    if onnx_path.exists():
        try:
            res = benchmark_timing_memory(
                "ONNX (CPU)", lambda p=str(onnx_path): create_onnx_runner(p), x_perf
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            logger.error(f"Failed to benchmark ONNX: {e}")

    # 4. TensorRT Engines
    for engine_path in dep_path.glob("*.engine"):
        try:
            # E.g. model_fp16.engine -> TensorRT FP16 (GPU)
            name_suffix = engine_path.stem.split("_")[-1] if "_" in engine_path.stem else "fp16"
            res = benchmark_timing_memory(
                f"TensorRT {name_suffix.upper()} (GPU)",
                lambda p=str(engine_path): create_tensorrt_runner(p),
                x_perf,
            )
            if res is not None:
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
