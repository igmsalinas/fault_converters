"""
Deploy Trained Model to Edge Platforms and Benchmark Performance
================================================================

CLI entry-point to compile, quantize, and benchmark trained models.

Usage::

    python -m src.deploy --model-dir experiments/conv1d_ae --data-dir data/buck/buck_data --run-all
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

from .deployment import (
    load_deployment_datasets,
    get_calibration_dataset,
    quantize_keras_native,
    convert_to_tflite,
    convert_keras_to_onnx,
    compile_onnx_to_tensorrt,
    quantize_for_vitis_ai,
    run_deployment_benchmarks,
    save_benchmark_report,
    run_full_performance_suite,
    save_performance_report,
    run_deployment_evaluations,
    save_evaluation_report,
    run_batch_size_study,
    save_batch_size_report,
)
from .inference.predictor import AnomalyPredictor
from .utils.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Deploy and benchmark power converter anomaly detection models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the trained model directory containing configuration and weights.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/buck/buck_data",
        help="Directory containing simulation files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for caching preprocessed data.",
    )
    parser.add_argument(
        "--normal-threshold",
        type=float,
        default=5.0,
        help="Tolerance threshold for data classification.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files to load for calibration/testing (None for all).",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=100,
        help="Number of calibration samples for quantization.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for deployment files (default: model_dir/deployment).",
    )
    parser.add_argument(
        "--vitis-target",
        type=str,
        default="zcu102",
        choices=["zcu102", "zcu104", "kv260", "ultra96", "pynq_z2"],
        help="Target hardware profile for Xilinx Vitis AI quantization.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all available optimization and quantization conversions.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated list of batch sizes for scaling study.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        sys.exit(1)

    # 1. Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "deployment"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Deployment output directory: {output_dir}")

    # 2. Load the baseline predictor and Keras model
    try:
        predictor = AnomalyPredictor(model_dir=str(model_dir))
        keras_model = predictor.model.autoencoder
        logger.info(f"Loaded baseline model '{keras_model.name}' successfully.")
    except Exception as e:
        logger.error(f"Failed to load baseline predictor: {e}")
        sys.exit(1)

    # 3. Load datasets
    try:
        _, splits = load_deployment_datasets(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            normal_threshold=args.normal_threshold,
            max_files=args.max_files,
            preprocessor=predictor.preprocessor,
        )
        test_data = splits["test"]

        # Extract calibration subset
        calibration_data = get_calibration_dataset(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            normal_threshold=args.normal_threshold,
            num_samples=args.num_calib,
            max_files=args.max_files,
            preprocessor=predictor.preprocessor,
        )
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        sys.exit(1)

    converted_models = {}

    # ---- 1. Keras backend and TFLite conversions ----
    logger.info("--- Starting Pipeline 1: Keras & TFLite Quantization ---")

    # 1a. Keras-native quantization
    try:
        quantize_keras_native(
            model=keras_model,
            mode="int8",
            output_dir=str(output_dir),
        )
    except Exception as e:
        logger.warning(f"Native Keras quantization encountered warning/error: {e}")

    # 1b. TFLite dynamic range
    path_dynamic = str(output_dir / "model_dynamic.tflite")
    try:
        convert_to_tflite(keras_model, "dynamic", output_path=path_dynamic)
        converted_models["tflite_dynamic"] = path_dynamic
    except Exception as e:
        logger.error(f"TFLite dynamic conversion failed: {e}")

    # 1c. TFLite FP16
    path_fp16 = str(output_dir / "model_fp16.tflite")
    try:
        convert_to_tflite(keras_model, "float16", output_path=path_fp16)
        converted_models["tflite_fp16"] = path_fp16
    except Exception as e:
        logger.error(f"TFLite float16 conversion failed: {e}")

    # 1d. TFLite Full Integer (INT8)
    path_int8 = str(output_dir / "model_int8.tflite")
    try:
        convert_to_tflite(
            keras_model,
            "int8",
            calibration_data=calibration_data,
            output_path=path_int8
        )
        converted_models["tflite_int8"] = path_int8
    except Exception as e:
        logger.error(f"TFLite INT8 quantization failed: {e}")

    # ---- 2. Keras to ONNX and ONNX to TensorRT ----
    logger.info("--- Starting Pipeline 2: Keras to ONNX & TensorRT Engine ---")
    path_onnx = str(output_dir / "model.onnx")
    onnx_success = convert_keras_to_onnx(keras_model, path_onnx)
    if onnx_success:
        converted_models["onnx"] = path_onnx

        # Compile to TensorRT Engine (FP32)
        path_engine_fp32 = str(output_dir / "model_fp32.engine")
        trt_success_fp32 = compile_onnx_to_tensorrt(
            onnx_path=path_onnx,
            engine_path=path_engine_fp32,
            precision_mode="FP32",
        )
        if trt_success_fp32:
            converted_models["tensorrt_fp32"] = path_engine_fp32

        # Compile to TensorRT Engine (FP16)
        path_engine_fp16 = str(output_dir / "model_fp16.engine")
        trt_success_fp16 = compile_onnx_to_tensorrt(
            onnx_path=path_onnx,
            engine_path=path_engine_fp16,
            precision_mode="FP16",
        )
        if trt_success_fp16:
            converted_models["tensorrt_fp16"] = path_engine_fp16

        # Compile to TensorRT Engine (INT8)
        path_engine_int8 = str(output_dir / "model_int8.engine")
        trt_success_int8 = compile_onnx_to_tensorrt(
            onnx_path=path_onnx,
            engine_path=path_engine_int8,
            precision_mode="INT8",
            calibration_data=calibration_data,
        )
        if trt_success_int8:
            converted_models["tensorrt_int8"] = path_engine_int8

    # ---- 3. Vitis AI Quantization ----
    logger.info("--- Starting Pipeline 3: Xilinx Vitis AI Quantization ---")
    try:
        quantize_for_vitis_ai(
            model=keras_model,
            calibration_data=calibration_data,
            output_dir=str(output_dir / "vitis_ai"),
            target_hardware=args.vitis_target,
        )
    except Exception as e:
        logger.warning(f"Vitis AI Quantization skipped or failed: {e}")

    # ---- Benchmarking ----
    logger.info("--- Starting Benchmarking Suite ---")
    try:
        benchmark_results = run_deployment_benchmarks(
            model_dir=str(model_dir),
            test_data=test_data,
            converted_models=converted_models,
        )

        report_path = str(output_dir / "deployment_benchmark_report")
        save_benchmark_report(benchmark_results, report_path)

        # Run timing and memory performance benchmark
        logger.info("--- Running Timing and Memory Performance Benchmark ---")
        perf_results = run_full_performance_suite(
            model_dir=str(model_dir),
            deployment_dir=str(output_dir),
            test_data=test_data,
        )
        perf_report_path = str(output_dir / "performance_benchmark_report")
        save_performance_report(perf_results, perf_report_path)

        # Run timing and classification evaluation
        logger.info("--- Running Timing and Classification Evaluation Benchmark ---")
        eval_results = run_deployment_evaluations(
            model_dir=str(model_dir),
            deployment_dir=str(output_dir),
            test_data=test_data,
            test_labels=splits["test_labels"],
        )
        eval_report_path = str(output_dir / "evaluation_benchmark_report")
        save_evaluation_report(eval_results, eval_report_path)

        # Run batch size scaling study
        logger.info("--- Running Batch Size Scaling Study ---")
        try:
            bs_list = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip().isdigit()]
        except Exception:
            bs_list = [1, 2, 4, 8, 16, 32, 64, 128]

        study_results = run_batch_size_study(
            model_dir=str(model_dir),
            deployment_dir=str(output_dir),
            test_data=test_data,
            batch_sizes=bs_list,
        )
        study_report_path = str(output_dir / "batch_size_study_report")
        save_batch_size_report(study_results, study_report_path)

        # Generate Unified Markdown Report
        logger.info("--- Generating Unified Markdown Report ---")
        generate_unified_markdown_report(
            benchmark_results, perf_results, eval_results, study_results,
            str(output_dir / "unified_deployment_report.md")
        )

    except Exception as e:
        logger.error(f"Benchmarking suite run failed: {e}")

    logger.info(f"Deployment processing complete! All files generated in: {output_dir}")

def generate_unified_markdown_report(
    benchmark_results: Dict[str, Any],
    perf_results: List[Dict[str, Any]],
    eval_results: List[Dict[str, Any]],
    study_results: Dict[str, Any],
    output_path: str
):
    """Aggregate all JSON outputs into a single comprehensive Markdown report."""
    md = ["# Edge Compute Unified Deployment Report", "", "This report aggregates optimizations across size, latency, memory, and classification fidelity.", ""]

    # Section 1: Fidelity & Basic Latency
    md.extend(["## 1. Inference Fidelity & Baseline Latency", "", "| Format | Size (MB) | Batch 1 Latency (ms) | Batch 32 Latency (ms) | Reconstruction MSE | MSE Shift from Baseline |", "| :--- | :---: | :---: | :---: | :---: | :---: |"])
    for k, v in benchmark_results.items():
        if k == "baseline": continue
        bs32_str = f"{v.get('latency_bs32_mean', 0):.3f} ± {v.get('latency_bs32_std', 0):.3f}" if v.get('latency_bs32_mean', 0) > 0 else "N/A"
        md.append(f"| **{v['name']}** | {v['size_mb']:.3f} MB | {v['latency_bs1_mean']:.3f} ± {v['latency_bs1_std']:.3f} ms | {bs32_str} ms | {v['mse']:.6e} | {v['mse_diff_from_baseline']:.6e} |")
    md.append("")

    # Section 2: Hardware Performance (Memory & VRAM)
    md.extend(["## 2. Hardware Resource Profiling", "", "| Model Format | Mean Latency | Min / Max Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |", "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"])
    for r in perf_results:
        latency_str = f"{r['latency_mean_ms']:.3f} ± {r['latency_std_ms']:.3f} ms"
        min_max_str = f"{r['latency_min_ms']:.2f} / {r['latency_max_ms']:.2f} ms"
        md.append(f"| **{r['model_name']}** | {latency_str} | {min_max_str} | {r['net_ram_mb']:.3f} MB | {r['net_vram_mb']:.3f} MB | {r['peak_ram_mb']:.3f} MB | {r['peak_vram_mb']:.3f} MB |")
    md.append("")

    # Section 3: Classification Metrics
    md.extend(["## 3. Classification Degradation Evaluation", "", "| Model Format | Size (MB) | Latency Per Sample (ms) | Accuracy | Precision | Recall | F1-Score | AUC-ROC |", "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"])
    for r in eval_results:
        md.append(f"| **{r['model_name']}** | {r['size_mb']:.3f} MB | {r['latency_per_sample_ms']:.4f} ms | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['auc_roc']:.4f} |")
    md.append("")

    # Section 4: Batch Size Scaling Study
    md.extend(["## 4. Batch Size Scaling Dynamics", ""])
    for format_name, bs_data in study_results.items():
        md.append(f"### {format_name}")
        md.append("| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |")
        md.append("| :---: | :---: | :---: | :---: | :---: |")
        sorted_bs = sorted([int(k) for k in bs_data.keys()])
        for bs in sorted_bs:
            data = bs_data[bs] if bs in bs_data else bs_data[str(bs)]
            md.append(f"| {bs} | {data['latency_batch_ms']:.2f} | {data['latency_sample_ms']:.3f} | {data['net_ram_mb']:.2f} | {data['net_vram_mb']:.2f} |")
        md.append("")

    out_p = Path(output_path)
    with open(out_p, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Saved unified benchmark report to {out_p}")


if __name__ == "__main__":
    main()
