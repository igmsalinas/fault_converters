"""
Deploy Trained Model to Edge Platforms and Benchmark Performance
================================================================

CLI entry-point to compile, quantize, and benchmark trained models.

Usage::

    python -m src.deploy --model-dir experiments/conv1d_ae --data-dir data/buck/buck_data
"""

import argparse
import sys
from pathlib import Path

from .deployment import (
    load_deployment_datasets,
    get_calibration_dataset,
    quantize_keras_native,
    convert_to_tflite,
    convert_keras_to_onnx,
    compile_onnx_to_tensorrt,
    quantize_for_vitis_ai,
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

    # 1b. TFLite Conversions
    for mode in ["dynamic", "float16", "int8"]:
        path_tflite = str(output_dir / f"model_{mode}.tflite")
        kwargs = {"calibration_data": calibration_data} if mode == "int8" else {}
        try:
            convert_to_tflite(keras_model, mode, output_path=path_tflite, **kwargs)
        except Exception as e:
            logger.error(f"TFLite {mode} conversion failed: {e}")

    # ---- 2. Keras to ONNX and ONNX to TensorRT ----
    logger.info("--- Starting Pipeline 2: Keras to ONNX & TensorRT Engine ---")
    path_onnx = str(output_dir / "model.onnx")
    onnx_success = convert_keras_to_onnx(keras_model, path_onnx)
    if onnx_success:
        for mode in ["FP32", "FP16", "INT8"]:
            path_engine = str(output_dir / f"model_{mode.lower()}.engine")
            kwargs = {"calibration_data": calibration_data} if mode == "INT8" else {}
            compile_onnx_to_tensorrt(onnx_path=path_onnx, engine_path=path_engine, precision_mode=mode, **kwargs)

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

    logger.info(f"Deployment optimization complete! All files generated in: {output_dir}")

if __name__ == "__main__":
    main()
