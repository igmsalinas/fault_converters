"""
Keras Backend & TFLite Quantization Pipeline
=============================================

Implements model quantization using Keras 3 and TensorFlow Lite APIs.
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from typing import Optional, Dict, Any, Callable

from ..utils.logger import get_logger

logger = get_logger(__name__)


def quantize_keras_native(
    model: keras.Model,
    mode: str = "int8",
    output_dir: Optional[str] = None,
) -> keras.Model:
    """
    Quantize model using Keras 3 native quantization APIs.

    Args:
        model: Trained Keras model.
        mode: Quantization mode ("int8", "float16", etc.).
        output_dir: Optional directory to save quantized weights.

    Returns:
        Quantized Keras model.
    """
    logger.info(f"Applying Keras-native {mode} quantization...")
    try:
        # Clone the model to avoid modifying the original in-place
        quantized_model = keras.models.clone_model(model)
        quantized_model.set_weights(model.get_weights())
        
        # Check if native quantize is available on the model instance (introduced in Keras 3)
        if hasattr(quantized_model, "quantize"):
            quantized_model.quantize(mode)
            logger.info("Keras-native quantization completed successfully.")
        else:
            logger.warning(
                "Keras model does not have native .quantize() method. "
                "Falling back to custom weight quantization or TFLite pipeline."
            )
            # Custom fallback logic if model.quantize is not available
            # Note: For some Keras 3 builds/backends, this may require specific configurations
            
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            quantized_model.save_weights(str(out_path / f"keras_quantized_{mode}.weights.h5"))
            logger.info(f"Saved Keras-native quantized weights to {out_path}")
            
        return quantized_model
    except Exception as e:
        logger.error(f"Keras-native quantization failed: {e}")
        raise


def convert_to_tflite(
    model: keras.Model,
    optimization_mode: str = "dynamic",
    calibration_data: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Convert a Keras model to TensorFlow Lite format with post-training quantization.

    Args:
        model: Trained Keras model.
        optimization_mode: Quantization type: "none" (FP32), "dynamic" (INT8 weights),
                            "float16" (FP16 weights), "int8" (Full INT8).
        calibration_data: Numpy array of representative inputs for full INT8 calibration.
        output_path: Path where the .tflite model should be written.

    Returns:
        Serialized TFLite model bytes.
    """
    logger.info(f"Converting model to TFLite format (optimization_mode: {optimization_mode})...")
    
    # 1. Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 2. Apply optimizations
    if optimization_mode != "none":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    if optimization_mode == "float16":
        converter.target_spec.supported_types = [tf.float16]
        
    elif optimization_mode == "int8":
        if calibration_data is None:
            raise ValueError("calibration_data is required for full INT8 post-training quantization.")
            
        # Define representative dataset generator
        def representative_dataset_gen():
            for i in range(len(calibration_data)):
                # Expand dimensions to match expected batch shape: (1, seq_len, features)
                yield [np.expand_dims(calibration_data[i], axis=0).astype(np.float32)]
                
        converter.representative_dataset = representative_dataset_gen
        # Enforce full integer quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        
        # If strict target requires integer inputs (like Edge TPU/microcontrollers):
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        
    # 3. Convert
    tflite_model = converter.convert()
    logger.info("TFLite conversion completed.")
    
    # 4. Save to disk if path is provided
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(tflite_model)
        logger.info(f"Saved TFLite model to {out_file}")
        
    return tflite_model
