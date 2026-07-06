"""
Keras to ONNX and TensorRT Compilation Pipeline
================================================

Handles conversion of Keras models to ONNX format, and subsequent
compilation into standalone NVIDIA TensorRT engines.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from typing import Optional, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


def convert_keras_to_onnx(
    model: keras.Model,
    output_path: str,
    opset: int = 15,
) -> bool:
    """
    Convert a Keras model to ONNX format using tf2onnx.

    Args:
        model: Trained Keras model.
        output_path: Path where the output .onnx model should be written.
        opset: ONNX opset version.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    logger.info("Initiating Keras to ONNX conversion...")
    try:
        import tf2onnx
        import onnx
    except ImportError:
        logger.warning(
            "Required packages for ONNX conversion (tf2onnx, onnx) are not installed. "
            "Please install them via: pip install tf2onnx onnx"
        )
        return False

    try:
        # Fix the batch dimension to 1 so the exported model is optimised for
        # single-sample edge inference (static input shape [1, seq_len, features]).
        input_shape = model.input_shape
        static_shape = (1,) + tuple(input_shape[1:])
        input_signature = [tf.TensorSpec(static_shape, tf.float32, name="input_1")]
        
        logger.info(f"Converting Keras model with fixed batch-1 input shape: {static_shape} and opset: {opset}...")
        
        # Convert model using tf2onnx from_keras
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
        )
        
        # Save ONNX model
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(onnx_model, str(out_file))
        logger.info(f"Successfully saved ONNX model to {out_file}")
        return True
    except Exception as e:
        logger.error(f"Keras to ONNX conversion failed with error: {e}")
        return False


def compile_onnx_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    precision_mode: str = "FP16",
    calibration_data: Optional[np.ndarray] = None,
) -> bool:
    """
    Compile an ONNX model into a standalone TensorRT engine.

    Args:
        onnx_path: Path to the input .onnx file.
        engine_path: Path where the output .engine file should be saved.
        precision_mode: "FP32", "FP16", or "INT8".
        calibration_data: Optional calibration data for INT8 quantization.

    Returns:
        bool: True if compilation succeeded, False otherwise.
    """
    logger.info(f"Initiating ONNX to TensorRT compilation (precision: {precision_mode})...")
    
    # (We rely on trt.BuilderFlag.FP16 natively instead of manual ONNX cast to prevent NaNs)
    # 1. Defensive Check: Check if TensorRT is available in Python
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning(
            "The 'tensorrt' Python package is not installed. "
            "You can compile this ONNX model using NVIDIA's command-line tool 'trtexec' instead:\n"
            f"  trtexec --onnx={onnx_path} --saveEngine={engine_path} --{precision_mode.lower()}\n"
            "To install Python bindings, please refer to TensorRT installation documentation."
        )
        return False

    try:
        # 2. Build the TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        
        # Create network with explicit batch dimension
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        else:
            # In TensorRT 10.0+, explicit batch is the only mode and the flag is deprecated/removed.
            network_flags = 0
        network = builder.create_network(network_flags)
        
        # Parse ONNX file
        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX Parser Error: {parser.get_error(error)}")
                return False
                
        # Configure the builder
        config = builder.create_builder_config()
        
        # We allow TRT to use available memory instead of restricting to 1GB to prevent OOM
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        # Configure optimization profile for dynamic shapes if any are present
        has_dynamic = False
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if any(dim == -1 or dim is None for dim in input_tensor.shape):
                has_dynamic = True
                break

        if has_dynamic:
            logger.info("Dynamic input shapes detected. Creating optimization profile...")
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                shape = list(input_tensor.shape)
                name = input_tensor.name
                
                # Replace any dynamic dimensions with a fixed batch of 1, since
                # the models are exported and optimised for single-sample inference.
                min_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                opt_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                max_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                
                logger.info(f"Setting profile shape for input '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        
        # Precision flags
        if precision_mode == "FP16":
            if hasattr(trt.BuilderFlag, "FP16"):
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled.")
            else:
                logger.info("FP16 builder flag not available in this TensorRT version (strongly typed network default). Pre-cast ONNX was loaded.")
        elif precision_mode == "INT8":
            if calibration_data is not None:
                try:
                    import onnxruntime as ort
                    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
                    
                    logger.info("Executing Post-Training Quantization (PTQ) to insert QDQ nodes for TensorRT 11+...")
                    
                    # Determine input name from ONNX model
                    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                    input_name = session.get_inputs()[0].name
                    
                    class NumpyCalibrationDataReader(CalibrationDataReader):
                        def __init__(self, data, input_name, batch_size=64):
                            self.data = data.astype(np.float32)
                            self.input_name = input_name
                            self.batch_size = batch_size
                            self.current_index = 0
                            
                        def get_next(self):
                            if self.current_index + self.batch_size > self.data.shape[0]:
                                return None
                            batch = self.data[self.current_index : self.current_index + self.batch_size]
                            self.current_index += self.batch_size
                            return {self.input_name: batch}

                    data_reader = NumpyCalibrationDataReader(calibration_data, input_name)
                    
                    int8_onnx_path = onnx_path.replace(".onnx", "_int8_qdq.onnx")
                    
                    # Perform static quantization to QDQ format
                    quantize_static(
                        model_input=onnx_path,
                        model_output=int8_onnx_path,
                        calibration_data_reader=data_reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True}
                    )
                    
                    onnx_path = int8_onnx_path
                    logger.info(f"QDQ Quantization completed. Natively typed INT8 ONNX saved to {onnx_path}")
                    
                    # We must reload the network and parser since onnx_path changed
                    network = builder.create_network(network_flags)
                    parser = trt.OnnxParser(network, TRT_LOGGER)
                    with open(onnx_path, "rb") as model_file:
                        if not parser.parse(model_file.read()):
                            for error in range(parser.num_errors):
                                logger.error(f"ONNX Parser Error (QDQ model): {parser.get_error(error)}")
                            return False
                    logger.info("Re-parsed QDQ ONNX model into TensorRT network.")
                    
                except ImportError:
                    logger.warning("onnxruntime is not installed. INT8 PTQ is skipped. "
                                   "Please install it to enable INT8 precision in TensorRT 11+.")
                except Exception as e:
                    logger.error(f"PTQ failed: {e}. Falling back to original FP32 ONNX.")
            else:
                logger.warning("No calibration data provided. Cannot perform INT8 PTQ. Falling back to FP32 execution.")
        elif precision_mode == "FP32":
            logger.info("FP32 precision selected. No extra flags needed.")

        # 4. Serialize and save the engine
        logger.info("Building and serializing TensorRT engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            logger.error("Failed to build serialized network engine.")
            return False
            
        out_engine = Path(engine_path)
        out_engine.parent.mkdir(parents=True, exist_ok=True)
        with open(out_engine, "wb") as f:
            f.write(serialized_engine)
            
        logger.info(f"Successfully saved TensorRT engine to {out_engine}")
        return True
    except Exception as e:
        logger.error(f"TensorRT compilation failed with error: {e}")
        return False

