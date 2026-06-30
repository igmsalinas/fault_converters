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
        # Determine input shape and build input signature
        input_shape = model.input_shape
        # If batch size is None, set it to None or 1 for dynamic/static graph
        input_signature = [tf.TensorSpec(input_shape, tf.float32, name="input_1")]
        
        logger.info(f"Converting Keras model with input shape: {input_shape} and opset: {opset}...")
        
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
                
        # 3. Configure the builder
        config = builder.create_builder_config()
        
        # Set workspace memory limit (1GB)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
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
                
                # Replace dynamic dimensions with sensible ranges for min, opt, and max.
                # Supporting up to batch size 128 to match the batch scaling study.
                min_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                opt_shape = [1 if dim == -1 or dim is None else dim for dim in shape]
                max_shape = [128 if dim == -1 or dim is None else dim for dim in shape]
                
                logger.info(f"Setting profile shape for input '{name}': min={min_shape}, opt={opt_shape}, max={max_shape}")
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
        
        # Precision flags
        if precision_mode == "FP16":
            if hasattr(trt.BuilderFlag, "FP16"):
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 precision enabled.")
            else:
                logger.info("FP16 builder flag not available in this TensorRT version (strongly typed network default).")
        elif precision_mode == "INT8":
            if hasattr(trt.BuilderFlag, "INT8"):
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("INT8 precision enabled.")
                # Configure calibrator if calibration data is available
                if calibration_data is not None:
                    class NumpyDataCalibrator(trt.IInt8EntropyCalibrator2):
                        def __init__(self, data, batch_size=64, cache_file="calibration.cache"):
                            trt.IInt8EntropyCalibrator2.__init__(self)
                            self.cache_file = cache_file
                            self.data = np.ascontiguousarray(data.astype(np.float32))
                            self.batch_size = min(batch_size, self.data.shape[0])
                            self.current_index = 0
                            
                            self.bytes_per_batch = self.batch_size * self.data.shape[1] * self.data.shape[2] * 4 # float32
                            
                            try:
                                from cuda.bindings import runtime as cudart
                                err, self.device_input = cudart.cudaMalloc(self.bytes_per_batch)
                                if err != 0:
                                    raise RuntimeError(f"cudaMalloc failed with error code {err}")
                                self.cudart = cudart
                                self._use_pycuda = False
                            except ImportError:
                                import pycuda.driver as cuda
                                import pycuda.autoinit  # noqa: F401
                                self.device_input = cuda.mem_alloc(self.bytes_per_batch)
                                self.cuda = cuda
                                self._use_pycuda = True

                        def get_batch_size(self):
                            return self.batch_size

                        def get_batch(self, names):
                            if self.current_index + self.batch_size > self.data.shape[0]:
                                return None

                            batch = self.data[self.current_index : self.current_index + self.batch_size]
                            
                            if self._use_pycuda:
                                self.cuda.memcpy_htod(self.device_input, batch)
                                ptr = int(self.device_input)
                            else:
                                self.cudart.cudaMemcpy(
                                    self.device_input, batch.ctypes.data,
                                    self.bytes_per_batch, self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
                                )
                                ptr = self.device_input

                            self.current_index += self.batch_size
                            return [ptr]

                        def read_calibration_cache(self):
                            import os
                            if os.path.exists(self.cache_file):
                                with open(self.cache_file, "rb") as f:
                                    return f.read()
                            return None

                        def write_calibration_cache(self, cache):
                            with open(self.cache_file, "wb") as f:
                                f.write(cache)

                        def free(self):
                            if self.device_input is not None:
                                if self._use_pycuda:
                                    self.device_input.free()
                                else:
                                    self.cudart.cudaFree(self.device_input)
                    
                    calibrator = NumpyDataCalibrator(calibration_data)
                    config.int8_calibrator = calibrator
                    logger.info("Custom IInt8EntropyCalibrator2 attached.")
            else:
                logger.info("INT8 builder flag not available in this TensorRT version (strongly typed network default).")
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

