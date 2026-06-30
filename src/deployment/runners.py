"""
Inference Runner Factories
==========================

Provides reusable callable runner factories for each model format
(Keras, TFLite, ONNX, TensorRT). These runners encapsulate the
format-specific setup (session loading, buffer allocation, quant/dequant)
behind a single ``Callable[[np.ndarray], np.ndarray]`` interface.

Used by the benchmark, evaluation, and performance modules to avoid
duplicating inference code per format.
"""

import numpy as np
import tensorflow as tf
import keras
from typing import Optional, Callable

from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_keras_runner(model: keras.Model) -> Callable[[np.ndarray], np.ndarray]:
    """Create a callable that runs inference on a Keras model.

    Args:
        model: Compiled Keras model.

    Returns:
        Callable accepting an input array and returning predictions.
    """
    def runner(x: np.ndarray) -> np.ndarray:
        return model(x.astype(np.float32), training=False).numpy()
    return runner


def create_tflite_runner(tflite_path: str) -> Callable[[np.ndarray], np.ndarray]:
    """Create a callable that runs inference on a TFLite model.

    Handles INT8 quantization/dequantization transparently.

    Args:
        tflite_path: Path to a .tflite model file.

    Returns:
        Callable accepting a batch array and returning float32 predictions.
        Supports dynamic batch sizes by resizing the interpreter internally.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details["index"]
    output_index = output_details["index"]

    is_int8_input = input_details["dtype"] == np.int8
    input_quant = input_details.get("quantization", (1.0, 0))
    is_int8_output = output_details["dtype"] == np.int8
    output_quant = output_details.get("quantization", (1.0, 0))

    # Track the current allocated batch size to avoid unnecessary resizes
    _current_batch_size = [input_details["shape"][0]]

    def runner(x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        # Resize interpreter if batch size changed
        if batch_size != _current_batch_size[0]:
            new_shape = list(input_details["shape"])
            new_shape[0] = batch_size
            interpreter.resize_tensor_input(input_index, new_shape)
            interpreter.allocate_tensors()
            _current_batch_size[0] = batch_size

        # Prepare input (handle INT8 quantization)
        if is_int8_input:
            scale, zero_point = input_quant
            x_prep = np.round(x.astype(np.float32) / scale + zero_point).astype(np.int8)
        else:
            x_prep = x.astype(np.float32)

        interpreter.set_tensor(input_index, x_prep)
        interpreter.invoke()
        out_raw = interpreter.get_tensor(output_index)

        # Dequantize output if needed
        if is_int8_output:
            out_scale, out_zero_point = output_quant
            return (out_raw.astype(np.float32) - out_zero_point) * out_scale
        return out_raw.astype(np.float32)

    return runner


def create_onnx_runner(onnx_path: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a callable that runs inference on an ONNX model via ORT.

    Args:
        onnx_path: Path to a .onnx model file.

    Returns:
        Callable accepting a batch array and returning predictions,
        or None if onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime is not installed. Cannot create ONNX runner.")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def runner(x: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: x.astype(np.float32)})[0]

    return runner


def create_tensorrt_runner(
    engine_path: str,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a callable that runs inference on a TensorRT engine.

    Allocates CUDA device buffers matching the engine's I/O bindings
    and runs synchronous inference.

    Args:
        engine_path: Path to a serialized .engine file.

    Returns:
        Callable accepting a batch array (float32) and returning predictions,
        or None if tensorrt/pycuda is not available.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning(
            "The 'tensorrt' Python package is not installed. Cannot create TensorRT runner."
        )
        return None

    try:
        from cuda import cudart
    except ImportError:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit  # noqa: F401
                _use_pycuda = True
            except ImportError:
                logger.warning(
                    "Neither 'cuda-python' nor 'pycuda' is installed. "
                    "Cannot create TensorRT runner (CUDA memory management required)."
                )
                return None
        else:
            _use_pycuda = False
    else:
        _use_pycuda = False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    try:
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    except FileNotFoundError:
        logger.error(f"TensorRT engine file not found: {engine_path}")
        return None

    if engine is None:
        logger.error(f"Failed to deserialize TensorRT engine from {engine_path}")
        return None

    context = engine.create_execution_context()

    # Identify input/output tensor names and shapes
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            output_name = name

    if input_name is None or output_name is None:
        logger.error("Could not identify input/output tensors in TensorRT engine.")
        return None

    def runner(x: np.ndarray) -> np.ndarray:
        x_contiguous = np.ascontiguousarray(x.astype(np.float32))
        batch_size = x_contiguous.shape[0]

        # Set the input shape for dynamic batching
        context.set_input_shape(input_name, x_contiguous.shape)

        # Determine output shape
        output_shape = context.get_tensor_shape(output_name)
        output_array = np.empty(output_shape, dtype=np.float32)

        if _use_pycuda:
            # pycuda path
            d_input = cuda.mem_alloc(x_contiguous.nbytes)
            d_output = cuda.mem_alloc(output_array.nbytes)

            cuda.memcpy_htod(d_input, x_contiguous)

            context.set_tensor_address(input_name, int(d_input))
            context.set_tensor_address(output_name, int(d_output))
            context.execute_async_v3(stream_handle=0)

            cuda.memcpy_dtoh(output_array, d_output)
            d_input.free()
            d_output.free()
        else:
            # cuda-python (cudart) path
            err, d_input = cudart.cudaMalloc(x_contiguous.nbytes)
            err, d_output = cudart.cudaMalloc(output_array.nbytes)

            cudart.cudaMemcpy(
                d_input, x_contiguous.ctypes.data,
                x_contiguous.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )

            context.set_tensor_address(input_name, d_input)
            context.set_tensor_address(output_name, d_output)
            context.execute_async_v3(stream_handle=0)

            cudart.cudaMemcpy(
                output_array.ctypes.data, d_output,
                output_array.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            cudart.cudaFree(d_input)
            cudart.cudaFree(d_output)

        return output_array

    return runner


def run_inference_loop(
    runner: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run inference across a full dataset using a runner callable.

    Splits data into batches to keep memory usage bounded.

    Args:
        runner: Callable produced by one of the ``create_*_runner`` factories.
        data: Full dataset array of shape ``(N, seq_len, features)``.
        batch_size: Maximum batch size per runner call.

    Returns:
        Concatenated reconstruction array of shape ``(N, seq_len, features)``.
    """
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size].astype(np.float32)
        out = runner(batch)
        results.append(out)
    return np.concatenate(results, axis=0)

