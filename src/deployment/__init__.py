"""
Deployment Pipelines and Quantization Toolkit
=============================================

A submodule containing tools for quantizing and compiling trained autoencoders
for deployment in edge compute environments. Exposes multiple target pipelines:
1. Model quantization and optimization using Keras/TFLite
2. Keras-to-ONNX and ONNX-to-TensorRT Engine compilation
3. AMD/Xilinx Vitis AI quantization and compilation setup
"""

from .utils import (
    load_deployment_datasets,
    get_calibration_dataset,
    get_file_size_mb,
)

from .keras_quantization import (
    quantize_keras_native,
    convert_to_tflite,
)

from .onnx_trt import (
    convert_keras_to_onnx,
    compile_onnx_to_tensorrt,
)

from .vitis_ai import (
    quantize_for_vitis_ai,
)

from .runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
    run_inference_loop,
)

from .benchmark import (
    run_deployment_benchmarks,
    save_benchmark_report,
)

from .timing_memory_benchmark import (
    run_full_performance_suite,
    save_performance_report,
    run_batch_size_study,
    save_batch_size_report,
)

from .timing_evaluation import (
    run_deployment_evaluations,
    save_evaluation_report,
)
