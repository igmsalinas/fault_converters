"""
Xilinx Vitis AI Quantization and Compilation Pipeline
======================================================

Handles optimization and quantization of Keras/TensorFlow models
targeting Xilinx/AMD Edge DPUs (Deep Learning Processing Units)
on Zynq UltraScale+ MPSoC and custom Zynq-7000 PYNQ architectures.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from typing import Optional, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Supported Target Hardware Profiles
HARDWARE_PROFILES = {
    "zcu102": {
        "target": "DPUCZDX8G_ISA1_B4096",
        "cpu_arch": "arm64",
        "name": "Zynq UltraScale+ MPSoC ZCU102",
        "dpu_arch_path": "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"
    },
    "zcu104": {
        "target": "DPUCZDX8G_ISA1_B4096",
        "cpu_arch": "arm64",
        "name": "Zynq UltraScale+ MPSoC ZCU104",
        "dpu_arch_path": "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"
    },
    "kv260": {
        "target": "DPUCZDX8G_ISA1_B4096",
        "cpu_arch": "arm64",
        "name": "Kria KV260 Vision AI Starter Kit",
        "dpu_arch_path": "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
    },
    "ultra96": {
        "target": "DPUCZDX8G_ISA1_B2304",
        "cpu_arch": "arm64",
        "name": "Ultra96-V2 Development Board",
        "dpu_arch_path": "/opt/vitis_ai/compiler/arch/DPUCZDX8G/Ultra96/arch.json"
    },
    "pynq_z2": {
        "target": "DPUCZDX8G_ISA1_B1024",
        "cpu_arch": "arm32",
        "name": "PYNQ-Z2 (Zynq-7000 XC7Z020 custom DPU overlay)",
        "dpu_arch_path": None  # Generates local arch.json
    }
}


def generate_pynq_deployment_script(output_dir: Path, input_shape: tuple) -> None:
    """Generate a Python script template for deploying on PYNQ board using pynq-dpu."""
    script_path = output_dir / "deploy_pynq.py"
    
    # Keras input shape: (seq_len, features) -> DPU runner shape: [batch_size, seq_len, features]
    # Standard batch size for single-inference evaluation is 1
    dpu_shape = [1, input_shape[0], input_shape[1]]
    
    script_content = f"""# PYNQ Board Deployment Script
# ==========================================
# Run this script directly on your PYNQ board (e.g. ZCU104, KV260, or PYNQ-Z2).
# Requirements: pip install pynq-dpu

import os
import time
import numpy as np
from pynq_dpu import DpuOverlay

# 1. Load the DPU Bitstream overlay
# Place 'dpu.bit', 'dpu.hwh', and 'dpu.xclbin' in this directory.
overlay_name = "dpu.bit"
print(f"Loading DPU overlay: {{overlay_name}}...")
try:
    overlay = DpuOverlay(overlay_name)
    print("DPU overlay loaded successfully.")
except Exception as e:
    print(f"Error loading overlay: {{e}}")
    print("Ensure dpu.bit, dpu.hwh, and dpu.xclbin are in the current folder.")
    exit(1)

# 2. Instantiate Vitis AI Runtime DPU Runner
dpu_runner = overlay.runner
print("DPU Runner instantiated.")

# 3. Retrieve DPU Input/Output Tensors
input_tensors = dpu_runner.get_input_tensors()
output_tensors = dpu_runner.get_output_tensors()

print("DPU Input Tensors:")
for i, tensor in enumerate(input_tensors):
    print(f"  Input {{i}}: name={{tensor.name}}, shape={{tensor.dims}}, dtype={{tensor.dtype}}")

print("DPU Output Tensors:")
for i, tensor in enumerate(output_tensors):
    print(f"  Output {{i}}: name={{tensor.name}}, shape={{tensor.dims}}, dtype={{tensor.dtype}}")

# 4. Prepare Mock Input Data
# The network expects shape matching input_tensors[0].dims
input_shape = input_tensors[0].dims
print(f"Model expects input shape: {{input_shape}}")

# Generate a mock batch matching the DPU layout (e.g. [batch_size, seq_len, features])
batch_size = input_shape[0]
seq_len = input_shape[1]
features = input_shape[2]

# In production, replace this with your actual preprocessed frequency-domain sweeps
# (Amplitude & Phase) after applying the saved DataPreprocessor scaler.
mock_input = np.random.normal(0, 1.0, size=input_shape).astype(np.float32)

# 5. Allocate buffers
input_data = [mock_input]
output_data = [np.empty(list(tensor.dims), dtype=np.float32) for tensor in output_tensors]

# 6. Execute inference
print("Running DPU inference...")
t0 = time.perf_counter()
job_id = dpu_runner.execute_async(input_data, output_data)
dpu_runner.wait(job_id)
t1 = time.perf_counter()

print(f"Inference complete in {{(t1 - t0)*1000.0:.3f}} ms.")

# 7. Postprocess output
reconstructed_data = output_data[0]
print(f"Output shape: {{reconstructed_data.shape}}")

# Anomaly detection decision:
# Compute reconstruction Mean Squared Error (MSE)
# If MSE > threshold, classify as Anomaly.
mse = np.mean(np.square(mock_input - reconstructed_data))
print(f"Reconstruction MSE: {{mse:.6f}}")
"""
    with open(script_path, "w") as f:
        f.write(script_content)
    logger.info(f"Saved PYNQ python deployment script template to {script_path}")


def quantize_for_vitis_ai(
    model: keras.Model,
    calibration_data: np.ndarray,
    output_dir: str,
    target_hardware: str = "zcu102",
) -> bool:
    """
    Quantize Keras model targeting Xilinx DPU using Vitis AI Quantizer API.

    Args:
        model: Trained Keras model.
        calibration_data: Numpy array of calibration inputs.
        output_dir: Directory where the quantized model will be saved.
        target_hardware: Target platform ('zcu102', 'zcu104', 'kv260', 'ultra96', 'pynq_z2').

    Returns:
        bool: True if quantization succeeded, False otherwise.
    """
    logger.info(f"Initiating Vitis AI quantization flow for hardware target: {target_hardware}...")
    
    if target_hardware not in HARDWARE_PROFILES:
        logger.error(
            f"Unsupported hardware target: '{target_hardware}'. "
            f"Supported options are: {list(HARDWARE_PROFILES.keys())}"
        )
        return False
        
    profile = HARDWARE_PROFILES[target_hardware]
    logger.info(f"Target Hardware Configuration: {profile['name']} (ISA: {profile['target']}, CPU: {profile['cpu_arch']})")

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate target-specific arch.json
    arch_json_path = out_path / f"{target_hardware}_arch.json"
    arch_data = {
        "target": profile["target"],
        "cpu_arch": profile["cpu_arch"]
    }
    with open(arch_json_path, "w") as f:
        json.dump(arch_data, f, indent=2)
    logger.info(f"Generated target architecture config file: {arch_json_path}")
    
    # 2. Generate PYNQ deployment script template
    # model.input_shape format: (None, seq_len, features)
    input_shape = (model.input_shape[1], model.input_shape[2])
    generate_pynq_deployment_script(out_path, input_shape)

    # 3. Defensive Check: Check if Vitis AI packages are available
    try:
        from tensorflow_model_optimization.quantization.keras import vitis_quantize
    except ImportError:
        logger.warning(
            f"Vitis AI Quantizer ('vitis_quantize') is not installed in this environment.\n"
            f"To compile the model for '{profile['name']}', run this script inside the Vitis AI Docker container:\n"
            f"  1. Launch Vitis AI CPU/GPU Docker container.\n"
            f"  2. Activate environment: 'conda activate vitis-ai-tensorflow2'\n"
            f"  3. Execute this python script.\n\n"
            f"Once quantized_model.h5 is generated, compile it to .xmodel using:\n"
            f"  vai_c_tensorflow2 \\\n"
            f"    -m {out_path / 'quantized_model.h5'} \\\n"
            f"    -a {arch_json_path} \\\n"
            f"    -o {out_path / 'compiled_dpu'} \\\n"
            f"    -n fault_converter_dpu"
        )
        return False

    try:
        # 4. Instantiate Vitis Quantizer
        logger.info("Initializing VitisQuantizer with trained model...")
        quantizer = vitis_quantize.VitisQuantizer(model)
        
        # 5. Prepare calibration dataset
        calib_dataset = tf.data.Dataset.from_tensor_slices(calibration_data).batch(10)
        
        # 6. Run calibration and quantization
        logger.info("Running Vitis AI calibration and quantization...")
        quantized_model = quantizer.quantize_model(
            calib_dataset=calib_dataset,
            output_dir=str(out_path),
        )
        
        # 7. Save the quantized model in H5 format
        quantized_model_path = out_path / "quantized_model.h5"
        quantized_model.save(str(quantized_model_path))
        logger.info(f"Successfully saved Vitis AI quantized model to {quantized_model_path}")
        
        # 8. Generate compilation script
        compiler_cmd = (
            "vai_c_tensorflow2 \\\n"
            f"  -m {quantized_model_path} \\\n"
            f"  -a {arch_json_path} \\\n"
            f"  -o {out_path / 'compiled_dpu'} \\\n"
            "  -n fault_converter_dpu"
        )
        
        comp_script_path = out_path / "compile_dpu.sh"
        with open(comp_script_path, "w") as f:
            f.write("#!/bin/bash\n# Run this script inside Vitis AI Docker container\n")
            f.write("conda activate vitis-ai-tensorflow2\n\n")
            f.write(compiler_cmd)
            f.write("\n")
        logger.info(f"DPU compiler script written to: {comp_script_path}")
            
        return True
    except Exception as e:
        logger.error(f"Vitis AI quantization failed with error: {e}")
        return False
