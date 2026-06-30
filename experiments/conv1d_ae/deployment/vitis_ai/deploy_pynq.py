# PYNQ Board Deployment Script
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
print(f"Loading DPU overlay: {overlay_name}...")
try:
    overlay = DpuOverlay(overlay_name)
    print("DPU overlay loaded successfully.")
except Exception as e:
    print(f"Error loading overlay: {e}")
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
    print(f"  Input {i}: name={tensor.name}, shape={tensor.dims}, dtype={tensor.dtype}")

print("DPU Output Tensors:")
for i, tensor in enumerate(output_tensors):
    print(f"  Output {i}: name={tensor.name}, shape={tensor.dims}, dtype={tensor.dtype}")

# 4. Prepare Mock Input Data
# The network expects shape matching input_tensors[0].dims
input_shape = input_tensors[0].dims
print(f"Model expects input shape: {input_shape}")

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

print(f"Inference complete in {(t1 - t0)*1000.0:.3f} ms.")

# 7. Postprocess output
reconstructed_data = output_data[0]
print(f"Output shape: {reconstructed_data.shape}")

# Anomaly detection decision:
# Compute reconstruction Mean Squared Error (MSE)
# If MSE > threshold, classify as Anomaly.
mse = np.mean(np.square(mock_input - reconstructed_data))
print(f"Reconstruction MSE: {mse:.6f}")
