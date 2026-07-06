# Edge Compute Unified Deployment Report

This report aggregates optimizations across size, latency, memory, and classification fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Inference Fidelity & Baseline Latency

| Format | Size (MB) | Latency n=1 (ms) | Reconstruction MSE | MSE Shift from Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **TFLite Dynamic** | 0.349 MB | 0.140 ± 0.013 ms | 1.311519e-01 | 3.732592e-04 |
| **TFLite FP16** | 0.578 MB | 0.162 ± 0.049 ms | 1.315157e-01 | 9.432435e-06 |
| **TFLite INT8** | 0.388 MB | 0.121 ± 0.042 ms | 6.766404e-01 | 5.451152e-01 |
| **ONNX (CPU)** | 1.129 MB | 0.412 ± 0.082 ms | 1.315296e-01 | 4.455447e-06 |
| **TensorRT FP32 (GPU)** | 1.454 MB | 0.428 ± 0.196 ms | 1.315319e-01 | 6.765127e-06 |
| **TensorRT FP16 (GPU)** | 1.438 MB | 0.385 ± 0.093 ms | 1.315289e-01 | 3.755093e-06 |
| **TensorRT INT8 (GPU)** | 0.691 MB | 0.532 ± 0.234 ms | 1.347818e-01 | 3.256604e-03 |

## 2. Hardware Resource Profiling

| Model Format | Mean Latency | Min / Max Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 19.603 ± 1.378 ms | 17.24 / 25.95 ms | 0.000 MB | 0.000 MB | 3438.008 MB | 2.672 MB |
| **Keras FP32 (CPU)** | 14.455 ± 1.928 ms | 12.78 / 35.58 ms | 0.500 MB | 0.000 MB | 810.273 MB | 10231.000 MB |
| **TFLite FP16** | 0.172 ± 0.056 ms | 0.14 / 0.55 ms | 0.000 MB | 0.000 MB | 3438.508 MB | 2.672 MB |
| **TFLite Dynamic** | 0.140 ± 0.017 ms | 0.13 / 0.25 ms | 0.000 MB | 0.000 MB | 3438.379 MB | 2.672 MB |
| **TFLite INT8** | 0.112 ± 0.017 ms | 0.11 / 0.25 ms | 0.000 MB | 0.000 MB | 3438.418 MB | 2.672 MB |
| **ONNX (CPU)** | 0.406 ± 0.111 ms | 0.36 / 1.41 ms | 0.000 MB | 0.000 MB | 3438.918 MB | 2.672 MB |
| **TensorRT FP32 (GPU)** | 0.800 ± 0.437 ms | 0.52 / 2.77 ms | 0.000 MB | 0.000 MB | 3440.043 MB | 2.672 MB |
| **TensorRT FP16 (GPU)** | 0.682 ± 0.221 ms | 0.44 / 1.59 ms | 0.000 MB | 0.000 MB | 3440.418 MB | 2.672 MB |
| **TensorRT INT8 (GPU)** | 0.630 ± 0.263 ms | 0.48 / 2.40 ms | 0.000 MB | 0.000 MB | 3440.418 MB | 2.672 MB |

## 3. Classification Degradation Evaluation

| Model Format | Size (MB) | Latency Per Sample (ms) | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 3.420 MB | 19.9439 ± 1.6718 ms | 0.9555 | 0.9702 | 0.9597 | 0.9649 | 0.9851 |
| **Keras FP32 (CPU)** | 3.420 MB | 14.1625 ± 0.7660 ms | 0.9600 | 0.9800 | 0.9567 | 0.9682 | 0.9861 |
| **TFLite FP16** | 0.578 MB | 0.1646 ± 0.0489 ms | 0.9576 | 0.9731 | 0.9599 | 0.9665 | 0.9854 |
| **TFLite Dynamic** | 0.349 MB | 0.1525 ± 0.0285 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.9446 |
| **TFLite INT8** | 0.388 MB | 0.1125 ± 0.0133 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.3213 |
| **ONNX (CPU)** | 1.129 MB | 0.4403 ± 0.2897 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |
