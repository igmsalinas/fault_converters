# Edge Compute Unified Deployment Report

This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.

All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.

## 1. Timing & Memory Profiling (Single-Sample, n=1)

| Model Format | Mean Latency | Min / Max Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 19.040 ± 1.688 ms | 16.58 / 28.21 ms | 325.625 MB | 1.319 MB | 2986.562 MB | 2.661 MB |
| **Keras FP32 (CPU)** | 13.457 ± 0.722 ms | 11.84 / 16.13 ms | 8.500 MB | 0.000 MB | 810.570 MB | 10289.000 MB |
| **TFLite FP16** | 0.142 ± 0.017 ms | 0.13 / 0.27 ms | 5.750 MB | 0.000 MB | 2992.312 MB | 1.343 MB |
| **TFLite Dynamic** | 0.137 ± 0.016 ms | 0.13 / 0.22 ms | 0.375 MB | 0.000 MB | 2992.207 MB | 1.343 MB |
| **TFLite INT8** | 0.108 ± 0.006 ms | 0.11 / 0.14 ms | 0.375 MB | 0.000 MB | 2992.266 MB | 1.343 MB |
| **ONNX (CPU)** | 0.379 ± 0.048 ms | 0.35 / 0.64 ms | 24.500 MB | 0.000 MB | 3016.453 MB | 1.343 MB |
| **TensorRT FP32 (GPU)** | 0.786 ± 0.371 ms | 0.52 / 2.58 ms | 268.250 MB | 0.000 MB | 3284.703 MB | 1.343 MB |
| **TensorRT FP16 (GPU)** | 0.630 ± 0.262 ms | 0.45 / 1.61 ms | 2.000 MB | 0.000 MB | 3286.703 MB | 1.343 MB |
| **TensorRT INT8 (GPU)** | 0.645 ± 0.297 ms | 0.45 / 2.63 ms | 1.750 MB | 0.000 MB | 3288.453 MB | 1.343 MB |

## 2. Classification Metrics Evaluation

| Model Format | Size (MB) | Latency Per Sample (ms) | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 3.420 MB | 19.4007 ± 1.1938 ms | 0.9586 | 0.9774 | 0.9572 | 0.9672 | 0.9856 |
| **Keras FP32 (CPU)** | 3.420 MB | 13.3143 ± 0.6372 ms | 0.9600 | 0.9800 | 0.9567 | 0.9682 | 0.9861 |
| **TFLite FP16** | 0.576 MB | 0.1458 ± 0.0188 ms | 0.9576 | 0.9731 | 0.9599 | 0.9665 | 0.9854 |
| **TFLite Dynamic** | 0.346 MB | 0.1343 ± 0.0143 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.9446 |
| **TFLite INT8** | 0.385 MB | 0.1198 ± 0.0391 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.3213 |
| **ONNX (CPU)** | 1.126 MB | 0.4277 ± 0.2313 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |
| **TensorRT FP32 (GPU)** | 1.711 MB | 1.1837 ± 0.7735 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |
| **TensorRT FP16 (GPU)** | 1.702 MB | 0.5802 ± 0.2619 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |
| **TensorRT INT8 (GPU)** | 1.743 MB | 0.3767 ± 0.1758 ms | 0.9589 | 0.9776 | 0.9575 | 0.9674 | 0.9857 |
