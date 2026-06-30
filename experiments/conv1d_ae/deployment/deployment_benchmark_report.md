# Edge Compute Deployment Benchmark Report

This report studies the hardware deployment optimizations across size, latency, and reconstruction accuracy.

| Format | Size (MB) | Batch Size 1 Latency (ms) | Batch Size 32 Latency (ms) | Reconstruction MSE | MSE Shift from Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 3.420 MB | 19.829 ± 1.256 ms | 19.381 ± 1.088 ms | 1.315244e-01 | 0.000000e+00 |
| **TFLite DYNAMIC** | 0.349 MB | 0.133 ± 0.010 ms | 4.423 ± 0.222 ms | 1.311519e-01 | 3.724843e-04 |
| **TFLite FP16** | 0.578 MB | 0.161 ± 0.038 ms | 4.450 ± 0.296 ms | 1.315157e-01 | 8.657575e-06 |
| **TFLite INT8** | 0.388 MB | 0.114 ± 0.011 ms | 3.353 ± 0.130 ms | 6.766404e-01 | 5.451160e-01 |
| **ONNX (CPU)** | 1.129 MB | 0.398 ± 0.072 ms | 2.217 ± 0.374 ms | 1.315296e-01 | 5.230308e-06 |
| **TensorRT (GPU)** | 1.445 MB | 0.936 ± 0.241 ms | 0.671 ± 0.269 ms | 1.315227e-01 | 1.654029e-06 |