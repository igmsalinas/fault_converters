# Edge Compute Unified Deployment Report

This report aggregates optimizations across size, latency, memory, and classification fidelity.

## 1. Inference Fidelity & Baseline Latency

| Format | Size (MB) | Batch 1 Latency (ms) | Batch 32 Latency (ms) | Reconstruction MSE | MSE Shift from Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TFLite DYNAMIC** | 0.349 MB | 0.142 ± 0.039 ms | 4.360 ± 0.295 ms | 1.311519e-01 | 3.795773e-04 |
| **TFLite INT8** | 0.388 MB | 0.110 ± 0.004 ms | 3.416 ± 0.207 ms | 6.766404e-01 | 5.451089e-01 |
| **ONNX (CPU)** | 1.129 MB | 0.404 ± 0.070 ms | 2.334 ± 0.366 ms | 1.315296e-01 | 1.862645e-06 |
| **TensorRT FP32 (GPU)** | 1.454 MB | 0.332 ± 0.081 ms | 0.332 ± 0.060 ms | 1.315319e-01 | 4.470348e-07 |
| **TensorRT FP16 (GPU)** | 1.438 MB | 0.540 ± 0.163 ms | 0.649 ± 0.106 ms | 1.315289e-01 | 2.563000e-06 |
| **TensorRT INT8 (GPU)** | 0.691 MB | 0.471 ± 0.111 ms | 0.655 ± 0.200 ms | 1.347818e-01 | 3.250286e-03 |

## 2. Hardware Resource Profiling

| Model Format | Mean Latency | Min / Max Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 18.859 ± 1.043 ms | 16.85 / 23.97 ms | 0.000 MB | 0.000 MB | 3446.461 MB | 2.672 MB |
| **Keras FP32 (CPU)** | 289.203 ± 19.316 ms | 242.70 / 357.25 ms | 29.094 MB | 0.000 MB | 1039.957 MB | 10395.000 MB |
| **TFLite Dynamic** | 0.133 ± 0.010 ms | 0.13 / 0.23 ms | 0.000 MB | 0.000 MB | 3446.836 MB | 2.672 MB |
| **TFLite INT8** | 0.118 ± 0.016 ms | 0.11 / 0.20 ms | 0.000 MB | 0.000 MB | 3446.961 MB | 2.672 MB |
| **ONNX (CPU)** | 0.393 ± 0.068 ms | 0.35 / 0.82 ms | 0.000 MB | 0.000 MB | 3447.336 MB | 2.672 MB |
| **TensorRT FP32 (GPU)** | 1.124 ± 0.392 ms | 0.85 / 2.43 ms | 0.000 MB | 0.000 MB | 3447.336 MB | 2.672 MB |
| **TensorRT FP16 (GPU)** | 0.885 ± 0.276 ms | 0.64 / 2.99 ms | 0.000 MB | 0.000 MB | 3448.336 MB | 2.672 MB |
| **TensorRT INT8 (GPU)** | 0.833 ± 0.210 ms | 0.62 / 1.63 ms | 0.000 MB | 0.000 MB | 3448.961 MB | 2.672 MB |

## 3. Classification Degradation Evaluation

| Model Format | Size (MB) | Latency Per Sample (ms) | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 3.420 MB | 0.0406 ± 0.0581 ms | 0.9555 | 0.9702 | 0.9596 | 0.9649 | 0.9851 |
| **Keras FP32 (CPU)** | 3.420 MB | 0.1162 ± 0.0090 ms | 0.9600 | 0.9800 | 0.9567 | 0.9682 | 0.9861 |
| **TFLite Dynamic** | 0.349 MB | 0.1395 ± 0.0043 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.9446 |
| **TFLite INT8** | 0.388 MB | 0.1136 ± 0.0016 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.3213 |
| **ONNX (CPU)** | 1.129 MB | 0.1208 ± 0.0102 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |
| **TensorRT FP32 (GPU)** | 1.454 MB | 0.0045 ± 0.0012 ms | 0.9590 | 0.9784 | 0.9568 | 0.9675 | 0.9854 |
| **TensorRT FP16 (GPU)** | 1.438 MB | 0.0062 ± 0.0013 ms | 0.9588 | 0.9780 | 0.9569 | 0.9673 | 0.9852 |
| **TensorRT INT8 (GPU)** | 0.691 MB | 0.0088 ± 0.0033 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.8946 |

## 4. Batch Size Scaling Dynamics

### Keras FP32 (Baseline)
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 23.25 | 23.245 | 0.00 | 0.00 |
| 2 | 18.24 | 9.118 | 0.00 | 0.00 |
| 4 | 18.39 | 4.598 | 0.00 | 0.00 |
| 8 | 18.40 | 2.301 | 0.00 | 0.00 |
| 16 | 19.64 | 1.228 | 0.00 | 0.00 |
| 32 | 18.72 | 0.585 | 0.00 | 0.00 |
| 64 | 20.16 | 0.315 | 0.00 | 0.00 |
| 128 | 18.82 | 0.147 | 0.00 | 0.00 |

### TFLite Dynamic
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.14 | 0.136 | 0.00 | 0.00 |
| 2 | 0.27 | 0.137 | 0.00 | 0.00 |
| 4 | 0.52 | 0.129 | 0.00 | 0.00 |
| 8 | 1.03 | 0.128 | 0.00 | 0.00 |
| 16 | 2.09 | 0.131 | 0.00 | 0.00 |
| 32 | 4.34 | 0.136 | 0.00 | 0.00 |
| 64 | 8.66 | 0.135 | 0.00 | 0.00 |
| 128 | 17.59 | 0.137 | 0.00 | 0.00 |

### TFLite INT8
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.11 | 0.112 | 0.00 | 0.00 |
| 2 | 0.23 | 0.117 | 0.00 | 0.00 |
| 4 | 0.43 | 0.107 | 0.00 | 0.00 |
| 8 | 0.85 | 0.106 | 0.00 | 0.00 |
| 16 | 1.67 | 0.104 | 0.00 | 0.00 |
| 32 | 3.33 | 0.104 | 0.00 | 0.00 |
| 64 | 6.69 | 0.105 | 0.00 | 0.00 |
| 128 | 13.67 | 0.107 | 0.00 | 0.00 |

### ONNX (CPU)
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.45 | 0.446 | 0.00 | 0.00 |
| 2 | 0.57 | 0.286 | 0.00 | 0.00 |
| 4 | 0.67 | 0.167 | 0.00 | 0.00 |
| 8 | 0.92 | 0.115 | 0.00 | 0.00 |
| 16 | 1.39 | 0.087 | 0.00 | 0.00 |
| 32 | 2.48 | 0.078 | 0.00 | 0.00 |
| 64 | 4.43 | 0.069 | 0.00 | 0.00 |
| 128 | 8.27 | 0.065 | 0.00 | 0.00 |
