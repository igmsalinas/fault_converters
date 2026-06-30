# Edge Compute Unified Deployment Report

This report aggregates optimizations across size, latency, memory, and classification fidelity.

## 1. Inference Fidelity & Baseline Latency

| Format | Size (MB) | Batch 1 Latency (ms) | Batch 32 Latency (ms) | Reconstruction MSE | MSE Shift from Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TFLite DYNAMIC** | 0.349 MB | 0.133 ± 0.006 ms | 4.633 ± 0.276 ms | 1.311519e-01 | 3.732592e-04 |
| **TFLite FP16** | 0.578 MB | 0.161 ± 0.036 ms | 4.682 ± 0.290 ms | 1.315157e-01 | 9.432435e-06 |
| **TFLite INT8** | 0.388 MB | 0.115 ± 0.010 ms | 3.481 ± 0.238 ms | 6.766404e-01 | 5.451152e-01 |
| **ONNX (CPU)** | 1.129 MB | 0.528 ± 0.720 ms | 2.327 ± 0.411 ms | 1.315296e-01 | 4.455447e-06 |
| **TensorRT FP32 (GPU)** | 1.434 MB | 0.573 ± 0.219 ms | 0.852 ± 0.357 ms | 1.315254e-01 | 1.788139e-07 |
| **TensorRT FP16 (GPU)** | 1.470 MB | 0.731 ± 0.188 ms | 0.777 ± 0.288 ms | 1.315275e-01 | 2.339482e-06 |
| **TensorRT INT8 (GPU)** | 1.464 MB | 0.440 ± 0.128 ms | 0.500 ± 0.134 ms | 1.315183e-01 | 6.824732e-06 |

## 2. Hardware Resource Profiling

| Model Format | Mean Latency | Min / Max Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 20.870 ± 2.132 ms | 17.80 / 30.77 ms | 0.000 MB | 0.000 MB | 4190.672 MB | 2.695 MB |
| **TFLite FP16** | 0.182 ± 0.076 ms | 0.14 / 0.56 ms | 0.000 MB | 0.000 MB | 4192.797 MB | 2.695 MB |
| **TFLite Dynamic** | 0.157 ± 0.049 ms | 0.13 / 0.37 ms | 0.000 MB | 0.000 MB | 4193.105 MB | 2.695 MB |
| **TFLite INT8** | 0.133 ± 0.040 ms | 0.11 / 0.32 ms | 0.000 MB | 0.000 MB | 4193.520 MB | 2.695 MB |
| **ONNX (CPU)** | 0.417 ± 0.105 ms | 0.36 / 1.08 ms | 0.000 MB | 0.000 MB | 4196.145 MB | 2.695 MB |
| **TensorRT FP16 (GPU)** | 0.792 ± 0.395 ms | 0.47 / 3.66 ms | 0.000 MB | 0.000 MB | 4199.395 MB | 2.695 MB |
| **TensorRT FP32 (GPU)** | 0.937 ± 0.414 ms | 0.45 / 2.04 ms | 0.000 MB | 0.000 MB | 4201.770 MB | 2.695 MB |
| **TensorRT FP16 (GPU)** | 0.659 ± 0.351 ms | 0.40 / 2.01 ms | 0.000 MB | 0.000 MB | 4202.645 MB | 2.695 MB |
| **TensorRT INT8 (GPU)** | 0.753 ± 0.423 ms | 0.38 / 2.52 ms | 0.000 MB | 0.000 MB | 4203.270 MB | 2.695 MB |

## 3. Classification Degradation Evaluation

| Model Format | Size (MB) | Latency Per Sample (ms) | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 3.420 MB | 0.0306 ms | 0.9555 | 0.9702 | 0.9597 | 0.9649 | 0.9851 |
| **TFLite FP16** | 0.578 MB | 0.1568 ms | 0.9576 | 0.9731 | 0.9599 | 0.9665 | 0.9854 |
| **TFLite Dynamic** | 0.349 MB | 0.1454 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.9446 |
| **TFLite INT8** | 0.388 MB | 0.1291 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.3213 |
| **ONNX (CPU)** | 1.129 MB | 0.1441 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |

## 4. Batch Size Scaling Dynamics

### Keras FP32 (Baseline)
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 21.73 | 21.732 | 0.00 | 0.00 |
| 2 | 20.79 | 10.396 | 0.00 | 0.00 |
| 4 | 22.56 | 5.639 | 0.00 | 0.00 |
| 8 | 23.33 | 2.916 | 0.00 | 0.00 |
| 16 | 22.76 | 1.422 | 0.00 | 0.00 |
| 32 | 20.37 | 0.637 | 0.00 | 0.00 |
| 64 | 20.86 | 0.326 | 0.00 | 0.00 |
| 128 | 22.20 | 0.173 | 0.00 | 0.00 |

### TFLite FP16
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.18 | 0.178 | 0.00 | 0.00 |
| 2 | 0.31 | 0.157 | 0.00 | 0.00 |
| 4 | 0.56 | 0.141 | 0.00 | 0.00 |
| 8 | 1.13 | 0.142 | 0.00 | 0.00 |
| 16 | 2.39 | 0.149 | 0.00 | 0.00 |
| 32 | 5.24 | 0.164 | 0.00 | 0.00 |
| 64 | 10.16 | 0.159 | 0.00 | 0.00 |
| 128 | 20.14 | 0.157 | 0.00 | 0.00 |

### TFLite Dynamic
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.15 | 0.151 | 0.00 | 0.00 |
| 2 | 0.29 | 0.146 | 0.00 | 0.00 |
| 4 | 0.54 | 0.135 | 0.00 | 0.00 |
| 8 | 1.14 | 0.143 | 0.00 | 0.00 |
| 16 | 2.38 | 0.149 | 0.00 | 0.00 |
| 32 | 4.50 | 0.141 | 0.00 | 0.00 |
| 64 | 10.20 | 0.159 | 0.00 | 0.00 |
| 128 | 19.91 | 0.156 | 0.00 | 0.00 |

### TFLite INT8
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.12 | 0.119 | 0.00 | 0.00 |
| 2 | 0.23 | 0.114 | 0.00 | 0.00 |
| 4 | 0.46 | 0.114 | 0.00 | 0.00 |
| 8 | 0.88 | 0.110 | 0.00 | 0.00 |
| 16 | 1.76 | 0.110 | 0.00 | 0.00 |
| 32 | 3.48 | 0.109 | 0.00 | 0.00 |
| 64 | 7.03 | 0.110 | 0.00 | 0.00 |
| 128 | 13.71 | 0.107 | 0.00 | 0.00 |

### ONNX (CPU)
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.43 | 0.426 | 0.00 | 0.00 |
| 2 | 0.61 | 0.304 | 0.00 | 0.00 |
| 4 | 0.76 | 0.189 | 0.00 | 0.00 |
| 8 | 0.99 | 0.123 | 0.00 | 0.00 |
| 16 | 1.60 | 0.100 | 0.00 | 0.00 |
| 32 | 2.97 | 0.093 | 0.00 | 0.00 |
| 64 | 4.55 | 0.071 | 0.00 | 0.00 |
| 128 | 9.67 | 0.076 | 0.00 | 0.00 |

### TensorRT (GPU)
| Batch Size | Batch Latency (ms) | Sample Latency (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.94 | 0.939 | 0.00 | 0.00 |
| 2 | 0.90 | 0.449 | 0.00 | 0.00 |
| 4 | 0.71 | 0.178 | 0.00 | 0.00 |
| 8 | 0.88 | 0.110 | 0.00 | 0.00 |
| 16 | 0.99 | 0.062 | 0.00 | 0.00 |
| 32 | 1.70 | 0.053 | 0.00 | 0.00 |
| 64 | 1.07 | 0.017 | 0.00 | 0.00 |
| 128 | 1.62 | 0.013 | 0.00 | 0.00 |
