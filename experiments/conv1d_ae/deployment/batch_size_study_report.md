# Edge Compute Batch Size Scaling Study Report

This report studies the impact of varying execution batch size on latency (per batch and per sample) and memory usage.

## Keras FP32 (Baseline)

| Batch Size | Batch Latency (ms) | Latency Per Sample (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 20.231 ms | 20.2306 ms | 0.000 MB | 0.000 MB |
| 2 | 19.620 ms | 9.8101 ms | 0.000 MB | 0.000 MB |
| 4 | 20.271 ms | 5.0678 ms | 0.000 MB | 0.000 MB |
| 8 | 21.069 ms | 2.6337 ms | 0.000 MB | 0.000 MB |
| 16 | 20.714 ms | 1.2946 ms | 0.000 MB | 0.000 MB |
| 32 | 20.617 ms | 0.6443 ms | 0.000 MB | 0.000 MB |
| 64 | 19.734 ms | 0.3083 ms | 0.000 MB | 0.000 MB |
| 128 | 20.473 ms | 0.1599 ms | 0.000 MB | 0.000 MB |

## TFLite FP16

| Batch Size | Batch Latency (ms) | Latency Per Sample (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.157 ms | 0.1567 ms | 0.000 MB | 0.000 MB |
| 2 | 0.315 ms | 0.1573 ms | 0.000 MB | 0.000 MB |
| 4 | 0.572 ms | 0.1431 ms | 0.000 MB | 0.000 MB |
| 8 | 1.060 ms | 0.1325 ms | 0.000 MB | 0.000 MB |
| 16 | 2.378 ms | 0.1486 ms | 0.000 MB | 0.000 MB |
| 32 | 4.662 ms | 0.1457 ms | 0.000 MB | 0.000 MB |
| 64 | 9.289 ms | 0.1451 ms | 0.000 MB | 0.000 MB |
| 128 | 18.632 ms | 0.1456 ms | 0.000 MB | 0.000 MB |

## TFLite Dynamic

| Batch Size | Batch Latency (ms) | Latency Per Sample (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.142 ms | 0.1420 ms | 0.000 MB | 0.000 MB |
| 2 | 0.304 ms | 0.1522 ms | 0.000 MB | 0.000 MB |
| 4 | 0.532 ms | 0.1330 ms | 0.000 MB | 0.000 MB |
| 8 | 1.040 ms | 0.1300 ms | 0.000 MB | 0.000 MB |
| 16 | 2.118 ms | 0.1324 ms | 0.000 MB | 0.000 MB |
| 32 | 5.125 ms | 0.1601 ms | 0.000 MB | 0.000 MB |
| 64 | 9.919 ms | 0.1550 ms | 0.000 MB | 0.000 MB |
| 128 | 19.561 ms | 0.1528 ms | 0.000 MB | 0.000 MB |

## TFLite INT8

| Batch Size | Batch Latency (ms) | Latency Per Sample (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.134 ms | 0.1341 ms | 0.000 MB | 0.000 MB |
| 2 | 0.221 ms | 0.1105 ms | 0.000 MB | 0.000 MB |
| 4 | 0.490 ms | 0.1226 ms | 0.000 MB | 0.000 MB |
| 8 | 0.924 ms | 0.1154 ms | 0.000 MB | 0.000 MB |
| 16 | 2.048 ms | 0.1280 ms | 0.000 MB | 0.000 MB |
| 32 | 3.437 ms | 0.1074 ms | 0.000 MB | 0.000 MB |
| 64 | 7.082 ms | 0.1106 ms | 0.000 MB | 0.000 MB |
| 128 | 14.675 ms | 0.1146 ms | 0.000 MB | 0.000 MB |

## ONNX (CPU)

| Batch Size | Batch Latency (ms) | Latency Per Sample (ms) | Net RAM (MB) | Net VRAM (MB) |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.408 ms | 0.4081 ms | 0.000 MB | 0.000 MB |
| 2 | 0.566 ms | 0.2828 ms | 0.000 MB | 0.000 MB |
| 4 | 0.729 ms | 0.1822 ms | 0.000 MB | 0.000 MB |
| 8 | 1.066 ms | 0.1333 ms | 0.000 MB | 0.000 MB |
| 16 | 2.254 ms | 0.1409 ms | 0.000 MB | 0.000 MB |
| 32 | 2.418 ms | 0.0756 ms | 0.000 MB | 0.000 MB |
| 64 | 4.604 ms | 0.0719 ms | 0.000 MB | 0.000 MB |
| 128 | 10.269 ms | 0.0802 ms | 0.000 MB | 0.000 MB |
