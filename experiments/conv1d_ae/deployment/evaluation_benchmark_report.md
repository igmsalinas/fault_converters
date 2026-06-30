# Edge Compute Deployment Evaluation Benchmark Report

This report studies the hardware deployment optimizations across size, latency, and classification metrics.

| Model Format | Size (MB) | Latency Per Sample (ms) | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 3.420 MB | 0.0293 ms | 0.9586 | 0.9774 | 0.9572 | 0.9672 | 0.9856 |
| **TFLite FP16** | 0.578 MB | 0.1522 ms | 0.9576 | 0.9731 | 0.9599 | 0.9665 | 0.9854 |
| **TFLite Dynamic** | 0.349 MB | 0.1463 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.9446 |
| **TFLite INT8** | 0.388 MB | 0.1170 ms | 0.6371 | 0.6371 | 1.0000 | 0.7783 | 0.3213 |
| **ONNX (CPU)** | 1.129 MB | 0.1117 ms | 0.9600 | 0.9801 | 0.9568 | 0.9683 | 0.9861 |