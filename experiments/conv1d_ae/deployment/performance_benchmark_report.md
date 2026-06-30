# Edge Compute Performance Benchmark (Timing & Memory)

This report profiles the execution speed and memory footprints across optimized model formats.

| Model Format | Mean Latency | Min / Max Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keras FP32 (Baseline)** | 20.618 ± 2.993 ms | 17.24 / 45.80 ms | 0.000 MB | 0.000 MB | 4184.898 MB | 2.671 MB |
| **TFLite FP16** | 0.156 ± 0.039 ms | 0.13 / 0.38 ms | 0.000 MB | 0.000 MB | 4186.523 MB | 2.671 MB |
| **TFLite Dynamic** | 0.153 ± 0.036 ms | 0.13 / 0.31 ms | 0.000 MB | 0.000 MB | 4186.816 MB | 2.671 MB |
| **TFLite INT8** | 0.112 ± 0.013 ms | 0.11 / 0.24 ms | 0.000 MB | 0.000 MB | 4187.230 MB | 2.671 MB |
| **ONNX (CPU)** | 0.420 ± 0.104 ms | 0.36 / 1.23 ms | 0.000 MB | 0.000 MB | 4189.105 MB | 2.671 MB |
| **TensorRT (GPU)** | 1.207 ± 0.809 ms | 0.54 / 4.66 ms | 0.000 MB | 0.000 MB | 4190.730 MB | 2.671 MB |