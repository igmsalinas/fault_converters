# Buck Converter Anomaly Detection

Machine learning project for anomaly detection on simulated transfer functions of Buck Power Converters using autoencoder architectures and contrastive learning (CARLA).

Extended domain-specific knowledge and physics-based parameters can be found in the `docs/` folder.

## Key Features

- **6 Autoencoder Architectures**: Conv1D-AE, LSTM-AE, GRU-AE, MLP-AE, VAE, Transformer-AE
- **CARLA Contrastive Learning**: State-of-the-art self-supervised anomaly detection (Conv1D, Transformer, and MLP encoders)
- **Hyperparameter Optimization**: Bayesian, Random, Grid, and Hyperband search methods native integrations
- **Edge Deployment**: Quantization and compilation to TFLite, ONNX, TensorRT, and Xilinx Vitis AI with full latency/memory/fidelity benchmarking

## Setup & Installation

This project utilizes [uv](https://github.com/astral-sh/uv) to ensure reproducible and lightning-fast dependency management.

```bash
# Clone the repository
git clone https://github.com/igmsalinas/fault_converters.git
cd fault_converters

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create a virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

## Quick Start (Playing with the Repo)

Training runs a hyperparameter search followed by a final training run. Supported
models are `conv1d_ae`, `lstm_ae`, `gru_ae`, `mlp_ae`, `vae`, `transformer_ae`, and `carla`.

### Train a basic model:
```bash
python -m src.train --model conv1d_ae --n-trials 15 --final-epochs 100 --data-dir data/buck/buck_data
```

### Contrastive Learning (CARLA):
```bash
python -m src.train --model carla --n-trials 10 --data-dir data/buck/buck_data
```

### Quick debug run:
```bash
python -m src.train --model conv1d_ae --debug
```

### Automated Training Pipelines:
If you want to run the full training regimes across all architectures:
```bash
bash scripts/train_all.sh
```

## Evaluation

Evaluate a single trained model and write metrics/thresholds to its experiment directory:
```bash
python -m src.evaluate --model-dir experiments/conv1d_ae
```

Evaluate every trained model under `experiments/`:
```bash
bash scripts/evaluate_all.sh
```

Aggregated results can be explored interactively in `notebooks/explore_results.ipynb`.

## Edge Deployment & Benchmarking

Trained autoencoders can be quantized, compiled, and benchmarked for edge compute
environments. The deployment toolkit lives in `src/deployment/` and is driven by two
entry points.

### Supported Target Formats

| Backend | Variants | Target Hardware |
| :--- | :--- | :--- |
| **Keras / TFLite** | Native INT8, Dynamic, Float16, INT8 | CPU / Mobile / MCU |
| **ONNX Runtime** | FP32, INT8 (QDQ) | CPU |
| **TensorRT** | FP32, FP16, INT8 | NVIDIA GPU |
| **Xilinx Vitis AI** | INT8 quantized + compiled | FPGA (`zcu102`, `zcu104`, `kv260`, `ultra96`, `pynq_z2`) |

### 1. Optimize & Quantize a Model

Compile and quantize a trained model into all available formats:
```bash
python -m src.deployment_optimization \
  --model-dir experiments/conv1d_ae \
  --data-dir data/buck/buck_data \
  --vitis-target pynq_z2 \
  --run-all
```

Or use the helper script:
```bash
bash scripts/optimize.sh
```

Converted artifacts (`.tflite`, `.onnx`, `.engine`, Vitis AI outputs) are written to
`<model-dir>/deployment/` by default.

### 2. Evaluate & Benchmark Deployed Models

Benchmark latency, memory footprint, and classification fidelity across every
converted format:
```bash
python -m src.deployment_evaluation \
  --model-dir experiments/conv1d_ae \
  --data-dir data/buck/buck_data
```

Or use the helper script:
```bash
bash scripts/evaluate_inference.sh
```

### Deployment Reports

Evaluation produces a consolidated `unified_deployment_report.md` (plus per-stage
JSON/Markdown reports) inside `<model-dir>/deployment/`, aggregating four dimensions:

1. **Inference Fidelity & Baseline Latency** — model size, batch-1/batch-32 latency, and reconstruction MSE shift per format.
2. **Hardware Resource Profiling** — mean/min/max latency plus net and peak RAM/VRAM usage.
3. **Classification Degradation** — accuracy, precision, recall, F1, and AUC-ROC to quantify quantization impact.
4. **Batch Size Scaling Dynamics** — latency and memory scaling across batch sizes (1–128).

## Development Guide

If you are developing new models or features, follow these guidelines.

### Code Quality & Formatting
We strictly enforce formatting standards. Run the following before committing:
```bash
# Auto-format code
uv run ruff format .
uv run isort .

# Run linter
uv run ruff check .
```

### Testing (TDD)
The project adheres to Test-Driven Development. Make sure to implement and pass unit tests located in `tests/`:
```bash
# Run all tests natively
uv run pytest

# Run with coverage reporting
uv run pytest --cov=src tests/
```

### Common Architectural Rules:
- **Separation of Concerns**: Data handling (`src/data`), model architectures (`src/models`), and training loops (`src/training`) are strictly separated. Do not mix Keras loops directly into dataset parsing files.
- **Base Classes**: All autoencoders must inherit from `src/models/base.py:BaseAutoencoder` for uniformity.
- **Keras Integration**: Datasets use `src/data/generator.py` for batch yielding natively decoupled from underlying filesystem I/O.

## Documentation Reference

- **`docs/CONVERTER_PARAMETERS.md`**: Physical definitions of the Buck converter components and normal operating thresholds.
- **`docs/FUTURE_RESEARCH.md`**: Academic context and immediate next steps.
- **`docs/REFERENCES.md`**: Reference library and papers.
