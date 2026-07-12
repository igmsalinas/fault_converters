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

## Data Pipeline

The dataset is a set of small-signal **control-to-output Bode responses** (amplitude
+ phase over a frequency sweep) of a power converter, each labelled `normal`,
`anomalous`, or `unknown` (gray zone). Everything is **converter-agnostic**: a new
topology only needs its own data folder — no code changes.

### Per-converter data folder

```
data/<converter>/
├── <converter>.psimsch          # PSIM schematic (simulation template)
├── parameters.txt               # nominal component values
├── component_ranges.json        # healthy + faulty multiplier bands (+ sampling steps)
└── <converter>_data/            # generated .txt Bode files + manifests
    ├── manifest_grid.csv
    ├── manifest_lhs.csv
    └── lhs_000000.txt, ...
```

### Component ranges (`component_ranges.json`)

Each component declares **multiplier bands** on its nominal value (`1.0` = nominal),
grounded in `docs/REFERENCES.md`:

- `normal`  — healthy envelope (manufacturing tolerance + temperature + ageing);
- `anomalous` — degradation band (or `null` for an operating-point knob, not a fault);
- `normal_step` / `anomalous_step` — optional additive sampling steps (fine for the
  narrow tolerance band, coarse for the wide fault band).

A sample is `anomalous` if **any** component is in/beyond its fault band, `normal`
if **all** are within their healthy band, else `unknown` (excluded from training/test).
The same bands drive the online synthetic fault injector used by CARLA.

### Generating data (PSIM)

```bash
# Estimate the run (no PSIM needed) — prints the per-component sampling plan
PYTHONPATH=. python data/generate_data.py --estimate

# Generate (on the PSIM host)
PYTHONPATH=. python data/generate_data.py --converter buck
```

- **Healthy set** (`--normal-mode lhs|random|grid`, default `lhs`) — Latin-hypercube
  sampling over the joint tolerance box.
- **Faulty set** (`--fault-mode lhs|grid`, default `lhs`) — each sample has a guaranteed
  primary fault plus independent secondary faults (`--fault-prob`), so **multiple
  simultaneous failures** occur; non-faulted components keep realistic tolerance spread.
- Counts: `--n-normal` (default 1000), `--n-fault` (default 300).
- **Real-time manifests**: each simulation is written as an opaque id (`lhs_000042.txt`)
  and its metadata (component multipliers + label) is appended to `manifest_*.csv`
  as it completes.
- **Resume**: re-running skips grid combinations already present and tops up LHS to the
  requested count.

### Synthetic (no PSIM) debug dataset

```bash
PYTHONPATH=. python scripts/generate_synthetic_buck.py \
    --out data/buck/buck_data_debug --n-normal 30 --n-anomaly 20
```
Fabricates a physically-consistent buck dataset from the analytic model so the full
pipeline can run end-to-end without PSIM.

### Loading, migration & multiple datasets

- **Labels come from the manifest** when present; legacy percentage-encoded filenames
  (`Cout_-20__Rds_1_-5.txt`) still load via a backward-compatible fallback.
- **Migrate a legacy dataset** to the manifest format (non-destructive by default):
  ```bash
  python -m src.data.migrate --data-dir data/buck/buck_data          # add manifest
  python -m src.data.migrate --data-dir data/buck/buck_data --rename # + opaque ids
  ```
- **Combine multiple dataset directories** (concatenated) anywhere `--data-dir` is used:
  ```bash
  python -m src.train --model carla --data-dir data/buck/run_lhs data/buck/run_grid
  ```

For the full data-model, sampling theory, manifest schema, and module APIs see
[`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md).

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
JSON/Markdown reports) inside `<model-dir>/deployment/`, aggregating three dimensions.
Every latency is measured under a uniform protocol — single-sample batch (n=1),
15 warm-up iterations, 150 timed runs — so all backends are directly comparable:

1. **Inference Fidelity & Baseline Latency** — model size, single-sample (n=1) latency, and reconstruction MSE shift per format.
2. **Hardware Resource Profiling** — mean/min/max latency plus net and peak RAM/VRAM usage.
3. **Classification Degradation** — accuracy, precision, recall, F1, and AUC-ROC to quantify quantization impact.

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

- **[`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md)**: Full technical reference — architecture, data pipeline, model zoo, training/evaluation/deployment internals, CLI reference, and how to add a new converter.
- **`docs/CONVERTER_PARAMETERS.md`**: Physical definitions of the Buck converter components and normal operating thresholds.
- **`docs/REFERENCES.md`**: Reference library and papers (incl. component tolerance / degradation ranges).
- **`docs/FUTURE_RESEARCH.md`**: Academic context and immediate next steps.
