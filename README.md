# Buck Converter Anomaly Detection

Machine learning project for anomaly detection on simulated transfer functions of Buck Power Converters using autoencoder architectures and contrastive learning (CARLA).

Extended domain-specific knowledge and physics-based parameters can be found in the `docs/` folder.

## Key Features

- **6 Model Architectures**: Conv1D-AE, LSTM-AE, VAE, Transformer-AE, CARLA-Conv1D, CARLA-LSTM
- **CARLA Contrastive Learning**: State-of-the-art self-supervised anomaly detection
- **Hyperparameter Optimization**: Bayesian, Random, and Grid search methods native integrations

## Setup & Installation

This project utilizes [uv](https://github.com/astral-sh/uv) to ensure reproducible and lightning-fast dependency management.

```bash
# Clone the repository
git clone <repository_url>
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

### Train a basic model:
```bash
python examples/train_anomaly_detector.py --model conv1d_ae --epochs 20 --data-dir data/simulation_results
```

### Contrastive Learning (CARLA):
```bash
python examples/train_carla.py --encoder-type conv1d --epochs 20 --temperature 0.1 --anomaly-ratio 0.5
```

### Automated Training Pipelines:
If you want to run the full training regimes across all architectures:
```bash
bash scripts/train_all.sh
```

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
- **`REFERENCES.md`**: Reference library and papers.
