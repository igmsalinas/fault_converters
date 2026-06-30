#!/bin/bash
# Run edge compute inference evaluation and benchmarking

echo "Starting Edge Compute Evaluation Pipeline..."
python -m src.deployment_evaluation \
  --model-dir experiments/conv1d_ae \
  --data-dir data/buck/buck_data
