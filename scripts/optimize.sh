#!/bin/bash
# Run deployment optimization and quantization

echo "Starting Edge Compute Deployment Optimization Pipeline..."
python -m src.deployment_optimization \
  --model-dir experiments/conv1d_ae \
  --data-dir data/buck/buck_data \
  --vitis-target pynq_z2
