@echo off
REM Run all deployment optimization, quantization, and benchmark pipelines

echo Starting Edge Compute Deployment Pipeline...
python -m src.deploy --model-dir experiments/conv1d_ae --data-dir data/buck/buck_data --vitis-target pynq_z2
