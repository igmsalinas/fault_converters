#!/usr/bin/env bash
# ==============================================================================
# Run deployment optimization and quantization
# ==============================================================================

set -euo pipefail

# Default values
MODEL_DIR=""
EXPERIMENT_NAME=""
DATA_DIR="data/buck/buck_data"
VITIS_TARGET="pynq_z2"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        --vitis-target) VITIS_TARGET="$2"; shift 2 ;;
        --vitis-target=*) VITIS_TARGET="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_optimize() {
    local target_dir="$1"
    echo "--------------------------------------------------------"
    echo "Optimizing Model: $(basename "$target_dir")"
    echo "--------------------------------------------------------"
    uv run python -m src.deployment_optimization \
      --model-dir "$target_dir" \
      --data-dir "$DATA_DIR" \
      --vitis-target "$VITIS_TARGET"
}

echo "Starting Edge Compute Deployment Optimization Pipeline..."

if [ -n "$MODEL_DIR" ]; then
    run_optimize "$MODEL_DIR"
elif [ -n "$EXPERIMENT_NAME" ]; then
    EXP_DIR="experiments/$EXPERIMENT_NAME"
    if [ ! -d "$EXP_DIR" ]; then
        echo "Error: Experiment directory not found: $EXP_DIR"
        exit 1
    fi
    for m_dir in "$EXP_DIR"/*/; do
        if [[ -d "$m_dir" && -f "$m_dir/model_config.json" ]]; then
            run_optimize "${m_dir%/}"
        fi
    done
else
    # Default fallback to conv1d_ae
    run_optimize "experiments/conv1d_ae"
fi

echo "Optimization pipeline complete!"
