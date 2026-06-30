#!/usr/bin/env bash
# ==============================================================================
# Train All Models
# ==============================================================================
#
# Consecutively trains all model architectures with default HP search settings.
#
# Usage:
#   bash scripts/train_all.sh
#   bash scripts/train_all.sh --debug          # Quick debug run
#   bash scripts/train_all.sh --dry-run        # Print commands without executing
#
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Configuration ----
MODELS=("conv1d_ae" "lstm_ae" "gru_ae" "mlp_ae" "vae" "transformer_ae" "carla")
DATA_DIR="data/buck/buck_data"
OUTPUT_DIR="experiments"
N_TRIALS=30
EPOCHS_PER_TRIAL=30
FINAL_EPOCHS=100
SEED=42

# ---- Parse script-level flags ----
DEBUG=false
DRY_RUN=false
OVERWRITE=false

EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

# ---- Banner ----
echo "============================================================"
echo "  Train All Models — Hyperparameter Search Pipeline"
echo "============================================================"
echo "  Models:          ${MODELS[*]}"
echo "  Data dir:        $DATA_DIR"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Trials:          $N_TRIALS"
echo "  Epochs/trial:    $EPOCHS_PER_TRIAL"
echo "  Final epochs:    $FINAL_EPOCHS"
echo "  Debug mode:      $DEBUG"
echo "  Overwrite:       $OVERWRITE"
echo "  Experiment:      ${EXPERIMENT_NAME:-default}"
echo "============================================================"
echo ""

FAILED=()
SUCCEEDED=()

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Training: $MODEL"
    echo "============================================================"

    if $DEBUG; then
        CURRENT_OUTPUT_DIR="tmp/experiments"
    else
        CURRENT_OUTPUT_DIR="$OUTPUT_DIR"
    fi

    if [ -n "$EXPERIMENT_NAME" ]; then
        CURRENT_OUTPUT_DIR="$CURRENT_OUTPUT_DIR/$EXPERIMENT_NAME"
    fi

    if ! $OVERWRITE && [[ -f "$CURRENT_OUTPUT_DIR/$MODEL/model_config.json" ]]; then
        echo "  Model '$MODEL' is already fully trained (model_config.json exists). Skipping."
        echo "  Pass --overwrite to restart training and hyperparameter search."
        SUCCEEDED+=("$MODEL")
        continue
    fi

    CMD=(
        uv run python -m src.train
        --model "$MODEL"
        --data-dir "$DATA_DIR"
        --output-dir "$OUTPUT_DIR"
        --n-trials "$N_TRIALS"
        --batch-size 512
        --epochs-per-trial "$EPOCHS_PER_TRIAL"
        --final-epochs "$FINAL_EPOCHS"
        --seed "$SEED"
    )

    if $DEBUG; then
        CMD+=(--debug)
    fi

    if $OVERWRITE; then
        CMD+=(--overwrite)
    fi

    if [ -n "$EXPERIMENT_NAME" ]; then
        CMD+=(--experiment-name "$EXPERIMENT_NAME")
    fi

    echo "  Command: ${CMD[*]}"
    echo ""

    if $DRY_RUN; then
        echo "  [DRY RUN] Skipping execution."
        SUCCEEDED+=("$MODEL")
        continue
    fi

    if "${CMD[@]}"; then
        echo ""
        echo "  ✓ $MODEL completed successfully."
        SUCCEEDED+=("$MODEL")
    else
        echo ""
        echo "  ✗ $MODEL failed (exit code $?)."
        FAILED+=("$MODEL")
    fi
done

# ---- Summary ----
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]:-none}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed    (${#FAILED[@]}): ${FAILED[*]}"
    echo "============================================================"
    exit 1
else
    echo "  Failed    (0): none"
    echo "============================================================"
    echo "  All models trained successfully!"
    exit 0
fi
