#!/bin/bash
# Evaluates all available trained models in the experiments directory

EXPERIMENTS_DIR="experiments"
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        *) shift ;;
    esac
done

if [ -n "$EXPERIMENT_NAME" ]; then
    EXPERIMENTS_DIR="$EXPERIMENTS_DIR/$EXPERIMENT_NAME"
fi

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Experiments directory not found: $EXPERIMENTS_DIR"
    exit 1
fi

echo "Evaluating all models in $EXPERIMENTS_DIR..."

for model_dir in "$EXPERIMENTS_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        
        # Check if the model has a model_config.json to confirm it is actually trained
        if [ ! -f "$model_dir/model_config.json" ]; then
            continue
        fi
        
        # In case we only want to evaluate if test_results are not already there (like in evaluate_all.py)
        # Uncomment below if that behavior is desired, although `src.evaluate` allows overwriting.
        if [ -f "$model_dir/test_results.csv" ]; then
            echo "Skipping $model_name (test_results.csv already exists)"
            continue
        fi

        echo "--------------------------------------------------------"
        echo "Evaluating Model: $model_name"
        echo "--------------------------------------------------------"
        
        # Python module invocation for src.evaluate
        uv run python -m src.evaluate --model-dir "${model_dir%/}"
        
        echo -e "\nDetailed results saved for $model_name."
    fi
done

echo "All evaluations complete!"
