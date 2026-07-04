"""
Standalone script to run Keras benchmarking on CPU only.
It enforces CUDA_VISIBLE_DEVICES="-1" before loading any TensorFlow components.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Enforce CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Now we can import the rest safely
import numpy as np
from src.inference.predictor import AnomalyPredictor
from src.deployment.timing_memory_benchmark import benchmark_timing_memory
from src.deployment.runners import create_keras_runner
from src.deployment.timing_evaluation import evaluate_keras_model
from src.deployment.utils import get_file_size_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    # Load Model
    predictor = AnomalyPredictor(model_dir=args.model_dir)
    keras_model = predictor.model.autoencoder
    threshold = predictor.threshold if predictor.threshold is not None else 0.001

    # Load Data
    test_data = np.load(args.data_file)
    test_labels = np.load(args.labels_file)

    runner = create_keras_runner(keras_model)

    # 1. Timing Benchmark
    print("Running Timing Benchmark on CPU...")
    # Multi-sample inference test (mean/std) — must match the batch used by
    # run_full_performance_suite so CPU and GPU latencies are comparable.
    x_perf = test_data[:min(2000, len(test_data))]
    bench_res = benchmark_timing_memory("Keras FP32 (CPU)", runner, x_perf)
    bench_res["size_mb"] = get_file_size_mb(str(Path(args.model_dir) / "best_model.weights.h5"))

    # 2. Evaluation Benchmark
    print("Running Evaluation Benchmark on CPU...")
    eval_res = evaluate_keras_model(keras_model, test_data, test_labels, threshold)
    eval_res["model_name"] = "Keras FP32 (CPU)"
    eval_res["size_mb"] = bench_res["size_mb"]

    # Save outputs
    out_data = {
        "benchmark": bench_res,
        "evaluation": eval_res
    }
    with open(args.output_file, "w") as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()
