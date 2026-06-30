import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import keras
import shutil

from src.models.conv1d_ae import Conv1DAutoencoder
from src.deployment.keras_quantization import quantize_keras_native, convert_to_tflite
from src.deployment.onnx_trt import convert_keras_to_onnx, compile_onnx_to_tensorrt
from src.deployment.vitis_ai import quantize_for_vitis_ai
from src.deployment.utils import get_file_size_mb
from src.deployment.runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
    run_inference_loop,
)
from src.deployment.benchmark import (
    benchmark_keras_model,
    benchmark_tflite_model,
    benchmark_onnx_model,
    benchmark_tensorrt_model,
    run_deployment_benchmarks,
)
from src.deployment.timing_memory_benchmark import (
    run_full_performance_suite,
    save_performance_report,
    run_batch_size_study,
    save_batch_size_report,
)
from src.deployment.timing_evaluation import (
    run_deployment_evaluations,
    save_evaluation_report,
)


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_model(sequence_length, n_features):
    ae = Conv1DAutoencoder(
        input_shape=(sequence_length, n_features),
        latent_dim=4,
        filters=[8],
        kernel_size=3,
    ).build()
    return ae


@pytest.fixture
def model_dir_with_config(sample_model, temp_dir):
    """Save a minimal model directory with config and weights (shared fixture).

    This eliminates the copy-paste of model_config.json + save_weights
    across multiple test functions.
    """
    with open(temp_dir / "model_config.json", "w") as f:
        json.dump({
            "name": "conv1d_ae",
            "type": "standard",
            "input_shape": [24, 5],
            "latent_dim": 4,
            "filters_idx": 0,
            "kernel_size": 3,
        }, f)
    sample_model.autoencoder.save_weights(str(temp_dir / "best_model.weights.h5"))
    return temp_dir


@pytest.fixture
def deployment_dir_with_tflite(sample_model, model_dir_with_config):
    """Save a TFLite dynamic model alongside the model dir for benchmark tests."""
    path_dynamic = str(model_dir_with_config / "model_dynamic.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="dynamic",
        output_path=path_dynamic,
    )
    return model_dir_with_config


# ---------------------------------------------------------------------------
# Component 1: Keras & TFLite Quantization
# ---------------------------------------------------------------------------

def test_keras_and_tflite_quantization(sample_model, mock_data, temp_dir):
    # Test Keras-native quantization
    quant_keras = quantize_keras_native(
        sample_model.autoencoder,
        mode="int8",
        output_dir=str(temp_dir),
    )
    assert isinstance(quant_keras, keras.Model)
    assert (temp_dir / "keras_quantized_int8.weights.h5").exists()

    # Test TFLite conversions
    path_dynamic = str(temp_dir / "model_dynamic.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="dynamic",
        output_path=path_dynamic,
    )
    assert Path(path_dynamic).exists()
    assert Path(path_dynamic).stat().st_size > 0

    path_fp16 = str(temp_dir / "model_fp16.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="float16",
        output_path=path_fp16,
    )
    assert Path(path_fp16).exists()

    # Full INT8 calibration conversion
    path_int8 = str(temp_dir / "model_int8.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="int8",
        calibration_data=mock_data,
        output_path=path_int8,
    )
    assert Path(path_int8).exists()


# ---------------------------------------------------------------------------
# Component 2: ONNX Conversion
# ---------------------------------------------------------------------------

def test_onnx_conversion(sample_model, temp_dir):
    path_onnx = str(temp_dir / "model.onnx")
    success = convert_keras_to_onnx(sample_model.autoencoder, path_onnx)

    # Try importing tf2onnx; if import fails, success is False but execution is safe
    try:
        import tf2onnx
        assert success is True
        assert Path(path_onnx).exists()
    except ImportError:
        assert success is False


# ---------------------------------------------------------------------------
# Component 3: ONNX to TensorRT
# ---------------------------------------------------------------------------

def test_onnx_to_tensorrt_compilation(sample_model, temp_dir):
    # Convert Keras model to ONNX
    path_onnx = str(temp_dir / "model.onnx")
    convert_keras_to_onnx(sample_model.autoencoder, path_onnx)

    # Attempt to compile ONNX to TensorRT
    path_engine = str(temp_dir / "model.engine")
    success = compile_onnx_to_tensorrt(
        onnx_path=path_onnx,
        engine_path=path_engine,
        precision_mode="FP16",
    )
    # The return value should be a boolean (either True on successful build or False if TRT not installed/supported)
    assert isinstance(success, bool)


# ---------------------------------------------------------------------------
# Component 4: Vitis AI Targets
# ---------------------------------------------------------------------------

def test_vitis_ai_targets(sample_model, mock_data, temp_dir):
    # Verify Vitis AI quantizer fallback handles environment issues gracefully
    # and generates matching configs for different target boards.
    for target in ["kv260", "pynq_z2"]:
        target_dir = temp_dir / target
        success = quantize_for_vitis_ai(
            model=sample_model.autoencoder,
            calibration_data=mock_data,
            output_dir=str(target_dir),
            target_hardware=target,
        )
        assert isinstance(success, bool)

        # Verify configuration files were generated
        assert (target_dir / f"{target}_arch.json").exists()
        assert (target_dir / "deploy_pynq.py").exists()

        # Read arch.json to verify target parameters
        with open(target_dir / f"{target}_arch.json", "r") as f:
            arch_data = json.load(f)
            assert "target" in arch_data
            assert "cpu_arch" in arch_data
            if target == "pynq_z2":
                assert arch_data["cpu_arch"] == "arm32"
            else:
                assert arch_data["cpu_arch"] == "arm64"


# ---------------------------------------------------------------------------
# Component 5: Shared Utilities
# ---------------------------------------------------------------------------

def test_get_file_size_mb(temp_dir):
    # Test file size
    test_file = temp_dir / "test.bin"
    test_file.write_bytes(b"\x00" * 1024)  # 1KB
    size = get_file_size_mb(str(test_file))
    assert abs(size - 1.0 / 1024) < 0.001

    # Test nonexistent path
    size = get_file_size_mb(str(temp_dir / "nonexistent"))
    assert size == 0.0

    # Test directory size
    size = get_file_size_mb(str(temp_dir))
    assert size > 0.0


# ---------------------------------------------------------------------------
# Component 6: Inference Runners
# ---------------------------------------------------------------------------

def test_keras_runner(sample_model, mock_data):
    runner = create_keras_runner(sample_model.autoencoder)
    output = runner(mock_data)
    assert output.shape == mock_data.shape


def test_tflite_runner(sample_model, mock_data, temp_dir):
    path = str(temp_dir / "model_dynamic.tflite")
    convert_to_tflite(sample_model.autoencoder, "dynamic", output_path=path)

    runner = create_tflite_runner(path)
    # Test single sample
    single = np.expand_dims(mock_data[0], axis=0)
    output = runner(single)
    assert output.shape == single.shape

    # Test batch (dynamic resize)
    output_batch = runner(mock_data)
    assert output_batch.shape == mock_data.shape


def test_onnx_runner(sample_model, mock_data, temp_dir):
    path = str(temp_dir / "model.onnx")
    success = convert_keras_to_onnx(sample_model.autoencoder, path)
    if not success:
        pytest.skip("tf2onnx not installed")

    runner = create_onnx_runner(path)
    assert runner is not None
    output = runner(mock_data)
    assert output.shape == mock_data.shape


def test_tensorrt_runner(sample_model, temp_dir, mock_data):
    """Test TRT runner creation — handles multiple unavailability scenarios.

    TensorRT engine compilation (needs `tensorrt`) and runner creation
    (additionally needs `cuda-python` or `pycuda`) can fail independently.
    """
    path_onnx = str(temp_dir / "model.onnx")
    success = convert_keras_to_onnx(sample_model.autoencoder, path_onnx)
    if not success:
        pytest.skip("tf2onnx not installed")

    path_engine = str(temp_dir / "model.engine")
    trt_success = compile_onnx_to_tensorrt(path_onnx, path_engine, "FP16")

    if not trt_success:
        # TensorRT package not available — runner should also fail gracefully
        runner = create_tensorrt_runner(path_engine)
        assert runner is None
        return

    # Engine compiled successfully — runner may or may not work
    # depending on whether cuda-python/pycuda is installed
    runner = create_tensorrt_runner(path_engine)
    if runner is None:
        # TRT compiled but CUDA memory packages not available — acceptable
        pytest.skip("TensorRT engine compiled but cuda-python/pycuda not installed for runtime")
    else:
        # Full TRT stack available — verify runner produces correct output
        output = runner(mock_data)
        assert output.shape == mock_data.shape


def test_run_inference_loop(sample_model, mock_data):
    runner = create_keras_runner(sample_model.autoencoder)
    output = run_inference_loop(runner, mock_data, batch_size=2)
    assert output.shape == mock_data.shape


# ---------------------------------------------------------------------------
# Component 7: Benchmarking Suite
# ---------------------------------------------------------------------------

def test_benchmarking_runners(sample_model, mock_data, temp_dir):
    # Benchmark baseline
    res_keras = benchmark_keras_model(sample_model.autoencoder, mock_data, num_warmup=2, num_runs=5)
    assert "latency_bs1_mean" in res_keras
    assert "mse" in res_keras
    assert res_keras["mse"] >= 0.0

    # Benchmark TFLite
    path_dynamic = str(temp_dir / "model_dynamic.tflite")
    convert_to_tflite(
        sample_model.autoencoder,
        optimization_mode="dynamic",
        output_path=path_dynamic,
    )

    res_tflite = benchmark_tflite_model(path_dynamic, mock_data, num_warmup=2, num_runs=5)
    assert "latency_bs1_mean" in res_tflite
    assert "mse" in res_tflite
    assert res_tflite["mse"] >= 0.0


def test_tensorrt_benchmark_graceful(temp_dir):
    """TRT benchmark returns empty dict when engine doesn't exist."""
    fake_engine = str(temp_dir / "nonexistent.engine")
    res = benchmark_tensorrt_model(fake_engine, np.zeros((4, 24, 5), dtype=np.float32))
    assert res == {}


# ---------------------------------------------------------------------------
# Component 8: Performance Suite (Timing & Memory)
# ---------------------------------------------------------------------------

def test_timing_memory_suite(mock_data, deployment_dir_with_tflite):
    res = run_full_performance_suite(
        model_dir=str(deployment_dir_with_tflite),
        deployment_dir=str(deployment_dir_with_tflite),
        test_data=mock_data,
    )
    assert len(res) >= 1
    # Check that baseline has results
    assert res[0]["model_name"] == "Keras FP32 (Baseline)"
    assert "latency_mean_ms" in res[0]

    # Save results
    report_path = str(deployment_dir_with_tflite / "performance_benchmark_report")
    save_performance_report(res, report_path)

    # Assert reports exist
    assert Path(report_path + ".json").exists()


# ---------------------------------------------------------------------------
# Component 9: Classification Evaluation
# ---------------------------------------------------------------------------

def test_timing_evaluation_suite(mock_data, deployment_dir_with_tflite):
    mock_labels = np.random.choice([0, 1], size=(mock_data.shape[0],))

    res = run_deployment_evaluations(
        model_dir=str(deployment_dir_with_tflite),
        deployment_dir=str(deployment_dir_with_tflite),
        test_data=mock_data,
        test_labels=mock_labels,
    )
    assert len(res) >= 1
    # Check that baseline has results
    assert res[0]["model_name"] == "Keras FP32 (Baseline)"
    assert "latency_per_sample_ms" in res[0]
    assert "f1" in res[0]

    # Save results
    report_path = str(deployment_dir_with_tflite / "evaluation_benchmark_report")
    save_evaluation_report(res, report_path)

    # Assert reports exist
    assert Path(report_path + ".json").exists()


# ---------------------------------------------------------------------------
# Component 10: Batch Size Scaling Study
# ---------------------------------------------------------------------------

def test_batch_size_study(mock_data, deployment_dir_with_tflite):
    res = run_batch_size_study(
        model_dir=str(deployment_dir_with_tflite),
        deployment_dir=str(deployment_dir_with_tflite),
        test_data=mock_data,
        batch_sizes=[1, 2],
    )
    assert len(res) >= 1
    assert "Keras FP32 (Baseline)" in res
    assert 1 in res["Keras FP32 (Baseline)"]
    assert 2 in res["Keras FP32 (Baseline)"]
    assert "latency_batch_ms" in res["Keras FP32 (Baseline)"][1]

    # Save results
    report_path = str(deployment_dir_with_tflite / "batch_size_study_report")
    save_batch_size_report(res, report_path)

    # Assert reports exist
    assert Path(report_path + ".json").exists()
