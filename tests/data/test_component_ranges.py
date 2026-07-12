import numpy as np

from src.data.component_ranges import (
    ANOMALOUS,
    NORMAL,
    UNKNOWN,
    classify_component,
    classify_variations,
    load_ranges,
)
from src.data.loader import DataLoader, SimulationMetadata

# Buck ranges are declared in the converter's data folder (converter-agnostic).
RANGES = load_ranges("data/buck/component_ranges.json")


# ---------------------------------------------------------------------------
# Per-component classification
# ---------------------------------------------------------------------------
def test_capacitance_bands():
    assert classify_component("Cout", -10, RANGES) == NORMAL      # 0.90x, within ±20%
    assert classify_component("Cout", +20, RANGES) == NORMAL      # 1.20x, edge
    assert classify_component("Cout", -50, RANGES) == ANOMALOUS   # 0.50x, in fault band
    assert classify_component("Cout", -80, RANGES) == ANOMALOUS   # 0.20x, beyond (worse)
    assert classify_component("Cout", -25, RANGES) == UNKNOWN     # 0.75x, gray gap
    assert classify_component("Cout", +50, RANGES) == UNKNOWN     # 1.50x, not a fault


def test_esr_bands_temperature_is_normal():
    assert classify_component("Esr_C", +80, RANGES) == NORMAL     # 1.8x, cold temp
    assert classify_component("Esr_C", -40, RANGES) == NORMAL     # 0.6x, hot temp
    assert classify_component("Esr_C", +400, RANGES) == ANOMALOUS  # 5x, wear-out
    assert classify_component("Esr_C", +900, RANGES) == ANOMALOUS  # 10x, beyond fault hi


def test_rds_temperature_normal_vs_degradation():
    assert classify_component("Rds_1", +100, RANGES) == NORMAL     # 2.0x from temperature
    assert classify_component("Rds_1", +150, RANGES) == UNKNOWN    # 2.5x, gray gap (2.2-3.0)
    assert classify_component("Rds_1", +400, RANGES) == ANOMALOUS  # 5x, degradation


def test_unknown_component_falls_back_to_flat_threshold():
    assert classify_component("Xyz", 3, RANGES, fallback_threshold=5) == NORMAL
    assert classify_component("Xyz", 10, RANGES, fallback_threshold=5) == ANOMALOUS


def test_no_ranges_falls_back_to_flat_threshold():
    # Converter without a declared spec still labels via the flat threshold.
    assert classify_component("Cout", 3, None, fallback_threshold=5) == NORMAL
    assert classify_component("Cout", 10, None, fallback_threshold=5) == ANOMALOUS


# ---------------------------------------------------------------------------
# Whole-sample classification
# ---------------------------------------------------------------------------
def test_sample_normal_when_all_components_healthy():
    assert classify_variations({"Cout": -15, "Lout": +10, "Rds_1": +80}, RANGES) == NORMAL


def test_sample_anomalous_when_any_component_faulted():
    # capacitance healthy, but ESR is at 5x -> fault dominates
    assert classify_variations({"Cout": -10, "Esr_C": +400}, RANGES) == ANOMALOUS


def test_sample_unknown_when_gray_and_no_fault():
    assert classify_variations({"Cout": -25}, RANGES) == UNKNOWN  # gray, no fault
    assert classify_variations({}, RANGES) == NORMAL              # nominal


# ---------------------------------------------------------------------------
# Metadata + loader integration
# ---------------------------------------------------------------------------
def test_metadata_classify():
    healthy = SimulationMetadata("f", {"Cout": -18, "Lout": 12})
    faulty = SimulationMetadata("f", {"Esr_C": 500})
    gray = SimulationMetadata("f", {"Lout": -25})
    assert healthy.classify(RANGES) == NORMAL
    assert faulty.classify(RANGES) == ANOMALOUS
    assert gray.classify(RANGES) == UNKNOWN


def _loader_with(metas):
    loader = DataLoader(data_dir=".", component_ranges=RANGES)
    loader._metadata = metas
    loader._data = np.zeros((len(metas), 4, 2), dtype=np.float32)
    loader._is_loaded = True
    return loader


def test_loader_range_labelling_excludes_gray():
    metas = [
        SimulationMetadata("a", {"Cout": -10}),      # normal
        SimulationMetadata("b", {"Cout": +15, "Lout": -12}),  # normal
        SimulationMetadata("c", {"Esr_C": 500}),     # anomalous
        SimulationMetadata("d", {"Rds_1": 400}),     # anomalous
        SimulationMetadata("e", {"Cout": -25}),      # gray -> excluded
    ]
    loader = _loader_with(metas)
    normal = set(loader.get_normal_indices().tolist())
    anomaly = set(loader.get_anomaly_indices().tolist())
    assert normal == {0, 1}
    assert anomaly == {2, 3}
    assert 4 not in normal and 4 not in anomaly  # gray excluded from both


def test_loader_legacy_flat_threshold():
    metas = [
        SimulationMetadata("a", {"Cout": -3}),   # normal under flat 5%
        SimulationMetadata("b", {"Cout": -15}),  # anomalous under flat 5%
    ]
    loader = DataLoader(data_dir=".", normal_threshold=5.0, use_component_ranges=False)
    loader._metadata = metas
    loader._data = np.zeros((2, 4, 2), dtype=np.float32)
    loader._is_loaded = True
    assert loader.get_normal_indices().tolist() == [0]
    assert loader.get_anomaly_indices().tolist() == [1]
