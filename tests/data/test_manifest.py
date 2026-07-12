import numpy as np

from src.data import manifest as mf
from src.data.component_ranges import load_ranges
from src.data.loader import DataLoader


def _write_txt(path, n=5):
    with open(path, "w") as f:
        f.write("Frequency amp phase\n")
        for i in range(n):
            f.write(f"{100.0*(i+1):.6e} {-1.0*i:.6e} {-2.0*i:.6e}\n")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def test_make_key_is_deterministic():
    cols = ["Cout", "Esr_C"]
    k1 = mf.make_key({"Cout": 0.5, "Esr_C": 1.0}, cols)
    k2 = mf.make_key({"Esr_C": 1.0, "Cout": 0.5}, cols)
    assert k1 == k2
    assert mf.make_key({"Cout": 0.5}, cols) != mf.make_key({"Cout": 0.6}, cols)


def test_manifest_name_routing():
    assert mf.manifest_name("grid") == "manifest_grid.csv"
    assert mf.manifest_name("lhs") == "manifest_lhs.csv"
    assert mf.manifest_name("random") == "manifest_lhs.csv"


def test_writer_roundtrip_and_helpers(tmp_path):
    path = tmp_path / "manifest_grid.csv"
    w = mf.ManifestWriter(path, ["Cout", "Esr_C"])
    w.append("grid_000000.txt", mf.HEALTHY, "normal", 0, "grid", "k0",
             {"Cout": 1.0, "Esr_C": 1.0})
    w.append("grid_000001.txt", mf.FAULT, "anomalous", 1, "grid", "k1",
             {"Cout": 0.5, "Esr_C": 1.0})
    w.close()

    fields, rows = mf.read_manifest(path)
    assert fields == mf.BASE_FIELDS + ["Cout", "Esr_C"]
    assert len(rows) == 2
    assert mf.component_columns(fields) == ["Cout", "Esr_C"]

    assert mf.existing_keys(rows) == {"k0", "k1"}
    assert mf.existing_keys(rows, mf.FAULT) == {"k1"}
    assert mf.count_set(rows, mf.HEALTHY) == 1
    assert mf.count_set(rows, mf.FAULT) == 1
    assert mf.next_index(rows) == 2

    index = mf.load_manifest_index(tmp_path)
    assert index["grid_000001.txt"]["label"] == "anomalous"
    assert index["grid_000001.txt"]["multipliers"]["Cout"] == 0.5


def test_writer_append_resumes_index(tmp_path):
    path = tmp_path / "manifest_lhs.csv"
    with mf.ManifestWriter(path, ["Cout"]) as w:
        w.append("lhs_000000.txt", mf.HEALTHY, "normal", 0, "lhs", "", {"Cout": 1.0})
    _, rows = mf.read_manifest(path)
    assert mf.next_index(rows) == 1
    # Re-open and append: header is reused, no duplicate header row.
    with mf.ManifestWriter(path, ["Cout"]) as w:
        w.append("lhs_000001.txt", mf.FAULT, "anomalous", 1, "lhs", "", {"Cout": 0.4})
    _, rows = mf.read_manifest(path)
    assert [r["filename"] for r in rows] == ["lhs_000000.txt", "lhs_000001.txt"]


# ---------------------------------------------------------------------------
# Loader reads labels from the manifest
# ---------------------------------------------------------------------------
def test_loader_uses_manifest_labels(tmp_path):
    _write_txt(tmp_path / "lhs_000000.txt")
    _write_txt(tmp_path / "lhs_000001.txt")
    with mf.ManifestWriter(tmp_path / "manifest_lhs.csv", ["Cout", "Esr_C"]) as w:
        w.append("lhs_000000.txt", mf.HEALTHY, "normal", 0, "lhs", "",
                 {"Cout": 1.0, "Esr_C": 1.0})
        w.append("lhs_000001.txt", mf.FAULT, "anomalous", 1, "lhs", "",
                 {"Cout": 0.4, "Esr_C": 1.0})

    dl = DataLoader(data_dir=str(tmp_path))
    dl.load(use_cache=False)
    assert dl.get_normal_indices().tolist() == [0]
    assert dl.get_anomaly_indices().tolist() == [1]
    # Metadata carries the manifest label and reconstructed pct variations.
    meta = dl.metadata[1]
    assert meta.label == "anomalous"
    assert meta.variations["Cout"] == -60.0  # 0.4x -> -60%


# ---------------------------------------------------------------------------
# Generator planning / resume (no PSIM needed)
# ---------------------------------------------------------------------------
def _rows_from_plan(rows_by_file):
    """Turn planned rows into manifest-style dict rows (as read_manifest yields)."""
    out = []
    for fname, r in rows_by_file.items():
        out.append({
            "filename": fname, "set": r["set_name"], "label": r["label"],
            "n_faults": str(r["n_faults"]), "mode": r["mode"], "key": r["key"],
        })
    return out


def test_plan_generation_grid_resume():
    from data.generate_data import plan_generation
    comps = ["Cout"]
    nominal = {"Cout": 100e-6}
    ranges = load_ranges("data/buck/component_ranges.json")
    normal_combos = [{"Cout": 1.0}, {"Cout": 1.1}]
    fault_combos = [{"Cout": 0.5}]

    tasks, rbf = plan_generation(normal_combos, fault_combos, comps, nominal, ranges,
                                 "grid", "grid", [], [])
    assert len(tasks) == 3  # 2 healthy + 1 fault, all new
    assert {t[0] for t in tasks} == set(rbf)

    grid_rows = _rows_from_plan(rbf)
    tasks2, _ = plan_generation(normal_combos, fault_combos, comps, nominal, ranges,
                                "grid", "grid", grid_rows, [])
    assert len(tasks2) == 0  # everything already present -> resume skips all


def test_plan_generation_lhs_topup():
    from data.generate_data import plan_generation
    comps = ["Cout"]
    nominal = {"Cout": 100e-6}
    ranges = load_ranges("data/buck/component_ranges.json")
    normal_combos = [{"Cout": 1.0}] * 5   # request 5 healthy
    fault_combos = [{"Cout": 0.5}] * 3    # request 3 fault
    lhs_rows = [
        {"filename": "lhs_000000.txt", "set": "healthy", "key": ""},
        {"filename": "lhs_000001.txt", "set": "healthy", "key": ""},
        {"filename": "lhs_000002.txt", "set": "fault", "key": ""},
    ]
    tasks, rbf = plan_generation(normal_combos, fault_combos, comps, nominal, ranges,
                                 "lhs", "lhs", [], lhs_rows)
    # top-up: (5-2) healthy + (3-1) fault = 5 new; ids continue from index 3
    assert len(tasks) == 5
    assert min(rbf) == "lhs_000003.txt"

