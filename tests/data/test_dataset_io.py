import numpy as np

from src.data import manifest as mf
from src.data.component_ranges import load_ranges
from src.data.loader import DataLoader
from src.data.migrate import migrate_legacy_dataset

RANGES = load_ranges("data/buck/component_ranges.json")


def _write_txt(path, n=5):
    with open(path, "w") as f:
        f.write("Frequency amp phase\n")
        for i in range(n):
            f.write(f"{100.0*(i+1):.6e} {-1.0*i:.6e} {-2.0*i:.6e}\n")


def _legacy_dataset(tmp_path):
    """A directory of legacy pct-encoded files (no manifest)."""
    _write_txt(tmp_path / "Cout_-10.txt")   # 0.90x -> normal
    _write_txt(tmp_path / "Cout_-60.txt")   # 0.40x -> anomalous
    return tmp_path


# ---------------------------------------------------------------------------
# Backward compatibility: legacy pct filenames still load + label
# ---------------------------------------------------------------------------
def test_legacy_dataset_loads_without_manifest(tmp_path):
    _legacy_dataset(tmp_path)
    dl = DataLoader(data_dir=str(tmp_path), component_ranges=RANGES)
    dl.load(use_cache=False)
    # Labels derived from the parsed pct filename + ranges (no manifest present).
    fnames = [m.filename for m in dl.metadata]
    normal = {fnames[i] for i in dl.get_normal_indices()}
    anomaly = {fnames[i] for i in dl.get_anomaly_indices()}
    assert normal == {"Cout_-10.txt"}
    assert anomaly == {"Cout_-60.txt"}


# ---------------------------------------------------------------------------
# Migration: legacy -> manifest (non-destructive and rename)
# ---------------------------------------------------------------------------
def test_migrate_non_destructive(tmp_path):
    _legacy_dataset(tmp_path)
    n = migrate_legacy_dataset(tmp_path, ranges=RANGES, mode="grid")
    assert n == 2

    fields, rows = mf.read_manifest(tmp_path / "manifest_grid.csv")
    assert "Cout" in mf.component_columns(fields)
    by_name = {r["filename"]: r for r in rows}
    assert set(by_name) == {"Cout_-10.txt", "Cout_-60.txt"}  # names unchanged
    assert by_name["Cout_-60.txt"]["label"] == "anomalous"
    assert by_name["Cout_-10.txt"]["label"] == "normal"
    assert abs(float(by_name["Cout_-60.txt"]["Cout"]) - 0.40) < 1e-6

    # Files are untouched on disk.
    assert (tmp_path / "Cout_-10.txt").exists()

    # Re-running is idempotent (already-listed files are skipped).
    assert migrate_legacy_dataset(tmp_path, ranges=RANGES, mode="grid") == 0

    # Loader now uses the manifest labels.
    dl = DataLoader(data_dir=str(tmp_path), component_ranges=RANGES)
    dl.load(use_cache=False)
    assert all(m.label is not None for m in dl.metadata)


def test_migrate_rename_to_opaque_ids(tmp_path):
    _legacy_dataset(tmp_path)
    migrate_legacy_dataset(tmp_path, ranges=RANGES, mode="grid", rename=True)
    names = sorted(p.name for p in tmp_path.glob("*.txt"))
    assert names == ["grid_000000.txt", "grid_000001.txt"]
    _, rows = mf.read_manifest(tmp_path / "manifest_grid.csv")
    assert {r["filename"] for r in rows} == set(names)


# ---------------------------------------------------------------------------
# Multiple-dataset loading (concatenate several directories)
# ---------------------------------------------------------------------------
def test_multi_directory_loading(tmp_path):
    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    d1.mkdir()
    d2.mkdir()
    for d, prefix in ((d1, "lhs"), (d2, "grid")):
        _write_txt(d / f"{prefix}_000000.txt")
        _write_txt(d / f"{prefix}_000001.txt")
        with mf.ManifestWriter(d / mf.manifest_name(prefix), ["Cout"]) as w:
            w.append(f"{prefix}_000000.txt", mf.HEALTHY, "normal", 0, prefix, "", {"Cout": 1.0})
            w.append(f"{prefix}_000001.txt", mf.FAULT, "anomalous", 1, prefix, "", {"Cout": 0.4})

    dl = DataLoader(data_dir=[str(d1), str(d2)], component_ranges=RANGES)
    dl.load(use_cache=False)
    assert dl.num_samples == 4
    assert len(dl.get_normal_indices()) == 2
    assert len(dl.get_anomaly_indices()) == 2
