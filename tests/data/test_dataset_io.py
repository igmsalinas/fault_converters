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


def test_resolve_data_dir(tmp_path):
    from src.data.loader import resolve_data_dir
    
    # 1. Test dataset directory directly (has txts subdirectory)
    ds_dir = tmp_path / "dataset_A"
    txts_dir = ds_dir / "txts"
    txts_dir.mkdir(parents=True)
    with open(ds_dir / "manifest.csv", "w") as f:
        f.write("filename,set,label,n_faults,mode,key\n")
    assert resolve_data_dir(ds_dir) == ds_dir

    # 2. Test base directory containing dataset subfolders with txts/
    base_dir = tmp_path
    (base_dir / "dataset_B" / "txts").mkdir(parents=True)
    assert resolve_data_dir(base_dir) == base_dir / "dataset_B"

    # 3. Test fallback to legacy flat directory
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    assert resolve_data_dir(legacy_dir) == legacy_dir


def test_loader_loads_versioned_dataset(tmp_path):
    import json
    ds_dir = tmp_path / "my_dataset"
    txts_dir = ds_dir / "txts"
    txts_dir.mkdir(parents=True)
    
    _write_txt(txts_dir / "lhs_000000.txt")
    _write_txt(txts_dir / "lhs_000001.txt")
    
    with mf.ManifestWriter(ds_dir / "manifest.csv", ["Cout"]) as w:
        w.append("lhs_000000.txt", mf.HEALTHY, "normal", 0, "lhs", "", {"Cout": 1.0})
        w.append("lhs_000001.txt", mf.FAULT, "anomalous", 1, "lhs", "", {"Cout": 0.4})
        
    meta = {
        "converter": "buck",
        "dataset_name": "my_dataset",
        "seed": 42
    }
    with open(ds_dir / "dataset.json", "w") as f:
        json.dump(meta, f)
        
    # Load pointing directly to dataset folder
    dl = DataLoader(data_dir=str(ds_dir), component_ranges=RANGES)
    dl.load(use_cache=False)
    assert dl.num_samples == 2
    assert len(dl.get_normal_indices()) == 1
    assert len(dl.get_anomaly_indices()) == 1
    assert dl.frequencies is not None
    assert len(dl.frequencies) == 5
    
    # Load pointing to base folder (auto-resolves to my_dataset)
    dl_base = DataLoader(data_dir=str(tmp_path), component_ranges=RANGES)
    dl_base.load(use_cache=False)
    assert dl_base.num_samples == 2


def test_generate_data_creates_nested_structure(tmp_path, monkeypatch):
    import sys
    from unittest.mock import patch, MagicMock
    from data.generate_data import main as generate_main
    
    # Setup folders
    input_dir = tmp_path / "buck"
    input_dir.mkdir()
    
    # write mock parameters.txt
    with open(input_dir / "parameters.txt", "w") as f:
        f.write("Cout=100u\nL=10u\n")
        
    # write mock component_ranges.json
    import json
    ranges_data = {
        "Cout": {
            "normal": [0.9, 1.1],
            "anomalous": [0.3, 0.7]
        },
        "L": {
            "normal": [0.9, 1.1],
            "anomalous": [0.3, 0.7]
        }
    }
    with open(input_dir / "component_ranges.json", "w") as f:
        json.dump(ranges_data, f)
        
    # write mock psimsch file
    with open(input_dir / "buck.psimsch", "w") as f:
        f.write("mock psim schema")
        
    # Mock multiprocessing.Pool and run_simulation
    mock_pool = MagicMock()
    mock_pool.__enter__.return_value = mock_pool
    mock_pool.imap_unordered.return_value = ["lhs_000000.txt"]
    
    with patch("data.generate_data.Pool", return_value=mock_pool), \
         patch("data.generate_data.PSIM_PATH", ""), \
         patch("data.generate_data.run_simulation", return_value="lhs_000000.txt"):
         
        # Setup sys.argv
        test_args = [
            "generate_data.py",
            "--converter", "buck",
            "--input_dir", str(input_dir),
            "--output", str(tmp_path / "buck_data"),
            "--normal-mode", "lhs",
            "--n-normal", "1",
            "--mode", "normal",
            "--dataset-name", "test_ds"
        ]
        monkeypatch.setattr(sys, "argv", test_args)
        
        generate_main()
        
    # Verify the created structure
    dataset_dir = tmp_path / "buck_data" / "test_ds"
    assert dataset_dir.exists()
    assert (dataset_dir / "dataset.json").exists()
    assert (dataset_dir / "manifest.csv").exists()
    assert (dataset_dir / "txts").exists()
    
    # Read manifest.csv
    _, rows = mf.read_manifest(dataset_dir / "manifest.csv")
    assert len(rows) == 1
    assert rows[0]["filename"] == "lhs_000000.txt"
    
    # Read dataset.json
    with open(dataset_dir / "dataset.json", "r") as f:
        meta_loaded = json.load(f)
    assert meta_loaded["dataset_name"] == "test_ds"


def test_balanced_max_files_loading(tmp_path):
    txts_dir = tmp_path / "txts"
    txts_dir.mkdir()
    
    # Write 4 healthy and 4 faulty files
    for i in range(4):
        _write_txt(txts_dir / f"lhs_00000{i}.txt")
        _write_txt(txts_dir / f"lhs_01000{i}.txt")
        
    # Write a manifest.csv
    with mf.ManifestWriter(tmp_path / "manifest_lhs.csv", ["Cout"]) as w:
        for i in range(4):
            w.append(f"lhs_00000{i}.txt", mf.HEALTHY, "normal", 0, "lhs", "", {"Cout": 1.0})
            w.append(f"lhs_01000{i}.txt", mf.FAULT, "anomalous", 1, "lhs", "", {"Cout": 0.4})
            
    # Load with max_files=4
    dl = DataLoader(data_dir=str(tmp_path), component_ranges=RANGES)
    dl.load(max_files=4, use_cache=False)
    
    assert dl.num_samples == 4
    normal_indices = dl.get_normal_indices()
    anomaly_indices = dl.get_anomaly_indices()
    
    assert len(normal_indices) == 2
    assert len(anomaly_indices) == 2


def test_corrupted_file_handling(tmp_path):
    txts_dir = tmp_path / "txts"
    txts_dir.mkdir()
    
    # Write 1 normal file and 1 corrupted (empty) file
    _write_txt(txts_dir / "lhs_000000.txt")
    with open(txts_dir / "lhs_000001.txt", "w") as f:
        pass
        
    # Write manifest
    with mf.ManifestWriter(tmp_path / "manifest_lhs.csv", ["Cout"]) as w:
        w.append("lhs_000000.txt", mf.HEALTHY, "normal", 0, "lhs", "", {"Cout": 1.0})
        w.append("lhs_000001.txt", mf.HEALTHY, "normal", 0, "lhs", "", {"Cout": 1.0})
        
    # Loader should skip the empty file and load only the first one without crashing
    dl = DataLoader(data_dir=str(tmp_path), component_ranges=RANGES)
    dl.load(use_cache=False)
    
    assert dl.num_samples == 1
    assert dl.metadata[0].filename == "lhs_000000.txt"
