"""
Legacy dataset migration
========================

Convert an **old pct-encoded** dataset (filenames like ``Cout_-20__Rds_1_-5.txt``)
into the new **manifest** format.

By default this is **non-destructive**: it only writes a manifest that references
the existing filenames (which keep working). Pass ``rename=True`` to also rename
the files to opaque identifiers (``grid_000001.txt`` …) matching freshly-generated
datasets.

Usage::

    python -m src.data.migrate --data-dir data/buck/buck_data
    python -m src.data.migrate --data-dir data/buck/buck_data --rename
    
    # Or to migrate into the new versioned structured dataset format:
    python -m src.data.migrate --data-dir data/buck/buck_data --dataset-name dataset_00
"""
import os
from pathlib import Path
from typing import Optional, Union

from . import manifest as mf
from .component_ranges import (
    ANOMALOUS,
    Ranges,
    classify_variations,
    load_ranges_for,
    pct_to_mult,
)
from .loader import parse_filename


def migrate_legacy_dataset(
    data_dir: Union[str, Path],
    ranges: Optional[Ranges] = None,
    mode: str = "grid",
    rename: bool = False,
    dataset_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    log=print,
) -> int:
    """
    Write a manifest for a legacy pct-encoded dataset.

    Args:
        data_dir: Directory holding the legacy ``.txt`` files.
        ranges: Component ranges for labelling (auto-discovered if ``None``).
        mode: Which manifest to write (``grid`` or ``lhs``).
        rename: If True, rename files to opaque ids and reference the new names.
        dataset_name: Optional target dataset name.
        output_dir: Optional target output root directory.
        log: Logging callable.

    Returns:
        Number of files migrated.
    """
    data_dir = Path(data_dir)
    ranges = ranges if ranges is not None else (load_ranges_for(data_dir) or {})

    # Resolve nominal parameters
    param_file = data_dir.parent / "parameters.txt"
    nominal_params = {}
    if param_file.exists():
        try:
            from data.generate_data import read_nominal_params
            nominal_params = read_nominal_params(param_file)
        except ImportError:
            pass

    # Note: don't load manifest from the directory if we are migrating to a fresh target structure
    already = {}
    if not dataset_name:
        already = mf.load_manifest_index(data_dir)
    
    files = sorted(p for p in data_dir.glob("*.txt") if p.name != "parameters.txt")

    legacy = []
    for p in files:
        if p.name in already:
            continue
        meta = parse_filename(p.name)  # tolerant: opaque ids -> {} (skipped)
        if not meta.variations:
            continue
        legacy.append((p, meta.variations))

    if not legacy:
        log(f"No legacy pct-encoded files to migrate in {data_dir}.")
        return 0

    # Setup target paths
    if dataset_name:
        out_root = Path(output_dir) if output_dir else data_dir
        dataset_dir = out_root / dataset_name
        txts_dir = dataset_dir / "txts"
        txts_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = dataset_dir / "manifest.csv"
    else:
        dataset_dir = data_dir
        txts_dir = data_dir
        manifest_file = data_dir / mf.manifest_name(mode)

    # Component columns: ranges order first, then any extras, restricted to seen.
    seen = []
    for _, variations in legacy:
        for c in variations:
            if c not in seen:
                seen.append(c)
    ordered = [c for c in ranges if c in seen] + [c for c in seen if c not in ranges]

    idx = 0
    if not dataset_name and manifest_file.exists():
        idx = mf.next_index(mf.read_manifest(manifest_file)[1])

    writer = mf.ManifestWriter(manifest_file, ordered)

    n = 0
    for p, variations in legacy:
        full = {c: pct_to_mult(float(variations.get(c, 0.0))) for c in ordered}
        
        # Use legacy flat rule if dataset_name is provided; otherwise fallback to custom ranges if available.
        if dataset_name:
            devs = {c: float(v) for c, v in variations.items()}
            max_dev = max(abs(v) for v in devs.values()) if devs else 0.0
            if max_dev <= 5.0:
                label = "normal"
                set_name = mf.HEALTHY
                n_faults = 0
            else:
                label = "anomalous"
                set_name = mf.FAULT
                n_faults = sum(1 for v in devs.values() if abs(v) > 5.0)
        else:
            label = classify_variations(variations, ranges)
            set_name = mf.HEALTHY if label == "normal" else mf.FAULT
            n_faults = sum(
                1
                for c, m in full.items()
                if c in ranges
                and ranges[c].anomalous is not None
                and ranges[c].classify_multiplier(m) == ANOMALOUS
            )
            
        key = mf.make_key(full, ordered)
        
        # In target structured migration, we always rename to opaque IDs
        if rename or dataset_name:
            fname = mf.make_filename(mode, idx)
            idx += 1
            p.rename(txts_dir / fname)
        else:
            fname = p.name
            
        writer.append(fname, set_name, label, n_faults, mode, key, full)
        n += 1
        
        if n % 10000 == 0:
            log(f"  Processed {n}/{len(legacy)} files...")

    writer.close()

    # Write dataset.json if we are in the new structured layout
    if dataset_name:
        import json
        from datetime import datetime, timezone

        ranges_dict = {}
        for c in ordered:
            ranges_dict[c] = {
                "normal": [0.95, 1.05],
                "anomalous": [1.05, 999.0]
            }

        dataset_json_data = {
            "converter": data_dir.parent.name,
            "dataset_name": dataset_name,
            "normal_threshold": 5.0,
            "use_component_ranges": False,
            "seed": 42,
            "components": ordered,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nominal_parameters": nominal_params,
            "component_ranges": ranges_dict,
        }

        with open(dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset_json_data, f, indent=2)

    log(f"Migrated {n} legacy files to {dataset_dir}")
    return n


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert a legacy pct-filename dataset to the manifest format."
    )
    ap.add_argument("--data-dir", required=True, help="Directory with legacy .txt files")
    ap.add_argument("--mode", choices=["grid", "lhs"], default="grid",
                    help="Which manifest to write (default: grid)")
    ap.add_argument("--rename", action="store_true",
                    help="Also rename files to opaque ids (destructive)")
    ap.add_argument("--dataset-name", type=str, default=None,
                    help="Target dataset name under <converter>_data/")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Target output root directory")
                    
    args = ap.parse_args()
    migrate_legacy_dataset(
        args.data_dir,
        mode=args.mode,
        rename=args.rename,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
