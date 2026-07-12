"""
Legacy dataset migration
========================

Convert an **old pct-encoded** dataset (filenames like ``Cout_-20__Rds_1_-5.txt``)
into the new **manifest** format (a ``manifest_grid.csv`` / ``manifest_lhs.csv``
mapping *filename -> component multipliers + label*).

By default this is **non-destructive**: it only writes a manifest that references
the existing filenames (which keep working). Pass ``rename=True`` to also rename
the files to opaque identifiers (``grid_000001.txt`` …) matching freshly-generated
datasets.

Usage::

    python -m src.data.migrate --data-dir data/buck/buck_data
    python -m src.data.migrate --data-dir data/buck/buck_data --rename
"""

from __future__ import annotations

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
    log=print,
) -> int:
    """
    Write a manifest for a legacy pct-encoded dataset.

    Args:
        data_dir: Directory holding the legacy ``.txt`` files.
        ranges: Component ranges for labelling (auto-discovered if ``None``).
        mode: Which manifest to write (``grid`` or ``lhs``).
        rename: If True, rename files to opaque ids and reference the new names.
        log: Logging callable.

    Returns:
        Number of files migrated.
    """
    data_dir = Path(data_dir)
    ranges = ranges if ranges is not None else (load_ranges_for(data_dir) or {})

    already = mf.load_manifest_index(data_dir)  # skip files already in a manifest
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

    # Component columns: ranges order first, then any extras, restricted to seen.
    seen = []
    for _, variations in legacy:
        for c in variations:
            if c not in seen:
                seen.append(c)
    ordered = [c for c in ranges if c in seen] + [c for c in seen if c not in ranges]

    manifest_file = data_dir / mf.manifest_name(mode)
    idx = mf.next_index(mf.read_manifest(manifest_file)[1])
    writer = mf.ManifestWriter(manifest_file, ordered)

    n = 0
    for p, variations in legacy:
        full = {c: pct_to_mult(float(variations.get(c, 0.0))) for c in ordered}
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
        if rename:
            fname = mf.make_filename(mode, idx)
            idx += 1
            p.rename(data_dir / fname)
        else:
            fname = p.name
        writer.append(fname, set_name, label, n_faults, mode, key, full)
        n += 1

    writer.close()
    log(f"Migrated {n} legacy files -> {mf.manifest_name(mode)} in {data_dir}"
        + (" (renamed to opaque ids)" if rename else ""))
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
    args = ap.parse_args()
    migrate_legacy_dataset(args.data_dir, mode=args.mode, rename=args.rename)


if __name__ == "__main__":
    main()
