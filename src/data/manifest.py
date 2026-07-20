"""
Dataset Manifest (CSV) — sample identity, component values, and labels
======================================================================

Generated simulation files are named by an **opaque identifier** (e.g.
``lhs_000042.txt``) instead of encoding component deviations in the filename.
The mapping *identifier -> component multipliers + label* lives in a CSV
**manifest** written next to the ``.txt`` files, one manifest per generation
mode so grid and Latin-hypercube runs stay separate:

- ``manifest_grid.csv`` — deterministic step-grid samples
- ``manifest_lhs.csv``  — Latin-hypercube (and plain random) samples

Schema (one row per simulation)::

    filename,set,label,n_faults,mode,key,<Comp1>,<Comp2>,...

- ``set``      : ``healthy`` or ``fault`` (generation intent)
- ``label``    : ``normal`` / ``anomalous`` / ``unknown`` (from classification)
- ``n_faults`` : # components in their anomalous band
- ``mode``     : ``grid`` / ``lhs`` / ``random``
- ``key``      : deterministic combo signature (grid dedup; empty for lhs/random)
- ``<Comp*>``  : the component **multiplier** on nominal (1.0 = nominal)

The module is dependency-free (standard library only) so the generator can write
manifests on the PSIM host, and the training loader can read them without pandas.
"""
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Fixed (non-component) columns, in order.
BASE_FIELDS = ["filename", "set", "label", "n_faults", "mode", "key"]

# Which manifest a mode writes to (random shares the lhs manifest).
_MODE_TO_MANIFEST = {"grid": "manifest_grid.csv", "lhs": "manifest_lhs.csv",
                     "random": "manifest_lhs.csv"}

HEALTHY = "healthy"
FAULT = "fault"


def manifest_name(mode: str) -> str:
    """Manifest filename for a generation ``mode`` (grid / lhs / random)."""
    return _MODE_TO_MANIFEST.get(mode, "manifest_lhs.csv")


def manifest_path(data_dir: Union[str, Path], mode: str) -> Path:
    """Full path to the manifest for ``mode`` inside ``data_dir``."""
    return Path(data_dir) / manifest_name(mode)


def all_manifest_paths(data_dir: Union[str, Path]) -> List[Path]:
    """Existing manifest files (manifest.csv, grid and/or lhs) present in ``data_dir``."""
    d = Path(data_dir)
    seen: List[Path] = []
    p_new = d / "manifest.csv"
    if p_new.is_file():
        seen.append(p_new)
    for name in dict.fromkeys(_MODE_TO_MANIFEST.values()):
        p = d / name
        if p.is_file() and p not in seen:
            seen.append(p)
    return seen


def make_key(combo: Dict[str, float], component_cols: Sequence[str]) -> str:
    """Deterministic signature of a combo (rounded multipliers) for grid dedup."""
    return "|".join(
        f"{c}={round(float(combo.get(c, 1.0)), 6):.6f}" for c in component_cols
    )


def read_manifest(path: Union[str, Path]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a manifest CSV -> ``(fieldnames, rows)`` (rows are str->str dicts)."""
    path = Path(path)
    if not path.is_file():
        return [], []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return list(fields), rows


def component_columns(fieldnames: Sequence[str]) -> List[str]:
    """Component columns = any header column that is not a fixed base field."""
    return [f for f in fieldnames if f not in BASE_FIELDS]


def existing_keys(rows: Sequence[Dict[str, str]], set_name: Optional[str] = None) -> set:
    """Set of ``key`` values already present (optionally filtered by ``set``)."""
    return {
        r.get("key", "")
        for r in rows
        if r.get("key") and (set_name is None or r.get("set") == set_name)
    }


def count_set(rows: Sequence[Dict[str, str]], set_name: str) -> int:
    """How many rows belong to a generation ``set`` (healthy / fault)."""
    return sum(1 for r in rows if r.get("set") == set_name)


def next_index(rows: Sequence[Dict[str, str]]) -> int:
    """Next free integer index given existing ``<mode>_<idx>.txt`` filenames."""
    mx = -1
    for r in rows:
        stem = os.path.splitext(r.get("filename", ""))[0]
        digits = stem.rsplit("_", 1)[-1]
        if digits.isdigit():
            mx = max(mx, int(digits))
    return mx + 1


def make_filename(mode: str, index: int) -> str:
    """Opaque sample filename, e.g. ``lhs_000042.txt``."""
    return f"{mode}_{index:06d}.txt"


class ManifestWriter:
    """Append-only manifest writer that flushes each row (real-time updates).

    The component columns are fixed at construction. If the manifest already
    exists its header is reused (and must contain every component column), so
    resumed runs stay consistent.
    """

    def __init__(self, path: Union[str, Path], component_cols: Sequence[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        if not is_new:
            existing_fields, _ = read_manifest(self.path)
            existing_comps = component_columns(existing_fields)
            missing = [c for c in component_cols if c not in existing_comps]
            if existing_fields and missing:
                raise ValueError(
                    f"Manifest {self.path} lacks columns {missing}; use a fresh "
                    f"output directory or matching --components."
                )
            self.fieldnames = existing_fields or (BASE_FIELDS + list(component_cols))
        else:
            self.fieldnames = BASE_FIELDS + list(component_cols)
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def append(
        self,
        filename: str,
        set_name: str,
        label: str,
        n_faults: int,
        mode: str,
        key: str,
        multipliers: Dict[str, float],
    ) -> None:
        """Append one simulation row and flush immediately."""
        row = {
            "filename": filename,
            "set": set_name,
            "label": label,
            "n_faults": n_faults,
            "mode": mode,
            "key": key,
        }
        for c in component_columns(self.fieldnames):
            row[c] = f"{float(multipliers.get(c, 1.0)):.6f}"
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_manifest_index(
    data_dir: Union[str, Path],
) -> Dict[str, Dict[str, object]]:
    """
    Read every manifest in ``data_dir`` -> ``{filename: {...}}``.

    Each value has ``label`` (str) and ``multipliers`` ({component: float}).
    Empty dict if no manifest is present (caller falls back to filename parsing).
    """
    index: Dict[str, Dict[str, object]] = {}
    for path in all_manifest_paths(data_dir):
        fields, rows = read_manifest(path)
        comps = component_columns(fields)
        for r in rows:
            fname = r.get("filename")
            if not fname:
                continue
            mults: Dict[str, float] = {}
            for c in comps:
                val = r.get(c, "")
                if val not in ("", None):
                    try:
                        mults[c] = float(val)
                    except ValueError:
                        pass
            index[fname] = {
                "label": r.get("label") or None,
                "set": r.get("set") or None,
                "multipliers": mults,
            }
    return index
