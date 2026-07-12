"""
Component Tolerance / Degradation Specification (converter-agnostic)
===================================================================

This module provides the **generic machinery** for describing and classifying
component deviations; the actual per-component ranges are **not** hardcoded here.
Instead each converter declares its own ranges in a data file next to its
``parameters.txt``::

    data/<converter>/component_ranges.json

so adding a converter (with a different component set) needs no change to ``src``.
The loader auto-discovers this file from the dataset directory; the generator
reads it from the converter folder.

File format (multipliers on the nominal value; ``1.0`` = nominal)::

    {
      "converter": "buck",
      "components": {
        "Cout":  {"normal": [0.80, 1.20], "anomalous": [0.30, 0.70], "note": "..."},
        "Esr_C": {"normal": [0.50, 2.00], "anomalous": [2.00, 8.00], "note": "..."},
        "Rout":  {"normal": [0.50, 2.00], "anomalous": null,         "note": "op. point"}
      }
    }

``anomalous: null`` marks an operating-point knob (not a fault). Ranges are
grounded in ``docs/REFERENCES.md`` ("Component Tolerances & Degradation Ranges").

The module is intentionally dependency-free (standard library only) so it can be
imported from the PSIM host as well as the training environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

# Classification labels
NORMAL = "normal"
ANOMALOUS = "anomalous"
UNKNOWN = "unknown"  # gray zone: outside the healthy band but not a modelled fault

#: Conventional filename for a converter's range spec (in its data folder).
RANGES_FILENAME = "component_ranges.json"

# A ranges spec maps component name -> ComponentRange.
Ranges = Dict[str, "ComponentRange"]


@dataclass(frozen=True)
class ComponentRange:
    """Healthy and faulty multiplier bands for one component.

    Attributes:
        normal: ``(low, high)`` multiplier band for a *healthy* component
            (tolerance + temperature + ageing).
        anomalous: ``(low, high)`` multiplier band for a *degraded* component,
            or ``None`` if the component is an operating-point knob (not a fault).
        note: short human-readable rationale.
        normal_step: optional additive step (multiplier units) for the healthy
            GRID sweep; fine because tolerance bands are narrow. ``None`` -> the
            generator falls back to a level count. Ignored by classification and
            by random/LHS sampling.
        anomalous_step: optional additive step (multiplier units) for the fault
            severity sweep; coarse because degradation bands are wide.
    """

    normal: Tuple[float, float]
    anomalous: Optional[Tuple[float, float]]
    note: str = ""
    normal_step: Optional[float] = None
    anomalous_step: Optional[float] = None

    def classify_multiplier(self, m: float) -> str:
        """Classify a single component multiplier as normal / anomalous / unknown."""
        lo, hi = self.normal
        if lo <= m <= hi:
            return NORMAL
        if self.anomalous is None:
            return UNKNOWN
        a_lo, a_hi = self.anomalous
        if a_lo <= m <= a_hi:
            return ANOMALOUS
        # Beyond the fault band *in the fault direction* is still a fault
        # (even more degraded); the opposite side / the gap is the gray zone.
        if a_hi <= lo:  # fault direction is downward (e.g. capacitance, inductance)
            return ANOMALOUS if m < a_lo else UNKNOWN
        if a_lo >= hi:  # fault direction is upward (e.g. ESR, Rds_on)
            return ANOMALOUS if m > a_hi else UNKNOWN
        return UNKNOWN


# ---------------------------------------------------------------------------
# Loading a converter's ranges from its data folder
# ---------------------------------------------------------------------------
def load_ranges(path: Union[str, Path]) -> Ranges:
    """Load a ranges spec (JSON) into ``{name: ComponentRange}``."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    comps = raw.get("components", raw)  # allow {"components": {...}} or a flat dict
    out: Ranges = {}
    for name, spec in comps.items():
        normal = tuple(float(x) for x in spec["normal"])
        anom = spec.get("anomalous")
        anomalous = tuple(float(x) for x in anom) if anom else None
        n_step = spec.get("normal_step")
        a_step = spec.get("anomalous_step")
        out[name] = ComponentRange(
            normal=normal,
            anomalous=anomalous,
            note=spec.get("note", ""),
            normal_step=float(n_step) if n_step is not None else None,
            anomalous_step=float(a_step) if a_step is not None else None,
        )
    return out


def find_ranges_file(
    data_dir: Union[str, Path], filename: str = RANGES_FILENAME, max_up: int = 3
) -> Optional[Path]:
    """Search ``data_dir`` and up to ``max_up`` parents for the ranges file.

    e.g. ``data/buck/buck_data`` -> finds ``data/buck/component_ranges.json``.
    """
    p = Path(data_dir).resolve()
    for _ in range(max_up):
        candidate = p / filename
        if candidate.is_file():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


def load_ranges_for(data_dir: Union[str, Path]) -> Optional[Ranges]:
    """Auto-discover and load the ranges for a dataset directory (or ``None``)."""
    found = find_ranges_file(data_dir)
    return load_ranges(found) if found is not None else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def pct_to_mult(pct: float) -> float:
    """Percentage deviation from nominal -> multiplier (e.g. -20 -> 0.80)."""
    return 1.0 + pct / 100.0


def mult_to_pct(mult: float) -> float:
    """Multiplier -> percentage deviation from nominal (e.g. 5.0 -> +400)."""
    return (mult - 1.0) * 100.0


def classify_component(
    name: str,
    pct: float,
    ranges: Optional[Ranges] = None,
    fallback_threshold: float = 5.0,
) -> str:
    """
    Classify one component's percentage deviation.

    Components absent from ``ranges`` (or when ``ranges`` is ``None``) fall back
    to a symmetric ``|pct| <= fallback_threshold`` rule, so arbitrary converters
    keep working even without a spec.
    """
    cr = (ranges or {}).get(name)
    if cr is None:
        return NORMAL if abs(pct) <= fallback_threshold else ANOMALOUS
    return cr.classify_multiplier(pct_to_mult(pct))


def classify_variations(
    variations: Dict[str, float],
    ranges: Optional[Ranges] = None,
    fallback_threshold: float = 5.0,
) -> str:
    """
    Classify a whole simulation from its per-component percentage deviations.

    - ``anomalous`` if ANY component is in (or beyond) its fault band,
    - ``normal`` if EVERY component is within its healthy band,
    - ``unknown`` otherwise (a component sits in the gray zone between the two).
    """
    if not variations:
        return NORMAL
    labels = [
        classify_component(k, v, ranges, fallback_threshold)
        for k, v in variations.items()
    ]
    if any(lbl == ANOMALOUS for lbl in labels):
        return ANOMALOUS
    if all(lbl == NORMAL for lbl in labels):
        return NORMAL
    return UNKNOWN
