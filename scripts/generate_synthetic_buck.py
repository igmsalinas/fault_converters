"""
Synthetic Buck Dataset Generator (DEBUG stand-in for the PSIM dataset)
=====================================================================

The real training set is produced on Windows by ``data/generate_data.py`` (PSIM)
and is **not** present in this workspace. This script fabricates a small,
physically consistent buck dataset directly from the analytic control-to-output
model (:class:`src.data.physics_anomaly.BuckConverter`) so the CARLA pipeline can
be exercised end-to-end.

It is ALSO a concrete illustration of "what counts as a normal sample": the
healthy units are drawn from *component-specific* tolerance bands (not a flat
+/-5 %), while the faulty units perturb a single component well beyond its
tolerance envelope (see NORMAL_TOLERANCE / FAULTS below).

Output matches the loader format: whitespace-separated ``freq  amp[dB]  phase[deg]``
with one header line, filenames encoding per-component % deviations so the loader
labels normal vs anomaly with the per-component bands declared in
``data/buck/component_ranges.json``.

Usage::

    uv run python scripts/generate_synthetic_buck.py \
        --out data/buck/buck_data_debug --n-normal 34 --n-anomaly 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.component_ranges import classify_variations, load_ranges, mult_to_pct
from src.data import manifest as mf
from src.data.physics_anomaly import BuckConverter


# Frequency grid source (the real AC sweep) — falls back to a log grid.
FRA_FILE = Path("data/buck/buck(AC-Sweep).fra")

# ---------------------------------------------------------------------------
# Healthy and faulty draws come from the converter's declared component spec so
# the loader labels this debug data exactly. Multipliers on nominal: normal band
# = healthy; anomalous band = fault.
# ---------------------------------------------------------------------------
COMPONENT_RANGES = load_ranges("data/buck/component_ranges.json")

# Filename tokens -> BuckConverter component fields (subset we perturb here).
FIELD = {
    "Cout": "capacitance",
    "Lout": "inductance",
    "Esr_C": "esr_cap",
    "Esr_L": "esr_ind",
    "Rds_1": "rds",
}
VARIED = [c for c in FIELD if c in COMPONENT_RANGES]
FAULT_TOKENS = [c for c in VARIED if COMPONENT_RANGES[c].anomalous is not None]


def load_frequency_grid() -> np.ndarray:
    if FRA_FILE.exists():
        arr = np.loadtxt(FRA_FILE, skiprows=1)
        return np.asarray(arr[:, 0], dtype=float)
    return np.logspace(2, np.log10(2e5), 300)


def bode(converter: BuckConverter, freqs: np.ndarray, rng: np.random.Generator):
    """Return (amp_dB, phase_deg) with small measurement noise."""
    h = converter.transfer_function().frequency_response(2 * np.pi * freqs)
    amp_db = 20.0 * np.log10(np.abs(h))
    phase_deg = np.unwrap(np.angle(h)) * 180.0 / np.pi
    amp_db += rng.normal(0.0, 0.05, amp_db.shape)      # ~0.05 dB sensor noise
    phase_deg += rng.normal(0.0, 0.10, phase_deg.shape)  # ~0.1 deg sensor noise
    return amp_db, phase_deg


def write_sample(path: Path, freqs, amp_db, phase_deg) -> None:
    with open(path, "w") as f:
        f.write("Frequency        amp(Vo1)          phase(Vo1)\n")
        for fr, a, p in zip(freqs, amp_db, phase_deg):
            f.write(f"  {fr:.9E}   {a:.9E}  {p:.9E}\n")


def build_converter(mults: dict) -> BuckConverter:
    return BuckConverter().scaled(**{FIELD[k]: float(m) for k, m in mults.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/buck/buck_data_debug")
    ap.add_argument("--n-normal", type=int, default=34)
    ap.add_argument("--n-anomaly", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    freqs = load_frequency_grid()

    # Opaque IDs + real-time manifest (matches data/generate_data.py convention).
    writer = mf.ManifestWriter(out / mf.manifest_name("lhs"), VARIED)
    existing = mf.read_manifest(out / mf.manifest_name("lhs"))[1]
    idx = mf.next_index(existing)

    def _emit(mults: dict, set_name: str) -> bool:
        nonlocal idx
        full = {c: float(mults.get(c, 1.0)) for c in VARIED}
        variations = {c: mult_to_pct(m) for c, m in full.items()}
        label = classify_variations(variations, COMPONENT_RANGES)
        n_faults = sum(
            1 for c, m in full.items()
            if COMPONENT_RANGES[c].anomalous is not None
            and COMPONENT_RANGES[c].classify_multiplier(m) == "anomalous"
        )
        fname = mf.make_filename("lhs", idx)
        idx += 1
        conv = build_converter(mults)
        amp_db, phase_deg = bode(conv, freqs, rng)
        write_sample(out / fname, freqs, amp_db, phase_deg)
        writer.append(fname, set_name, label, n_faults, "lhs", "", full)
        return True

    n_written = 0

    # ---- Normal units: 1-3 components jitter within their healthy band ----
    for _ in range(args.n_normal):
        tokens = rng.choice(VARIED, size=rng.integers(1, 4), replace=False)
        mults = {t: float(rng.uniform(*COMPONENT_RANGES[t].normal)) for t in tokens}
        n_written += _emit(mults, mf.HEALTHY)

    # ---- Faulty units: ONE component driven into its anomalous band ----
    for _ in range(args.n_anomaly):
        t = str(rng.choice(FAULT_TOKENS))
        mults = {t: float(rng.uniform(*COMPONENT_RANGES[t].anomalous))}
        n_written += _emit(mults, mf.FAULT)

    writer.close()
    print(f"Wrote {n_written} files + manifest to {out} over {freqs.size} frequency "
          f"points ({freqs[0]:.1f} Hz .. {freqs[-1]:.1f} Hz)")


if __name__ == "__main__":
    main()
