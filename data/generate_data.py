"""
PSIM Data Generation for Power-Converter Small-Signal Datasets
==============================================================

Sweeps converter component values in PSIM and exports the control-to-output AC
response (Bode) per combination. Each simulation is written under an **opaque
identifier** (e.g. ``lhs_000042.txt``); the mapping *identifier -> component
multipliers + label* is recorded in a CSV **manifest** next to the ``.txt``
files (``manifest_grid.csv`` / ``manifest_lhs.csv``, one per generation mode).
The manifest is updated **in real time** as each simulation completes, and a
re-run **resumes**: grid skips combinations already present, while
Latin-hypercube tops up to the requested count. Downstream, the training loader
reads the manifest to label each sample normal / anomalous / gray with the
per-component bands declared in ``data/<converter>/component_ranges.json``.

The generator draws HEALTHY samples inside each component's normal band and
FAULTY samples inside each component's anomalous band. By default both sets use
**Latin-hypercube sampling** (stratified, gap-free) over the component box:
healthy = LHS over the joint tolerance box; faulty = a guaranteed primary fault
plus independent secondary faults (``--fault-prob``) so **multiple simultaneous
failures may occur**, with every non-faulted component drawn from its normal
band. A deterministic step grid is also available for either set. Run with
``--estimate`` to print the resulting simulation count without launching PSIM.

How finely each band is sampled (the "step")
--------------------------------------------
The sampling granularity is a *data* decision, declared per component in the
ranges file, so the script stays converter-agnostic. Each component may carry:

- ``normal_step``  — fine additive step across the narrow tolerance band (used by
  ``--normal-mode grid``; random/LHS mode samples the box continuously);
- ``anomalous_step`` — coarse additive step across the wide degradation band.

Precedence (backward compatible): per-component step in the ranges file >
global ``--normal-step`` / ``--fault-step`` > level count ``--normal-levels`` /
``--fault-levels``. ``--estimate`` prints, per component, the band, the step (or
fallback count) and the resulting number of levels, so the step decision is
reviewable before any PSIM time is spent.

What counts as a *normal* operating point?
------------------------------------------
A healthy converter is **not** confined to a flat ±5 %. The healthy spread of a
component is dominated by *manufacturing tolerance + temperature + ageing*, and
these differ strongly by component. The recommended, source-grounded envelopes
(healthy) and degradation ranges (fault) live in this converter's data folder,
``data/<converter>/component_ranges.json`` (multipliers on nominal), loaded here
and by the training loader so generation and labelling stay in sync.
References are in ``docs/REFERENCES.md`` ("Component Tolerances & Degradation
Ranges"). To adapt to another converter, edit ``parameters.txt`` and copy /
edit ``component_ranges.json`` in that converter's folder — no code changes.

The same anomalous multipliers drive the on-line synthetic fault injector
(``src/data/physics_anomaly.py``, ``DEFAULT_FAULT_MODES``); keeping the two in
sync guarantees the injected negatives sit *outside* the healthy envelope.

Is it worth sweeping the inductor L? (cost vs. information)
----------------------------------------------------------
Generating *normal* data is the expensive part (a full grid costs the product of
the per-component level counts). For a **buck**, both L and C move the LC
resonance ``ω0 = 1/√(LC)`` and the damping ``Q = R√(C/L)``; only C also moves the
ESR zero ``ωz = 1/(Rc·C)``. So L adds an *independent* Bode direction (shift ω0/Q
without moving ωz) **only if ωz is inside the measured band**.

For this project ωz ≈ 1/(20 mΩ · 100 µF) ≈ 5×10⁵ rad/s ≈ **80 kHz**, while the AC
sweep spans **100 Hz – 50 kHz** — so ωz is *out of band*. Within the measured
band L and C are therefore nearly **degenerate** (both merely shift the ~1.1 kHz
resonance), and a full L sweep adds little over the C sweep. Practical guidance:

- Keep L at nominal (or a coarse 3-level ±tolerance) for the buck normal set, and
  spend the sweep budget on C — this roughly divides the grid size by the number
  of L levels at negligible information loss.
- OR prefer **random / Latin-hypercube sampling** of the tolerance box over a full
  Cartesian product: cost becomes a chosen sample count (independent of the number
  of components), covering the joint L×C manifold — then including L is free.
- Because faults are injected synthetically, you generally do **not** need to PSIM
  the anomalous grid for *training* (CARLA trains on normal + synthetic negatives);
  only a modest real anomaly set is needed for the *test* split. That removes the
  largest combinatorial cost.
- If you extend the AC sweep past ~100 kHz (so ωz becomes observable), L stops
  being degenerate with C and *is* worth a dedicated ±tolerance sweep.

Degradation overlaps (why single-component labels are a simplification)
-----------------------------------------------------------------------
Different degradations produce overlapping Bode signatures and, physically,
co-occur — see the notes in ``data/<converter>/component_ranges.json`` and the
generic machinery in ``src/data/component_ranges.py``.
"""

import argparse
import os
import shutil
import sys
import time
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np

# Make ``src`` importable when this script is run directly (e.g. on the PSIM host).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.component_ranges import (  # noqa: E402
    ANOMALOUS,
    classify_variations,
    load_ranges,
    mult_to_pct,
)
from src.data import manifest as mf  # noqa: E402


# Rough per-simulation wall-clock (seconds) used only for the time ESTIMATE.
# Tune to your PSIM host; it does not affect the generated data.
PER_SIM_SECONDS = 3.0

# Component normal/anomalous ranges (multipliers on nominal) are declared per
# converter in ``data/<converter>/component_ranges.json`` and loaded at runtime,
# shared with the training loader so generation and labelling stay in sync. See
# docs/REFERENCES.md ("Component Tolerances & Degradation Ranges").

# Degradation OVERLAPS (why per-component single-fault labels are a simplification):
#   1. C-drop vs L-drop: both raise ω0 = 1/√(LC); in-band (with ωz out of band)
#      they are nearly indistinguishable — the resonance just moves up.
#   2. Cap-ESR rise vs switch/DCR (Rds/Esr_L) rise: both add series resistance,
#      lowering Q and flattening the resonance peak; ESR additionally drags ωz
#      toward the origin, but that signature is lost if ωz is out of band.
#   3. C-drop AND ESR-rise co-occur in a real ageing electrolytic (dry-out) — they
#      are physically correlated, not independent single-component faults.
#   4. Because tolerances stack, a *joint* healthy excursion (e.g. L −20 % and
#      C −20 %) can shift ω0 as much as a single moderate fault — which is why
#      the anomalous ranges start clearly beyond the healthy envelope.



def parse_value(val_str):
    val_str = val_str.strip()
    multiplier = 1.0
    if val_str.endswith("m"):
        multiplier = 1e-3
        val_str = val_str[:-1]
    elif val_str.endswith("u"):
        multiplier = 1e-6
        val_str = val_str[:-1]
    elif val_str.endswith("n"):
        multiplier = 1e-9
        val_str = val_str[:-1]
    elif val_str.endswith("k") or val_str.endswith("K"):
        multiplier = 1e3
        val_str = val_str[:-1]
    elif val_str.endswith("M") or val_str.lower().endswith("meg"):
        multiplier = 1e6
        if val_str.lower().endswith("meg"):
            val_str = val_str[:-3]
        else:
            val_str = val_str[:-1]
    return float(val_str) * multiplier


def read_nominal_params(param_file):
    params = {"Simview": 0}
    
    try:
        with open(param_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(param_file, "r", encoding="utf-16") as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            params[key.strip()] = parse_value(val)
    return params


# PSIM Path
PSIM_PATH = os.environ.get("PSIM_PATH", r"C:\Altair\Altair_PSIM_2026")

# Global variables for worker processes
p1 = None
worker_temp_dir = None
template_path = None
converter_name = None


def init_worker(psim_path, template_file, temp_root, converter):
    global p1, worker_temp_dir, template_path, converter_name
    converter_name = converter

    # Initialize PSIM once per worker (imported lazily so estimation / imports
    # work on hosts without psimapipy installed).
    try:
        from psimapipy import PSIM

        p1 = PSIM(psim_path)
    except Exception as e:
        print(f"Error initializing PSIM: {e}")
        p1 = None

    # Setup worker directory
    pid = os.getpid()
    worker_temp_dir = os.path.join(temp_root, f"worker_{pid}")
    os.makedirs(worker_temp_dir, exist_ok=True)

    template_path = os.path.abspath(template_file)


def run_simulation(task, component_names, output_dir, nominal_params):
    """Run one PSIM sim and write it under the given opaque ``output_filename``.

    ``task`` is ``(output_filename, values)``. Returns ``output_filename`` on
    success (so the parent can append its manifest row in real time), else
    ``None``.
    """
    t_start = time.time()
    global p1, worker_temp_dir, template_path, converter_name

    output_filename, values = task
    if p1 is None:
        return None

    # Use worker-specific paths
    psimsch_path = os.path.join(worker_temp_dir, f"{converter_name}.psimsch")
    output_file = os.path.join(worker_temp_dir, f"{converter_name}.txt")

    try:
        shutil.copy(template_path, psimsch_path)
    except FileNotFoundError:
        print(f"\nError: Template file not found at {template_path}")
        return None

    # Prepare parameters
    params = nominal_params.copy()
    for name, val in zip(component_names, values):
        params[name] = val

    # Run Simulation
    try:
        res = p1.PsimSimulate(psimsch_path, output_file, **params)

        if res.Result == 0:
            print(
                f"\nError in simulation: {res.ErrorMessage} (Took {time.time() - t_start:.4f}s)"
            )
            return None

    except Exception as e:
        print(f"\nException in simulation: {e} (Took {time.time() - t_start:.4f}s)")
        return None

    # Check if output exists
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        print(f"\nFile not generated (Took {time.time() - t_start:.4f}s)")
        return None

    destination = os.path.join(output_dir, output_filename)
    try:
        shutil.copy(output_file, destination)
    except Exception as e:
        print(f"\nError saving result: {e}")
        return None
    return output_filename


def _band_levels(lo, hi, n):
    """n multipliers evenly spaced across [lo, hi] (n >= 1)."""
    if n <= 1:
        return [round((lo + hi) / 2.0, 6)]
    return [round(float(x), 6) for x in np.linspace(lo, hi, n)]


def _levels_for(lo, hi, step=None, count=None):
    """Multipliers across [lo, hi] using a fixed STEP if given, else COUNT levels.

    Linear/additive spacing. ``step`` (fine for narrow tolerance bands, coarse for
    wide fault bands) is the reviewable knob declared per component in
    ``component_ranges.json``; ``count`` is the fallback (``--normal-levels`` /
    ``--fault-levels``). The upper edge is always included.
    """
    lo, hi = float(lo), float(hi)
    if lo > hi:
        lo, hi = hi, lo
    if step and step > 0:
        n = int(round((hi - lo) / step))
        vals = [round(lo + i * step, 6) for i in range(n + 1)]
        if vals[-1] < hi - 1e-9:
            vals.append(round(hi, 6))
        return vals
    return _band_levels(lo, hi, count if count and count > 0 else 1)


def _norm_step(cr, global_step):
    """Per-component healthy step (JSON) overrides the global ``--normal-step``."""
    return cr.normal_step if cr.normal_step is not None else global_step


def _anom_step(cr, global_step):
    """Per-component fault step (JSON) overrides the global ``--fault-step``."""
    return cr.anomalous_step if cr.anomalous_step is not None else global_step


def _lhs(n, d, rng):
    """Latin-hypercube samples: ``(n, d)`` array in [0, 1), stratified per column.

    Dependency-free (numpy only) so it runs on the PSIM host. Each column places
    exactly one point in each of the ``n`` equal-probability bins, then the bin
    order is shuffled independently per dimension.
    """
    if n <= 0:
        return np.empty((0, d))
    cut = np.arange(n) / n
    out = cut[:, None] + rng.random((n, d)) / n
    for j in range(d):
        rng.shuffle(out[:, j])
    return out


def _scale(u, lo, hi):
    """Map a unit sample u in [0,1) to the multiplier band [lo, hi]."""
    return float(lo + u * (hi - lo))


def build_normal_combinations(varied, mode, n_samples, n_levels, rng, ranges, step=None):
    """Healthy samples: every varied component within its NORMAL band.

    - ``lhs``    : Latin-hypercube over the tolerance box (stratified, default);
    - ``random`` : independent uniform draws (may clump);
    - ``grid``   : full Cartesian product honouring each component's ``normal_step``.
    Returns a list of ``{component: multiplier}`` dicts.
    """
    if mode == "grid":
        per_comp = [
            _levels_for(*ranges[c].normal, _norm_step(ranges[c], step), n_levels)
            for c in varied
        ]
        return [dict(zip(varied, tup)) for tup in product(*per_comp)]
    if mode == "lhs":
        L = _lhs(n_samples, len(varied), rng)
        return [
            {c: _scale(L[i, k], *ranges[c].normal) for k, c in enumerate(varied)}
            for i in range(n_samples)
        ]
    combos = []
    for _ in range(n_samples):
        combos.append(
            {c: float(rng.uniform(*ranges[c].normal)) for c in varied}
        )
    return combos


def build_fault_combinations_lhs(varied, n_fault, rng, ranges, fault_prob=0.0):
    """Faulty samples via LHS, allowing MULTIPLE simultaneous failures.

    The ``n_fault`` budget is split evenly across the fault components; in each
    block that component is the guaranteed *primary* fault (so every sample is
    anomalous and each component gets even coverage). On top of that, every other
    fault-capable component *independently* also fails with probability
    ``fault_prob`` (a *secondary* fault) — so a sample may have 1, 2, ... failures
    (multiplicity ``1 + Binomial(C-1, fault_prob)``). Faulted components draw from
    their anomalous band, the rest from their normal band; all values are LHS-
    stratified. ``fault_prob=0`` reproduces strict single-component faults.
    Returns a list of ``{component: multiplier}`` dicts.
    """
    fault_comps = [c for c in varied if ranges[c].anomalous is not None]
    if not fault_comps or n_fault <= 0:
        return []
    C = len(fault_comps)
    counts = [n_fault // C + (1 if i < n_fault % C else 0) for i in range(C)]
    combos = []
    for ci, c in enumerate(fault_comps):
        m = counts[ci]
        L = _lhs(m, len(varied), rng)
        for i in range(m):
            faulted = {c}  # guaranteed primary fault
            if fault_prob > 0:
                for other in fault_comps:
                    if other != c and rng.random() < fault_prob:
                        faulted.add(other)  # independent secondary fault
            combo = {}
            for k, comp in enumerate(varied):
                band = ranges[comp].anomalous if comp in faulted else ranges[comp].normal
                combo[comp] = _scale(L[i, k], *band)
            combos.append(combo)
    return combos


def build_fault_combinations(varied, fault_levels, backgrounds, include_correlated, rng, ranges, step=None):
    """Faulty samples: ONE component in its ANOMALOUS band (others healthy).

    Each component's severity grid honours its ``anomalous_step`` (coarse step
    across the wide degradation band). Optionally appends a correlated
    electrolytic-ageing trajectory (C-down & ESR-up).
    Returns a list of ``{component: multiplier}`` dicts.
    """
    fault_comps = [c for c in varied if ranges[c].anomalous is not None]
    combos = []
    for c in fault_comps:
        for sev in _levels_for(*ranges[c].anomalous, _anom_step(ranges[c], step), fault_levels):
            for b in range(max(1, backgrounds)):
                combo = {}
                if b > 0:  # jitter the healthy background too
                    for other in fault_comps:
                        if other != c:
                            combo[other] = float(
                                rng.uniform(*ranges[other].normal)
                            )
                combo[c] = float(sev)
                combos.append(combo)
    if include_correlated and "Cout" in varied and "Esr_C" in varied:
        c_lo, c_hi = ranges["Cout"].anomalous   # e.g. (0.30, 0.70)
        e_lo, e_hi = ranges["Esr_C"].anomalous  # e.g. (3.00, 8.00)
        for t in _band_levels(0.0, 1.0, fault_levels):
            combos.append(
                {
                    "Cout": float(c_hi + t * (c_lo - c_hi)),   # 0.70 -> 0.30 (worsening)
                    "Esr_C": float(e_lo + t * (e_hi - e_lo)),  # 2x -> 8x
                }
            )
    return combos


def plan_generation(normal_combos, fault_combos, components, nominal_params, ranges,
                    normal_mode, fault_mode, grid_rows, lhs_rows, log=print):
    """Assign opaque filenames + manifest rows to combos, applying resume.

    - grid: skip combos whose deterministic key is already in the manifest;
    - lhs/random: top-up to the requested count (skip if already satisfied).

    Returns ``(tasks, rows_by_file)`` where each task is ``(filename, values)``
    and ``rows_by_file[filename]`` is the kwargs for ``ManifestWriter.append``.
    Pure/deterministic given its inputs, so it is unit-testable without PSIM.
    """
    manifest_rows = {"grid": grid_rows, "lhs": lhs_rows}
    counters = {
        mf.manifest_name("grid"): mf.next_index(grid_rows),
        mf.manifest_name("lhs"): mf.next_index(lhs_rows),
    }

    def full_combo(combo):
        return {c: float(combo.get(c, 1.0)) for c in components}

    def to_values(full):
        return tuple(nominal_params[c] * full[c] for c in components)

    def label_and_nfaults(full):
        variations = {c: mult_to_pct(m) for c, m in full.items()}
        label = classify_variations(variations, ranges)
        n = sum(
            1
            for c, m in full.items()
            if ranges[c].anomalous is not None
            and ranges[c].classify_multiplier(m) == ANOMALOUS
        )
        return label, n

    def build(combos, set_name, mode):
        mname = mf.manifest_name(mode)
        rows = manifest_rows["grid" if mode == "grid" else "lhs"]
        tasks, rbf = [], {}
        if mode == "grid":
            done = mf.existing_keys(rows, set_name)
            seen = set()
            for combo in combos:
                full = full_combo(combo)
                key = mf.make_key(full, components)
                if key in done or key in seen:
                    continue
                seen.add(key)
                fname = mf.make_filename(mode, counters[mname])
                counters[mname] += 1
                label, nf = label_and_nfaults(full)
                tasks.append((fname, to_values(full)))
                rbf[fname] = dict(filename=fname, set_name=set_name, label=label,
                                  n_faults=nf, mode=mode, key=key, multipliers=full)
        else:  # lhs / random: top-up to the requested count
            have = mf.count_set(rows, set_name)
            need = max(0, len(combos) - have)
            for combo in combos[:need]:
                full = full_combo(combo)
                fname = mf.make_filename(mode, counters[mname])
                counters[mname] += 1
                label, nf = label_and_nfaults(full)
                tasks.append((fname, to_values(full)))
                rbf[fname] = dict(filename=fname, set_name=set_name, label=label,
                                  n_faults=nf, mode=mode, key="", multipliers=full)
            if have:
                log(f"  {set_name} ({mode}): {have} existing -> generating {need} more")
        return tasks, rbf

    ntasks, nrows = build(normal_combos, mf.HEALTHY, normal_mode)
    ftasks, frows = build(fault_combos, mf.FAULT, fault_mode)
    return ntasks + ftasks, {**nrows, **frows}


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(
        description="Generate PSIM data over grounded normal + anomalous component "
        "ranges (data/<converter>/component_ranges.json)."
    )
    parser.add_argument(
        "--components",
        nargs="+",
        help="Components to vary (default: all degradation components declared in "
        "data/<converter>/component_ranges.json that exist in parameters.txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "normal", "fault"],
        default="both",
        help="Which sets to generate",
    )
    parser.add_argument(
        "--normal-mode",
        choices=["lhs", "random", "grid"],
        default="lhs",
        help="Cover the healthy box by Latin-hypercube (default, stratified), plain "
        "random, or a full grid",
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=1000,
        help="# healthy samples (lhs/random normal-mode)",
    )
    parser.add_argument(
        "--normal-levels",
        type=int,
        default=3,
        help="# levels per component (grid normal-mode); fallback when a component "
        "has no 'normal_step' in the ranges file",
    )
    parser.add_argument(
        "--normal-step",
        type=float,
        default=None,
        help="Global additive step (multiplier units) for the healthy GRID; a "
        "per-component 'normal_step' in the ranges file overrides it",
    )
    parser.add_argument(
        "--fault-mode",
        choices=["lhs", "grid"],
        default="lhs",
        help="Cover the fault bands by Latin-hypercube severities (default, "
        "stratified, uses --n-fault) or a deterministic step grid (uses "
        "--fault-step / --fault-levels)",
    )
    parser.add_argument(
        "--n-fault",
        type=int,
        default=300,
        help="# faulty samples (lhs fault-mode); split evenly across fault components",
    )
    parser.add_argument(
        "--fault-prob",
        type=float,
        default=0.1,
        help="lhs fault-mode: probability each OTHER fault component also fails in a "
        "sample (secondary fault), allowing multiple simultaneous failures. "
        "0 = strict single-component faults",
    )
    parser.add_argument(
        "--fault-levels",
        type=int,
        default=5,
        help="# severities per component (grid fault-mode; fallback when a component "
        "has no 'anomalous_step' in the ranges file)",
    )
    parser.add_argument(
        "--fault-step",
        type=float,
        default=None,
        help="Global additive step (multiplier units) across the anomalous band; a "
        "per-component 'anomalous_step' in the ranges file overrides it",
    )
    parser.add_argument(
        "--fault-backgrounds",
        type=int,
        default=1,
        help="# healthy backgrounds per single-fault sample",
    )
    parser.add_argument(
        "--correlated",
        action="store_true",
        help="Add a correlated C-down & ESR-up electrolytic ageing trajectory",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print the simulation-count estimate and exit (no PSIM)",
    )
    parser.add_argument(
        "--converter", type=str, default="buck", help="Converter name (buck, boost, ...)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with parameters.txt and <converter>.psimsch",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = os.path.join(".", "data", args.converter)
    if args.output is None:
        args.output = os.path.join(args.input_dir, f"{args.converter}_data")

    param_file = os.path.join(args.input_dir, "parameters.txt")
    if not os.path.exists(param_file):
        print(f"Error: Parameter file not found at {param_file}")
        return
    nominal_params = read_nominal_params(param_file)

    psimsch_template = os.path.join(args.input_dir, f"{args.converter}.psimsch")

    # Per-converter component ranges (declared in the converter's data folder).
    ranges_file = os.path.join(args.input_dir, "component_ranges.json")
    if not os.path.exists(ranges_file):
        print(f"Error: Component ranges file not found at {ranges_file}")
        return
    ranges = load_ranges(ranges_file)

    # Components to vary: degradation components declared for this converter
    # (``anomalous`` band present) that also exist in its parameters.
    if args.components:
        components_to_vary = list(args.components)
    else:
        components_to_vary = [
            c
            for c in nominal_params
            if c in ranges and ranges[c].anomalous is not None
        ]
    for c in components_to_vary:
        if c not in nominal_params:
            print(f"Error: Component {c} not found in parameters.txt.")
            return
        if c not in ranges:
            print(
                f"Error: Component {c} has no entry in {ranges_file}."
            )
            return

    rng = np.random.default_rng(args.seed)
    normal_combos = (
        build_normal_combinations(
            components_to_vary, args.normal_mode, args.n_normal, args.normal_levels,
            rng, ranges, step=args.normal_step,
        )
        if args.mode in ("both", "normal")
        else []
    )
    fault_combos = (
        (
            build_fault_combinations_lhs(
                components_to_vary, args.n_fault, rng, ranges, fault_prob=args.fault_prob
            )
            if args.fault_mode == "lhs"
            else build_fault_combinations(
                components_to_vary,
                args.fault_levels,
                args.fault_backgrounds,
                args.correlated,
                rng,
                ranges,
                step=args.fault_step,
            )
        )
        if args.mode in ("both", "fault")
        else []
    )

    # ---- Estimation -------------------------------------------------------
    fault_comps = [
        c for c in components_to_vary if ranges[c].anomalous is not None
    ]

    def _fault_sev(c):
        return _levels_for(*ranges[c].anomalous, _anom_step(ranges[c], args.fault_step),
                           args.fault_levels)

    def _fmt_step(step, count):
        return f"step {step:g}" if step else f"count {count}"

    n_single = sum(len(_fault_sev(c)) for c in fault_comps) * max(1, args.fault_backgrounds)
    total = len(normal_combos) + len(fault_combos)
    print("=" * 66)
    print(f"Converter: {args.converter}   varied: {components_to_vary}")
    print(f"Fault components ({len(fault_comps)}): {fault_comps}")
    print("-" * 66)
    if args.mode in ("both", "normal"):
        if args.normal_mode == "grid":
            per_comp = {
                c: _levels_for(*ranges[c].normal, _norm_step(ranges[c], args.normal_step),
                               args.normal_levels)
                for c in components_to_vary
            }
            print("Healthy (grid)  [band | step/count -> #levels]:")
            for c in components_to_vary:
                lo, hi = ranges[c].normal
                s = _norm_step(ranges[c], args.normal_step)
                print(f"  {c:<7} [{lo:.2f}, {hi:.2f}]  {_fmt_step(s, args.normal_levels):<10} "
                      f"-> {len(per_comp[c])}")
            print(f"  => grid product = {len(normal_combos)}")
        else:
            label = "LHS (stratified)" if args.normal_mode == "lhs" else "random (uniform)"
            print(f"Healthy [{label}]: {len(normal_combos)}  (continuous over the "
                  f"{len(components_to_vary)}-dim tolerance box)")
    if args.mode in ("both", "fault"):
        if args.fault_mode == "lhs":
            C = len(fault_comps)
            counts = [args.n_fault // C + (1 if i < args.n_fault % C else 0)
                      for i in range(C)] if C else []
            exp_mult = 1.0 + (C - 1) * args.fault_prob if C else 0.0
            kind = ("multi-component" if args.fault_prob > 0 else "single-component")
            print(f"Faults [LHS {kind}]  primary + LHS-stratified severity/background:")
            for c, m in zip(fault_comps, counts):
                lo, hi = ranges[c].anomalous
                print(f"  {c:<7} [{lo:.2f}, {hi:.2f}]  -> {m} primary samples")
            print(f"  => faulty sims = {len(fault_combos)}  "
                  f"(n_fault={args.n_fault} split across {C} components)")
            if args.fault_prob > 0:
                print(f"  secondary fault prob = {args.fault_prob:g}  "
                      f"=> ~{exp_mult:.2f} simultaneous faults/sample "
                      f"(1 + Binomial({C - 1}, {args.fault_prob:g}))")
        else:
            print("Faults [grid single-component]  [anomalous band | step/count -> #sev]:")
            for c in fault_comps:
                lo, hi = ranges[c].anomalous
                s = _anom_step(ranges[c], args.fault_step)
                sev = _fault_sev(c)
                print(f"  {c:<7} [{lo:.2f}, {hi:.2f}]  {_fmt_step(s, args.fault_levels):<10} "
                      f"-> {len(sev)}   {sev}")
            print(f"  => single-fault sims = {n_single}  "
                  f"(x{max(1, args.fault_backgrounds)} background/s)")
            if args.correlated:
                print(f"  + correlated (C-drop & ESR-rise) trajectory: {args.fault_levels}")
    print("-" * 66)
    print(
        f"TOTAL simulations: {total}  "
        f"(healthy {len(normal_combos)} + faulty {len(fault_combos)})"
    )
    print(
        f"Rough wall-clock @ {PER_SIM_SECONDS:.1f}s/sim over 8 workers: "
        f"~{total * PER_SIM_SECONDS / 8 / 60:.1f} min"
    )
    print("=" * 66)

    if args.estimate:
        return
    if not os.path.exists(psimsch_template):
        print(f"Error: PSIM template not found at {psimsch_template}")
        return

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ---- Plan tasks (opaque ids + manifest rows) with resume -------------
    grid_rows = mf.read_manifest(mf.manifest_path(output_dir, "grid"))[1]
    lhs_rows = mf.read_manifest(mf.manifest_path(output_dir, "lhs"))[1]
    tasks, rows_by_file = plan_generation(
        normal_combos, fault_combos, components_to_vary, nominal_params, ranges,
        args.normal_mode, args.fault_mode, grid_rows, lhs_rows,
    )

    n_norm = sum(1 for r in rows_by_file.values() if r["set_name"] == mf.HEALTHY)
    print(f"New simulations to run: {len(tasks)} "
          f"(healthy {n_norm} + faulty {len(tasks) - n_norm})")
    if not tasks:
        print("Nothing to generate; manifests already satisfy the requested counts.")
        return

    # ---- Open manifest writers (real-time, one per mode used) ------------
    writers = {}
    for row in rows_by_file.values():
        m = mf.manifest_name(row["mode"])
        if m not in writers:
            writers[m] = mf.ManifestWriter(
                os.path.join(output_dir, m), components_to_vary
            )

    # Create temp root for workers in input_dir instead of output_dir
    workers_temp_root = os.path.join(args.input_dir, "temp_workers")
    os.makedirs(workers_temp_root, exist_ok=True)

    print("Launching simulations with 8 processes (manifests update in real time)...")

    worker_func = partial(
        run_simulation,
        component_names=components_to_vary,
        output_dir=output_dir,
        nominal_params=nominal_params,
    )

    processed_count = 0
    written_count = 0
    t_sims = time.time()
    try:
        with Pool(
            processes=8,
            initializer=init_worker,
            initargs=(PSIM_PATH, psimsch_template, workers_temp_root, args.converter),
        ) as pool:
            for result in pool.imap_unordered(worker_func, tasks):
                processed_count += 1
                if result is not None:
                    row = rows_by_file.get(result)
                    if row is not None:
                        writers[mf.manifest_name(row["mode"])].append(**row)
                        written_count += 1
                if processed_count % 100 == 0 or processed_count == len(tasks):
                    print(
                        f"\rProcessed {processed_count}/{len(tasks)} "
                        f"(written {written_count})...",
                        flush=True,
                    )
        print(f"\nAll parallel simulations completed in {time.time() - t_sims:.2f}s")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Partial results kept in manifests.")
    finally:
        for w in writers.values():
            w.close()
        try:
            shutil.rmtree(workers_temp_root)
        except Exception:
            pass

    print(f"Total execution time: {time.time() - t0:.2f}s")
    print(f"Total simulations written: {written_count}/{len(tasks)}")


if __name__ == "__main__":
    main()
