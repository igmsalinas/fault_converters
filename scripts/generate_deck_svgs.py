"""
Generate deck SVG charts
========================

Renders the data-driven figures of the CMMSE 2026 Results cluster as themeable
vector SVGs using :mod:`src.visualization.svg_deck` (no matplotlib). Output goes
to ``docs/plots/<name>.svg`` alongside the existing ``.png`` files, which are
left untouched. Run:

    uv run python scripts/generate_deck_svgs.py

Data sources (all already in the repo):
  - experiments/<model>/metrics.json                    (model comparison, CM)
  - experiments/conv1d_ae/deployment/unified_deployment_report.json (deployment)
  - experiments/conv1d_ae/test_results.csv              (diagnostics)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization import svg_deck as svg  # noqa: E402

EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Display names + deck ordering (matches the deck data table, by F1 desc).
MODEL_ORDER = [
    ("conv1d_ae", "Conv1D-AE"),
    ("lstm_ae", "LSTM-AE"),
    ("transformer_ae", "Transformer-AE"),
    ("vae", "VAE"),
    ("carla", "CARLA"),
    ("mlp_ae", "MLP-AE"),
    ("gru_ae", "GRU-AE"),
]


def _write(name: str, content: str) -> None:
    path = OUT / name
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(content):,} bytes)")


def final_metrics_comparison() -> None:
    metrics = ["precision", "recall", "f1", "auc_roc"]
    labels = {"precision": "Precision", "recall": "Recall", "f1": "F1", "auc_roc": "AUC-ROC"}
    series, values = [], {}
    for key, disp in MODEL_ORDER:
        d = json.loads((EXP / key / "metrics.json").read_text())
        series.append(disp)
        values[disp] = {labels[m]: float(d.get(m, 0.0)) for m in metrics}
    doc = svg.grouped_bar(
        groups=[labels[m] for m in metrics],
        series=series,
        values=values,
        title="Detection metrics across architectures",
        subtitle="Best test threshold \u00b7 higher is better",
        y_max=1.0,
        highlight="Conv1D-AE",
    )
    _write("final_metrics_comparison.svg", doc)


def confusion_matrix_conv1d() -> None:
    d = json.loads((EXP / "conv1d_ae" / "metrics.json").read_text())
    doc = svg.confusion_matrix(
        tn=int(d["true_negatives"]), fp=int(d["false_positives"]),
        fn=int(d["false_negatives"]), tp=int(d["true_positives"]),
        title="Confusion matrix \u00b7 Conv1D-AE",
    )
    _write("confusion_matrix_conv1d_ae.svg", doc)


def _deployment() -> dict:
    return json.loads((EXP / "conv1d_ae" / "deployment" / "unified_deployment_report.json").read_text())


def deploy_hardware() -> None:
    rep = _deployment()
    rows = []
    for name, v in rep.items():
        ev = v.get("evaluation") or {}
        lat = ev.get("latency_per_sample_mean_ms")
        if lat is None:
            continue
        rows.append((name, lat / 1000.0))  # ms -> seconds
    rows.sort(key=lambda r: r[1])  # fastest first
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    highlight = [n for n in labels if n.startswith("TensorRT")]
    doc = svg.hbar_log(
        labels=labels,
        values=values,
        title="Per-sample inference time by export format",
        subtitle="Lower is faster \u00b7 TensorRT engines highlighted",
        highlight_labels=highlight,
        axis_min=1e-6,
        axis_max=1e-3,
    )
    _write("deploy_hardware_conv1d_ae.svg", doc)


def deploy_tradeoff() -> None:
    rep = _deployment()
    # Only label the informative outliers; the tight high-AUC cluster is left
    # unlabelled (the adjacent deck table lists every value) to avoid overlap.
    # Per-label placement keeps text off the points and inside the canvas.
    label_place = {
        "TensorRT FP16 (GPU)": {"label_anchor": "middle", "label_dy": -18},
        "TensorRT INT8 (GPU)": {"label_anchor": "start", "label_dy": 4},
        "TFLite Dynamic": {"label_anchor": "end", "label_dx": -12, "label_dy": 4},
        "TFLite INT8": {"label_anchor": "start", "label_dy": 4},
        "Keras FP32 (Baseline)": {"label_anchor": "end", "label_dx": -12, "label_dy": -10},
    }
    pts = []
    for name, v in rep.items():
        ev = v.get("evaluation") or {}
        lat = ev.get("latency_per_sample_mean_ms")
        auc = ev.get("auc_roc")
        if lat is None or auc is None:
            continue
        point = {
            "x": lat / 1000.0,
            "y": auc,
            "label": name if name in label_place else None,
            "color": svg.ACCENT if name.startswith("TensorRT") else "#4a6670",
            "highlight": name == "TensorRT FP16 (GPU)",
            "r": 7,
        }
        point.update(label_place.get(name, {}))
        pts.append(point)
    doc = svg.scatter(
        pts,
        title="Latency vs. fidelity trade-off",
        subtitle="Pareto-optimal corner: fast and faithful (top-left)",
        x_label="Seconds per sample (log scale)",
        y_label="ROC-AUC",
        x_range=(1e-6, 1e-3),
        y_range=(0.3, 1.0),
        log_x=True,
    )
    _write("deploy_tradeoff_conv1d_ae.svg", doc)


def _load_results() -> pd.DataFrame:
    return pd.read_csv(EXP / "conv1d_ae" / "test_results.csv")


def score_vs_deviation() -> None:
    df = _load_results()

    def signed_max_dev(row):
        try:
            if pd.notna(row.get("variations_json")):
                vd = json.loads(row["variations_json"])
                if vd:
                    return max(vd.values(), key=abs)
        except Exception:
            pass
        return row["max_deviation"]

    df["signed"] = df.apply(signed_max_dev, axis=1)

    def status(row):
        if row["label"] == row["prediction"]:
            return "correct"
        if row["label"] == 1 and row["prediction"] == 0:
            return "fn"
        return "fp"

    df["status"] = df.apply(status, axis=1)
    palette = {"correct": "#9a938a", "fp": "#b79155", "fn": svg.ACCENT}

    tn = df[(df.label == 0) & (df.prediction == 0)]["score"]
    tp = df[(df.label == 1) & (df.prediction == 1)]["score"]
    threshold = (tn.max() + tp.min()) / 2 if len(tn) and len(tp) else None

    # subsample the dominant "correct" class to keep the SVG light + legible
    frames = []
    for st, g in df.groupby("status"):
        cap = 700 if st == "correct" else 1500
        frames.append(g.sample(min(len(g), cap), random_state=0))
    sub = pd.concat(frames)
    sub = sub[sub["score"] > 0]

    pts = [{
        "x": float(r.signed),
        "y": float(r.score),
        "color": palette[r.status],
        "r": 3.5,
        "opacity": 0.6,
    } for r in sub.itertuples()]

    doc = svg.scatter(
        pts,
        title="Anomaly score vs. deviation magnitude \u00b7 Conv1D-AE",
        subtitle="Error scales with physical drift \u00b7 misses cluster near nominal",
        x_label="Maximum component deviation (%)",
        y_label="Reconstruction score (log scale)",
        y_range=(max(sub["score"].min(), 1e-9), sub["score"].max()),
        log_y=True,
        threshold=threshold,
        legend=[("Correct (TP/TN)", palette["correct"]),
                ("False alarm (FP)", palette["fp"]),
                ("Missed anomaly (FN)", palette["fn"])],
    )
    _write("score_vs_deviation_conv1d_ae.svg", doc)


def error_by_component() -> None:
    df = _load_results()
    df["is_fn"] = (df["label"] == 1) & (df["prediction"] == 0)
    df["is_fp"] = (df["label"] == 0) & (df["prediction"] == 1)

    comps = set()
    for c in df["varied_components"].dropna().unique():
        if c:
            comps.update(c.split(","))
    comps = sorted(comps)

    cats, vals, cols = [], [], []
    for comp in comps:
        mask = df["varied_components"].fillna("").apply(lambda x: comp in x.split(","))
        anom = df[mask & (df["label"] == 1)]
        rate = (anom["is_fn"].sum() / len(anom) * 100) if len(anom) else 0.0
        cats.append(comp)
        vals.append(rate)
        cols.append(svg.ACCENT)

    normal = df[df["label"] == 0]
    fp_rate = (normal["is_fp"].sum() / len(normal) * 100) if len(normal) else 0.0
    cats.append("All normal")
    vals.append(fp_rate)
    cols.append("#b79155")

    doc = svg.rate_bars(
        categories=cats,
        values=vals,
        colors=cols,
        title="Error rate by component influence \u00b7 Conv1D-AE",
        subtitle="Missed-anomaly rate per component (terracotta) \u00b7 false alarms (ochre)",
        y_label="Error rate (%)",
    )
    _write("error_by_component_conv1d_ae.svg", doc)


def main() -> None:
    print("Generating deck SVG charts ->", OUT.relative_to(ROOT))
    final_metrics_comparison()
    confusion_matrix_conv1d()
    deploy_hardware()
    deploy_tradeoff()
    score_vs_deviation()
    error_by_component()
    print("Done.")


if __name__ == "__main__":
    main()
