"""
Deck SVG charts
===============

A dependency-free, hand-authored **SVG** charting toolkit that renders the
data-driven figures of the CMMSE 2026 deck as crisp, themeable vector graphics
instead of matplotlib rasters.

Every chart is emitted as a self-contained ``<svg>`` document whose colours,
type and gridlines mirror the deck's CSS design tokens (see ``deck_style.py``
and the ``:root`` custom properties in
``docs/presentations/CMMSE_2026_presentation.html``). Styling lives in an inline
``<style>`` block using semantic classes, so the look can be tuned with CSS
rather than re-running Python. The same SVG can be embedded in the HTML deck
(``<img src="...svg">`` or inlined) and rasterised to PNG/PDF for the paper.

The toolkit intentionally uses only the standard library.
"""
import html
import math
from typing import Dict, List, Optional, Sequence, Tuple

# --- Deck design tokens (mirror deck_style.py / the deck :root variables) -----
PAPER = "#f7f4ef"
PAPER_SOFT = "#fbf9f5"
INK = "#121212"
INK_SOFT = "#43413d"
MUTED = "#8a837b"
LINE = "rgba(17,17,17,0.08)"
LINE_STRONG = "rgba(17,17,17,0.18)"
ACCENT = "#a7623f"

CATEGORICAL: List[str] = [
    "#1f1d1b",  # ink
    "#a7623f",  # terracotta
    "#4a6670",  # slate teal
    "#b79155",  # ochre
    "#6d6f5a",  # sage
    "#8c5361",  # dusty plum
    "#3f5a52",  # pine
    "#9a938a",  # warm grey
    "#5b6b8a",  # muted indigo
    "#c08a5a",  # amber
]

FONT = '"ABC Monument Grotesk Light","Helvetica Neue",Helvetica,Arial,sans-serif'
FONT_DISPLAY = '"ABC Monument Grotesk Thin","ABC Monument Grotesk Light","Helvetica Neue",Helvetica,Arial,sans-serif'

# Mono ramp (cream -> ink) for heatmaps, matching deck_style.MONO_CMAP stops.
_MONO_STOPS = ["#fbf9f5", "#d8cdbb", "#8f8578", "#4a453f", "#121212"]


# --- small helpers ------------------------------------------------------------
def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def categorical_colors(n: int) -> List[str]:
    if n <= len(CATEGORICAL):
        return CATEGORICAL[:n]
    reps = (n // len(CATEGORICAL)) + 1
    return (CATEGORICAL * reps)[:n]


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mono_color(t: float) -> str:
    """Sample the cream->ink mono ramp at t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    n = len(_MONO_STOPS) - 1
    idx = min(int(t * n), n - 1)
    local = (t * n) - idx
    c0 = _hex_to_rgb(_MONO_STOPS[idx])
    c1 = _hex_to_rgb(_MONO_STOPS[idx + 1])
    r = round(_lerp(c0[0], c1[0], local))
    g = round(_lerp(c0[1], c1[1], local))
    b = round(_lerp(c0[2], c1[2], local))
    return f"#{r:02x}{g:02x}{b:02x}"


def text_on(hex_color: str) -> str:
    """Pick ink or paper for legible text on a given fill (WCAG luminance)."""
    r, g, b = (c / 255 for c in _hex_to_rgb(hex_color))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    return INK if lum > 0.42 else PAPER_SOFT


def _fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def _fmt_sci(value: float) -> str:
    """Compact scientific-ish label, e.g. 4.4e-6."""
    if value == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / (10 ** exp)
    return f"{mant:.1f}\u00d710{_superscript(exp)}"


_SUP = {"-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
        "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078",
        "9": "\u2079"}


def _superscript(n: int) -> str:
    return "".join(_SUP[c] for c in str(n))


# --- base stylesheet shared by every chart ------------------------------------
_BASE_CSS = f"""
    .bg {{ fill: none; }}
    .title {{ font: 600 22px {FONT}; fill: {INK}; }}
    .subtitle {{ font: 400 14px {FONT}; fill: {MUTED}; }}
    .axis-title {{ font: 500 14px {FONT}; fill: {INK_SOFT}; }}
    .tick {{ font: 400 13px {FONT}; fill: {MUTED}; }}
    .tick-strong {{ font: 500 14px {FONT}; fill: {INK_SOFT}; }}
    .value {{ font: 500 12px {FONT}; fill: {INK_SOFT}; }}
    .legend {{ font: 400 14px {FONT}; fill: {INK_SOFT}; }}
    .grid {{ stroke: {LINE}; stroke-width: 1; }}
    .axis {{ stroke: {LINE_STRONG}; stroke-width: 1.2; }}
""".rstrip()


def _document(width: int, height: int, body: str, extra_css: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'font-family=\'{FONT}\' preserveAspectRatio="xMidYMid meet">\n'
        f"  <style>{_BASE_CSS}{extra_css}</style>\n"
        f'  <rect class="bg" x="0" y="0" width="{width}" height="{height}"/>\n'
        f"{body}\n"
        f"</svg>\n"
    )


def _text(x: float, y: float, s: str, cls: str = "tick",
          anchor: str = "start", rotate: Optional[float] = None) -> str:
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
            f'text-anchor="{anchor}"{tr}>{_esc(s)}</text>')


# ==============================================================================
# 1. Grouped bar chart  (final_metrics_comparison)
# ==============================================================================
def grouped_bar(
    groups: Sequence[str],
    series: Sequence[str],
    values: Dict[str, Dict[str, float]],
    *,
    title: str = "",
    subtitle: str = "",
    y_max: float = 1.0,
    colors: Optional[Sequence[str]] = None,
    highlight: Optional[str] = None,
    width: int = 1040,
    height: int = 620,
) -> str:
    """Grouped bars: one cluster per ``group``, one bar per ``series``."""
    colors = list(colors) if colors else categorical_colors(len(series))
    color_of = {s: colors[i % len(colors)] for i, s in enumerate(series)}
    if highlight and highlight in color_of:
        color_of[highlight] = ACCENT

    m_left, m_right, m_top, m_bottom = 64, 236, 74, 58
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    x0, y0 = m_left, m_top

    def yv(v: float) -> float:
        return y0 + plot_h * (1 - v / y_max)

    parts: List[str] = []
    if title:
        parts.append(_text(x0, 34, title, "title"))
    if subtitle:
        parts.append(_text(x0, 56, subtitle, "subtitle"))

    # y grid + ticks
    steps = 5
    for i in range(steps + 1):
        v = y_max * i / steps
        y = yv(v)
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}"/>')
        parts.append(_text(x0 - 12, y + 4, _fmt(v, 2), "tick", "end"))
    # baseline axis
    parts.append(f'<line class="axis" x1="{x0}" y1="{yv(0):.1f}" x2="{x0 + plot_w}" y2="{yv(0):.1f}"/>')

    n_groups = len(groups)
    group_w = plot_w / n_groups
    inner = group_w * 0.82
    bar_w = inner / len(series)
    for gi, g in enumerate(groups):
        gx = x0 + gi * group_w + (group_w - inner) / 2
        for si, s in enumerate(series):
            v = float(values.get(s, {}).get(g, 0.0))
            bx = gx + si * bar_w
            by = yv(v)
            bh = yv(0) - by
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bar_w - 1.5, 1):.1f}" '
                f'height="{bh:.1f}" fill="{color_of[s]}" rx="1"/>'
            )
        parts.append(_text(x0 + gi * group_w + group_w / 2, y0 + plot_h + 30, g, "tick-strong", "middle"))

    # legend (right column)
    lx = x0 + plot_w + 34
    ly = y0 + 6
    parts.append(_text(lx, ly, "Architecture", "axis-title"))
    for i, s in enumerate(series):
        yy = ly + 26 + i * 26
        parts.append(f'<rect x="{lx}" y="{yy - 11:.1f}" width="16" height="16" rx="3" fill="{color_of[s]}"/>')
        parts.append(_text(lx + 24, yy + 2, s, "legend"))

    return _document(width, height, "\n".join(parts))


# ==============================================================================
# 2. Horizontal bar chart with optional log scale  (deploy_hardware)
# ==============================================================================
def hbar_log(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    subtitle: str = "",
    highlight_labels: Sequence[str] = (),
    value_fmt=None,
    axis_min: Optional[float] = None,
    axis_max: Optional[float] = None,
    log: bool = True,
    width: int = 1040,
    height: int = 620,
) -> str:
    """Horizontal bars sorted as given; log x-axis by default (for latency)."""
    value_fmt = value_fmt or _fmt_sci
    vmin = axis_min or min(v for v in values if v > 0)
    vmax = axis_max or max(values)
    if log:
        lo, hi = math.log10(vmin), math.log10(vmax)
    else:
        lo, hi = 0.0, vmax

    m_left, m_right, m_top, m_bottom = 232, 118, 74, 54
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    x0, y0 = m_left, m_top

    def xv(v: float) -> float:
        if log:
            t = (math.log10(max(v, vmin)) - lo) / (hi - lo)
        else:
            t = v / hi if hi else 0
        return x0 + plot_w * t

    parts: List[str] = []
    if title:
        parts.append(_text(24, 34, title, "title"))
    if subtitle:
        parts.append(_text(24, 56, subtitle, "subtitle"))

    # x gridlines (decades if log)
    if log:
        d0, d1 = math.floor(lo), math.ceil(hi)
        for d in range(d0, d1 + 1):
            gx = xv(10 ** d)
            if gx < x0 - 0.5 or gx > x0 + plot_w + 0.5:
                continue
            parts.append(f'<line class="grid" x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + plot_h}"/>')
            parts.append(_text(gx, y0 + plot_h + 24, f"10{_superscript(d)}", "tick", "middle"))
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>')
    parts.append(_text(x0 + plot_w / 2, height - 12, "Seconds per sample (log scale)", "axis-title", "middle"))

    n = len(labels)
    row_h = plot_h / n
    bar_h = row_h * 0.62
    hset = set(highlight_labels)
    for i, (lab, v) in enumerate(zip(labels, values)):
        cy = y0 + i * row_h + row_h / 2
        bx = xv(v)
        color = ACCENT if lab in hset else "#4a6670"
        parts.append(
            f'<rect x="{x0:.1f}" y="{cy - bar_h / 2:.1f}" width="{max(bx - x0, 1):.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="2"/>'
        )
        parts.append(_text(x0 - 14, cy + 4, lab, "tick-strong", "end"))
        parts.append(_text(bx + 10, cy + 4, value_fmt(v), "value", "start"))

    return _document(width, height, "\n".join(parts))


# ==============================================================================
# 3. Confusion matrix heatmap  (confusion_matrix_*)
# ==============================================================================
def confusion_matrix(
    tn: int, fp: int, fn: int, tp: int,
    *,
    title: str = "",
    x_labels: Tuple[str, str] = ("Normal", "Anomaly"),
    y_labels: Tuple[str, str] = ("Normal", "Anomaly"),
    width: int = 620,
    height: int = 560,
) -> str:
    cm = [[tn, fp], [fn, tp]]
    row_tot = [tn + fp, fn + tp]
    vmax = max(tn, fp, fn, tp) or 1

    m_left, m_right, m_top, m_bottom = 132, 40, 92, 84
    grid_w = width - m_left - m_right
    grid_h = height - m_top - m_bottom
    cell_w = grid_w / 2
    cell_h = grid_h / 2
    x0, y0 = m_left, m_top

    parts: List[str] = []
    if title:
        parts.append(_text(width / 2, 40, title, "title", "middle"))

    for r in range(2):
        for c in range(2):
            val = cm[r][c]
            frac = (val / row_tot[r]) if row_tot[r] else 0.0
            fill = mono_color(val / vmax)
            tx = x0 + c * cell_w
            ty = y0 + r * cell_h
            parts.append(
                f'<rect x="{tx:.1f}" y="{ty:.1f}" width="{cell_w - 3:.1f}" '
                f'height="{cell_h - 3:.1f}" fill="{fill}" rx="4"/>'
            )
            tcol = text_on(fill)
            cx = tx + cell_w / 2
            cy = ty + cell_h / 2
            parts.append(
                f'<text x="{cx:.1f}" y="{cy - 4:.1f}" text-anchor="middle" '
                f'class="cm-count" fill="{tcol}">{val:,}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{cy + 26:.1f}" text-anchor="middle" '
                f'class="cm-frac" fill="{tcol}">{frac:.1%}</text>'
            )

    # axis labels
    for c, lab in enumerate(x_labels):
        parts.append(_text(x0 + c * cell_w + cell_w / 2, y0 - 16, lab, "tick-strong", "middle"))
    parts.append(_text(x0 + grid_w / 2, height - 24, "Predicted label", "axis-title", "middle"))
    for r, lab in enumerate(y_labels):
        yy = y0 + r * cell_h + cell_h / 2
        parts.append(_text(x0 - 18, yy + 4, lab, "tick-strong", "end"))
    parts.append(
        f'<text x="34" y="{y0 + grid_h / 2:.1f}" text-anchor="middle" class="axis-title" '
        f'transform="rotate(-90 34 {y0 + grid_h / 2:.1f})">True label</text>'
    )
    extra = (f"\n    .cm-count {{ font: 600 30px {FONT}; }}"
             f"\n    .cm-frac {{ font: 400 17px {FONT}; opacity: 0.85; }}")
    return _document(width, height, "\n".join(parts), extra)
# ==============================================================================
# 4. Scatter plot  (deploy_tradeoff, score_vs_deviation)
# ==============================================================================
def scatter(
    points: Sequence[dict],
    *,
    title: str = "",
    subtitle: str = "",
    x_label: str = "",
    y_label: str = "",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    log_x: bool = False,
    log_y: bool = False,
    threshold: Optional[float] = None,
    legend: Optional[Sequence[Tuple[str, str]]] = None,
    width: int = 1040,
    height: int = 620,
) -> str:
    """Generic scatter. Each point is a dict:
    {x, y, color, r?, label?, highlight?}
    """
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    xr = x_range or (min(xs), max(xs))
    yr = y_range or (min(ys), max(ys))

    m_left = 78
    m_right = 210 if legend else 40
    m_top, m_bottom = 74, 66
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    x0, y0 = m_left, m_top

    def xv(v: float) -> float:
        if log_x:
            t = (math.log10(max(v, xr[0])) - math.log10(xr[0])) / (math.log10(xr[1]) - math.log10(xr[0]))
        else:
            t = (v - xr[0]) / (xr[1] - xr[0]) if xr[1] != xr[0] else 0.5
        return x0 + plot_w * t

    def yv(v: float) -> float:
        if log_y:
            t = (math.log10(max(v, yr[0])) - math.log10(yr[0])) / (math.log10(yr[1]) - math.log10(yr[0]))
        else:
            t = (v - yr[0]) / (yr[1] - yr[0]) if yr[1] != yr[0] else 0.5
        return y0 + plot_h * (1 - t)

    parts: List[str] = []
    if title:
        parts.append(_text(24, 34, title, "title"))
    if subtitle:
        parts.append(_text(24, 56, subtitle, "subtitle"))

    # y gridlines
    if log_y:
        d0, d1 = math.floor(math.log10(yr[0])), math.ceil(math.log10(yr[1]))
        for d in range(d0, d1 + 1):
            gy = yv(10 ** d)
            if gy < y0 - 0.5 or gy > y0 + plot_h + 0.5:
                continue
            parts.append(f'<line class="grid" x1="{x0}" y1="{gy:.1f}" x2="{x0 + plot_w}" y2="{gy:.1f}"/>')
            parts.append(_text(x0 - 12, gy + 4, f"10{_superscript(d)}", "tick", "end"))
    else:
        steps = 5
        for i in range(steps + 1):
            v = yr[0] + (yr[1] - yr[0]) * i / steps
            gy = yv(v)
            parts.append(f'<line class="grid" x1="{x0}" y1="{gy:.1f}" x2="{x0 + plot_w}" y2="{gy:.1f}"/>')
            parts.append(_text(x0 - 12, gy + 4, _fmt(v, 2), "tick", "end"))

    # x gridlines
    if log_x:
        d0, d1 = math.floor(math.log10(xr[0])), math.ceil(math.log10(xr[1]))
        for d in range(d0, d1 + 1):
            gx = xv(10 ** d)
            if gx < x0 - 0.5 or gx > x0 + plot_w + 0.5:
                continue
            parts.append(f'<line class="grid" x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + plot_h}"/>')
            parts.append(_text(gx, y0 + plot_h + 24, f"10{_superscript(d)}", "tick", "middle"))
    else:
        steps = 5
        for i in range(steps + 1):
            v = xr[0] + (xr[1] - xr[0]) * i / steps
            gx = xv(v)
            parts.append(f'<line class="grid" x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + plot_h}"/>')
            parts.append(_text(gx, y0 + plot_h + 24, _fmt(v, 1), "tick", "middle"))

    parts.append(f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>')

    if threshold is not None:
        ty = yv(threshold)
        parts.append(
            f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0 + plot_w}" y2="{ty:.1f}" '
            f'stroke="{INK}" stroke-width="1.3" stroke-dasharray="6 4" opacity="0.55"/>'
        )
        parts.append(_text(x0 + plot_w - 6, ty - 8, "threshold", "tick", "end"))

    # points (highlighted drawn last)
    ordered = sorted(points, key=lambda p: 1 if p.get("highlight") else 0)
    for p in ordered:
        cx, cy = xv(p["x"]), yv(p["y"])
        r = p.get("r", 6)
        color = p.get("color", INK_SOFT)
        opacity = p.get("opacity", 0.9)
        if p.get("highlight"):
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 5:.1f}" fill="none" stroke="{ACCENT}" stroke-width="2"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}" opacity="{opacity}"/>')
        if p.get("label"):
            anchor = p.get("label_anchor", "start")
            gap = r + 6
            if "label_dx" in p:
                dx = p["label_dx"]
            elif anchor == "end":
                dx = -gap
            elif anchor == "middle":
                dx = 0
            else:
                dx = gap
            dy = p.get("label_dy", 4)
            parts.append(_text(cx + dx, cy + dy, p["label"], "value", anchor))

    # axis titles
    parts.append(_text(x0 + plot_w / 2, height - 14, x_label, "axis-title", "middle"))
    parts.append(
        f'<text x="24" y="{y0 + plot_h / 2:.1f}" text-anchor="middle" class="axis-title" '
        f'transform="rotate(-90 24 {y0 + plot_h / 2:.1f})">{_esc(y_label)}</text>'
    )

    if legend:
        lx = x0 + plot_w + 30
        ly = y0 + 10
        for i, (lab, col) in enumerate(legend):
            yy = ly + i * 28
            parts.append(f'<circle cx="{lx + 7}" cy="{yy}" r="7" fill="{col}"/>')
            parts.append(_text(lx + 22, yy + 5, lab, "legend"))

    return _document(width, height, "\n".join(parts))


# ==============================================================================
# 5. Vertical grouped/segmented bar for error rates  (error_by_component)
# ==============================================================================
def rate_bars(
    categories: Sequence[str],
    values: Sequence[float],
    colors: Sequence[str],
    *,
    title: str = "",
    subtitle: str = "",
    y_label: str = "",
    y_max: Optional[float] = None,
    width: int = 1040,
    height: int = 560,
) -> str:
    y_max = y_max or max(max(values) * 1.3, 5.0)
    m_left, m_right, m_top, m_bottom = 68, 40, 74, 66
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    x0, y0 = m_left, m_top

    def yv(v: float) -> float:
        return y0 + plot_h * (1 - v / y_max)

    parts: List[str] = []
    if title:
        parts.append(_text(24, 34, title, "title"))
    if subtitle:
        parts.append(_text(24, 56, subtitle, "subtitle"))

    steps = 5
    for i in range(steps + 1):
        v = y_max * i / steps
        gy = yv(v)
        parts.append(f'<line class="grid" x1="{x0}" y1="{gy:.1f}" x2="{x0 + plot_w}" y2="{gy:.1f}"/>')
        parts.append(_text(x0 - 12, gy + 4, f"{v:.0f}", "tick", "end"))
    parts.append(f'<line class="axis" x1="{x0}" y1="{yv(0):.1f}" x2="{x0 + plot_w}" y2="{yv(0):.1f}"/>')

    n = len(categories)
    slot = plot_w / n
    bar_w = slot * 0.56
    for i, (cat, v, col) in enumerate(zip(categories, values, colors)):
        bx = x0 + i * slot + (slot - bar_w) / 2
        by = yv(v)
        parts.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{yv(0) - by:.1f}" '
            f'fill="{col}" rx="2"/>'
        )
        parts.append(_text(bx + bar_w / 2, by - 9, f"{v:.1f}%", "value", "middle"))
        parts.append(_text(x0 + i * slot + slot / 2, y0 + plot_h + 28, cat, "tick-strong", "middle"))

    parts.append(
        f'<text x="24" y="{y0 + plot_h / 2:.1f}" text-anchor="middle" class="axis-title" '
        f'transform="rotate(-90 24 {y0 + plot_h / 2:.1f})">{_esc(y_label)}</text>'
    )
    return _document(width, height, "\n".join(parts))
