"""
Deck Visual Style
=================

Shared matplotlib styling that makes every generated figure match the editorial
aesthetic of the CMMSE 2026 slide deck (``docs/presentations``): a warm off-white
paper background, near-black ink text, thin grotesk-style type, hairline gridlines,
and a small curated palette of muted, print-friendly hues.

Usage
-----
Call :func:`apply_deck_style` once before plotting (the plotting helpers in this
package already do this), then use the palette helpers for colours::

    from .deck_style import apply_deck_style, categorical_colors, DIVERGING_CMAP
    apply_deck_style()
    colors = categorical_colors(n_models)
"""
from typing import List

import numpy as np
import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# --- Core deck palette (mirrors the CSS custom properties in the deck) --------
PAPER = "#f7f4ef"        # --bg-primary : warm off-white slide background
PAPER_SOFT = "#fbf9f5"   # --bg-secondary
INK = "#121212"          # --text-primary
INK_SOFT = "#43413d"     # --text-secondary
MUTED = "#8a837b"        # --text-muted
LINE = "#e4ded4"         # hairline gridlines  (~rgba(17,17,17,0.08) over paper)
LINE_STRONG = "#c9c1b5"  # spines / stronger rules

# --- Curated qualitative palette ---------------------------------------------
# Muted, editorial hues chosen to stay legible on the warm paper background and
# to remain distinguishable when printed. Ordered for maximum separation between
# adjacent entries (used for models / export formats / grouped bars).
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

# Accent used for "the best / highlighted" markers.
ACCENT = "#a7623f"


def _mono_cmap() -> LinearSegmentedColormap:
    """Sequential cream -> ink ramp for heatmaps."""
    return LinearSegmentedColormap.from_list(
        "deck_mono", [PAPER_SOFT, "#d8cdbb", "#8f8578", "#4a453f", INK], N=256
    )


def _diverging_cmap() -> LinearSegmentedColormap:
    """Muted slate-blue (negative) -> paper -> terracotta (positive)."""
    return LinearSegmentedColormap.from_list(
        "deck_diverging",
        ["#33506f", "#6e88a6", "#b7c2cf", PAPER_SOFT, "#e0b49c", "#c07a53", "#9a3f22"],
        N=256,
    )


MONO_CMAP = _mono_cmap()
DIVERGING_CMAP = _diverging_cmap()
# Sequential ramp for scalar-encoded scatter/lines (e.g. accuracy), kept warm.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "deck_seq", ["#33506f", "#7d8ca0", "#c7a98a", "#a7623f", "#6e2f18"], N=256
)


def apply_deck_style() -> None:
    """Apply the deck's editorial look to matplotlib's global rcParams."""
    mpl.rcParams.update(
        {
            # Canvas
            "figure.facecolor": PAPER,
            "figure.edgecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.edgecolor": PAPER,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            # Default colour cycle -> curated deck palette
            "axes.prop_cycle": cycler(color=CATEGORICAL),
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.weight": "light",
            "text.color": INK,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            # Sizing
            "axes.titlesize": 15,
            "axes.titleweight": "regular",
            "axes.labelsize": 12,
            "axes.labelweight": "regular",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "figure.titlesize": 17,
            "figure.titleweight": "regular",
            # Spines: keep only left + bottom, hairline weight
            "axes.edgecolor": LINE_STRONG,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Titles hung on the left, editorial style
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            # Grid: faint horizontal hairlines only
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": LINE,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            # Lines / markers
            "lines.linewidth": 1.9,
            "lines.markersize": 6,
            "patch.edgecolor": INK,
            "patch.linewidth": 0.7,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.facecolor": PAPER_SOFT,
            "legend.edgecolor": LINE_STRONG,
            "legend.borderpad": 0.6,
            "legend.labelcolor": INK_SOFT,
        }
    )


def categorical_colors(n: int) -> List[str]:
    """Return ``n`` colours cycling the curated qualitative palette."""
    return [CATEGORICAL[i % len(CATEGORICAL)] for i in range(n)]


def deviation_color(value: float, max_abs: float) -> tuple:
    """Map a signed deviation percentage to a colour on the diverging ramp."""
    max_abs = max(max_abs, 1e-9)
    normalized = (np.clip(value / max_abs, -1.0, 1.0) + 1.0) / 2.0
    return DIVERGING_CMAP(normalized)


def style_axis(ax) -> None:
    """Apply per-axis finishing touches not covered by rcParams."""
    ax.tick_params(length=3.5, width=0.9)
    for spine in ("left", "bottom"):
        if spine in ax.spines:
            ax.spines[spine].set_color(LINE_STRONG)


def text_on(color) -> str:
    """Return a legible text colour (paper or ink) for a given background."""
    r, g, b, _ = to_rgba(color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return PAPER_SOFT if luminance < 0.5 else INK
