"""Colours and matplotlib defaults for the dashboard."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

BG = "#06080c"
PANEL = "#0a0f15"
GRID = "#18222d"
FRAME = "#223040"
TEXT = "#cfdae6"
DIM = "#5f6e7e"
FAINT = "#334250"
ACCENT = "#2ee6c5"  # teal
ACCENT_2 = "#ff4d8d"  # pink
TRACK = "#ffcf5a"  # amber
TRUTH = "#8696a7"

FONT = ["DejaVu Sans Mono", "Menlo", "Consolas", "monospace"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": FRAME,
        "axes.labelcolor": DIM,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.labelsize": 8,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "font.family": "monospace",
        "font.monospace": FONT,
        "text.color": TEXT,
        "savefig.facecolor": BG,
    })


def radar_glow() -> ListedColormap:
    """Transparent at the noise floor, through deep teal to near white."""
    base = LinearSegmentedColormap.from_list(
        "glow", ["#000000", "#06343a", "#0b7f7a", "#2ee6c5", "#e8fff9"], N=256)(np.linspace(0, 1, 256))
    base[:, 3] = np.clip(np.linspace(-0.05, 1.6, 256), 0, 1) ** 0.8
    return ListedColormap(base, "radar_glow")


def heat() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "heat", ["#06080c", "#1a0b2e", "#4b0f5c", "#a3165f", "#f0465a", "#ff9e4a", "#fff3b0"], N=256)


def velocity_cmap() -> LinearSegmentedColormap:
    """Approaching (negative radial velocity) warm, receding cool."""
    return LinearSegmentedColormap.from_list(
        "vel", ["#ff3b5c", "#ff9e4a", "#f2f2f2", "#4fc3ff", "#3a6bff"], N=256)
