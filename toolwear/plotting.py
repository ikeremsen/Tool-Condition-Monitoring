"""Shared matplotlib style so every figure in the project looks the same."""

import matplotlib as mpl

# Colour-blind-checked categorical order: blue, orange, aqua. Use in this order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SHADE = "#f0efec"        # neutral fill for highlighted regions
CRITICAL = "#d03b3b"     # reserved for "bad data" markers, never a series colour


def use_project_style():
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })
