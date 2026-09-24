"""Shared report style (per strategy_style.md) for all charts/tables going forward.

Usage:
    import report_style as rs
    rs.apply()                      # set matplotlib rcParams
    ... ax.plot(..., color=rs.NAVY)
"""
from __future__ import annotations
import matplotlib as mpl

BG = "#FFFFFF"; NAVY = "#163A5F"; BLUE = "#2E86C1"; GREEN = "#3CA370"
ORANGE = "#F5A623"; RED = "#C94C4C"; GRID = "#E5E7EB"; GREY = "#6B7280"
PALETTE = dict(bg=BG, navy=NAVY, blue=BLUE, green=GREEN, orange=ORANGE, red=RED, grid=GRID, grey=GREY)


def apply():
    mpl.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Open Sans", "Roboto", "Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": NAVY, "axes.titlecolor": NAVY, "axes.titleweight": "bold",
        "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
        "axes.edgecolor": GREY, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9, "legend.frameon": False,
    })


def title(ax, title_text, subtitle=None):
    """Navy bold title + grey subtitle on white, per the spec."""
    ax.set_title(title_text, color=NAVY, fontweight="bold", fontsize=16, loc="left", pad=18)
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, color=GREY, fontsize=11, va="bottom", ha="left")
