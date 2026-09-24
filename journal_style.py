"""Journal figure style — drop-in replacement for report_style.

Conventions for print journals: serif text (Times New Roman, STIX math), a muted
colorblind-safe palette (Wong), thin rules, light dotted grid, no in-figure titles
(captions live in the manuscript), frameless compact legends.
"""
import matplotlib as mpl

# Wong colorblind-safe palette, muted for print
NAVY = "#1A1A1A"    # primary line: near-black
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#D55E00"
GRID = "#DDDDDD"
GREY = "#7F7F7F"

IN_FIGURE_TITLES = False   # schematic scripts check this before drawing header text


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": ":",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
    })


def title(ax, main, sub=None):
    """No in-figure titles in journal format; captions carry this information."""
    return None
