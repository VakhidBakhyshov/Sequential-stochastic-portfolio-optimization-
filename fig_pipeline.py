"""Pipeline diagram for the paper (house style): filtering -> engine -> allocation -> controller -> results."""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import report_style as rs; rs.apply()

NAVY, BLUE, GREEN, ORANGE, RED, GRID, GREY = rs.NAVY, rs.BLUE, rs.GREEN, rs.ORANGE, rs.RED, rs.GRID, rs.GREY

fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 100); ax.set_ylim(-1, 110); ax.axis("off")


def box(x, y, w, h, title, lines, fc="#FFFFFF", ec=NAVY, title_c=None, lw=1.4, fs=8.0, tfs=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=tfs,
            fontweight="bold", color=title_c or ec, zorder=4)
    if lines:
        ax.text(x + w / 2, y + h - 7.2, "\n".join(lines), ha="center", va="top",
                fontsize=fs, color="#333333", zorder=4, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=NAVY, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13,
                                 color=color, linewidth=lw, linestyle=ls, zorder=2,
                                 shrinkA=2, shrinkB=2))


# ---------------- top row: data -> screen -> universe ----------------
box(2, 82, 24, 15, "Raw ETF panel",
    ["~3,100 US-listed funds", "daily prices, volumes,", "total returns"], ec=GREY, title_c=GREY)
box(33, 82, 30, 15, "Monthly eligibility screen (causal)",
    ["traded value  A = ((Pᵒ+Pᶜ)/2)·V",
     "36-month continuous trading  AND",
     "36m mean dollar-volume share (lag 1m) > 5×10⁻⁵"], ec=NAVY)
box(70, 82, 28, 15, "Point-in-time universe  Aₜ",
    ["211–254 eligible funds / month", "median 242 · entries and exits", "each month; no survivorship"],
    ec=GREEN, title_c=GREEN)
arrow(26, 89.5, 33, 89.5); arrow(63, 89.5, 70, 89.5)

# ---------------- monthly loop container ----------------
ax.add_patch(FancyBboxPatch((2, 22), 96, 54, boxstyle="round,pad=0.8,rounding_size=1.8",
                            facecolor="#F7F9FC", edgecolor=GRID, linewidth=1.2, zorder=1))
ax.text(4, 73.5, "EACH REBALANCE  t   (walk-forward, information set F_t only)",
        fontsize=9, fontweight="bold", color=NAVY, zorder=4)

# engine row
box(4, 52, 20, 16, "Estimation window",
    ["3 years of daily returns", "for eligible funds"], ec=GREY, title_c=GREY)
box(28, 52, 24, 16, "Scenario engine",
    ["μ̂: shrinkage + momentum tilt",
     "Σ̂: Ledoit–Wolf → L",
     "R̃ = 1μ̂′ + ZL′   (1000×Nₜ)"], ec=BLUE)
box(56, 52, 24, 16, "CVaR-budget allocation (LP)",
    ["max μ̂′w − λₜ·turnover",
     "s.t. CVaR_{βₜ}(w) ≤ γ·CVaR(w^{EW})",
     "0 ≤ w ≤ 12%,  1′w = 1"], ec=NAVY)
box(84, 52, 12, 16, "Execution",
    ["partial:", "ηₜ toward wₜ", "10 bp costs"], ec=GREY, title_c=GREY)
arrow(24, 60, 28, 60); arrow(52, 60, 56, 60); arrow(80, 60, 84, 60)

# adaptive controller (feeds LP)
box(56, 27.5, 24, 13, "Adaptive parameters",
    ["score gₜ: own record + regime",
     "βₜ = 0.95 − 0.05gₜ  in [0.92, 0.97]",
     "λₜ = 0.01·e^{0.55gₜ}"], ec=ORANGE, title_c=ORANGE)
arrow(68, 40.5, 68, 52, color=ORANGE)

# exposure controller row
box(4, 27.5, 22, 13, "Fast dial  kᶠ",
    ["21-day realized vol", "of the held book", "kᶠ = clip(12%/σ̂₂₁)"], ec=BLUE)
box(30, 27.5, 22, 13, "Slow dial  kˢ",
    ["internal CVaR at β* = 0.95", "centred, median-normalized", "(read off the LP's own fan)"], ec=ORANGE, title_c=ORANGE)
arrow(40, 52, 41, 40.5, color=ORANGE, ls="--", lw=1.2)   # fan -> slow dial
ax.text(35.5, 46.5, "same scenario fan", fontsize=7, color=ORANGE, rotation=0)

box(13, 8, 30, 13, "HDRC exposure",
    ["kₜ = clip(kᶠ·kˢ, 0.30, 1)",
     "book: kₜ · wₜᵉˣᵉᶜ → cash: 1−kₜ"], ec=NAVY, fc="#EEF3F9", lw=2.0)
arrow(15, 27.5, 22, 21, color=BLUE); arrow(41, 27.5, 34, 21, color=ORANGE)

box(52, 8, 24, 13, "Realized month  t+1",
    ["r = kₜ rᵇ + (1−kₜ) r_f", "net of costs"], ec=GREEN, title_c=GREEN)
arrow(43, 14.5, 52, 14.5, color=NAVY, lw=2.0)

# note under execution
ax.text(90, 50.4, "executed book is held\nthrough month t+1", fontsize=7.0, color=GREY, ha="center", va="top")

# feedback
arrow(64, 8, 64, 5.2, color=GREY, lw=1.1)
arrow(64, 5.2, 10, 5.2, color=GREY, lw=1.1, ls="--")
arrow(10, 5.2, 10, 8, color=GREY, lw=1.1)
ax.text(37, 2.8, "state feedback into month t+1: executed weights, controller score", fontsize=7.2, color=GREY, ha="center")
arrow(84, 82, 84, 76.8, color=GREEN, lw=1.4)        # universe into loop

# evaluation box (right, outside loop)
box(80, 8, 18, 13, "Evaluation",
    ["benchmarks + controls", "paired bootstrap", "PSR / DSR"], ec=RED, title_c=RED)
arrow(76, 14.5, 80, 14.5, color=RED)

if getattr(rs, "IN_FIGURE_TITLES", True):
    ax.text(2, 109, "Monthly decision pipeline: from raw panel to evaluated returns",
            fontsize=13, fontweight="bold", color=NAVY, va="top")
    ax.text(2, 104.5, "Universe screening is offline and causal; allocation, exposure control and evaluation run inside the walk-forward loop",
            fontsize=9, color=GREY, va="top")

fig.tight_layout()
fig.savefig("fig_pipeline.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved fig_pipeline.png")
