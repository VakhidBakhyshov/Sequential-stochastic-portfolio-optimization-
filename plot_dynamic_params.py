"""Line charts of the dynamic optimizer parameters over the rebalance months.

Reads results/main_dyn_strong/dynamic_parameter_history.csv (written by run.py for the Dynamic arm)
and plots how the two LIVE parameters moved and what drove them:
  Panel 1: confidence_level  (alpha, the CVaR tail level)   -- base 0.95, band [0.90, 0.995]
  Panel 2: turnover_penalty  (lambda, L1 trade cost)        -- base 0.01
  Panel 3: the driving scores: total = 0.70*performance + 0.30*regime
Saves dynamic_params_timeseries.png
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "main_dyn_strong" / "dynamic_parameter_history.csv"
BASE_ALPHA, A_MIN, A_MAX = 0.95, 0.90, 0.995
BASE_LAMBDA = 0.01

d = pd.read_csv(CSV)
d["date"] = pd.to_datetime(d["date"])
d = d[d["date"] >= "2019-01-01"].reset_index(drop=True)
x = d["date"].values

fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# --- Panel 1: confidence_level (alpha) ---
ax[0].axhspan(A_MIN, A_MAX, color="0.92", zorder=0, label=f"band [{A_MIN}, {A_MAX}]")
ax[0].axhline(BASE_ALPHA, color="0.45", ls="--", lw=1, label=f"base {BASE_ALPHA}")
ax[0].plot(x, d["confidence_level"], color="#c0392b", lw=2.0, marker="o", ms=3, label="confidence_level α")
ax[0].fill_between(x, BASE_ALPHA, d["confidence_level"],
                   where=d["confidence_level"] >= BASE_ALPHA, color="#c0392b", alpha=0.15)
ax[0].set_ylabel("α  (CVaR tail level)")
ax[0].set_title("Dynamic optimizer parameters over the rebalance — Dynamic arm (main_dyn_strong)")
ax[0].legend(loc="upper left", fontsize=8, ncol=3); ax[0].grid(alpha=0.25)
ax[0].annotate("higher α = stricter / deeper tail\n(controller tightens in stress)",
               xy=(0.992, 0.04), xycoords="axes fraction", ha="right", va="bottom", fontsize=8, color="#c0392b")

# --- Panel 2: turnover_penalty (lambda) ---
ax[1].axhline(BASE_LAMBDA, color="0.45", ls="--", lw=1, label=f"base {BASE_LAMBDA}")
ax[1].plot(x, d["turnover_penalty"], color="#2471a3", lw=2.0, marker="o", ms=3, label="turnover_penalty λ")
ax[1].set_ylabel("λ  (L1 trade cost)")
ax[1].legend(loc="upper left", fontsize=8); ax[1].grid(alpha=0.25)
ax[1].annotate("floored at 0.01 in stress (cheaper to re-trade);\nrises only in calm",
               xy=(0.992, 0.92), xycoords="axes fraction", ha="right", va="top", fontsize=8, color="#2471a3")

# --- Panel 3: scores ---
ax[2].axhline(0.0, color="0.6", lw=0.8)
ax[2].plot(x, d["total_score"], color="black", lw=2.2, marker="o", ms=3, label="total_score (0.7·perf + 0.3·regime)")
ax[2].plot(x, d["performance_score"], color="#27ae60", lw=1.3, alpha=0.9, label="performance_score")
ax[2].plot(x, d["regime_score"], color="#e67e22", lw=1.3, alpha=0.9, label="regime_score")
ax[2].fill_between(x, 0, d["total_score"], where=d["total_score"] < 0, color="red", alpha=0.10)
ax[2].set_ylabel("score  [-1, +1]"); ax[2].set_ylim(-1.05, 1.05)
ax[2].legend(loc="upper left", fontsize=8, ncol=3); ax[2].grid(alpha=0.25)
ax[2].annotate("negative score = stress → raises α, lowers λ", xy=(0.992, 0.05),
               xycoords="axes fraction", ha="right", va="bottom", fontsize=8)

import matplotlib.dates as mdates
ax[2].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax[2].get_xticklabels(), rotation=90, fontsize=8)
fig.tight_layout()
out = ROOT / "dynamic_params_timeseries.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out.name)
print("α: min=%.4f median=%.4f max=%.4f" % (d.confidence_level.min(), d.confidence_level.median(), d.confidence_level.max()))
print("λ: min=%.4f median=%.4f max=%.4f" % (d.turnover_penalty.min(), d.turnover_penalty.median(), d.turnover_penalty.max()))
