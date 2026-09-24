"""Drawdown (underwater) chart for the model arms + 1/N, styled to match res_performance.png."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import report_style as rs; rs.apply()

ROOT = Path(__file__).resolve().parent; RES = ROOT / "results"


def load_pnl(folder):
    p = pd.read_csv(RES / folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float).iloc[1:]


def oneN():
    real = pd.read_csv(RES / "main_dyn_off" / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"])
    real = real.set_index("Unnamed: 0"); rsimple = np.exp(real) - 1.0
    return pd.Series({d: float(rsimple.loc[d].dropna().mean()) for d in rsimple.index[1:]})


def dd(r):
    eq = (1 + r).cumprod(); return (eq / eq.cummax() - 1) * 100


arms = {"Combined (managed)": ("main_combined", rs.NAVY, 2.4),
        "Realized-vol overlay": ("main_voltarget", rs.BLUE, 1.6),
        "Dynamic base": ("main_dyn_strong", rs.GREEN, 1.4),
        "Static allocator": ("main_dyn_off", rs.GREY, 1.4),
        "1/N benchmark": ("__1n__", rs.ORANGE, 1.6)}

fig, ax = plt.subplots(figsize=(12, 5))
for lbl, (f, col, lw) in arms.items():
    d = dd(oneN() if f == "__1n__" else load_pnl(f))
    if lbl == "Combined (managed)":
        ax.fill_between(d.index, 0, d.values, color=rs.NAVY, alpha=0.08)
    ax.plot(d.index, d.values, color=col, lw=lw, label=lbl)
ax.axhline(0, color=rs.GRID, lw=1)
ax.set_ylabel("Drawdown (%)"); ax.grid(axis="y")
ax.legend(loc="lower left", fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
rs.title(ax, "Drawdowns, 2019-2025", "Underwater curve (peak-to-trough), out-of-sample, net of costs")
fig.tight_layout(); fig.savefig(ROOT / "res_drawdown.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved res_drawdown.png")
for lbl, (f, col, lw) in arms.items():
    d = dd(oneN() if f == "__1n__" else load_pnl(f))
    print(f"  {lbl:22s} worst = {d.min():6.2f}%")
