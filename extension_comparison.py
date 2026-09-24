"""Extension section artifacts: multi-signal (colleague engine) vs HDRC (main study).

Outputs: fig_extension.png (growth + drawdown), fig_riskreturn_map.png (operating points),
extension_comparison.csv (metrics), and bootstrap comparisons printed.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import report_style as rs; rs.apply()

OURS = Path(__file__).resolve().parent / "results"
THEIRS = Path(__file__).resolve().parent / "companion" / "results"
RF = 0.02; SEED = 42


def load(root, arm):
    p = pd.read_csv(root / arm / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float).iloc[1:]


def one_n():
    real = pd.read_csv(OURS / "main_dyn_off" / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"])
    real = real.set_index("Unnamed: 0"); rsx = np.exp(real) - 1.0
    return pd.Series({d: float(rsx.loc[d].dropna().mean()) for d in rsx.index[1:]})


def met(r):
    rv = np.asarray(pd.Series(r).dropna(), float); n = len(rv)
    tot = float(np.prod(1 + rv) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = rv.std(ddof=1) * np.sqrt(12); ex = rv - RF / 12
    sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    eq = np.cumprod(1 + rv); dr = eq / np.maximum.accumulate(eq) - 1; mdd = float(dr.min())
    dn = rv[rv < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    v95 = np.percentile(rv, 5); cv = float(rv[rv <= v95].mean())
    return dict(Cumul=tot * 100, Ann=ann * 100, Vol=vol * 100, Sharpe=sh,
                Sortino=(ann - RF) / ddv if ddv and ddv > 0 else np.nan,
                MaxDD=mdd * 100, Calmar=ann / abs(mdd) if mdd < 0 else np.nan, CVaR95=cv * 100)


def ann_sharpe(r):
    r = np.asarray(r, float); ex = r - RF / 12; return ex.mean() / ex.std(ddof=1) * np.sqrt(12)


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


series = {
    "HDRC combined (main study)": load(OURS, "main_combined"),
    "Multi-signal (extension)": load(THEIRS, "advanced_smart_bayesian"),
    "Extension-engine base": load(THEIRS, "main_dyn_strong"),
    "Extension-engine fast dial": load(THEIRS, "main_voltarget"),
    "Extension-engine two-signal": load(THEIRS, "main_combined"),
    "Main-study base": load(OURS, "main_dyn_strong"),
    "1/N": one_n(),
}
T = pd.DataFrame({k: met(v) for k, v in series.items()}).T
T.round(3).to_csv(Path(__file__).resolve().parent / "extension_comparison.csv")
print(T.round(3).to_string())

# ---------------- bootstrap ----------------
rng = np.random.default_rng(SEED); B = 5000
print()
for lbl, a_name, b_name in [
        ("Multi-signal vs HDRC", "HDRC combined (main study)", "Multi-signal (extension)"),
        ("Multi-signal vs own base", "Extension-engine base", "Multi-signal (extension)"),
        ("Ext two-signal vs ext base", "Extension-engine base", "Extension-engine two-signal")]:
    a, b = series[a_name], series[b_name]
    j = a.index.intersection(b.index); x, y = a.loc[j].values, b.loc[j].values; n = len(j)
    sh = np.empty(B); dd = np.empty(B)
    for t in range(B):
        ii = rng.integers(0, n, n)
        sh[t] = ann_sharpe(y[ii]) - ann_sharpe(x[ii]); dd[t] = max_dd(y[ii]) - max_dd(x[ii])
    print(f"  {lbl:28s} n={n}  dSharpe={ann_sharpe(y)-ann_sharpe(x):+.3f} P={float((sh>0).mean()):.3f} | "
          f"dMaxDD={max_dd(y)-max_dd(x):+.3f} P(shallower)={float((dd>0).mean()):.3f}")

# ---------------- figure 10: growth + drawdown ----------------
show = [("Multi-signal (extension)", rs.RED, 2.0),
        ("Extension-engine base", rs.GREY, 1.3),
        ("HDRC combined (main study)", rs.NAVY, 2.2),
        ("1/N", rs.ORANGE, 1.4)]
fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.5, 1]})
for lbl, col, lw in show:
    r = series[lbl]
    eq = (1 + r).cumprod()
    axes[0].plot(eq.index, eq.values, color=col, lw=lw, label=lbl)
    dd = (eq / eq.cummax() - 1) * 100
    axes[1].plot(dd.index, dd.values, color=col, lw=lw)
axes[0].axhline(1, color=rs.GRID, lw=1); axes[0].set_ylabel("growth of 1.0")
axes[0].legend(loc="upper left", fontsize=8.5); axes[0].grid(axis="y")
axes[1].axhline(0, color=rs.GRID, lw=1); axes[1].set_ylabel("drawdown (%)"); axes[1].grid(axis="y")
axes[1].xaxis.set_major_locator(mdates.YearLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
rs.title(axes[0], "Extension vs main study", "Multi-signal arm pursues return; HDRC pursues drawdown control; both net of costs")
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "fig_extension.png", dpi=150, bbox_inches="tight")

# ---------------- figure 11: risk-return operating map ----------------
pts = {
    "HDRC (main)": ("main", rs.NAVY),
    "Multi-signal (ext)": ("ext", rs.RED),
    "Ext base": ("ext", rs.GREY),
    "Ext fast dial": ("ext", rs.BLUE),
    "Ext two-signal": ("ext", rs.GREEN),
    "Main base": ("main", rs.GREY),
    "1/N": ("bench", rs.ORANGE),
}
name_map = {"HDRC (main)": "HDRC combined (main study)", "Multi-signal (ext)": "Multi-signal (extension)",
            "Ext base": "Extension-engine base", "Ext fast dial": "Extension-engine fast dial",
            "Ext two-signal": "Extension-engine two-signal", "Main base": "Main-study base", "1/N": "1/N"}
fig, ax = plt.subplots(figsize=(9.2, 6))
for short, (fam, col) in pts.items():
    m = met(series[name_map[short]])
    marker = "o" if fam == "main" else ("s" if fam == "ext" else "D")
    ax.scatter(-m["MaxDD"], m["Ann"], s=110, color=col, marker=marker, zorder=3,
               edgecolor="white", linewidth=1.2)
    dx, dy = 0.35, 0.15
    ax.annotate(short, (-m["MaxDD"] + dx, m["Ann"] + dy), fontsize=8.6, color=col, fontweight="bold")
ax.set_xlabel("maximum drawdown (%, absolute)"); ax.set_ylabel("annualized return (%)")
ax.grid(True, alpha=0.5)
rs.title(ax, "Operating points: return vs drawdown", "Circles: main study - squares: extension engine - diamond: benchmark")
fig.tight_layout(); fig.savefig(Path(__file__).resolve().parent / "fig_riskreturn_map.png", dpi=150, bbox_inches="tight")
print("\nsaved fig_extension.png, fig_riskreturn_map.png, extension_comparison.csv")
