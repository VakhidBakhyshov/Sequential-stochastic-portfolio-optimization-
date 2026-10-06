"""Figures of the companion-engine sections built from saved run folders.

    python make_companion_figures.py [--results results] [--out companion_figures]

Figure 6   forward path-risk state of the full CDaR arm (raw CDaR, centred CDaR, centred terminal CVaR)
Figure 7   drift contribution to predictive CDaR (raw minus centred)
Figure 8   realized wealth: full CDaR arm, clean two-signal CDaR arm, matched CVaR arm
Figure 14  execution fraction and monthly transaction cost of the multi-signal arm
Figure L.6 matched CVaR and CDaR: wealth, drawdown, rolling 12-month volatility
Figure L.7 CVaR vs CDaR: L1 weight distance and active-set Jaccard by month
Figure L.8 turnover of the CVaR and CDaR arms (wealth bought plus sold)
Run folders: bayessian_cvar_paired_walk_forward, bayessian_cdar_walk_forward, bayessian_cdar_two_signal_walk_forward,
advanced_smart_bayesian (each with pnl.csv, forecast_risk.csv, weights.xlsx).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CVAR, CDAR, CDAR2, MULTI = ("bayessian_cvar_paired_walk_forward", "bayessian_cdar_walk_forward",
                            "bayessian_cdar_two_signal_walk_forward", "advanced_smart_bayesian")
COL = {"cvar": "#1F3A5F", "cdar": "#C0504D", "cdar2": "#4F81BD", "grey": "#7F7F7F"}


def load_pnl(folder: Path) -> pd.DataFrame:
    p = pd.read_csv(folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c]); p = p.set_index(c)
    return p


def returns(folder: Path) -> pd.Series:
    r = load_pnl(folder)["Returns"].astype(float).dropna()
    return r.iloc[1:] if len(r) and abs(r.iloc[0]) < 1e-15 else r


def weights(folder: Path) -> dict:
    xl = pd.ExcelFile(folder / "weights.xlsx"); out = {}
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        if "weights" not in d.columns:
            continue
        key = "Key" if "Key" in d.columns else d.columns[0]
        s = pd.Series(pd.to_numeric(d["weights"], errors="coerce").fillna(0.0).values, index=d[key].astype(str))
        try:
            out[pd.to_datetime(sh)] = s
        except Exception:
            pass
    return out


def turnover(W: dict) -> pd.Series:
    m = sorted(W); out = {}
    for i in range(1, len(m)):
        ix = W[m[i - 1]].index.union(W[m[i]].index)
        out[m[i]] = float(np.abs(W[m[i]].reindex(ix).fillna(0) - W[m[i - 1]].reindex(ix).fillna(0)).sum())
    return pd.Series(out)


def drawdown(r: pd.Series) -> pd.Series:
    eq = (1 + r).cumprod(); return eq / eq.cummax() - 1


def years(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="companion_figures")
    a = ap.parse_args(); res = Path(a.results); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})

    # Figures 6 and 7: forward path-risk state of the full CDaR arm
    f = pd.read_csv(res / CDAR / "forecast_risk.csv"); f["date"] = pd.to_datetime(f["date"])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(f.date, f.cdar_model_raw, label="Predictive CDaR, raw", lw=1.5, color=COL["grey"])
    ax.plot(f.date, f.cdar_model_centered, label="Predictive CDaR, centred", lw=2, color=COL["cdar"])
    ax.plot(f.date, f.cvar_model_centered, label="Predictive terminal CVaR, centred", lw=1.3, color=COL["cvar"], alpha=0.85)
    if "vol_model" in f:
        ax.plot(f.date, f.vol_model, label="Model volatility", lw=1.0, color=COL["cdar2"], alpha=0.8)
    ax.set_ylabel("Model risk"); ax.set_title("Full CDaR arm: forward path-risk state"); ax.legend(ncol=2); years(ax)
    fig.tight_layout(); fig.savefig(out / "fig06_cdar_path_risk_state.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.2))
    gap = f.cdar_model_raw - f.cdar_model_centered
    ax.bar(f.date, gap, width=20, color=COL["cdar"]); ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("CDaR difference"); ax.set_title("Drift contribution to predictive CDaR (raw minus centred)"); years(ax)
    fig.tight_layout(); fig.savefig(out / "fig07_cdar_drift_contribution.png", dpi=180); plt.close(fig)

    # Figure 8: realized wealth of the three arms
    fig, ax = plt.subplots(figsize=(12, 5))
    for lab, folder, col, ls in [("CDaR, full multi-signal", CDAR, COL["cdar"], "-"), ("CDaR, clean two-signal", CDAR2, COL["cdar2"], "-"),
                                 ("Matched CVaR", CVAR, COL["cvar"], "--")]:
        r = returns(res / folder); ax.plot(r.index, (1 + r).cumprod(), label=lab, lw=2, color=col, ls=ls)
    ax.set_ylabel("Growth of $1"); ax.set_title("Matched CVaR and CDaR arms: realized wealth"); ax.legend(); years(ax)
    fig.tight_layout(); fig.savefig(out / "fig08_cdar_arms_wealth.png", dpi=180); plt.close(fig)

    # Figure 14: execution fraction and transaction cost of the multi-signal arm
    p = load_pnl(res / MULTI).iloc[1:]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].step(p.index, p["Alpha"].astype(float), where="mid", color=COL["cvar"], lw=1.5); ax[0].set_ylabel("Execution fraction")
    ax[0].set_title("Multi-signal arm: selected execution fraction and monthly transaction cost")
    ax[1].bar(p.index, 100 * p["Cost"].astype(float), width=20, color=COL["cdar"]); ax[1].set_ylabel("Cost (% of wealth)"); years(ax[1])
    fig.tight_layout(); fig.savefig(out / "fig14_multisignal_execution_cost.png", dpi=180); plt.close(fig)

    # Figures L.6-L.8: matched CVaR vs CDaR
    rc, rd = returns(res / CVAR), returns(res / CDAR)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for lab, r, col in [("CVaR", rc, COL["cvar"]), ("CDaR", rd, COL["cdar"])]:
        axes[0].plot(r.index, (1 + r).cumprod(), label=lab, lw=2, color=col)
        axes[1].plot(r.index, 100 * drawdown(r), label=lab, lw=2, color=col)
        axes[2].plot(r.index, 100 * r.rolling(12).std() * np.sqrt(12), label=lab, lw=2, color=col)
    axes[0].set_ylabel("Growth of $1"); axes[1].set_ylabel("Drawdown (%)"); axes[2].set_ylabel("12-month volatility (%)")
    axes[0].set_title("Matched risk geometry: terminal CVaR against path CDaR"); axes[0].legend(); years(axes[2])
    fig.tight_layout(); fig.savefig(out / "figL6_cvar_cdar_wealth_drawdown_vol.png", dpi=180); plt.close(fig)

    wc, wd = weights(res / CVAR), weights(res / CDAR); rows = []
    for dt in sorted(set(wc) & set(wd)):
        ix = wc[dt].index.union(wd[dt].index); x = wc[dt].reindex(ix).fillna(0); y = wd[dt].reindex(ix).fillna(0)
        A, B = set(x[x > 1e-9].index), set(y[y > 1e-9].index)
        rows.append({"date": dt, "l1": float(np.abs(x - y).sum()), "jaccard": len(A & B) / len(A | B) if A | B else 1.0})
    cmp_ = pd.DataFrame(rows).set_index("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axes[0].plot(cmp_.index, cmp_.l1, color=COL["cvar"]); axes[0].set_ylabel("L1 weight distance"); axes[0].set_title("CVaR against CDaR: portfolio-weight distance")
    axes[1].plot(cmp_.index, cmp_.jaccard, color=COL["cdar"]); axes[1].set_ylabel("Active-set Jaccard"); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("CVaR against CDaR: active-set similarity"); years(axes[1])
    fig.tight_layout(); fig.savefig(out / "figL7_cvar_cdar_weight_distance.png", dpi=180); plt.close(fig)

    tc, td = turnover(wc), turnover(wd)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tc.index, tc.values, label="CVaR", color=COL["cvar"]); ax.plot(td.index, td.values, label="CDaR", color=COL["cdar"])
    ax.set_ylabel("Turnover (wealth bought plus sold)"); ax.set_title("Monthly turnover: CVaR against CDaR"); ax.legend(); years(ax)
    fig.tight_layout(); fig.savefig(out / "figL8_cvar_cdar_turnover.png", dpi=180); plt.close(fig)

    cmp_.describe().T.to_csv(out / "cvar_cdar_weight_comparison_summary.csv")
    pd.DataFrame({"CVaR": tc, "CDaR": td}).describe().T.to_csv(out / "cvar_cdar_turnover_summary.csv")
    print("saved", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
