"""Build all styled Results-section charts + statistics (per strategy_style.md / report_style.py).

Outputs (package root):
  res_composition_static.png, res_composition_dynamic.png  (monthly 100% stacked composition)
  res_performance.png        (growth of $1, key arms + 1/N)
  res_turnover.png           (monthly one-way turnover of the held book)
  res_dynamic_params.png     (alpha, lambda, score over time)
and prints ETF-count + turnover statistics used in the write-up.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgb
import report_style as rs; rs.apply()

ROOT = Path(__file__).resolve().parent; RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; CHOSEN_SPAN, CHOSEN_TV = 21, 0.12; TOP_K = 16

# on-brand categorical palette for composition (Cash=navy, Other=light grey)
CAT = ["#2E86C1", "#3CA370", "#F5A623", "#C94C4C", "#163A5F", "#6FB7E0", "#7FC9A6", "#F7C271",
       "#D98A8A", "#4E6E8E", "#A6CEE3", "#8E5BA6", "#B5BD68", "#E08AB0", "#56A8A8", "#C08457"]


def load_pnl(folder):
    p = pd.read_csv(RES / folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def load_weights(folder):
    xl = pd.ExcelFile(RES / folder / "weights.xlsx"); out = {}
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        if "weights" not in d.columns: continue
        w = pd.to_numeric(d["weights"], errors="coerce").fillna(0.0)
        s = pd.Series(w.values, index=d[d.columns[0]].astype(str).values); s = s[s > 1e-9]
        if not s.empty:
            try: out[pd.to_datetime(sh)] = s / s.sum()
            except Exception: pass
    return dict(sorted(out.items()))


def daily_simple():
    df = pd.read_csv(DAILY); df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    return np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0


def realized_vol_k(weights, idx, dly):
    reb = list(weights.keys()); parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]; w = weights[d0]; cols = [t for t in w.index if t in dly.columns]
        win = dly.loc[(dly.index > d0) & (dly.index <= d1), cols].fillna(0.0)
        if not win.empty: parts.append(pd.Series(win.values @ w.reindex(cols).values, index=win.index))
    pd_ = pd.concat(parts).sort_index(); va = pd_.rolling(CHOSEN_SPAN, min_periods=10).std() * np.sqrt(252)
    return pd.Series({idx[j]: (float(np.clip(CHOSEN_TV / va.asof(idx[j-1]), KMIN, KMAX))
                               if (va.asof(idx[j-1]) == va.asof(idx[j-1]) and va.asof(idx[j-1]) > 0) else KMAX)
                      for j in range(1, len(idx))})


def comp_matrix(weights, kt=None):
    months = sorted(weights.keys()); cols = sorted({t for m in months for t in weights[m].index})
    M = pd.DataFrame(0.0, index=months, columns=cols)
    for m in months:
        M.loc[m, weights[m].index] = weights[m].values
        if kt is not None: M.loc[m] *= float(kt.get(m, 1.0))
    if kt is not None: M["Cash / T-bill"] = (1.0 - M.sum(axis=1)).clip(lower=0)
    return M


def plot_comp(M, title, sub, out, cash="Cash / T-bill"):
    etfs = [c for c in M.columns if c != cash]
    keep = list(M[etfs].sum().sort_values(ascending=False).head(TOP_K).index)
    rest = [c for c in etfs if c not in keep]
    P = M[keep].copy()
    if rest: P["Other"] = M[rest].sum(axis=1)
    order = keep + (["Other"] if rest else [])
    cmap = {c: CAT[i % len(CAT)] for i, c in enumerate(keep)}
    if rest: cmap["Other"] = "#D9DCE1"
    if cash in M.columns:
        P[cash] = M[cash]; order = [cash] + order; cmap[cash] = rs.NAVY
    x = np.arange(len(P.index)); bottom = np.zeros(len(P.index))
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for c in order:
        ax.bar(x, P[c].values, bottom=bottom, width=0.95, color=cmap[c], label=c, linewidth=0)
        bottom += P[c].values
    step = max(1, len(x) // 14)
    ax.set_xticks(x[::step]); ax.set_xticklabels([d.strftime("%Y-%m") for d in P.index[::step]], rotation=0)
    ax.set_xlim(-0.6, len(x) - 0.4); ax.set_ylim(0, 1.001); ax.set_ylabel("Portfolio share"); ax.grid(False)
    ax.legend(ncol=2, fontsize=7.5, loc="center left", bbox_to_anchor=(1.005, 0.5), title=f"top {TOP_K} ETFs" + (" + cash" if cash in M.columns else ""))
    rs.title(ax, title, sub); fig.tight_layout(); fig.savefig(ROOT / out, dpi=150, bbox_inches="tight"); plt.close(fig)


def held_stats(weights, label):
    n = pd.Series({m: int((w > 1e-9).sum()) for m, w in weights.items()})
    n = n[n.index >= pd.Timestamp("2019-01-01")]
    print(f"  {label:26s} mean={n.mean():.1f} median={int(n.median())} min={n.min()} max={n.max()}")
    return n


def turnover(weights):
    allm = sorted(weights.keys()); dts = [d for d in allm if d >= pd.Timestamp("2019-01-01")]
    t = []
    for i in range(1, len(allm)):
        if allm[i] < pd.Timestamp("2019-01-01"): continue
        a, b = weights[allm[i-1]], weights[allm[i]]; idx = a.index.union(b.index)
        t.append((allm[i], float(np.abs(b.reindex(idx).fillna(0) - a.reindex(idx).fillna(0)).sum()) / 2))
    return pd.Series(dict(t))


def main():
    dly = daily_simple()
    w_static = load_weights("main_dyn_off"); w_strong = load_weights("main_dyn_strong")
    r_strong = load_pnl("main_dyn_strong")
    k_rv = realized_vol_k(w_strong, r_strong.index, dly).reindex(r_strong.index[1:])
    fc = pd.read_csv(RES / "main_dyn_strong" / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")["cvar_model"].astype(float).reindex(r_strong.index[1:])
    k_mc = (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)
    k_comb = (k_rv * k_mc).clip(KMIN, KMAX)

    print("=== ETF-count per month (held book) ===")
    held_stats(w_static, "Static (allocator)"); held_stats(w_strong, "Dynamic / managed book")

    print("\n=== Turnover (one-way, held book) ===")
    tn = turnover(w_strong)
    print(f"  mean={tn.mean()*100:.1f}%  median={tn.median()*100:.1f}%  max={tn.max()*100:.1f}%  "
          f"zero-turnover months={int((tn<1e-9).sum())}/{len(tn)}  big(>=10%)={int((tn>=0.10).sum())}")
    cost = pd.read_csv(RES / "main_dyn_strong" / "pnl.csv")["Cost"].astype(float).iloc[1:]
    print(f"  total transaction cost over sample = {cost.sum()*100:.2f}% of balance")

    # ---- composition charts ----
    plot_comp(comp_matrix(w_static), "Static allocator — monthly portfolio composition",
              "Maximum-return-under-CVaR book, no overlay", "res_composition_static.png")
    plot_comp(comp_matrix(w_strong, k_comb), "Managed portfolio — composition with de-risked sleeve",
              "Book scaled by the combined exposure; de-risked slice in cash / T-bills", "res_composition_dynamic.png")

    # ---- performance (growth of $1) ----
    arms = {"Combined (managed)": ("main_combined", rs.NAVY, 2.4),
            "Realized-vol overlay": ("main_voltarget", rs.BLUE, 1.6),
            "Dynamic base": ("main_dyn_strong", rs.GREEN, 1.4),
            "Static allocator": ("main_dyn_off", rs.GREY, 1.4),
            "1/N benchmark": ("__1n__", rs.ORANGE, 1.6)}
    # 1/N series
    real = pd.read_csv(RES / "main_dyn_off" / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"]); real = real.set_index("Unnamed: 0")
    rs_ = np.exp(real) - 1.0; oneN = pd.Series({d: float(rs_.loc[d].dropna().mean()) for d in rs_.index[1:]})
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl, (f, col, lw) in arms.items():
        r = oneN if f == "__1n__" else load_pnl(f).iloc[1:]
        eq = (1 + r).cumprod();
        if lbl == "Combined (managed)": ax.fill_between(eq.index, 1, eq.values, color=rs.NAVY, alpha=0.07)
        ax.plot(eq.index, eq.values, color=col, lw=lw, label=f"{lbl}")
    ax.axhline(1, color=rs.GRID, lw=1); ax.set_ylabel("Growth of \\$1"); ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    rs.title(ax, "Cumulative performance, 2019-2025", "Growth of \\$1, out-of-sample, net of costs")
    fig.tight_layout(); fig.savefig(ROOT / "res_performance.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- turnover chart ----
    fig, ax = plt.subplots(figsize=(12, 3.8))
    colv = [rs.RED if v >= 0.10 else rs.BLUE for v in tn.values]
    ax.bar(tn.index, tn.values * 100, width=20, color=colv, linewidth=0)
    ax.set_ylabel("One-way turnover (%)"); ax.grid(axis="y")
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    rs.title(ax, "Monthly portfolio turnover", f"Median {tn.median()*100:.0f}%; large rebalances (>=10%, red) in {int((tn>=0.10).sum())} of {len(tn)} months")
    fig.tight_layout(); fig.savefig(ROOT / "res_turnover.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- dynamic parameters ----
    h = pd.read_csv(RES / "main_dyn_strong" / "dynamic_parameter_history.csv"); h["date"] = pd.to_datetime(h["date"])
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(h["date"], h["confidence_level"], color=rs.NAVY, lw=2); ax[0].axhline(0.95, color=rs.GREY, ls="--", lw=1)
    ax[0].set_ylabel("CVaR level alpha"); rs.title(ax[0], "Dynamic optimiser parameters over time", "Adapted monthly from a performance + market-regime score")
    ax[1].plot(h["date"], h["turnover_penalty"], color=rs.BLUE, lw=2); ax[1].axhline(0.01, color=rs.GREY, ls="--", lw=1)
    ax[1].set_ylabel("turnover penalty")
    ax[2].axhline(0, color=rs.GRID, lw=1)
    ax[2].plot(h["date"], h["total_score"], color=rs.NAVY, lw=2, label="total score")
    ax[2].plot(h["date"], h["performance_score"], color=rs.GREEN, lw=1.2, label="performance")
    ax[2].plot(h["date"], h["regime_score"], color=rs.ORANGE, lw=1.2, label="regime")
    ax[2].fill_between(h["date"], 0, h["total_score"], where=h["total_score"] < 0, color=rs.RED, alpha=0.10)
    ax[2].set_ylabel("score [-1, 1]"); ax[2].legend(loc="upper left", fontsize=8, ncol=3)
    for a in ax: a.grid(axis="y")
    ax[2].xaxis.set_major_locator(mdates.YearLocator()); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout(); fig.savefig(ROOT / "res_dynamic_params.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("\nsaved: res_composition_static/dynamic, res_performance, res_turnover, res_dynamic_params .png")
    print(f"alpha range [{h.confidence_level.min():.3f},{h.confidence_level.max():.3f}] | "
          f"lambda range [{h.turnover_penalty.min():.4f},{h.turnover_penalty.max():.4f}] | "
          f"score range [{h.total_score.min():.2f},{h.total_score.max():.2f}]")


if __name__ == "__main__":
    main()
