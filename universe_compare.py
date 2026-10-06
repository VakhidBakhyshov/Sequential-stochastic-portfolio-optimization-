"""Compare the full novel model set across eligibility universes (original ~220, matrix1, matrix2).

For each universe (a Static + Dynamic-base run pair), builds: Static, Dynamic base, realized-vol
overlay, model-CVaR overlay, Combined (Primary), plus 1/N and Markowitz benchmarks (all causal, same
realized returns from that universe's real.csv). Outputs a comparison table + a Sharpe bar chart.

Run from the package root: python universe_compare.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm
try:
    from sklearn.covariance import LedoitWolf; HAS_LW = True
except Exception:
    HAS_LW = False
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from overlay_arms import overlay

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; CHOSEN_SPAN, CHOSEN_TV = 21, 0.12; CAP = 0.12; CBPS = 1e-3; LB_Y = 3

UNIVERSES = {
    "original (~220)": ("main_dyn_off", "main_dyn_strong"),
    "matrix1 (~300)":  ("off_m1", "strong_m1"),
    "matrix2 (cap400)": ("off_m2", "strong_m2"),
}


def load_pnl(folder):
    p = pd.read_csv(RES / folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def metrics(r):
    r = np.asarray(r, float); n = len(r); tot = float(np.prod(1 + r) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12; sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    cal = ann / abs(mdd) if mdd < 0 else np.nan; v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Cumul=tot*100, Ann=ann*100, Vol=vol*100, Sharpe=sh, Sortino=sor, MaxDD=mdd*100, Calmar=cal, CVaR95=cv*100)


def daily_simple():
    df = pd.read_csv(DAILY); df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    return np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0


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


def realized_vol_k(weights, pnl_idx, dly):
    reb = list(weights.keys()); parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]; w = weights[d0]; cols = [t for t in w.index if t in dly.columns]
        win = dly.loc[(dly.index > d0) & (dly.index <= d1), cols].fillna(0.0)
        if win.empty: continue
        parts.append(pd.Series(win.values @ w.reindex(cols).values, index=win.index))
    pdaily = pd.concat(parts).sort_index(); vol_ann = pdaily.rolling(CHOSEN_SPAN, min_periods=10).std() * np.sqrt(252)
    k = {}
    for j in range(1, len(pnl_idx)):
        ve = vol_ann.asof(pnl_idx[j - 1]); k[pnl_idx[j]] = float(np.clip(CHOSEN_TV / ve, KMIN, KMAX)) if (ve == ve and ve > 0) else KMAX
    return pd.Series(k)


def model_cvar_k(folder, idx):
    fc = pd.read_csv(RES / folder / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")["cvar_model"].astype(float).reindex(idx)
    # CAUSAL normalizer (expanding median, min 6 months, early months back-filled) — same convention
    return (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)


def solve(mu, cov, kind):
    n = len(mu); x0 = np.ones(n) / n; bnds = [(0.0, CAP)] * n; cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    obj = (lambda w: float(w @ cov @ w)) if kind == "minvar" else (lambda w: -float((mu @ w - RF/12) / np.sqrt(max(w @ cov @ w, 1e-12))))
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    w = r.x if (r.success and np.all(np.isfinite(r.x))) else x0; w = np.clip(w, 0, CAP)
    return w / w.sum() if w.sum() > 0 else x0


def benchmarks(off_folder, dly):
    real = pd.read_csv(RES / off_folder / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"])
    real = real.set_index("Unnamed: 0"); real_s = np.exp(real) - 1.0; dts = list(real.index)
    ew, ms, mv = [], [], []
    for i in range(1, len(dts)):
        rd, bd = dts[i], dts[i - 1]; elig = real_s.loc[rd].dropna().index.tolist()
        win = dly.loc[(dly.index > bd - pd.DateOffset(years=LB_Y)) & (dly.index <= bd)]
        good = [t for t in elig if t in win.columns and win[t].notna().sum() >= 250] or [t for t in elig if t in win.columns]
        if len(good) < 5:
            ew.append((rd, np.nan)); ms.append((rd, np.nan)); mv.append((rd, np.nan)); continue
        rr = real_s.loc[rd, good].astype(float); W = win[good].dropna(how="all")
        mu = W.mean().values * 21.0
        cov = (LedoitWolf().fit(np.nan_to_num(W.values)).covariance_ if HAS_LW and W.shape[0] > len(good) else np.cov(np.nan_to_num(W.values), rowvar=False)) * 21.0
        cov = np.atleast_2d(cov)
        ew.append((rd, float((np.ones(len(good))/len(good)) @ rr.values)))
        ms.append((rd, float(solve(mu, cov, "ms") @ rr.values)))
        mv.append((rd, float(solve(mu, cov, "minvar") @ rr.values)))
    return {"1/N": pd.Series(dict(ew)), "Markowitz maxSh": pd.Series(dict(ms)), "Markowitz minVar": pd.Series(dict(mv))}


def main():
    dly = daily_simple(); rows = []
    for uni, (off_f, strong_f) in UNIVERSES.items():
        r_static = load_pnl(off_f).iloc[1:]; r_strong = load_pnl(strong_f).iloc[1:]
        kt_rv = realized_vol_k(load_weights(strong_f), load_pnl(strong_f).index, dly).reindex(r_strong.index)
        kt_mc = model_cvar_k(strong_f, r_strong.index)
        r_real = overlay(kt_rv, r_strong)
        r_mc = overlay(kt_mc, r_strong)
        r_comb = overlay((kt_rv * kt_mc).clip(KMIN, KMAX), r_strong)
        bench = benchmarks(off_f, dly)
        arms = {"Static": r_static, "Dynamic base": r_strong, "+Realized-vol": r_real,
                "+Model-CVaR": r_mc, "+Combined (Primary)": r_comb,
                "Benchmark 1/N": bench["1/N"], "Benchmark Markowitz-maxSh": bench["Markowitz maxSh"],
                "Benchmark Markowitz-minVar": bench["Markowitz minVar"]}
        for arm, s in arms.items():
            s = s.dropna()
            if len(s) < 10: continue
            m = metrics(s)
            rows.append({"universe": uni, "arm": arm, "Cumul%": round(m["Cumul"],1), "Ann%": round(m["Ann"],2),
                         "Vol%": round(m["Vol"],2), "Sharpe": round(m["Sharpe"],3), "Sortino": round(m["Sortino"],3),
                         "MaxDD%": round(m["MaxDD"],2), "Calmar": round(m["Calmar"],3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(ROOT / "universe_comparison.csv", index=False)
    print(tbl.to_string(index=False))

    # Sharpe bar chart grouped by arm
    piv = tbl.pivot(index="arm", columns="universe", values="Sharpe")
    order = ["Static","Dynamic base","+Realized-vol","+Model-CVaR","+Combined (Primary)",
             "Benchmark 1/N","Benchmark Markowitz-maxSh","Benchmark Markowitz-minVar"]
    piv = piv.reindex([a for a in order if a in piv.index])
    ax = piv.plot(kind="bar", figsize=(14, 6)); ax.set_ylabel("Sharpe"); ax.grid(alpha=0.25, axis="y")
    ax.set_title("Sharpe by arm across eligibility universes"); ax.axhline(0, color="0.5", lw=0.8)
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(ROOT / "_universe_comparison.png", dpi=130, bbox_inches="tight")
    print("\nsaved universe_comparison.csv + _universe_comparison.png")


if __name__ == "__main__":
    main()
