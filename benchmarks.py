"""Benchmarks on the SAME universe / dates / realized returns as the strategy.

  - Equal-weight (1/N): hold 1/N across the month's eligible ETFs, rebalanced monthly.
  - Markowitz mean-variance (max-Sharpe / tangency) and minimum-variance: estimate mean + Ledoit-Wolf
    covariance from the trailing 3y of DAILY returns, solve long-only with the same 0.12 weight cap,
    rebalanced monthly.

Realized returns use the pipeline's own real.csv (per-ETF monthly returns over the eligible universe
each month) -> identical universe, dates, and realization as the strategy. Same transaction cost
(c_bps = 1e-3) is charged on turnover. Outputs results/bench_*/pnl.csv and a comparison table that
also includes the strategy arms.

Run from the package root: python benchmarks.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
try:
    from sklearn.covariance import LedoitWolf
    HAS_LW = True
except Exception:
    HAS_LW = False

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; CAP = 0.12; CBPS = 1e-3; LOOKBACK_Y = 3
SEED = 42


def metrics(r) -> dict:
    r = np.asarray(r, float); n = len(r); yrs = n / 12
    tot = float(np.prod(1 + r) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12
    sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    cal = ann / abs(mdd) if mdd < 0 else np.nan
    v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Cumul=tot * 100, Ann=ann * 100, Vol=vol * 100, Sharpe=sh, Sortino=sor,
                MaxDD=mdd * 100, Calmar=cal, CVaR95=cv * 100)


def psr_vs(r, sr_star_m):
    r = np.asarray(r, float); ex = r - RF / 12; srm = ex.mean() / ex.std(ddof=1); n = len(r)
    sk = float(skew(r)); ku = float(kurtosis(r, fisher=False))
    den = np.sqrt(max(1 - sk * srm + (ku - 1) / 4 * srm ** 2, 1e-12))
    return float(norm.cdf((srm - sr_star_m) * np.sqrt(n - 1) / den))


def solve(mu, cov, kind):
    n = len(mu); x0 = np.ones(n) / n
    bnds = [(0.0, CAP)] * n; cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if kind == "minvar":
        obj = lambda w: float(w @ cov @ w)
    else:  # max-Sharpe (tangency)
        obj = lambda w: -float((mu @ w - RF / 12) / np.sqrt(max(w @ cov @ w, 1e-12)))
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 400, "ftol": 1e-10})
    w = r.x if (r.success and np.all(np.isfinite(r.x))) else x0
    w = np.clip(w, 0, CAP)
    return w / w.sum() if w.sum() > 0 else x0


def main():
    # ---- realized monthly per-ETF returns + eligible universe per month (from the pipeline) ----
    real = pd.read_csv(RES / "main_dyn_off" / "real.csv")
    real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"]); real = real.set_index("Unnamed: 0")
    real_simple = np.exp(real) - 1.0                         # monthly LOG -> simple
    dts = list(real.index)                                   # [seed, then 83 realize months]
    months = dts[1:]

    # ---- daily returns for Markowitz estimation ----
    dly = pd.read_csv(DAILY)
    dly = dly.drop(columns=[c for c in dly.columns if c.startswith("Unnamed")], errors="ignore")
    dly["Date"] = pd.to_datetime(dly["Date"]); dly = dly.set_index("Date").sort_index()
    dly = dly.apply(pd.to_numeric, errors="coerce")          # daily LOG returns
    dly_s = np.exp(dly) - 1.0

    arms = {"Equal-weight (1/N)": [], "Markowitz max-Sharpe": [], "Markowitz min-variance": []}
    prev = {k: pd.Series(dtype=float) for k in arms}

    for i in range(1, len(dts)):
        rd, bd = dts[i], dts[i - 1]                           # realize date; rebalance/estimation cutoff
        elig = real_simple.loc[rd].dropna().index.tolist()    # held universe (eligible at rebalance bd)
        # CAUSAL: estimate from the trailing window up to the REBALANCE date bd (NOT the realize date rd),
        # so the realized month rd is never in the estimation window (no look-ahead).
        win = dly_s.loc[(dly_s.index > bd - pd.DateOffset(years=LOOKBACK_Y)) & (dly_s.index <= bd)]
        good = [t for t in elig if t in win.columns and win[t].notna().sum() >= 250]
        if len(good) < 5:
            good = [t for t in elig if t in win.columns] or elig
        rr = real_simple.loc[rd, good].astype(float)
        W = win[good].dropna(how="all")

        # estimates (monthly scale)
        mu = W.mean().values * 21.0
        if HAS_LW and W.shape[0] > len(good):
            cov = LedoitWolf().fit(np.nan_to_num(W.values)).covariance_ * 21.0
        else:
            cov = np.cov(np.nan_to_num(W.values), rowvar=False) * 21.0
        cov = np.atleast_2d(cov)

        w_ew = pd.Series(1.0 / len(good), index=good)
        w_ms = pd.Series(solve(mu, cov, "maxsharpe"), index=good)
        w_mv = pd.Series(solve(mu, cov, "minvar"), index=good)

        for name, w in [("Equal-weight (1/N)", w_ew), ("Markowitz max-Sharpe", w_ms),
                        ("Markowitz min-variance", w_mv)]:
            gross = float((w * rr).sum())
            idx = w.index.union(prev[name].index)
            turn = float((w.reindex(idx).fillna(0) - prev[name].reindex(idx).fillna(0)).abs().sum())
            net = gross - CBPS * turn
            arms[name].append((rd, net))
            prev[name] = w

    # ---- save + metrics, alongside the strategy arms ----
    strat = {
        "Static (model, no overlay)": RES / "main_dyn_off" / "pnl.csv",
        "Dynamic base (no overlay)": RES / "main_dyn_strong" / "pnl.csv",
        "+ Realized-vol overlay": RES / "main_voltarget" / "pnl.csv",
        "+ Model-CVaR overlay (NEW)": RES / "main_cvartarget" / "pnl.csv",
        "+ Combined fwd x bwd (NEW)": RES / "main_combined" / "pnl.csv",
    }
    series = {}
    for name, recs in arms.items():
        s = pd.Series(dict(recs)); series[name] = s
        out = RES / f"bench_{name.split()[0].lower().replace('/','').replace('-','')}"
        out.mkdir(parents=True, exist_ok=True)
        rows = [(dts[0], 1000.0, 0.0)]
        b = 1000.0
        for dt, rv in s.items():
            b *= (1 + rv); rows.append((dt, b, rv))
        pd.DataFrame(rows, columns=["Date", "Balance", "Returns"]).drop_duplicates("Date").set_index("Date").to_csv(out / "pnl.csv")
    for name, f in strat.items():
        p = pd.read_csv(f); c = p.columns[0]; series[name] = p.set_index(c)["Returns"].astype(float).iloc[1:]

    sr = np.array([metrics(v)["Sharpe"] for v in series.values()])
    var_srm = np.var(sr / np.sqrt(12), ddof=1); Nt = len(sr); g = 0.5772156649
    z = (1 - g) * norm.ppf(1 - 1.0 / Nt) + g * norm.ppf(1 - 1.0 / (Nt * np.e))
    sr_star_m = float(np.sqrt(max(var_srm, 0.0)) * z)

    order = ["Equal-weight (1/N)", "Markowitz max-Sharpe", "Markowitz min-variance",
             "Static (model, no overlay)", "Dynamic base (no overlay)", "+ Realized-vol overlay",
             "+ Model-CVaR overlay (NEW)", "+ Combined fwd x bwd (NEW)"]
    rows = []
    for name in order:
        v = series[name]; m = metrics(v)
        rows.append({"strategy": name, "Cumul%": round(m["Cumul"], 1), "Ann%": round(m["Ann"], 2),
                     "Vol%": round(m["Vol"], 2), "Sharpe": round(m["Sharpe"], 3),
                     "Sortino": round(m["Sortino"], 3), "MaxDD%": round(m["MaxDD"], 2),
                     "Calmar": round(m["Calmar"], 3), "CVaR95%": round(m["CVaR95"], 2),
                     "PSR%": round(psr_vs(v, 0.0) * 100, 1), "DSR%": round(psr_vs(v, sr_star_m) * 100, 1)})
    tbl = pd.DataFrame(rows).set_index("strategy")
    print(f"\n=== Benchmarks vs strategy (83 OOS months, RF 2%, same universe; LedoitWolf={HAS_LW}) ===")
    print(tbl.to_string())
    tbl.to_csv(ROOT / "benchmarks_vs_strategy.csv")
    print("\nsaved benchmarks_vs_strategy.csv  + results/bench_*/pnl.csv")


if __name__ == "__main__":
    main()
