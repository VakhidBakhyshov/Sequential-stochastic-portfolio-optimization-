"""Final, self-contained verification of the ORIGINAL-universe results (all causal).

Loads the strategy arms from results/, computes the 1/N and Markowitz benchmarks INLINE (causally,
same universe/dates/realized returns), recomputes every metric, runs paired bootstraps, and reports a
Deflated Sharpe Ratio deflated over the FULL set of strategy variants explored. The combined/model-CVaR
arms use the CAUSAL expanding-median CVaR target (no look-ahead).

Run after: run.py main_dyn_off / main_dyn_strong ; fast_voltarget.py ; cvar_target_overlay.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
try:
    from sklearn.covariance import LedoitWolf; HAS_LW = True
except Exception:
    HAS_LW = False

ROOT = Path(__file__).resolve().parent; RES = ROOT / "results"; DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; CAP = 0.12; CBPS = 1e-3; LB_Y = 3; SEED = 42
import os
# + 18 grid cells (scenario count x history rule x seed) + 3 mechanism arms (executed-state reference,
# frozen universe, full execution) + 2 constructions of the internal signal (raw, centred) = 64.
# Override with the environment variable N_TRIALS for sensitivity.
N_TRIALS = int(os.environ.get("N_TRIALS", 64))


def load(folder):
    p = pd.read_csv(RES / folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float).iloc[1:]


def metrics(r):
    r = np.asarray(r, float); n = len(r); tot = float(np.prod(1 + r) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12; sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    cal = ann / abs(mdd) if mdd < 0 else np.nan; v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Cumul=tot*100, Ann=ann*100, Vol=vol*100, Sharpe=sh, Sortino=sor, MaxDD=mdd*100, Calmar=cal, CVaR95=cv*100,
                _sk=float(skew(r)), _ku=float(kurtosis(r, fisher=False)), _n=n, _srm=float(ex.mean()/ex.std(ddof=1)))


def psr(sr_m, sr_star_m, n, sk, ku):
    den = np.sqrt(max(1 - sk*sr_m + (ku-1)/4*sr_m**2, 1e-12)); return float(norm.cdf((sr_m - sr_star_m) * np.sqrt(n - 1) / den))
def ann_sharpe(r):
    ex = np.asarray(r, float) - RF/12; sd = ex.std(ddof=1); return float(ex.mean()/sd*np.sqrt(12)) if sd > 0 else np.nan
def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float(np.min(eq/np.maximum.accumulate(eq) - 1))


def daily_simple():
    df = pd.read_csv(DAILY); df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    return np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0


def solve(mu, cov, kind):
    n = len(mu); x0 = np.ones(n)/n; bnds = [(0.0, CAP)]*n; cons = [{"type": "eq", "fun": lambda w: w.sum()-1.0}]
    obj = (lambda w: float(w @ cov @ w)) if kind == "minvar" else (lambda w: -float((mu @ w - RF/12)/np.sqrt(max(w @ cov @ w, 1e-12))))
    r = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    w = r.x if (r.success and np.all(np.isfinite(r.x))) else x0; w = np.clip(w, 0, CAP)
    return w/w.sum() if w.sum() > 0 else x0


def benchmarks():
    real = pd.read_csv(RES / "main_dyn_off" / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"])
    real = real.set_index("Unnamed: 0"); real_s = np.exp(real) - 1.0; dts = list(real.index); dly = daily_simple()
    ew, ms, mv = [], [], []
    for i in range(1, len(dts)):
        rd, bd = dts[i], dts[i-1]; elig = real_s.loc[rd].dropna().index.tolist()      # held universe at rd
        win = dly.loc[(dly.index > bd - pd.DateOffset(years=LB_Y)) & (dly.index <= bd)]  # CAUSAL: up to prior rebalance
        good = [t for t in elig if t in win.columns and win[t].notna().sum() >= 250] or [t for t in elig if t in win.columns]
        if len(good) < 5: continue
        rr = real_s.loc[rd, good].astype(float); W = win[good].dropna(how="all")
        mu = W.mean().values * 21.0
        cov = (LedoitWolf().fit(np.nan_to_num(W.values)).covariance_ if HAS_LW and W.shape[0] > len(good) else np.cov(np.nan_to_num(W.values), rowvar=False)) * 21.0
        cov = np.atleast_2d(cov)
        ew.append((rd, float((np.ones(len(good))/len(good)) @ rr.values)))
        ms.append((rd, float(solve(mu, cov, "ms") @ rr.values)))
        mv.append((rd, float(solve(mu, cov, "minvar") @ rr.values)))
    return {"Benchmark 1/N": pd.Series(dict(ew)), "Benchmark Markowitz max-Sharpe": pd.Series(dict(ms)),
            "Benchmark Markowitz min-variance": pd.Series(dict(mv))}


def main():
    arms = {"Static (alloc, no overlay)": load("main_dyn_off"),
            "Dynamic base (no overlay)": load("main_dyn_strong"),
            "+ Realized-vol overlay": load("main_voltarget"),
            "+ Model-CVaR overlay (causal)": load("main_cvartarget"),
            "+ Combined (Primary, causal)": load("main_combined")}
    arms.update(benchmarks())

    ms = {lbl: metrics(r.dropna()) for lbl, r in arms.items()}
    # DSR dispersion is measured across OUR strategy configurations only; 1/N and Markowitz are
    # external comparators (not configs we searched), so they are excluded from the trial variance.
    sr_ann = np.array([m["Sharpe"] for lbl, m in ms.items() if not lbl.startswith("Benchmark")])
    var_srm = np.var(sr_ann/np.sqrt(12), ddof=1); g = 0.5772156649
    z = (1-g)*norm.ppf(1-1.0/N_TRIALS) + g*norm.ppf(1-1.0/(N_TRIALS*np.e)); sr_star_m = float(np.sqrt(max(var_srm,0))*z)

    rows = []
    for lbl, m in ms.items():
        rows.append({"strategy": lbl, "Cumul%": round(m["Cumul"],1), "Ann%": round(m["Ann"],2), "Vol%": round(m["Vol"],2),
                     "Sharpe": round(m["Sharpe"],3), "Sortino": round(m["Sortino"],3), "MaxDD%": round(m["MaxDD"],2),
                     "Calmar": round(m["Calmar"],3), "CVaR95%": round(m["CVaR95"],2),
                     "PSR%": round(psr(m["_srm"],0.0,m["_n"],m["_sk"],m["_ku"])*100,1),
                     "DSR%": round(psr(m["_srm"],sr_star_m,m["_n"],m["_sk"],m["_ku"])*100,1)})
    tbl = pd.DataFrame(rows).set_index("strategy")
    print("=== VERIFIED original-universe results (83 OOS months, all causal) ===")
    print(tbl.to_string())
    print(f"\nDSR deflated over N_TRIALS={N_TRIALS}; benchmark Sharpe* (annual) = {sr_star_m*np.sqrt(12):.3f}")
    tbl.to_csv(ROOT / "verified_results.csv")

    rng = np.random.default_rng(SEED); B = 5000
    def boot(base, treat, lbl):
        j = base.index.intersection(treat.index); a, b = base.loc[j].values, treat.loc[j].values; n = len(j)
        shd = np.empty(B); ddd = np.empty(B)
        for k in range(B):
            ix = rng.integers(0, n, n); shd[k] = ann_sharpe(b[ix]) - ann_sharpe(a[ix]); ddd[k] = max_dd(b[ix]) - max_dd(a[ix])
        print(f"  {lbl}: dSharpe={ann_sharpe(b)-ann_sharpe(a):+.3f} P(better)={float((shd>0).mean()):.3f} | "
              f"dMaxDD={max_dd(b)-max_dd(a):+.3f} P(shallower)={float((ddd>0).mean()):.3f}")
    print("\n=== Paired bootstraps (5000), treatment = Combined (Primary, causal) ===")
    boot(arms["+ Realized-vol overlay"], arms["+ Combined (Primary, causal)"], "Combined vs Realized-vol")
    boot(arms["Benchmark 1/N"], arms["+ Combined (Primary, causal)"], "Combined vs 1/N      ")
    boot(arms["Dynamic base (no overlay)"], arms["+ Combined (Primary, causal)"], "Combined vs Dynamic base")
    print("\nsaved verified_results.csv")


if __name__ == "__main__":
    main()
