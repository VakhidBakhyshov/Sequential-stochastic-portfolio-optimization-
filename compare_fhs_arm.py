"""Full comparison: parametric (Gaussian) vs filtered-historical (FHS) scenario generator.

Single change in the pipeline: distribution normal -> fhs (shocks are resampled real
EWMA-standardised days instead of Gaussian draws from the Ledoit-Wolf Cholesky). Everything
else -- mean, CVaR budget, optimizer, dynamic parameters, execution, seed -- is identical.

Reports, for BOTH arms:
  (A) whether the allocation actually changed (weight overlap, holdings, turnover);
  (B) whether the CVaR constraint became tail-aware (shape ratio c = CVaR/sigma of the book);
  (C) full risk/return metrics for the base arm and all three overlays;
  (D) paired bootstrap of FHS-combined vs Gaussian-combined.
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from overlay_arms import overlay

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; SPAN, TV = 21, 0.12; SEED = 42
BASE, FHS = "main_dyn_strong", "main_dyn_strong_fhs"


def load_pnl(arm):
    p = pd.read_csv(RES / arm / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def load_weights(arm):
    xl = pd.ExcelFile(RES / arm / "weights.xlsx"); out = {}
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        if "weights" not in d.columns: continue
        w = pd.to_numeric(d["weights"], errors="coerce").fillna(0.0)
        s = pd.Series(w.values, index=d[d.columns[0]].astype(str).values); s = s[s > 1e-9]
        if not s.empty:
            try: out[pd.to_datetime(sh)] = s / s.sum()
            except Exception: pass
    return dict(sorted(out.items()))


def metrics(r, mkt=None):
    r = pd.Series(r).dropna(); rv = np.asarray(r, float); n = len(rv)
    tot = float(np.prod(1 + rv) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = rv.std(ddof=1) * np.sqrt(12); ex = rv - RF / 12
    sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = rv[rv < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + rv); dr = eq / np.maximum.accumulate(eq) - 1; mdd = float(dr.min())
    v95 = np.percentile(rv, 5); cv = float(rv[rv <= v95].mean())
    ddev = np.sqrt(np.mean(np.minimum(rv, 0) ** 2)) * np.sqrt(12) * 100
    pain = -dr.mean(); painr = (ann - RF) / pain if pain > 0 else np.nan
    mad = np.mean(np.abs(rv - rv.mean())) * 100
    out = dict(Cumul=tot * 100, Ann=ann * 100, Vol=vol * 100, Sharpe=sh, Sortino=sor,
               MaxDD=mdd * 100, Calmar=ann / abs(mdd) if mdd < 0 else np.nan,
               CVaR95=cv * 100, DownDev=ddev, Pain=painr, MAD=mad)
    if mkt is not None:
        m = mkt.reindex(r.index).astype(float)
        beta = np.cov(rv, m.values, ddof=1)[0, 1] / np.var(m.values, ddof=1)
        out["Beta"] = beta; out["Treynor"] = (ann - RF) / beta * 100
    return out


def ann_sharpe(r):
    r = np.asarray(r, float); ex = r - RF / 12; return ex.mean() / ex.std(ddof=1) * np.sqrt(12)


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def build_arm(arm, DLY):
    """Return dict of the four series for one arm, plus diagnostics."""
    W = load_weights(arm); r_full = load_pnl(arm); idx = r_full.index; rs = r_full.iloc[1:]
    reb = list(W.keys()); parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]; w = W[d0]; cc = [t for t in w.index if t in DLY.columns]
        win = DLY.loc[(DLY.index > d0) & (DLY.index <= d1), cc].fillna(0.0)
        if not win.empty: parts.append(pd.Series(win.values @ w.reindex(cc).values, index=win.index))
    book = pd.concat(parts).sort_index()
    va = book.rolling(SPAN, min_periods=10).std() * np.sqrt(252)
    k_rv = pd.Series({idx[j]: (float(np.clip(TV / va.asof(idx[j - 1]), KMIN, KMAX))
                      if (va.asof(idx[j - 1]) == va.asof(idx[j - 1]) and va.asof(idx[j - 1]) > 0) else KMAX)
                      for j in range(1, len(idx))}).reindex(rs.index)
    fc = pd.read_csv(RES / arm / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    fcs = fc.set_index("date")
    cv = fcs["cvar_model"].astype(float).reindex(rs.index)
    k_mc = (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)
    k_cb = (k_rv * k_mc).clip(KMIN, KMAX)
    ov = lambda k: overlay(k, rs)
    series = {"Base (no overlay)": rs, "+ Realized-vol": ov(k_rv),
              "+ Model-CVaR": ov(k_mc), "+ Combined": ov(k_cb)}
    diag = {"W": W, "k_rv": k_rv, "k_mc": k_mc, "k_cb": k_cb,
            "cvar": cv, "vol_model": fcs["vol_model"].astype(float).reindex(rs.index)}
    return series, diag


def main():
    df = pd.read_csv(DAILY)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    DLY = np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0
    spy = DLY["SPY"]

    s_g, d_g = build_arm(BASE, DLY)
    s_f, d_f = build_arm(FHS, DLY)
    idxm = list(load_pnl(BASE).index)
    MKT = pd.Series({idxm[j]: float(np.prod(1 + spy[(spy.index > idxm[j-1]) & (spy.index <= idxm[j])].values) - 1)
                     for j in range(1, len(idxm))})

    # ---------- (A) did the allocation change? ----------
    print("=" * 100); print("(A) DID THE ALLOCATION ACTUALLY CHANGE?"); print("=" * 100)
    Wg, Wf = d_g["W"], d_f["W"]
    common = [d for d in Wg if d in Wf and d >= pd.Timestamp("2019-01-01")]
    ov_l1, jac, ng, nf = [], [], [], []
    for d in common:
        a, b = Wg[d], Wf[d]; u = a.index.union(b.index)
        av, bv = a.reindex(u).fillna(0), b.reindex(u).fillna(0)
        ov_l1.append(1 - 0.5 * float(np.abs(av - bv).sum()))          # weight overlap in [0,1]
        sa, sb = set(a.index), set(b.index)
        jac.append(len(sa & sb) / max(len(sa | sb), 1))
        ng.append(len(sa)); nf.append(len(sb))
    print(f"  months compared                     : {len(common)}")
    print(f"  mean weight overlap (1 = identical) : {np.mean(ov_l1):.3f}   [min {np.min(ov_l1):.3f}]")
    print(f"  mean ticker Jaccard overlap         : {np.mean(jac):.3f}")
    print(f"  holdings/month  Gaussian {np.mean(ng):.1f}   FHS {np.mean(nf):.1f}")
    print(f"  corr(monthly returns) base arms     : {s_g['Base (no overlay)'].corr(s_f['Base (no overlay)']):.4f}")

    # ---------- (B) did the constraint become tail-aware? ----------
    print(); print("=" * 100); print("(B) IS THE CVaR CONSTRAINT NOW TAIL-AWARE? (shape ratio c = CVaR/vol of the book)")
    print("=" * 100)
    for lab, d in [("Gaussian", d_g), ("FHS", d_f)]:
        c = (d["cvar"] / d["vol_model"]).replace([np.inf, -np.inf], np.nan).dropna()
        r2 = np.corrcoef(d["cvar"].dropna(), d["vol_model"].reindex(d["cvar"].dropna().index))[0, 1] ** 2
        print(f"  {lab:9s} c: mean {c.mean():6.3f}  sd {c.std():6.3f}  [{c.min():.3f}, {c.max():.3f}]   R2(CVaR~vol) = {r2:.3f}")
    print(f"  corr(k_mc Gaussian, k_mc FHS) = {d_g['k_mc'].corr(d_f['k_mc']):.3f}")
    print(f"  mean exposure  Gaussian: rv {d_g['k_rv'].mean():.3f} mc {d_g['k_mc'].mean():.3f} comb {d_g['k_cb'].mean():.3f}")
    print(f"  mean exposure  FHS     : rv {d_f['k_rv'].mean():.3f} mc {d_f['k_mc'].mean():.3f} comb {d_f['k_cb'].mean():.3f}")

    # ---------- (C) full metrics ----------
    print(); print("=" * 100); print("(C) FULL METRICS  (83 OOS months, net of costs, RF=2%)"); print("=" * 100)
    rows = {}
    for lab, s in [("Gaussian", s_g), ("FHS", s_f)]:
        for k, v in s.items():
            rows[f"{lab:9s} {k}"] = metrics(v, MKT)
    T = pd.DataFrame(rows).T
    print(T.round(3).to_string())
    T.round(4).to_csv(ROOT / "fhs_arm_comparison.csv")

    # ---------- (D) paired bootstrap ----------
    print(); print("=" * 100); print("(D) PAIRED BOOTSTRAP (5000): FHS vs Gaussian, like-for-like"); print("=" * 100)
    rng = np.random.default_rng(SEED); B = 5000
    for key in ["Base (no overlay)", "+ Realized-vol", "+ Model-CVaR", "+ Combined"]:
        a, b = s_g[key], s_f[key]
        j = a.index.intersection(b.index); x, y = a.loc[j].values, b.loc[j].values; n = len(j)
        sh = np.empty(B); dd = np.empty(B)
        for t in range(B):
            ii = rng.integers(0, n, n)
            sh[t] = ann_sharpe(y[ii]) - ann_sharpe(x[ii]); dd[t] = max_dd(y[ii]) - max_dd(x[ii])
        print(f"  {key:20s} dSharpe={ann_sharpe(y)-ann_sharpe(x):+.3f} P(FHS better)={float((sh>0).mean()):.3f} | "
              f"dMaxDD={max_dd(y)-max_dd(x):+.3f} P(FHS shallower)={float((dd>0).mean()):.3f}")
    print("\nsaved fhs_arm_comparison.csv")


if __name__ == "__main__":
    main()
