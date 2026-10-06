"""VIX benchmark arm — is the option-implied signal a better SECOND signal than our slow model one?

Claims to test (Part 1, section 2b/G2):
  - VIX is a FAST signal: its exposure dial should correlate highly with the realized-vol dial
    (both react to recent turbulence), unlike our slow model-implied dial.
  - Therefore, as the SECOND signal next to realized vol, VIX should add less than the slow signal:
    combined(fast realized x VIX) should beat neither combined(fast realized x slow model)
    nor add much over realized-only.
  - Bozovic-style substitution (VIX replacing realized vol as the fast leg) is also tested.

Dials (all causal, all clipped to [0.30, 1.00], target 12% like the realized dial):
  k_rv  : 12% / trailing 21d realized vol of the held book (asof previous rebalance)
  k_mc  : expanding-median(cvar_model) / cvar_model      (our slow model-implied dial)
  k_vix : 12% / (mean VIX over 21 trading days ending before t / 100)

Run from the package root: python vix_experiment.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from overlay_arms import overlay as net_overlay

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
VIXF = ROOT / "datasets" / "csv" / "vixcls.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; SPAN, TV = 21, 0.12; SEED = 42


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


def metrics(r):
    r = pd.Series(r).dropna(); rv = np.asarray(r, float); n = len(rv)
    tot = float(np.prod(1 + rv) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = rv.std(ddof=1) * np.sqrt(12); ex = rv - RF / 12
    sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    eq = np.cumprod(1 + rv); mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    v95 = np.percentile(rv, 5); cv = float(rv[rv <= v95].mean())
    return dict(Ann=ann * 100, Vol=vol * 100, Sharpe=sh, MaxDD=mdd * 100,
                Calmar=ann / abs(mdd) if mdd < 0 else np.nan, CVaR95=cv * 100)


def ann_sharpe(r):
    r = np.asarray(r, float); ex = r - RF / 12; return ex.mean() / ex.std(ddof=1) * np.sqrt(12)


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    # --- book daily returns (for the realized dial) ---
    df = pd.read_csv(DAILY)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    DLY = np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0

    W = load_weights("main_dyn_strong")
    r_full = load_pnl("main_dyn_strong"); idx = r_full.index; rs = r_full.iloc[1:]
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

    # --- slow model-implied dial ---
    fc = pd.read_csv(RES / "main_dyn_strong" / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")["cvar_model"].astype(float).reindex(rs.index)
    k_mc = (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)

    # --- VIX dial (causal: trailing 21 trading days strictly before t) ---
    vix = pd.read_csv(VIXF); vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"]); vix = vix.set_index("date")["vix"].astype(float).dropna()
    k_vix = {}
    for t in rs.index:
        wnd = vix[vix.index < t].tail(SPAN)
        k_vix[t] = float(np.clip(TV / (wnd.mean() / 100.0), KMIN, KMAX)) if len(wnd) >= 10 else KMAX
    k_vix = pd.Series(k_vix)

    # --- arms ---
    def overlay(k): return net_overlay(k, rs)
    arms = {
        "Base (no overlay)":            rs,
        "Realized-vol only":            overlay(k_rv),
        "Slow model only":              overlay(k_mc),
        "VIX only (Bozovic-style)":     overlay(k_vix),
        "Fast x Slow (OURS)":           overlay((k_rv * k_mc).clip(KMIN, KMAX)),
        "Fast x VIX":                   overlay((k_rv * k_vix).clip(KMIN, KMAX)),
        "VIX x Slow (VIX as fast leg)": overlay((k_vix * k_mc).clip(KMIN, KMAX)),
    }
    kmeans = {"Realized-vol only": k_rv.mean(), "Slow model only": k_mc.mean(),
              "VIX only (Bozovic-style)": k_vix.mean(),
              "Fast x Slow (OURS)": (k_rv * k_mc).clip(KMIN, KMAX).mean(),
              "Fast x VIX": (k_rv * k_vix).clip(KMIN, KMAX).mean(),
              "VIX x Slow (VIX as fast leg)": (k_vix * k_mc).clip(KMIN, KMAX).mean()}

    print("=" * 96)
    print("DIAL CORRELATIONS (the horizon argument in one table)")
    print("=" * 96)
    K = pd.DataFrame({"k_rv": k_rv, "k_vix": k_vix, "k_mc": k_mc}).dropna()
    print(K.corr().round(3).to_string())
    print("\nmean dials: k_rv=%.3f  k_vix=%.3f  k_mc=%.3f" % (k_rv.mean(), k_vix.mean(), k_mc.mean()))

    print()
    print("=" * 96)
    print("ARMS (83 OOS months, net of costs in the base series, RF=2%)")
    print("=" * 96)
    rows = {}
    for name, r in arms.items():
        m = metrics(r); m["mean_k"] = kmeans.get(name, 1.0); rows[name] = m
    T = pd.DataFrame(rows).T
    print(T.round(3).to_string())
    T.round(4).to_csv(ROOT / "vix_experiment_results.csv")

    print()
    print("=" * 96)
    print("PAIRED BOOTSTRAP (5000): OURS (fast x slow) vs VIX alternatives")
    print("=" * 96)
    rng = np.random.default_rng(SEED); B = 5000
    a_ours = arms["Fast x Slow (OURS)"]
    for label in ["Fast x VIX", "VIX only (Bozovic-style)", "VIX x Slow (VIX as fast leg)"]:
        b_arm = arms[label]
        j = a_ours.index.intersection(b_arm.index)
        x, y = b_arm.loc[j].values, a_ours.loc[j].values; n = len(j)
        shd = np.empty(B); ddd = np.empty(B)
        for k in range(B):
            ii = rng.integers(0, n, n)
            shd[k] = ann_sharpe(y[ii]) - ann_sharpe(x[ii]); ddd[k] = max_dd(y[ii]) - max_dd(x[ii])
        print(f"  OURS vs {label:28s} dSharpe={ann_sharpe(y)-ann_sharpe(x):+.3f} P(ours better)={float((shd>0).mean()):.3f} | "
              f"dMaxDD={max_dd(y)-max_dd(x):+.3f} P(ours shallower)={float((ddd>0).mean()):.3f}")
    print("\nsaved vix_experiment_results.csv")


if __name__ == "__main__":
    main()
