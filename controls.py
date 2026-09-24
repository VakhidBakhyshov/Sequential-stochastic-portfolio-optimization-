"""Referee-proofing controls for the risk-timing attribution argument.

CONTROL B (the decisive one): constant-exposure book matched to the SAME average gross exposure as
the dynamic overlay. Isolates TIMING from "merely holding less". Note the analytical point: for a
constant k, r = k*r_risky + (1-k)*rf  =>  excess = k*(r_risky - rf), so the Sharpe ratio is EXACTLY
unchanged. Hence any Sharpe improvement is attributable to time-variation in k alone; only the
drawdown reduction has a mechanical component, which this control quantifies.

CONTROL A: apply risk timing to the 1/N book -- (i) with its OWN realized-vol dial, and (ii) with our
strategy's combined exposure path -- to show the overlay is a general tool, not allocator-specific.

Run from the package root: python controls.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent; RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; SPAN, TARGET_VOL = 21, 0.12; SEED = 42


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
    eq = np.cumprod(1 + rv); dr = eq / np.maximum.accumulate(eq) - 1; mdd = float(dr.min())
    v95 = np.percentile(rv, 5); cv = float(rv[rv <= v95].mean())
    dn = rv[rv < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    return dict(Ann=ann * 100, Vol=vol * 100, Sharpe=sh, Sortino=sor, MaxDD=mdd * 100,
                Calmar=ann / abs(mdd) if mdd < 0 else np.nan, CVaR95=cv * 100)


def ann_sharpe(r):
    r = np.asarray(r, float); ex = r - RF / 12; return ex.mean() / ex.std(ddof=1) * np.sqrt(12)


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float((eq / np.maximum.accumulate(eq) - 1).min())


def vol_dial(daily_book, month_idx, target=TARGET_VOL):
    """Causal realized-vol exposure dial: vol known at the START of each month."""
    va = daily_book.rolling(SPAN, min_periods=10).std() * np.sqrt(252)
    out = {}
    for j in range(1, len(month_idx)):
        v = va.asof(month_idx[j - 1])
        out[month_idx[j]] = float(np.clip(target / v, KMIN, KMAX)) if (v == v and v > 0) else KMAX
    return pd.Series(out)


def main():
    df = pd.read_csv(DAILY); df["Date"] = pd.to_datetime(df["Date"])
    DLY = np.exp(df.set_index("Date").sort_index().apply(pd.to_numeric, errors="coerce")) - 1.0

    W = load_weights("main_dyn_strong"); r_strong_full = load_pnl("main_dyn_strong"); idx = r_strong_full.index
    reb = list(W.keys()); parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]; w = W[d0]; cc = [t for t in w.index if t in DLY.columns]
        win = DLY.loc[(DLY.index > d0) & (DLY.index <= d1), cc].fillna(0.0)
        if not win.empty: parts.append(pd.Series(win.values @ w.reindex(cc).values, index=win.index))
    book_daily = pd.concat(parts).sort_index()

    rs = r_strong_full.iloc[1:]
    k_rv = vol_dial(book_daily, idx).reindex(rs.index)
    fc = pd.read_csv(RES / "main_dyn_strong" / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")["cvar_model"].astype(float).reindex(rs.index)
    k_mc = (cv.expanding(min_periods=6).median().bfill() / cv).clip(KMIN, KMAX)
    k_comb = (k_rv * k_mc).clip(KMIN, KMAX)
    r_comb = k_comb * rs + (1 - k_comb) * RF / 12
    r_vt = k_rv * rs + (1 - k_rv) * RF / 12

    # ---------------- CONTROL B: constant exposure at the SAME average gross exposure ----------------
    kbar = float(k_comb.mean()); kbar_rv = float(k_rv.mean())
    r_const = kbar * rs + (1 - kbar) * RF / 12
    r_const_rv = kbar_rv * rs + (1 - kbar_rv) * RF / 12
    print("=" * 96)
    print("CONTROL B - does the gain come from TIMING or merely from holding less risk?")
    print("=" * 96)
    print(f"average gross exposure: combined overlay k_bar = {kbar:.3f} | realized-vol overlay = {kbar_rv:.3f}\n")
    rows = {"Dynamic base (unscaled, k=1)": metrics(rs),
            f"Constant exposure k={kbar:.2f} (matched)": metrics(r_const),
            "DYNAMIC combined overlay (same avg k)": metrics(r_comb),
            f"Constant exposure k={kbar_rv:.2f} (matched to RV)": metrics(r_const_rv),
            "DYNAMIC realized-vol overlay (same avg k)": metrics(r_vt)}
    print(pd.DataFrame(rows).T.round(3).to_string())

    mdd_unscaled, mdd_const, mdd_dyn = max_dd(rs), max_dd(r_const), max_dd(r_comb)
    mech = (mdd_const - mdd_unscaled) / (mdd_dyn - mdd_unscaled) * 100
    print(f"\n  drawdown decomposition (combined): unscaled {mdd_unscaled*100:.2f}%  ->  "
          f"constant-k {mdd_const*100:.2f}%  ->  dynamic {mdd_dyn*100:.2f}%")
    print(f"  share of the drawdown reduction that is MECHANICAL (holding less): {mech:.1f}%")
    print(f"  share attributable to TIMING: {100-mech:.1f}%")
    print(f"\n  Sharpe: unscaled {ann_sharpe(rs):.3f} | constant-k {ann_sharpe(r_const):.3f} "
          f"(identical by construction) | dynamic {ann_sharpe(r_comb):.3f}")
    print("  => constant scaling CANNOT change Sharpe; the entire Sharpe gain is from time-variation in k.")

    # paired bootstrap: dynamic vs matched-constant
    rng = np.random.default_rng(SEED); B = 5000
    for lbl, base, treat in [("Combined vs constant-k", r_const, r_comb),
                             ("Realized-vol vs constant-k", r_const_rv, r_vt)]:
        j = base.index.intersection(treat.index); a, b = base.loc[j].values, treat.loc[j].values; n = len(j)
        sh = np.empty(B); dd = np.empty(B)
        for t in range(B):
            ix = rng.integers(0, n, n)
            sh[t] = ann_sharpe(b[ix]) - ann_sharpe(a[ix]); dd[t] = max_dd(b[ix]) - max_dd(a[ix])
        print(f"  [bootstrap] {lbl:28s} dSharpe={ann_sharpe(b)-ann_sharpe(a):+.3f} P={float((sh>0).mean()):.3f} | "
              f"dMaxDD={max_dd(b)-max_dd(a):+.3f} P={float((dd>0).mean()):.3f}")

    # ---------------- CONTROL A: is the overlay a general tool? apply it to 1/N ----------------
    print("\n" + "=" * 96)
    print("CONTROL A - is the risk overlay a GENERAL tool, or specific to our allocator?")
    print("=" * 96)
    real = pd.read_csv(RES / "main_dyn_off" / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"])
    real = real.set_index("Unnamed: 0"); real_s = np.exp(real) - 1.0
    dts = list(real.index)
    one_n = pd.Series({d: float(real_s.loc[d].dropna().mean()) for d in dts[1:]})

    # daily 1/N book (equal weight over that month's eligible set)
    parts = []
    for i in range(len(dts) - 1):
        d0, d1 = dts[i], dts[i + 1]
        elig = [t for t in real_s.loc[d1].dropna().index if t in DLY.columns]
        if not elig: continue
        win = DLY.loc[(DLY.index > d0) & (DLY.index <= d1), elig].fillna(0.0)
        if not win.empty: parts.append(pd.Series(win.values @ (np.ones(len(elig)) / len(elig)), index=win.index))
    onen_daily = pd.concat(parts).sort_index()

    k_1n = vol_dial(onen_daily, dts).reindex(one_n.index)
    r_1n_vt = k_1n * one_n + (1 - k_1n) * RF / 12                       # 1/N with its OWN vol dial
    kc = k_comb.reindex(one_n.index).fillna(KMAX)
    r_1n_comb = kc * one_n + (1 - kc) * RF / 12                          # 1/N with OUR combined path

    rows2 = {"1/N (unmanaged)": metrics(one_n),
             f"1/N + own realized-vol dial (avg k={k_1n.mean():.2f})": metrics(r_1n_vt),
             f"1/N + our combined exposure path (avg k={kc.mean():.2f})": metrics(r_1n_comb),
             "Our combined (managed) strategy": metrics(r_comb)}
    print(pd.DataFrame(rows2).T.round(3).to_string())

    for lbl, base, treat in [("1/N+ownRV vs 1/N", one_n, r_1n_vt), ("1/N+ourK vs 1/N", one_n, r_1n_comb)]:
        j = base.index.intersection(treat.index); a, b = base.loc[j].values, treat.loc[j].values; n = len(j)
        sh = np.empty(B); dd = np.empty(B)
        for t in range(B):
            ix = rng.integers(0, n, n)
            sh[t] = ann_sharpe(b[ix]) - ann_sharpe(a[ix]); dd[t] = max_dd(b[ix]) - max_dd(a[ix])
        print(f"  [bootstrap] {lbl:20s} dSharpe={ann_sharpe(b)-ann_sharpe(a):+.3f} P={float((sh>0).mean()):.3f} | "
              f"dMaxDD={max_dd(b)-max_dd(a):+.3f} P={float((dd>0).mean()):.3f}")

    out = pd.DataFrame({**rows, **rows2}).T.round(4)
    out.to_csv(ROOT / "controls_results.csv")
    print("\nsaved controls_results.csv")


if __name__ == "__main__":
    main()
