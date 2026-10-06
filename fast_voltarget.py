"""Fast (daily) volatility-targeting cash-sleeve overlay on the policy run.

Reconstructs DAILY portfolio returns from the monthly executed weights x daily ETF returns,
estimates a fast trailing vol (EWMA ~10-day / rolling 21-day, annualized), and scales next
month's exposure toward cash. Causal: the vol used to scale month k is computed only from
daily data up to that month's rebalance date. Compares to STRONG (no overlay).

Run from the package root:  python fast_voltarget.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from overlay_arms import overlay

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results" / "main_dyn_strong"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RISK_FREE = 0.02
TARGET_VOLS = [0.14, 0.12, 0.10]
K_MIN, K_MAX = 0.30, 1.00
VOL_SPANS = {"EWMA10d": ("ewm", 10), "Roll21d": ("roll", 21)}  # fast estimators to compare


def perf(r):
    r = np.asarray(r, float); n = len(r); yrs = n / 12
    total = float(np.prod(1 + r) - 1); ann = (1 + total) ** (1 / yrs) - 1
    vol = np.std(r, ddof=1) * np.sqrt(12)
    ex = r - RISK_FREE / 12; sharpe = np.mean(ex) / np.std(ex, ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = np.std(dn, ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sortino = (ann - RISK_FREE) / ddv if ddv and ddv > 0 else np.nan
    eq = np.cumprod(1 + r); pk = np.maximum.accumulate(eq); mdd = float(np.min(eq / pk - 1))
    return dict(Total=round(total*100,1), Ann=round(ann*100,2), Vol=round(vol*100,2),
                Sharpe=round(float(sharpe),3), Sortino=round(float(sortino),3),
                MaxDD=round(mdd*100,2), Calmar=round(float(ann/abs(mdd)),3) if mdd<0 else np.nan)


def load_daily():
    df = pd.read_csv(DAILY)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")   # log returns


def load_weights():
    xl = pd.ExcelFile(RUN / "weights.xlsx")
    out = {}
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        if "weights" not in d.columns:
            continue
        w = pd.to_numeric(d["weights"], errors="coerce").fillna(0.0)
        s = pd.Series(w.values, index=d[d.columns[0]].values)
        s = s[s > 1e-9]
        if not s.empty:
            out[pd.to_datetime(sh)] = s / s.sum()
    return dict(sorted(out.items()))


def main():
    daily_log = load_daily()
    daily_simple = np.exp(daily_log) - 1.0
    weights = load_weights()
    reb = list(weights.keys())

    pnl = pd.read_csv(RUN / "pnl.csv")
    dcol = pnl.columns[0]; pnl[dcol] = pd.to_datetime(pnl[dcol])
    pnl = pnl.set_index(dcol)
    r_month = pnl["Returns"].astype(float)

    # ---- reconstruct continuous daily portfolio returns ----
    port_daily = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]
        w = weights[d0]
        cols = [t for t in w.index if t in daily_simple.columns]
        win = daily_simple.loc[(daily_simple.index > d0) & (daily_simple.index <= d1), cols].fillna(0.0)
        if win.empty:
            continue
        pr = win.values @ w.reindex(cols).values
        port_daily.append(pd.Series(pr, index=win.index))
    port_daily = pd.concat(port_daily).sort_index()

    results = {"STRONG (no overlay)": perf(r_month.iloc[1:].values)}
    for vname, (kind, span) in VOL_SPANS.items():
        if kind == "ewm":
            vol_d = port_daily.ewm(halflife=span, min_periods=10).std()
        else:
            vol_d = port_daily.rolling(span, min_periods=10).std()
        vol_ann = vol_d * np.sqrt(252)
        for tv in TARGET_VOLS:
            ks = []
            for k in range(1, len(pnl)):
                ve = vol_ann.asof(pnl.index[k - 1])    # vol known at month start (causal)
                ks.append(float(np.clip(tv / ve, K_MIN, K_MAX)) if (ve == ve and ve > 0) else K_MAX)
            results[f"{vname} @ {int(tv*100)}%"] = perf(overlay(np.array(ks), r_month.iloc[1:].values))

    table = pd.DataFrame(results).T
    print("\n=== FAST (daily) vol-target overlay vs STRONG ===")
    print(table.to_string())
    table.to_csv(RUN / "fast_voltarget_metrics.csv")

    # ---- save the chosen overlay (Roll21d @ 12%) as a pnl-like CSV (4th arm) and its exposure path ----
    CHOSEN_SPAN, CHOSEN_TV = 21, 0.12
    vol_ann = (port_daily.rolling(CHOSEN_SPAN, min_periods=10).std()) * np.sqrt(252)
    ks = []
    for k in range(1, len(pnl)):
        ve = vol_ann.asof(pnl.index[k - 1])
        ks.append(float(np.clip(CHOSEN_TV / ve, K_MIN, K_MAX)) if (ve == ve and ve > 0) else K_MAX)
    k_path = pd.Series(ks, index=pnl.index[1:], name="k").rename_axis("Date")
    net = overlay(k_path, r_month.iloc[1:])
    bal = 1000.0; rows = [(pnl.index[0], bal, 0.0)]
    for dt, rr in net.items():
        bal *= (1 + rr); rows.append((dt, bal, rr))
    out_dir = ROOT / "results" / "main_voltarget"; out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Date", "Balance", "Returns"]).set_index("Date").to_csv(out_dir / "pnl.csv")
    k_path.to_csv(out_dir / "exposure.csv")
    print(f"\nSaved chosen overlay (Roll21d @ {int(CHOSEN_TV*100)}%) -> {out_dir/'pnl.csv'} and exposure.csv  final balance {bal:,.2f}")


if __name__ == "__main__":
    main()
