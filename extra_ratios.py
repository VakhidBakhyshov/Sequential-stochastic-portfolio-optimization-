"""Additional risk ratios of the main arms (Section 7.2): beta against the S&P 500 fund, Treynor ratio,
downside deviation, pain ratio and mean absolute deviation. Market proxy = SPY, risk-free rate 2%.

Run from the package root after the arms and the overlay scripts:  python extra_ratios.py
Output: extra_ratios.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RF = 0.02

df = pd.read_csv(ROOT / "datasets" / "excel" / "new_etf_returns.csv")
df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
spy = np.exp(pd.to_numeric(df["SPY"], errors="coerce")) - 1.0


def full_index(folder):
    p = pd.read_csv(RES / folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def spy_monthly(idx_full):
    out = {}
    for j in range(1, len(idx_full)):
        d0, d1 = idx_full[j - 1], idx_full[j]; w = spy[(spy.index > d0) & (spy.index <= d1)]
        out[d1] = float(np.prod(1 + w.values) - 1) if len(w) else np.nan
    return pd.Series(out)


def ratios(r, mkt):
    r = r.dropna(); n = len(r); tot = np.prod(1 + r) - 1; ann = (1 + tot) ** (12 / n) - 1
    m = mkt.reindex(r.index); beta = np.cov(r.values, m.values, ddof=1)[0, 1] / np.var(m.values, ddof=1)
    treynor = (ann - RF) / beta * 100
    dd_dev = np.sqrt(np.mean(np.minimum(r.values, 0) ** 2)) * np.sqrt(12) * 100
    eq = np.cumprod(1 + r.values); ddraw = eq / np.maximum.accumulate(eq) - 1; pain = -ddraw.mean()
    pain_ratio = (ann - RF) / pain if pain > 0 else np.nan
    mad = np.mean(np.abs(r.values - r.values.mean())) * 100
    return dict(Beta=beta, Treynor=treynor, DownsideDev=dd_dev, PainRatio=pain_ratio, MAD=mad)


def main():
    arms = {"Static": "main_dyn_off", "Dynamic base": "main_dyn_strong", "+ Realized-vol": "main_voltarget",
            "+ Model-CVaR": "main_cvartarget", "+ Combined": "main_combined"}
    mkt = spy_monthly(list(full_index("main_dyn_off").index))
    rows = {k: ratios(full_index(f).iloc[1:], mkt) for k, f in arms.items()}
    real = pd.read_csv(RES / "main_dyn_off" / "real.csv"); real["Unnamed: 0"] = pd.to_datetime(real["Unnamed: 0"]); real = real.set_index("Unnamed: 0")
    rsimple = np.exp(real) - 1.0
    one_n = pd.Series({d: float(rsimple.loc[d].dropna().mean()) for d in rsimple.index[1:]})
    rows["1/N"] = ratios(one_n, mkt)
    T = pd.DataFrame(rows).T
    print(T.round(3).to_string())
    T.round(4).to_csv(ROOT / "extra_ratios.csv")
    print("saved extra_ratios.csv")


if __name__ == "__main__":
    main()
