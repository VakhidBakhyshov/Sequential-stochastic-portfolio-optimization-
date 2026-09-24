"""RQ1 signal validity for the slow signal, raw and centred: correlation of the forecast made at the
rebalance date with the book's subsequently realized risk (next-month realized volatility, realized
daily-CVaR, within-month drawdown) and with the next-month return. Book = target composition held over
the month (paper convention). Output: signal_validity.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from overlay_arms import load_weights, daily_simple, RES

ROOT = Path(__file__).resolve().parent
RUN = RES / "main_dyn_strong"


def main():
    dly = daily_simple(); W = load_weights(RUN, "weights"); reb = list(W.keys())
    fc = pd.read_csv(RUN / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"]); fc = fc.set_index("date")
    rows = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]
        if d1 not in fc.index:
            continue
        w = W[d0]; cols = [t for t in w.index if t in dly.columns]
        win = dly.loc[(dly.index > d0) & (dly.index <= d1), cols].fillna(0.0)
        if win.empty:
            continue
        pr = pd.Series(win.values @ w.reindex(cols).values, index=win.index)
        eq = (1 + pr).cumprod(); mdd = float((eq / eq.cummax() - 1).min())
        q = np.quantile(pr, 0.05); rcvar = float(-pr[pr <= q].mean())
        rows.append(dict(date=d1, raw=fc.loc[d1, "cvar_model_raw"], centred=fc.loc[d1, "cvar_model_centered"], vol_model=fc.loc[d1, "vol_model"],
                         rvol=float(pr.std(ddof=1) * np.sqrt(252)), rcvar=rcvar, mdd=mdd, ret=float(eq.iloc[-1] - 1)))
    D = pd.DataFrame(rows).set_index("date")
    out = []
    for sig in ["raw", "centred", "vol_model"]:
        for tgt, lab in [("rvol", "next-month realized vol"), ("rcvar", "next-month realized CVaR"), ("mdd", "within-month drawdown"), ("ret", "next-month return")]:
            r, p = pearsonr(D[sig], D[tgt]); rs, ps = spearmanr(D[sig], D[tgt])
            out.append(dict(signal=sig, target=lab, pearson=r, p=p, spearman=rs, p_s=ps, n=len(D)))
    T = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    T.to_csv(ROOT / "signal_validity.csv", index=False)
    D.to_csv(ROOT / "signal_validity_monthly.csv")


if __name__ == "__main__":
    main()
