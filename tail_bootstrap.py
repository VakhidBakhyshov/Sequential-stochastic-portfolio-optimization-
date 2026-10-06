"""Tail-risk bootstraps, Sharpe-difference power and the daily-path check for the HDRC arm.

Tail columns: paired iid bootstrap, 5,000 resamples, HDRC against the fast channel, 1/N and the unmanaged
base; probabilities that HDRC has the higher Sharpe ratio, the shallower drawdown, the smaller CVaR95, the
lower volatility and the higher Calmar and Sortino ratios.
Power: Jobson-Korkie / Memmel standard error of the annualized Sharpe difference and the smallest difference
detectable at the 5% level.
Daily path: the HDRC book rebuilt from daily fund returns at the monthly exposure (exposure moves costed on
the first day of the month) against daily 1/N of the eligible set; block bootstrap of the Sharpe difference
with 21-day blocks.

Run from the package root after benchmarks.py: python tail_bootstrap.py
Outputs: tail_bootstrap.csv, daily_path.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from overlay_arms import build, load_pnl, load_weights, daily_simple, ann_sharpe, max_dd, exposure_cost, RES, ROOT, RF

B = 5000; SEED = 42; BLOCK = 21


def cvar95(r):
    q = np.quantile(r, 0.05); return -r[r <= q].mean()


def calmar(r):
    e = np.cumprod(1 + r); mdd = -(e / np.maximum.accumulate(e) - 1).min(); ann = e[-1] ** (12 / len(r)) - 1
    return ann / mdd


def sortino(r):
    x = r - RF / 12; d = x[x < 0]; return x.mean() / np.sqrt((d ** 2).sum() / len(x)) * np.sqrt(12)


def vol(r):
    return r.std(ddof=1) * np.sqrt(12)


def se_diff(a, b):
    """Jobson-Korkie standard error with Memmel's correction, annualized; returns (se, b - a, rho)."""
    x = a - RF / 12; y = b - RF / 12; T = len(x); sa = x.mean() / x.std(ddof=1); sb = y.mean() / y.std(ddof=1); rho = x.corr(y)
    se = np.sqrt((2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / T) * np.sqrt(12)
    return se, (sb - sa) * np.sqrt(12), rho


def main():
    arms, ks = build(RES / "main_dyn_strong"); comb = arms["combined"]; base = arms["base"]; fast = arms["fast"]
    ew = load_pnl(RES / "bench_equalweight").iloc[1:].reindex(comb.index)
    n = len(comb); a = comb.values
    rows = []
    print("=== tail columns: paired iid bootstrap, 5,000 resamples, treatment = HDRC ===")
    for lab, x in [("fast channel", fast), ("1/N", ew), ("unmanaged base", base)]:
        rng = np.random.default_rng(SEED); b = x.values
        c = {k: 0 for k in ["Sharpe", "DD", "CVaR", "vol", "Calmar", "Sortino"]}
        for _ in range(B):
            idx = rng.integers(0, n, n); p, q = a[idx], b[idx]
            c["Sharpe"] += ann_sharpe(p) > ann_sharpe(q); c["DD"] += max_dd(p) > max_dd(q); c["CVaR"] += cvar95(p) < cvar95(q)
            c["vol"] += vol(p) < vol(q); c["Calmar"] += calmar(p) > calmar(q); c["Sortino"] += sortino(p) > sortino(q)
        se, d, rho = se_diff(x, comb)
        row = {"comparison": f"HDRC vs {lab}", "dSharpe": ann_sharpe(a) - ann_sharpe(b), "dMaxDD_pp": 100 * (max_dd(a) - max_dd(b)),
               **{f"P_{k}": v / B for k, v in c.items()},
               "CVaR95_HDRC": 100 * cvar95(a), "CVaR95_other": 100 * cvar95(b), "vol_HDRC": 100 * vol(a), "vol_other": 100 * vol(b),
               "Calmar_HDRC": calmar(a), "Calmar_other": calmar(b), "Sortino_HDRC": sortino(a), "Sortino_other": sortino(b),
               "se_dSharpe": se, "rho": rho, "min_detectable_dSharpe_5pct": 1.96 * se}
        rows.append(row)
        print(f"  vs {lab:15s}: dSharpe {row['dSharpe']:+.3f}  dMDD {row['dMaxDD_pp']:+.2f} pp | "
              + " ".join(f"P({k}) {v/B:.3f}" for k, v in c.items())
              + f" | se {se:.3f} rho {rho:.2f} min detectable {1.96*se:.2f}")
    T = pd.DataFrame(rows).set_index("comparison"); T.round(4).to_csv(ROOT / "tail_bootstrap.csv")

    print("=== daily-path check ===")
    dly = daily_simple(); W = load_weights(RES / "main_dyn_strong", "weights"); reb = list(W.keys())
    xl = pd.ExcelFile(RES / "main_dyn_strong" / "weights.xlsx")
    elig = {pd.Timestamp(sh): [str(k) for k, v in zip(*[xl.parse(sh)[c] for c in ("Key", "Value")]) if int(v) == 1] for sh in xl.sheet_names}
    kc = ks["combined"]; kcost = exposure_cost(kc); rc_parts = []; re_parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]
        if d1 not in kc.index: continue
        w = W[d0]; cols = [c for c in w.index if c in dly.columns]; win = dly.loc[(dly.index > d0) & (dly.index <= d1)]
        rb = pd.Series(win[cols].fillna(0).values @ w.reindex(cols).values, index=win.index); k = float(kc.loc[d1])
        r = k * rb + (1 - k) * RF / 252
        if len(r): r.iloc[0] -= float(kcost.loc[d1])
        rc_parts.append(r)
        ec = [c for c in elig.get(d0, []) if c in dly.columns]; re_parts.append(win[ec].fillna(0).mean(axis=1))
    rc = pd.concat(rc_parts); re_ = pd.concat(re_parts).reindex(rc.index)

    def sr_d(x): x = x - RF / 252; return x.mean() / x.std(ddof=1) * np.sqrt(252)
    def mdd_d(x): e = (1 + x).cumprod(); return 100 * (e / e.cummax() - 1).min()
    rng = np.random.default_rng(SEED); nd = len(rc); a = rc.values; b = re_.values; wins = 0; diffs = []
    for _ in range(B):
        idx = np.concatenate([np.arange(s, s + BLOCK) % nd for s in rng.integers(0, nd, int(np.ceil(nd / BLOCK)))])[:nd]
        x = a[idx] - RF / 252; y = b[idx] - RF / 252; ds = (x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(252); diffs.append(ds); wins += ds > 0
    out = {"daily_obs": nd, "Sharpe_HDRC": sr_d(rc), "Sharpe_1N": sr_d(re_), "P_HDRC_higher_Sharpe": wins / B,
           "CI90_low": float(np.percentile(diffs, 5)), "CI90_high": float(np.percentile(diffs, 95)),
           "MaxDD_HDRC": mdd_d(rc), "MaxDD_1N": mdd_d(re_)}
    pd.DataFrame([out]).round(4).to_csv(ROOT / "daily_path.csv", index=False)
    print(f"  daily n={nd}: HDRC Sharpe {out['Sharpe_HDRC']:.3f}  1/N {out['Sharpe_1N']:.3f}  P(HDRC > 1/N) {out['P_HDRC_higher_Sharpe']:.3f}  "
          f"90% CI [{out['CI90_low']:+.2f}, {out['CI90_high']:+.2f}] | daily MaxDD HDRC {out['MaxDD_HDRC']:.1f}%  1/N {out['MaxDD_1N']:.1f}%")
    print("saved tail_bootstrap.csv, daily_path.csv")


if __name__ == "__main__":
    main()
