"""Build the exposure arms (base / fast / slow / combined) for ANY run folder, with the exact conventions
of fast_voltarget.py and cvar_target_overlay.py, and report Table-1-style metrics plus paired bootstraps.

Usage:
  python overlay_arms.py <run_folder> [--weights-col weights] [--cvar-col cvar_model] [--label NAME] [--out file.csv]

  --weights-col executed_weights  -> fast signal measured on the EXECUTED book (sensitivity)
  --cvar-col cvar_model_raw       -> slow signal read uncentred (audit)
"""
from __future__ import annotations
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis, norm

ROOT = Path(__file__).resolve().parent; RES = ROOT / "results"
DAILY = ROOT / "datasets" / "excel" / "new_etf_returns.csv"
RF = 0.02; KMIN, KMAX = 0.30, 1.00; SPAN, TV = 21, 0.12; SEED = 42; NBOOT = 5000


def load_pnl(folder):
    p = pd.read_csv(folder / "pnl.csv"); c = p.columns[0]; p[c] = pd.to_datetime(p[c])
    return p.set_index(c)["Returns"].astype(float)


def load_weights(folder, col):
    xl = pd.ExcelFile(folder / "weights.xlsx"); out = {}
    for sh in xl.sheet_names:
        d = xl.parse(sh)
        if col not in d.columns:
            continue
        w = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
        s = pd.Series(w.values, index=d[d.columns[0]].values); s = s[s > 1e-9]
        if not s.empty:
            out[pd.to_datetime(sh)] = s / s.sum()
    return dict(sorted(out.items()))


def daily_simple():
    df = pd.read_csv(DAILY); df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    return np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0


def book_daily(weights, dly):
    reb = list(weights.keys()); parts = []
    for i in range(len(reb) - 1):
        d0, d1 = reb[i], reb[i + 1]; w = weights[d0]; cols = [t for t in w.index if t in dly.columns]
        win = dly.loc[(dly.index > d0) & (dly.index <= d1), cols].fillna(0.0)
        if not win.empty:
            parts.append(pd.Series(win.values @ w.reindex(cols).values, index=win.index))
    return pd.concat(parts).sort_index()


def metrics(r):
    r = np.asarray(r, float); n = len(r); tot = float(np.prod(1 + r) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12; sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    sor = (ann - RF) / ddv if (ddv and ddv > 0) else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    cal = ann / abs(mdd) if mdd < 0 else np.nan; v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Cumul=tot * 100, Ann=ann * 100, Vol=vol * 100, Sharpe=sh, Sortino=sor, MaxDD=mdd * 100, Calmar=cal, CVaR95=cv * 100)


def ann_sharpe(r):
    ex = np.asarray(r, float) - RF / 12; sd = ex.std(ddof=1); return float(ex.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan


def max_dd(r):
    eq = np.cumprod(1 + np.asarray(r, float)); return float(np.min(eq / np.maximum.accumulate(eq) - 1))


def paired_boot(a, b):
    rng = np.random.default_rng(SEED); a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    ws = []; wd = []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n); ws.append(ann_sharpe(b[idx]) > ann_sharpe(a[idx])); wd.append(max_dd(b[idx]) > max_dd(a[idx]))
    return float(np.mean(ws)), float(np.mean(wd))


def build(folder: Path, weights_col="weights", cvar_col="cvar_model"):
    r = load_pnl(folder); r_m = r.iloc[1:]
    dly = daily_simple(); W = load_weights(folder, weights_col); pdaily = book_daily(W, dly)
    vol_ann = pdaily.rolling(SPAN, min_periods=10).std() * np.sqrt(252)
    kF = pd.Series({r.index[k]: (float(np.clip(TV / vol_ann.asof(r.index[k - 1]), KMIN, KMAX))
                                 if (vol_ann.asof(r.index[k - 1]) == vol_ann.asof(r.index[k - 1]) and vol_ann.asof(r.index[k - 1]) > 0) else KMAX)
                    for k in range(1, len(r))})
    fc = pd.read_csv(folder / "forecast_risk.csv"); fc["date"] = pd.to_datetime(fc["date"])
    cv = fc.set_index("date")[cvar_col].astype(float).reindex(r_m.index)
    cstar = cv.expanding(min_periods=6).median().bfill()
    kS = (cstar / cv).clip(KMIN, KMAX)
    kC = (kF * kS).clip(KMIN, KMAX)
    arms = {
        "base": r_m,
        "fast": kF * r_m + (1 - kF) * RF / 12,
        "slow": kS * r_m + (1 - kS) * RF / 12,
        "combined": kC * r_m + (1 - kC) * RF / 12,
    }
    ks = {"base": pd.Series(1.0, index=r_m.index), "fast": kF, "slow": kS, "combined": kC}
    return arms, ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--weights-col", default="weights"); ap.add_argument("--cvar-col", default="cvar_model")
    ap.add_argument("--label", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    folder = RES / a.folder
    arms, ks = build(folder, a.weights_col, a.cvar_col)
    rows = []
    for name, rr in arms.items():
        m = metrics(rr.values); m["mean_k"] = float(ks[name].mean()); m["arm"] = name; rows.append(m)
    T = pd.DataFrame(rows).set_index("arm")
    pS, pD = paired_boot(arms["fast"].values, arms["combined"].values)
    pS2, pD2 = paired_boot(arms["base"].values, arms["combined"].values)
    T.loc["combined", "P>fast Sharpe"] = pS; T.loc["combined", "P>fast shallowerDD"] = pD
    T.loc["combined", "P>base Sharpe"] = pS2; T.loc["combined", "P>base shallowerDD"] = pD2
    T.loc["fast", "corr(kF,kS)"] = float(ks["fast"].corr(ks["slow"]))
    label = a.label or f"{a.folder} [w={a.weights_col}, cvar={a.cvar_col}]"
    pd.set_option("display.width", 220)
    print(f"=== {label} ===")
    print(T.round(3).to_string())
    if a.out:
        T.to_csv(ROOT / a.out)
    return T


if __name__ == "__main__":
    main()
