"""Equal-weight ablation of the feasible-set decision on OUR screen (generation A).

Holds the portfolio rule at monthly equal weight (positions held within the month, rebalanced to
equal weight at each rebalance date) and varies only the eligible set:
  original  : the production liquidity screen (last_filtered_weights.xlsx, Value == 1)
  matrix1   : the broader eligibility matrix (~300 funds)
  matrix2   : the second matrix with the top-400 traded-value cap, as used by the strong_m2 run
Reports return statistics net of 10 bp one-way costs, universe size, month-to-month Jaccard overlap
of the eligible set, and a paired block bootstrap (original vs each alternative). Uses only the
data files; no optimizer, no overlay. Output: ew_universe_ablation.csv, ew_universe_ablation_monthly.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "excel"
CSV = ROOT / "datasets" / "csv"
RF = 0.02; CBPS = 1e-3; SEED = 42; EVAL_START = pd.Timestamp("2019-01-01")


def daily_simple():
    df = pd.read_csv(DATA / "new_etf_returns.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date").sort_index()
    return np.exp(df.apply(pd.to_numeric, errors="coerce")) - 1.0


def rebalance_dates():
    fbd = pd.read_excel(DATA / "business_dates.xlsx", index_col=0)["Values"]
    return [pd.Timestamp(x) for x in fbd.tolist()]


def universe_original(dates):
    xl = pd.ExcelFile(DATA / "last_filtered_weights.xlsx"); out = {}
    for d in dates:
        sh = d.strftime("%Y-%m-%d")
        if sh in xl.sheet_names:
            df = xl.parse(sh)
            out[d] = [str(k) for k, v in zip(df["Key"], df["Value"]) if int(v) == 1]
    return out


def universe_from_run(dates, folder):
    """Eligible set as the optimizer arm saw it (Value == 1 in the run's own weights workbook): the
    screen AFTER the data-availability guard and the history rule. Keeps the equal-weight ablation on
    exactly the same feasible sets as the optimizer arms."""
    xl = pd.ExcelFile(ROOT / "results" / folder / "weights.xlsx"); out = {}
    for d in dates:
        sh = d.strftime("%Y-%m-%d")
        if sh in xl.sheet_names:
            df = xl.parse(sh)
            out[d] = [str(k) for k, v in zip(df["Key"], pd.to_numeric(df["Value"], errors="coerce").fillna(0)) if int(v) == 1]
    return out


def universe_matrix(dates, fname, ret_cols, cap_n=None, market_cap=None):
    from scripts.runs.run import build_portfolios_from_matrix  # same construction as the runs
    sheet_names = [d.strftime("%Y-%m-%d") for d in dates]
    ports = build_portfolios_from_matrix(DATA / fname, sheet_names, ret_cols, market_cap=market_cap, cap_n=cap_n)
    return {pd.Timestamp(ds): [str(k) for k, v in zip(p["Key"], p["Value"]) if int(v) == 1] for ds, p in ports.items()}


def ew_monthly(univ, dates, dly):
    """Equal weights set at each rebalance date, held through the month; net of one-way costs."""
    rows = []; prev_w = None
    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        names = [t for t in univ.get(d0, []) if t in dly.columns]
        if len(names) == 0:
            continue
        win = dly.loc[(dly.index > d0) & (dly.index <= d1), names].fillna(0.0)
        if win.empty:
            continue
        asset_month = (1.0 + win).prod(axis=0) - 1.0                     # buy-and-hold within the month
        w = pd.Series(1.0 / len(names), index=names)
        # one-way turnover from the drifted previous book to the new equal-weight book
        if prev_w is None:
            turn = 0.0
        else:
            idx = w.index.union(prev_w.index)
            turn = 0.5 * float((w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())
        gross = float((w * asset_month).sum())
        net = gross - CBPS * turn
        drifted = w * (1.0 + asset_month); drifted = drifted / drifted.sum()
        prev_w = drifted
        rows.append(dict(date=d1, ret=net, gross=gross, turnover=turn, n=len(names)))
    return pd.DataFrame(rows).set_index("date")


def jaccard(univ, dates):
    j = []
    for i in range(1, len(dates)):
        a, b = set(univ.get(dates[i - 1], [])), set(univ.get(dates[i], []))
        if a and b:
            j.append(len(a & b) / len(a | b))
    return float(np.mean(j)) if j else np.nan


def metrics(r):
    r = np.asarray(r, float); n = len(r); tot = float(np.prod(1 + r) - 1); ann = (1 + tot) ** (12 / n) - 1
    vol = r.std(ddof=1) * np.sqrt(12); ex = r - RF / 12; sh = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
    dn = r[r < 0]; ddv = dn.std(ddof=1) * np.sqrt(12) if len(dn) > 1 else np.nan
    eq = np.cumprod(1 + r); mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    v95 = np.percentile(r, 5); cv = float(r[r <= v95].mean())
    return dict(Ann=ann * 100, Vol=vol * 100, Sharpe=sh, Sortino=(ann - RF) / ddv if ddv and ddv > 0 else np.nan,
                MaxDD=mdd * 100, CVaR95=cv * 100, Calmar=ann / abs(mdd) if mdd < 0 else np.nan)


def block_boot(r_a, r_b, B=5000, L=6):
    rng = np.random.default_rng(SEED); n = len(r_a); wins = []; dds = []
    for _ in range(B):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n); idx.extend([(s + k) % n for k in range(L)])
        idx = np.asarray(idx[:n])
        a, b = r_a[idx], r_b[idx]
        sa = (a - RF / 12).mean() / (a - RF / 12).std(ddof=1); sb = (b - RF / 12).mean() / (b - RF / 12).std(ddof=1)
        ea = np.cumprod(1 + a); eb = np.cumprod(1 + b)
        da = np.min(ea / np.maximum.accumulate(ea) - 1); db = np.min(eb / np.maximum.accumulate(eb) - 1)
        wins.append(sb > sa); dds.append(db > da)
    return float(np.mean(wins)), float(np.mean(dds))


def main():
    dly = daily_simple(); dates = rebalance_dates(); ret_cols = list(dly.columns)
    px = pd.read_csv(CSV / "NewClosePrice.csv"); vol = pd.read_csv(CSV / "Volume.csv")
    from scripts.runs.run import _make_market_cap_frame
    mcap = _make_market_cap_frame(px, vol)
    # eligible sets exactly as the optimizer arms saw them (screen + guard + history rule)
    unis = {
        "original (production screen)": universe_from_run(dates, "main_dyn_strong"),
        "matrix1 (~300)": universe_from_run(dates, "strong_m1"),
        "matrix2 (top-400 cap)": universe_from_run(dates, "strong_m2"),
    }
    monthly = {}; table = []
    for name, u in unis.items():
        m = ew_monthly(u, dates, dly); m = m[m.index > EVAL_START]
        monthly[name] = m
        met = metrics(m["ret"].values)
        table.append(dict(universe=name, months=len(m), mean_n=m["n"].mean(), jaccard=jaccard({d: u[d] for d in u if d >= EVAL_START - pd.Timedelta(days=40)}, [d for d in dates if d >= EVAL_START - pd.Timedelta(days=40)]),
                          turnover_yr=m["turnover"].sum() / (len(m) / 12), cum_cost_pct=100 * CBPS * m["turnover"].sum(), **met))
    T = pd.DataFrame(table).set_index("universe")
    base = monthly["original (production screen)"]["ret"]
    for name in list(unis)[1:]:
        alt = monthly[name]["ret"].reindex(base.index).dropna(); bb = base.reindex(alt.index)
        p_sh, p_dd = block_boot(bb.values, alt.values)
        T.loc[name, "P(alt better Sharpe)"] = p_sh; T.loc[name, "P(alt shallower DD)"] = p_dd
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())
    T.to_csv(ROOT / "ew_universe_ablation.csv")
    pd.concat({k: v["ret"] for k, v in monthly.items()}, axis=1).to_csv(ROOT / "ew_universe_ablation_monthly.csv")
    print("\nsaved ew_universe_ablation.csv / ew_universe_ablation_monthly.csv")


if __name__ == "__main__":
    main()
