"""Sub-period table (split 2022-07-01; drawdown measured within each half) for the five Table-4 arms:
1/N, dynamic base, fast only, slow only, combined — from the canonical run and the equal-weight benchmark.
Output: subperiod_table.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from overlay_arms import build, load_pnl, ann_sharpe, max_dd, RES

ROOT = Path(__file__).resolve().parent; SPLIT = pd.Timestamp("2022-07-01")


def main():
    arms, ks = build(RES / "main_dyn_strong")
    arms["1/N"] = load_pnl(RES / "bench_equalweight").iloc[1:].reindex(arms["base"].index)
    order = ["1/N", "base", "fast", "slow", "combined"]
    rows = []
    for name in order:
        r = arms[name].dropna(); h1 = r[r.index < SPLIT]; h2 = r[r.index >= SPLIT]
        rows.append(dict(arm=name, n_H1=len(h1), Sharpe_H1=ann_sharpe(h1.values), MaxDD_H1=100 * max_dd(h1.values),
                         n_H2=len(h2), Sharpe_H2=ann_sharpe(h2.values), MaxDD_H2=100 * max_dd(h2.values)))
    T = pd.DataFrame(rows).set_index("arm")
    print(f"H1: {arms['base'].index[0].date()} .. {SPLIT.date()} (exclusive) | H2: from {SPLIT.date()}")
    print(T.round(3).to_string())
    T.to_csv(ROOT / "subperiod_table.csv")


if __name__ == "__main__":
    main()
