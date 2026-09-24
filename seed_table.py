"""Seed sensitivity of the canonical arm: identical configuration, Monte Carlo seed 42 (canonical), 7, 2024.
For each seed: Table-1 metrics of base / fast / slow / combined, mean exposure, holdings, turnover, and
the within-seed paired bootstraps (combined vs fast, combined vs base). Across seeds: correlation of
monthly base returns, weight overlap of the target books. Output: seed_sensitivity.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from overlay_arms import build, metrics, paired_boot, load_weights, RES

ROOT = Path(__file__).resolve().parent
# 2024 are the grid cells with exactly that configuration.
SEEDS = {"42 (canonical)": "main_dyn_strong", "7": "grid_S5000_H60_s7", "2024": "grid_S5000_H60_s2024"}


def main():
    rows = []; base = {}; books = {}
    for seed, folder in SEEDS.items():
        p = RES / folder
        if not (p / "pnl.csv").exists():
            print(f"  seed {seed}: folder {folder} missing"); continue
        arms, ks = build(p)
        f = pd.read_csv(p / "forecast_risk.csv"); yrs = len(f) / 12
        for arm in ["base", "fast", "slow", "combined"]:
            m = metrics(arms[arm].values); m.update(seed=seed, arm=arm, mean_k=float(ks[arm].mean()))
            if arm == "base":
                m.update(held=f.n_held_target.mean(), turn_target_yr=f.turnover_target_l1.iloc[1:].sum() / yrs,
                         binding_share=float((f.lp_shadow_price > 1e-10).mean()))
            rows.append(m)
        pS, pD = paired_boot(arms["fast"].values, arms["combined"].values)
        pS2, pD2 = paired_boot(arms["base"].values, arms["combined"].values)
        rows[-1].update({"P(comb>fast) Sharpe": pS, "P(comb>fast) shallowerDD": pD, "P(comb>base) Sharpe": pS2, "P(comb>base) shallowerDD": pD2,
                         "corr(kF,kS)": float(ks["fast"].corr(ks["slow"]))})
        base[seed] = arms["base"]; books[seed] = load_weights(p, "weights")
    T = pd.DataFrame(rows).set_index(["seed", "arm"])
    pd.set_option("display.width", 250)
    cols = ["Ann", "Vol", "Sharpe", "Sortino", "MaxDD", "CVaR95", "mean_k", "held", "turn_target_yr", "binding_share",
            "P(comb>fast) Sharpe", "P(comb>fast) shallowerDD", "P(comb>base) Sharpe", "P(comb>base) shallowerDD", "corr(kF,kS)"]
    print(T[[c for c in cols if c in T.columns]].round(3).to_string())
    seeds = list(base)
    print("\nacross seeds — corr of monthly BASE returns / mean weight overlap of target books (sum of min weights):")
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = seeds[i], seeds[j]
            r = base[a].corr(base[b])
            ov = []
            for d in books[a]:
                if d in books[b]:
                    wa, wb = books[a][d], books[b][d]; idx = wa.index.union(wb.index)
                    ov.append(float(np.minimum(wa.reindex(idx).fillna(0), wb.reindex(idx).fillna(0)).sum()))
            print(f"  {a} vs {b}: corr {r:.3f} | overlap mean {np.mean(ov):.3f} median {np.median(ov):.3f} min {np.min(ov):.3f}")
    T.to_csv(ROOT / "seed_sensitivity.csv")
    print("saved seed_sensitivity.csv")


if __name__ == "__main__":
    main()
