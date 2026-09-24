"""Grid experiment: canonical dynamic arm x scenarios {1000, 5000} x history rule
{36, 48, 60 months} x seed {42, 7, 2024}. For every cell: base / fast / slow / combined Sharpe and
MaxDD (paper conventions via overlay_arms.build), mean eligible names, holdings, and the paired
bootstrap P(combined shallower DD than base). Output: grid_results.csv / .md
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from overlay_arms import build, metrics, paired_boot, RES

ROOT = Path(__file__).resolve().parent
# history_m = 0 means "screen as is" (no enforced history rule) = the canonical configuration
CELLS = {(1000, 0, 42): "main_dyn_strong_S1000", (1000, 0, 7): "main_dyn_strong_seed7", (1000, 0, 2024): "main_dyn_strong_seed2024",
         (5000, 0, 42): "grid_S5000_Hnone_s42", (5000, 0, 7): "grid_S5000_Hnone_s7", (5000, 0, 2024): "grid_S5000_Hnone_s2024"}
for S in (1000, 5000):
    for H in (36, 48, 60):
        for seed in (42, 7, 2024):
            CELLS.setdefault((S, H, seed), f"grid_S{S}_H{H}_s{seed}")


def main():
    rows = []
    for (S, H, seed), folder in CELLS.items():
        p = RES / folder
        if not (p / "pnl.csv").exists():
            continue
        arms, ks = build(p)
        f = pd.read_csv(p / "forecast_risk.csv")
        r = dict(scenarios=S, history_m=H, seed=seed, folder=folder, n_eligible=f.n_assets.mean(), held=f.n_held_target.mean())
        for arm in ["base", "fast", "slow", "combined"]:
            m = metrics(arms[arm].values); r[f"{arm}_Sharpe"] = m["Sharpe"]; r[f"{arm}_MaxDD"] = m["MaxDD"]
        r["comb_mean_k"] = float(ks["combined"].mean())
        pS, pD = paired_boot(arms["base"].values, arms["combined"].values)
        r["P(comb>base) Sharpe"] = pS; r["P(comb shallower) DD"] = pD
        rows.append(r)
    T = pd.DataFrame(rows).set_index(["scenarios", "history_m", "seed"]).sort_index()
    pd.set_option("display.width", 250)
    cols = ["n_eligible", "held", "combined_Sharpe", "combined_MaxDD", "comb_mean_k", "P(comb shallower) DD"]
    print(T[cols].round(3).to_string())
    T.to_csv(ROOT / "grid_results.csv")   # full table (all arms) kept on disk for the ledger
    g = T.groupby(level=[0, 1])
    summ = pd.concat({"mean over seeds": g[["combined_Sharpe", "combined_MaxDD"]].mean(),
                      "best seed": g[["combined_Sharpe"]].max(),
                      "worst seed": g[["combined_Sharpe"]].min()}, axis=1)
    print("\n" + summ.round(3).to_string())
    md = ["# Grid results", "", T[cols].round(3).to_markdown(), "", "## Over seeds", "", summ.round(3).to_markdown()]
    (ROOT / "grid_results.md").write_text("\n".join(md), encoding="utf-8")
    print("saved grid_results.csv / .md")


if __name__ == "__main__":
    main()
