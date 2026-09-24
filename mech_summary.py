"""Mechanism experiments: canonical dynamic base vs executed-state carry, fixed universe, full execution.

For each arm builds base / fast / slow / combined overlays with overlay_arms.build (same conventions as
the paper), reports Table-1 metrics, mean exposure, holdings and turnover statistics, and paired
bootstraps of each arm's combined overlay against the canonical combined overlay.
Output: mech_experiments.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from pathlib import Path
from overlay_arms import build, metrics, paired_boot, RES

ROOT = Path(__file__).resolve().parent
ARMS = {
    "canonical (target state)": "main_dyn_strong",
    "executed-state carry": "mech_execstate_main_dyn_strong",
    "fixed universe (Jan-2019)": "main_dyn_strong_fixeduniv",
    "full execution (eta = 1)": "main_dyn_strong_eta1",
}


def holdings_and_turnover(folder: Path):
    f = pd.read_csv(folder / "forecast_risk.csv")
    yrs = len(f) / 12
    return dict(n_assets=f.n_assets.mean(), held_target=f.n_held_target.mean(), held_target_med=f.n_held_target.median(),
                eta=f.alpha_exec.mean(), turn_target_yr=f.turnover_target_l1.iloc[1:].sum() / yrs,
                turn_exec_yr=f.turnover_exec_l1.iloc[1:].sum() / yrs, binding_share=float((f.lp_shadow_price > 1e-10).mean()))


def main():
    rows = []; combined = {}
    for label, folder in ARMS.items():
        p = RES / folder
        if not (p / "pnl.csv").exists():
            print(f"  {label}: folder {folder} missing"); continue
        arms, ks = build(p)
        ht = holdings_and_turnover(p)
        pnl = pd.read_csv(p / "pnl.csv"); cost = float(pnl["Cost"].sum()) * 100
        for arm in ["base", "fast", "slow", "combined"]:
            m = metrics(arms[arm].values); m.update(dict(experiment=label, arm=arm, mean_k=float(ks[arm].mean())))
            if arm == "base":
                m.update(ht); m["cum_cost_pct"] = cost
            rows.append(m)
        combined[label] = arms["combined"]
    T = pd.DataFrame(rows).set_index(["experiment", "arm"])
    ref = combined.get("canonical (target state)")
    for label, rr in combined.items():
        if ref is None or label.startswith("canonical"):
            continue
        idx = ref.index.intersection(rr.index)
        pS, pD = paired_boot(ref.reindex(idx).values, rr.reindex(idx).values)
        T.loc[(label, "combined"), "P(> canonical) Sharpe"] = pS; T.loc[(label, "combined"), "P(> canonical) shallowerDD"] = pD
    pd.set_option("display.width", 250)
    cols = ["Ann", "Vol", "Sharpe", "Sortino", "MaxDD", "CVaR95", "mean_k", "held_target", "eta", "turn_target_yr", "turn_exec_yr", "cum_cost_pct", "binding_share", "P(> canonical) Sharpe", "P(> canonical) shallowerDD"]
    print(T[[c for c in cols if c in T.columns]].round(3).to_string())
    T.to_csv(ROOT / "mech_experiments.csv")


if __name__ == "__main__":
    main()
