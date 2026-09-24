"""Composition statistics of the target book for every cell of the scenario-count x history-rule x seed
grid (Section 7.8): holdings per month, effective number of positions 1 / sum(w^2), weight of the largest
and of the three largest positions, number of distinct funds held and the most frequently held funds.

Run from the package root after the grid arms:  python grid_composition.py
Output: grid_composition.csv
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from pathlib import Path

from overlay_arms import load_weights, RES
from grid_table import CELLS

ROOT = Path(__file__).resolve().parent
START = pd.Timestamp("2019-02-01")     # first month after the initialization month


def main():
    rows = []
    for (S, H, seed), folder in sorted(CELLS.items()):
        p = RES / folder
        if not (p / "weights.xlsx").exists():
            continue
        W = load_weights(p, "weights"); W = {d: w for d, w in W.items() if d >= START}
        n = []; top1 = []; top3 = []; effn = []; names = {}
        for d, w in W.items():
            w = w[w > 1e-6]; n.append(len(w)); ws = w.sort_values(ascending=False)
            top1.append(ws.iloc[0]); top3.append(ws.iloc[:3].sum()); effn.append(1.0 / float((w ** 2).sum()))
            for t, v in w.items():
                names[t] = names.get(t, 0) + 1
        top = sorted(names.items(), key=lambda kv: -kv[1])[:6]
        rows.append(dict(scen=S, hist=H, seed=seed, held_mean=np.mean(n), held_min=min(n), held_max=max(n),
                         effN_mean=np.mean(effn), effN_min=min(effn), top1_w=np.mean(top1), top3_w=np.mean(top3),
                         distinct=len(names), most_held=" ".join(f"{t}:{c}" for t, c in top)))
    T = pd.DataFrame(rows).set_index(["scen", "hist", "seed"])
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(T.round(2).to_string())
    T.to_csv(ROOT / "grid_composition.csv")
    print("saved grid_composition.csv")


if __name__ == "__main__":
    main()
