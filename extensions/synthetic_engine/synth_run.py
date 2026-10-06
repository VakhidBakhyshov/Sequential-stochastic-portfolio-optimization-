# -*- coding: utf-8 -*-
"""Run the PRODUCTION monthly policy (scripts/runs/run.py::model_computation, unchanged) on a synthetic
market with a known law. Usage (from the package root):
    python -m extensions.synthetic_engine.synth_run --law gbm    [--seed 20260913]
    python -m extensions.synthetic_engine.synth_run --law regime [--seed 20260913]
Outputs: results/known_law_<law>/ (pnl.csv, weights.xlsx, forecast_risk.csv, dynamic_parameter_history.csv,
the synthetic inputs and the truth)."""
from __future__ import annotations
import argparse, copy, sys, io, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]           # package root
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import yaml
from extensions.synthetic_engine.synth_market import MarketSpec, make_market, engine_inputs, eligibility, write_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", default="gbm", choices=["gbm", "regime"])
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--config", default="main_dyn_strong.yaml")
    ap.add_argument("--folder", default=None)
    a = ap.parse_args()
    spec = MarketSpec(law=a.law, seed=a.seed)
    folder = a.folder or f"known_law_{a.law}"
    m = make_market(spec)
    inp = engine_inputs(m)
    portfolios, audit = eligibility(m, spec, inp["first_business_days"])
    out = ROOT / "results" / folder
    write_inputs(m, spec, out, audit)
    print(f"[synth] law={a.law} seed={a.seed} assets={spec.n_assets} days={len(m['dates'])} "
          f"stress days={int(m['state'].sum())} eligible/month min={audit['eligible'].min()} max={audit['eligible'].max()}", flush=True)
    cfg = yaml.safe_load((ROOT / "scripts" / "configs" / a.config).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg); cfg["output_folder"] = folder
    from scripts.runs.run import model_computation         # production loop, imported unchanged
    t0 = time.time()
    model_computation(config=cfg, first_business_days=inp["first_business_days"], market_cap=inp["market_cap"],
                      ewma_returns_all=inp["ewma_returns_all"], returns_all=inp["returns_all"], portfolios=portfolios, output_folder=folder)
    print(f"[synth] done in {time.time() - t0:.0f} s -> results/{folder}", flush=True)


if __name__ == "__main__":
    main()
