"""Run the analytical/known-DGP benchmark used by the mathematical manuscript.

Example
-------
python -m scripts.runs.analytical_benchmark --output results/analytical_benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.theory.known_dgp_experiment import KnownDGPConfig, run_known_dgp_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/analytical_benchmark")
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--n-assets", type=int, default=24)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--execution-eta", type=float, default=0.55)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = KnownDGPConfig(
        seed=args.seed,
        n_assets=args.n_assets,
        top_k=args.top_k,
        execution_eta=args.execution_eta,
    )
    summary = run_known_dgp_experiment(Path(args.output), cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
