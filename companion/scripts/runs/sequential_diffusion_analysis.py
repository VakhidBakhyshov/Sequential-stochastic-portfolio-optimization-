"""Run the strengthened exact-law diffusion analysis used before synthetic and ETF data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.theory.sequential_diffusion_analysis import SequentialDiffusionConfig, run_sequential_diffusion_analysis


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/sequential_diffusion_analysis")
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--replications", type=int, default=120)
    args = p.parse_args()
    cfg = SequentialDiffusionConfig(seed=args.seed, n_estimation_replications=args.replications)
    print(json.dumps(run_sequential_diffusion_analysis(Path(args.output), cfg), indent=2))


if __name__ == "__main__":
    main()
