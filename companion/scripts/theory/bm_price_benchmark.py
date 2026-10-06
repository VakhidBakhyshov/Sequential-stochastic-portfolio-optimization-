"""Closed-form arithmetic-Brownian price benchmark for the manuscript.

This module complements ``known_dgp_experiment.py`` (which exercises the full
sequential architecture under correlated GBM) with a pure price-level BM test.
It is deliberately diagnostic: arithmetic BM has a Gaussian terminal price law
and therefore admits negative prices with a known probability.  The output is
used to verify formulas and to document why GBM, rather than BM, is the natural
positive-price benchmark for ETFs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .diffusion_benchmark import (
    ArithmeticBrownianSpec,
    bm_linear_portfolio_price_moments,
    bm_nonpositive_price_probability,
    bm_terminal_price_moments,
    correlated_bm_price_paths,
)


def run_bm_price_benchmark(output_dir: str | Path, *, seed: int = 20260909) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s0 = np.array([70.0, 85.0, 100.0, 120.0, 140.0, 160.0])
    drift = np.array([2.0, 2.5, 3.0, 1.0, -0.5, 1.5])
    abs_sigma = np.array([10.0, 12.0, 14.0, 18.0, 22.0, 25.0])
    rho = 0.35
    corr = np.full((len(s0), len(s0)), rho)
    np.fill_diagonal(corr, 1.0)
    spec = ArithmeticBrownianSpec(drift=drift, abs_sigma=abs_sigma, corr=corr, s0=s0)

    horizons = [21, 63, 126, 252, 756]
    rows = []
    for h in horizons:
        mean, cov = bm_terminal_price_moments(spec, h)
        pneg = bm_nonpositive_price_probability(spec, h)
        for i in range(spec.n_assets):
            rows.append({
                "horizon_days": h,
                "asset": f"BM{i:02d}",
                "terminal_mean": mean[i],
                "terminal_sd": np.sqrt(cov[i, i]),
                "p_terminal_price_le_zero": pneg[i],
            })
    terminal_table = pd.DataFrame(rows)
    terminal_table.to_csv(out / "bm_terminal_law.csv", index=False)

    # Monte Carlo verification at one year.
    n_paths = 100_000
    # Exact one-step draw over a one-year increment; avoids storing 252 intermediate states.
    paths = correlated_bm_price_paths(spec, n_steps=1, dt=1.0, n_paths=n_paths, seed=seed)
    terminal = paths[:, -1, :]
    analytic_mean, analytic_cov = bm_terminal_price_moments(spec, 252)
    mc_mean = terminal.mean(axis=0)
    mc_cov = np.cov(terminal.T)
    mean_err = float(np.max(np.abs(mc_mean - analytic_mean)))
    cov_err = float(np.max(np.abs(mc_cov - analytic_cov)))

    holdings = np.repeat(1.0 / spec.n_assets, spec.n_assets)
    pmean, pvar = bm_linear_portfolio_price_moments(holdings, spec, 252)
    portfolio_terminal = terminal @ holdings
    portfolio_mean_err = float(abs(portfolio_terminal.mean() - pmean))
    portfolio_var_err = float(abs(portfolio_terminal.var(ddof=1) - pvar))

    summary = {
        "label": "arithmetic_BM_price_level_known_DGP_diagnostic",
        "seed": seed,
        "n_paths": n_paths,
        "pairwise_brownian_correlation": rho,
        "one_year_max_abs_terminal_mean_error": mean_err,
        "one_year_max_abs_terminal_covariance_error": cov_err,
        "one_year_equal_unit_portfolio_mean_error": portfolio_mean_err,
        "one_year_equal_unit_portfolio_variance_error": portfolio_var_err,
        "max_nonpositive_probability_1y": float(bm_nonpositive_price_probability(spec, 252).max()),
        "max_nonpositive_probability_3y": float(bm_nonpositive_price_probability(spec, 756).max()),
        "interpretation": "BM has an exact Gaussian price law but violates global price positivity; GBM is therefore the primary positive-price diffusion benchmark.",
    }
    (out / "bm_price_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        sample_paths = correlated_bm_price_paths(spec, n_steps=252, n_paths=8, seed=seed + 1)
        sample = sample_paths[:, :, 0].T
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(sample)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title("Arithmetic-Brownian price benchmark: sample paths")
        ax.set_xlabel("trading day")
        ax.set_ylabel("price")
        fig.tight_layout()
        fig.savefig(out / "figure_bm_price_paths.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/known_dgp_bm")
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    print(json.dumps(run_bm_price_benchmark(args.output_dir, seed=args.seed), indent=2))
