"""Generate deterministic CDaR theorem-to-code diagnostics, tables and figures.

These outputs are analytical and controlled-synthetic evidence and are kept separate
from the historical ETF results.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.calculations.drawdown import drawdown_curve_from_returns, empirical_cdar, portfolio_path_returns
from scripts.optimizers.cdar import BayessianCDaR
from scripts.theory.cdar_benchmark import (
    CDaRBrownianSpec, brownian_average_drawdown_check, cdar_alpha_curve,
    convexity_check, homogeneity_check, path_order_counterexample, zero_drift_brownian_paths,
    terminal_drawdown_cdar_mc_check, exact_terminal_drawdown_cdar_zero_drift_bm,
    risk_geometry_ranking_reversal,
)


def synthetic_asset_paths(seed=20260911, S=180, H=21, N=8):
    rng = np.random.default_rng(seed)
    vol = np.linspace(0.008, 0.018, N)
    corr = 0.30 * np.ones((N, N)) + 0.70 * np.eye(N)
    cov = np.outer(vol, vol) * corr
    L = np.linalg.cholesky(cov)
    mu = np.linspace(0.0001, 0.0005, N)
    z = rng.standard_normal((S, H, N)) @ L.T
    return mu[None, None, :] + z


def run(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. Exact marginal-law consequence under zero-drift Brownian motion.
    spec = CDaRBrownianSpec(paths=25000)
    bm = zero_drift_brownian_paths(spec)
    bchk = brownian_average_drawdown_check(spec)
    rows.append({"check": "zero-drift BM expected average drawdown", "value": bchk["mc_average_drawdown"], "reference": bchk["continuous_exact"], "residual": bchk["absolute_error"]})
    terminal_checks = []
    for a in (0.50, 0.80, 0.90, 0.95, 0.975, 0.99):
        chk = terminal_drawdown_cdar_mc_check(spec, alpha=a)
        terminal_checks.append(chk)
    t95 = next(x for x in terminal_checks if abs(x["alpha"] - 0.95) < 1e-12)
    rows.append({"check": "zero-drift BM terminal CDaR95 exact law", "value": t95["mc_terminal_cdar"], "reference": t95["exact_terminal_cdar"], "residual": t95["absolute_cdar_error"]})

    # 2. CDaR alpha continuum and limiting cases.
    dd = drawdown_curve_from_returns(bm[:5000], axis=1)
    empirical_max_alpha = 1.0 - 1.0 / dd.size
    alphas = np.array([0.00, 0.50, 0.80, 0.90, 0.95, 0.975, 0.99, empirical_max_alpha])
    curve = cdar_alpha_curve(bm[:5000], alphas)
    rows.append({"check": "alpha=0 equals average drawdown", "value": curve[0], "reference": float(dd.mean()), "residual": abs(curve[0] - float(dd.mean()))})
    rows.append({"check": "high-alpha approaches maximum drawdown", "value": curve[-1], "reference": float(dd.max()), "residual": float(dd.max() - curve[-1])})

    # 3. Coherence-relevant properties and path-dependence identification.
    hom = homogeneity_check(bm[:2000], 0.95)
    rows.append({"check": "positive homogeneity max residual", "value": hom, "reference": 0.0, "residual": hom})
    paths = synthetic_asset_paths()
    w1 = np.array([.25,.20,.15,.10,.10,.08,.07,.05])
    w2 = w1[::-1].copy()
    cvx = convexity_check(paths, w1, w2)
    rows.append({"check": "convexity Jensen gap (must <=0)", "value": cvx, "reference": 0.0, "residual": max(cvx, 0.0)})
    counter = path_order_counterexample()
    rows.append({"check": "same-terminal path CDaR difference", "value": counter["cdar95_a"]-counter["cdar95_b"], "reference": 0.0, "residual": abs(counter["cdar95_a"]-counter["cdar95_b"])})
    reversal = risk_geometry_ranking_reversal(0.95)
    rows.append({"check": "CVaR-CDaR ranking reversal indicator", "value": reversal["terminal_prefers_a"] * reversal["cdar_prefers_b"], "reference": 1.0, "residual": abs(1.0 - reversal["terminal_prefers_a"] * reversal["cdar_prefers_b"])})

    # 4. End-to-end CDaR LP on a controlled predictive path cube.
    R_terminal = paths.sum(axis=1)
    N = paths.shape[2]
    cfg = {
        "task_type":"return_cdar_constraint", "confidence_level":0.95,
        "cdar_confidence_level":0.95, "cdar_budget_mult":1.0,
        "turnover_penalty":0.002, "penalty_type":"L1", "is_all_methods":False,
        "min_weight":0.0, "max_weight":0.35, "constraint_max_weight":True,
        "scenario_paths":paths,
    }
    opt = BayessianCDaR(cfg, np.ones(N), paths.reshape(-1,N), R_terminal, np.ones(N)/N)
    weights = opt.get_results()[1][0]
    diag = opt.last_lp_diagnostics
    ew = np.ones(N)/N
    cdar_opt = empirical_cdar(portfolio_path_returns(paths, weights), 0.95)
    cdar_ew = empirical_cdar(portfolio_path_returns(paths, ew), 0.95)
    rows.append({"check":"CDaR LP budget feasibility", "value":cdar_opt, "reference":cdar_ew, "residual":max(cdar_opt-cdar_ew,0.0)})
    rows.append({"check":"CDaR LP weight sum", "value":float(weights.sum()), "reference":1.0, "residual":abs(float(weights.sum())-1.0)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cdar_theorem_to_code_checks.csv", index=False)
    pd.DataFrame({"alpha":alphas, "cdar":curve}).to_csv(out_dir / "cdar_alpha_curve.csv", index=False)
    pd.DataFrame({"asset":np.arange(N), "equal_weight":ew, "cdar_optimal_weight":weights}).to_csv(out_dir / "cdar_weight_comparison.csv", index=False)
    pd.DataFrame(terminal_checks).to_csv(out_dir / "cdar_brownian_terminal_exactlaw.csv", index=False)
    pd.DataFrame([reversal]).to_csv(out_dir / "cvar_cdar_ranking_reversal.csv", index=False)

    # Scenario-count convergence for terminal CDaR against the exact Brownian oracle.
    conv_rows = []
    for n in (100, 250, 500, 1000, 2500, 5000, 10000, 25000):
        sub_spec = CDaRBrownianSpec(sigma=spec.sigma, horizon_years=spec.horizon_years, steps=spec.steps, paths=n, seed=spec.seed)
        chk = terminal_drawdown_cdar_mc_check(sub_spec, alpha=0.95)
        conv_rows.append({"paths": n, **chk})
    pd.DataFrame(conv_rows).to_csv(out_dir / "cdar_scenario_convergence.csv", index=False)

    # Figure A: alpha continuum.
    fig, ax = plt.subplots(figsize=(7.2,4.4)); ax.plot(alphas, curve, marker="o"); ax.set_xlabel("CDaR confidence alpha"); ax.set_ylabel("CDaR"); ax.set_title("CDaR interpolates average toward extreme drawdown"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir / "figure_cdar_alpha_continuum.png", dpi=180); plt.close(fig)

    # Figure B: exact-law average drawdown convergence across Monte Carlo path count.
    counts = np.array([250,500,1000,2500,5000,10000,25000]); vals=[]
    all_dd = drawdown_curve_from_returns(bm, axis=1).mean(axis=1)
    for n in counts: vals.append(float(all_dd[:n].mean()))
    fig, ax = plt.subplots(figsize=(7.2,4.4)); ax.plot(counts, vals, marker="o", label="Monte Carlo"); ax.axhline(bchk["continuous_exact"], linestyle="--", label="continuous-time exact expectation"); ax.set_xscale("log"); ax.set_xlabel("simulated paths"); ax.set_ylabel("expected average drawdown"); ax.set_title("Known-law CDaR(alpha=0) identification under zero-drift Brownian motion"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir / "figure_cdar_brownian_exactlaw.png", dpi=180); plt.close(fig)

    # Figure C: equal-weight vs CDaR-optimal underwater distribution.
    dd_ew = drawdown_curve_from_returns(portfolio_path_returns(paths, ew), axis=1).ravel()
    dd_opt = drawdown_curve_from_returns(portfolio_path_returns(paths, weights), axis=1).ravel()
    q = np.linspace(.50,.995,60)
    fig, ax = plt.subplots(figsize=(7.2,4.4)); ax.plot(q, np.quantile(dd_ew,q), label="equal weight"); ax.plot(q, np.quantile(dd_opt,q), label="CDaR constrained optimum"); ax.set_xlabel("drawdown quantile"); ax.set_ylabel("drawdown depth"); ax.set_title("Controlled predictive-path drawdown distribution"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir / "figure_cdar_drawdown_quantiles.png", dpi=180); plt.close(fig)

    conv = pd.DataFrame(conv_rows)
    fig, ax = plt.subplots(figsize=(7.2,4.4)); ax.plot(conv["paths"], conv["absolute_cdar_error"], marker="o"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Brownian paths"); ax.set_ylabel("absolute CDaR95 error"); ax.set_title("Finite-scenario convergence to exact terminal-drawdown CDaR"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir / "figure_cdar_scenario_convergence.png", dpi=180); plt.close(fig)

    summary = {
        "brownian": bchk, "brownian_terminal_cdar": terminal_checks,
        "path_order_counterexample": counter, "risk_geometry_ranking_reversal": reversal,
        "lp_diagnostics": diag, "weights": weights.tolist(),
        "note": "Synthetic/analytical diagnostics only; no new historical ETF CDaR performance is claimed without the frozen point-in-time data vintage."
    }
    (out_dir / "cdar_validation_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(df.to_string(index=False))
    print(f"\nWrote CDaR validation artifacts to {out_dir}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir', default='results/cdar_validation'); args=ap.parse_args(); run(Path(args.output_dir))
if __name__=='__main__': main()
