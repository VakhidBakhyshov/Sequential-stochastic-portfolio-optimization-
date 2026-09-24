"""Deterministic known-DGP calibration for the CDaR optimizer.

This is an executable calibration layer when the immutable ETF vintage is absent.
It does not replace the historical Optuna study; it verifies that alpha, budget
and turnover regularization create measurable out-of-sample trade-offs under a
controlled sequential path law, with executed holdings carried as state.
"""
from __future__ import annotations

import argparse, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.calculations.drawdown import empirical_cdar, portfolio_path_returns
from scripts.optimizers.cdar import BayessianCDaR


def _paths(rng: np.random.Generator, *, s: int, h: int, n: int, stress: bool) -> np.ndarray:
    base_vol = np.linspace(0.006, 0.014, n) * (1.7 if stress else 1.0)
    corr_level = 0.65 if stress else 0.25
    corr = corr_level * np.ones((n, n)) + (1.0 - corr_level) * np.eye(n)
    cov = np.outer(base_vol, base_vol) * corr
    L = np.linalg.cholesky(cov)
    mu = np.linspace(0.00005, 0.00045, n)
    if stress:
        mu = mu - np.linspace(0.0002, 0.0007, n)
    return mu[None, None, :] + rng.standard_normal((s, h, n)) @ L.T


def _run_one(alpha: float, budget_mult: float, turnover_penalty: float, *, seed: int = 20260911) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, h, months = 6, 10, 18
    w_state = np.ones(n) / n
    period_returns, test_cdars, turnovers, shadow_prices = [], [], [], []
    solve_count = 0
    for m in range(months):
        stress = (m % 6) in {4, 5}
        train = _paths(rng, s=48, h=h, n=n, stress=stress)
        test = _paths(rng, s=240, h=h, n=n, stress=stress)
        terminal = train.sum(axis=1)
        cfg = {
            "task_type": "return_cdar_constraint", "confidence_level": alpha,
            "cdar_confidence_level": alpha, "cdar_budget_mult": budget_mult,
            "turnover_penalty": turnover_penalty, "penalty_type": "L1",
            "is_all_methods": False, "min_weight": 0.0, "max_weight": 0.40,
            "constraint_max_weight": True, "scenario_paths": train,
        }
        opt = BayessianCDaR(cfg, np.ones(n), train.reshape(-1, n), terminal, w_state)
        weights = opt.get_results()[1]
        if not weights:
            continue
        target = np.asarray(weights[0], dtype=float)
        eta = 0.55
        executed = w_state + eta * (target - w_state)
        turnover = float(np.sum(np.abs(executed - w_state)))
        test_port = portfolio_path_returns(test, executed)
        test_terminal = test_port.sum(axis=1)
        # one-way proportional cost applied to the period return
        period_returns.append(float(np.mean(test_terminal) - 0.001 * turnover))
        test_cdars.append(float(empirical_cdar(test_port, alpha=0.95)))
        turnovers.append(turnover)
        diag = getattr(opt, "last_lp_diagnostics", {}) or {}
        shadow_prices.append(float(diag.get("cdar_budget_shadow_price_max", np.nan)))
        w_state = executed
        solve_count += 1
    r = np.asarray(period_returns, dtype=float)
    c = np.asarray(test_cdars, dtype=float)
    t = np.asarray(turnovers, dtype=float)
    mean_r = float(np.nanmean(r)) if r.size else np.nan
    vol_r = float(np.nanstd(r, ddof=1)) if r.size > 1 else np.nan
    sharpe = mean_r / vol_r * np.sqrt(12.0) if np.isfinite(vol_r) and vol_r > 0 else np.nan
    return {
        "alpha": alpha, "cdar_budget_mult": budget_mult, "turnover_penalty": turnover_penalty,
        "months_solved": solve_count, "mean_test_return": mean_r,
        "annualized_test_return": 12.0 * mean_r, "annualized_test_vol": np.sqrt(12.0) * vol_r,
        "test_sharpe_zero_rf": float(sharpe), "mean_test_cdar95": float(np.nanmean(c)),
        "max_test_cdar95": float(np.nanmax(c)), "mean_executed_turnover": float(np.nanmean(t)),
        "mean_shadow_price": float(np.nanmean(shadow_prices)),
    }


def run(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for alpha, budget, lam in itertools.product(
        (0.90, 0.95, 0.975), (0.85, 1.00, 1.15), (0.0, 0.001, 0.003, 0.006)
    ):
        rows.append(_run_one(alpha, budget, lam))
    df = pd.DataFrame(rows)
    # A return floor prevents risk control from choosing a trivial low-opportunity recipe.
    floor = float(df["annualized_test_return"].quantile(0.35))
    eligible = df[df["annualized_test_return"] >= floor].copy()
    for col in ("mean_test_cdar95", "mean_executed_turnover", "annualized_test_vol"):
        sd = max(float(eligible[col].std(ddof=0)), 1e-12)
        eligible[f"z_{col}"] = (eligible[col] - float(eligible[col].mean())) / sd
    sd_sr = max(float(eligible["test_sharpe_zero_rf"].std(ddof=0)), 1e-12)
    eligible["z_sharpe"] = (eligible["test_sharpe_zero_rf"] - float(eligible["test_sharpe_zero_rf"].mean())) / sd_sr
    eligible["risk_control_score"] = eligible["z_sharpe"] - 0.60*eligible["z_mean_test_cdar95"] - 0.20*eligible["z_annualized_test_vol"] - 0.10*eligible["z_mean_executed_turnover"]
    best_risk_idx = eligible["risk_control_score"].idxmax()
    best_return_idx = df["annualized_test_return"].idxmax()
    df["selected_risk_control"] = False; df.loc[best_risk_idx, "selected_risk_control"] = True
    df["selected_max_return"] = False; df.loc[best_return_idx, "selected_max_return"] = True
    df.to_csv(out_dir / "cdar_synthetic_calibration_trials.csv", index=False)
    best = {
        "risk_control": df.loc[best_risk_idx].to_dict(),
        "max_return": df.loc[best_return_idx].to_dict(),
        "return_floor": floor,
        "evidence_boundary": "Controlled known-DGP calibration only; historical ETF Optuna calibration remains data-pending.",
    }
    (out_dir / "cdar_synthetic_calibration_summary.json").write_text(json.dumps(best, indent=2, default=float))

    # Heatmap: best risk-control score within each alpha x budget cell across turnover penalties.
    scored = eligible.copy()
    pivot = scored.pivot_table(index="alpha", columns="cdar_budget_mult", values="risk_control_score", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=[f"{x:.3f}" for x in pivot.index])
    ax.set_xlabel("CDaR budget multiplier"); ax.set_ylabel("CDaR confidence alpha")
    ax.set_title("Known-DGP CDaR calibration: best risk-control score")
    fig.colorbar(im, ax=ax, label="risk-control score")
    fig.tight_layout(); fig.savefig(out_dir / "figure_cdar_calibration_heatmap.png", dpi=180); plt.close(fig)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", default="results/cdar_synthetic_calibration")
    args = ap.parse_args(); out = run(Path(args.output_dir)); print(f"Wrote {out}")


if __name__ == "__main__":
    main()
