"""Exact and plug-in benchmarks for the sequential portfolio policy under correlated diffusions.

This module strengthens the analytical layer of the research pipeline.  It does not
assume that historical ETF returns are Gaussian or that ETF prices are literally
GBM.  Instead it provides an identification environment in which the experimenter
knows the joint law and can compare the scenario program, filtering diagnostics,
risk signals and execution state against oracle quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .diffusion_benchmark import (
    DiffusionSpec,
    centered_normal_cvar,
    correlated_gbm_paths,
    empirical_cvar_loss,
    solve_return_cvar_lp,
)

EPS = 1e-12


@dataclass(frozen=True)
class SequentialDiffusionConfig:
    n_assets: int = 12
    horizon_days: int = 21
    beta: float = 0.95
    max_weight: float = 0.20
    turnover_penalty: float = 0.002
    scenario_counts: tuple[int, ...] = (250, 500, 1000, 2500, 5000)
    estimation_windows: tuple[int, ...] = (63, 126, 252, 504, 756)
    n_estimation_replications: int = 120
    seed: int = 20260909


def gaussian_portfolio_moments(
    weights: Iterable[float], mean: np.ndarray, covariance: np.ndarray
) -> tuple[float, float]:
    """Mean and standard deviation of a linear Gaussian portfolio return."""
    w = np.asarray(list(weights), dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (w.size, w.size) or mu.size != w.size:
        raise ValueError("weights, mean and covariance have incompatible shapes")
    return float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0.0)))


def gaussian_loss_cvar(
    weights: Iterable[float], mean: np.ndarray, covariance: np.ndarray, beta: float = 0.95, *, centered: bool = False
) -> float:
    """Exact CVaR of loss ``-w'R`` under ``R ~ N(mean,covariance)``."""
    m, s = gaussian_portfolio_moments(weights, mean, covariance)
    q = float(norm.pdf(norm.ppf(beta)) / (1.0 - beta))
    return float(q * s if centered else -m + q * s)


def gaussian_cvar_gradient(
    weights: Iterable[float], mean: np.ndarray, covariance: np.ndarray, beta: float = 0.95, *, centered: bool = False
) -> np.ndarray:
    """Gradient of exact Gaussian CVaR with respect to portfolio weights."""
    w = np.asarray(list(weights), dtype=float).reshape(-1)
    mu = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)
    s = np.sqrt(max(float(w @ cov @ w), EPS))
    q = float(norm.pdf(norm.ppf(beta)) / (1.0 - beta))
    grad = q * (cov @ w) / s
    if not centered:
        grad = grad - mu
    return np.asarray(grad, dtype=float)


def correlation_cvar_derivative(
    weights: Iterable[float], sigma: Iterable[float], corr: np.ndarray, i: int, j: int,
    *, beta: float = 0.95, horizon_years: float = 1.0
) -> float:
    """Derivative of centered Gaussian CVaR with respect to correlation rho_ij.

    For long-only weights the derivative is non-negative.  This directly quantifies
    the mathematical cost of correlation in the known-DGP benchmark.
    """
    w = np.asarray(list(weights), dtype=float)
    sig = np.asarray(list(sigma), dtype=float)
    rho = np.asarray(corr, dtype=float)
    cov = np.outer(sig, sig) * rho * float(horizon_years)
    s = np.sqrt(max(float(w @ cov @ w), EPS))
    q = float(norm.pdf(norm.ppf(beta)) / (1.0 - beta))
    return float(q * w[i] * w[j] * sig[i] * sig[j] * float(horizon_years) / s)


def solve_exact_gaussian_cvar_program(
    mean: np.ndarray,
    covariance: np.ndarray,
    previous_weights: np.ndarray,
    *,
    beta: float,
    cvar_budget: float,
    max_weight: float,
    turnover_penalty: float,
) -> dict[str, object]:
    """Solve the exact known-law return/CVaR/turnover program.

    The Gaussian CVaR constraint is ``-mu'w + q_beta sqrt(w' Sigma w) <= c``.
    The objective matches the scenario LP: maximize ``mu'w-lambda||w-w_prev||_1``.
    SLSQP is used only as an oracle benchmark; the production strategy remains the
    scenario-based linear program.
    """
    mu = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(covariance, dtype=float)
    prev = np.asarray(previous_weights, dtype=float).reshape(-1)
    n = len(mu)
    if cov.shape != (n, n) or prev.size != n:
        raise ValueError("incompatible shapes")
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError("max_weight makes the fully invested set infeasible")

    x0 = np.clip(prev, 0.0, max_weight)
    if x0.sum() <= EPS:
        x0[:] = 1.0 / n
    else:
        x0 /= x0.sum()
    # Equal weight is always a useful feasible fallback when the budget is defined from it.
    ew = np.ones(n) / n
    if gaussian_loss_cvar(x0, mu, cov, beta) > cvar_budget + 1e-8:
        x0 = ew

    def objective(w: np.ndarray) -> float:
        return float(-(mu @ w - turnover_penalty * np.sum(np.abs(w - prev))))

    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "ineq", "fun": lambda w: float(cvar_budget - gaussian_loss_cvar(w, mu, cov, beta))},
    ]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 3000, "disp": False},
    )
    w = np.asarray(res.x, dtype=float)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "weights": w,
        "expected_return": float(mu @ w),
        "turnover": float(np.sum(np.abs(w - prev))),
        "cvar": gaussian_loss_cvar(w, mu, cov, beta),
        "objective": float(mu @ w - turnover_penalty * np.sum(np.abs(w - prev))),
    }


def _make_spec(cfg: SequentialDiffusionConfig) -> DiffusionSpec:
    rng = np.random.default_rng(cfg.seed)
    mu = np.linspace(0.04, 0.11, cfg.n_assets) + rng.normal(0, 0.005, cfg.n_assets)
    sigma = np.linspace(0.10, 0.26, cfg.n_assets) + rng.normal(0, 0.006, cfg.n_assets)
    load1 = np.linspace(0.20, 0.75, cfg.n_assets)
    load2 = 0.25 * np.cos(np.linspace(0, 2 * np.pi, cfg.n_assets))
    raw = np.outer(load1, load1) + np.outer(load2, load2) + 0.55 * np.eye(cfg.n_assets)
    d = np.sqrt(np.diag(raw))
    corr = raw / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return DiffusionSpec(mu=mu, sigma=np.clip(sigma, 0.07, None), corr=corr, s0=100.0)


def _plot_results(out: Path, cvar_df: pd.DataFrame, est_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(cvar_df["scenario_count"], cvar_df["mean_abs_cvar_error"], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("scenario count")
    ax.set_ylabel("mean absolute CVaR error")
    ax.set_title("Scenario CVaR convergence to exact Gaussian CVaR")
    fig.tight_layout()
    fig.savefig(out / "figure_exact_cvar_scenario_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(est_df["window_days"], est_df["mean_error_mu_l2"], marker="o", label="mean-vector error")
    ax.plot(est_df["window_days"], est_df["mean_error_cov_op"], marker="o", label="covariance operator error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("estimation window (days)")
    ax.set_ylabel("estimation error")
    ax.set_title("Plug-in estimation error under the known GBM law")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure_parameter_estimation_convergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.plot(corr_df["rho"], corr_df["centered_cvar"])
    ax.set_xlabel("common off-diagonal correlation")
    ax.set_ylabel("exact centered CVaR")
    ax.set_title("Correlation sensitivity of exact portfolio tail risk")
    fig.tight_layout()
    fig.savefig(out / "figure_correlation_cvar_sensitivity.png", dpi=180)
    plt.close(fig)


def run_sequential_diffusion_analysis(
    output_dir: str | Path, cfg: SequentialDiffusionConfig | None = None
) -> dict[str, object]:
    cfg = cfg or SequentialDiffusionConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = _make_spec(cfg)
    rng = np.random.default_rng(cfg.seed + 11)
    dt = cfg.horizon_days / 252.0

    # Exact horizon law for log returns.
    mean_h = spec.log_drift * dt
    cov_h = spec.covariance * dt
    n = spec.n_assets
    prev = np.ones(n) / n
    ew = prev.copy()
    exact_budget = gaussian_loss_cvar(ew, mean_h, cov_h, cfg.beta)
    oracle = solve_exact_gaussian_cvar_program(
        mean_h, cov_h, prev, beta=cfg.beta, cvar_budget=exact_budget,
        max_weight=cfg.max_weight, turnover_penalty=cfg.turnover_penalty,
    )
    if not oracle["success"]:
        raise RuntimeError(f"exact Gaussian oracle failed: {oracle['message']}")
    w_oracle = np.asarray(oracle["weights"], dtype=float)

    # Scenario LP convergence to the exact known-law problem.
    cvar_rows = []
    for count in cfg.scenario_counts:
        errs = []
        obj_gaps = []
        weight_l1 = []
        n_rep = 8 if count <= 1000 else (5 if count <= 2500 else 3)
        for rep in range(n_rep):
            sc = rng.multivariate_normal(mean_h, cov_h, size=count)
            # Use the exact budget so the only approximation here is the finite scenario set.
            fit = solve_return_cvar_lp(
                sc, mean_h, prev, beta=cfg.beta, cvar_budget=exact_budget,
                max_weight=cfg.max_weight, turnover_penalty=cfg.turnover_penalty,
            )
            w = np.asarray(fit["weights"], dtype=float)
            exact_risk = gaussian_loss_cvar(w, mean_h, cov_h, cfg.beta)
            errs.append(abs(exact_risk - float(fit["cvar"])))
            fit_obj_exact = float(mean_h @ w - cfg.turnover_penalty * np.sum(np.abs(w - prev)))
            obj_gaps.append(float(oracle["objective"]) - fit_obj_exact)
            weight_l1.append(float(np.sum(np.abs(w - w_oracle))))
        cvar_rows.append({
            "scenario_count": count,
            "mean_abs_cvar_error": float(np.mean(errs)),
            "p95_abs_cvar_error": float(np.quantile(errs, 0.95)),
            "mean_oracle_objective_gap": float(np.mean(obj_gaps)),
            "mean_weight_l1_distance_to_oracle": float(np.mean(weight_l1)),
        })
    cvar_df = pd.DataFrame(cvar_rows)
    cvar_df.to_csv(out / "exact_vs_scenario_program.csv", index=False)

    # Plug-in estimation convergence: estimate daily log-return law from GBM paths.
    est_rows = []
    max_win = max(cfg.estimation_windows)
    for window in cfg.estimation_windows:
        mu_errs, cov_errs, risk_errs = [], [], []
        for rep in range(cfg.n_estimation_replications):
            path = correlated_gbm_paths(spec, n_steps=max_win, n_paths=1, seed=cfg.seed + 1000 + 31 * rep)[0]
            lr = np.diff(np.log(path), axis=0)[-window:]
            mu_hat = lr.mean(axis=0) * cfg.horizon_days
            cov_hat = np.cov(lr, rowvar=False, ddof=1) * cfg.horizon_days
            mu_errs.append(float(np.linalg.norm(mu_hat - mean_h)))
            cov_errs.append(float(np.linalg.norm(cov_hat - cov_h, ord=2)))
            risk_hat = gaussian_loss_cvar(w_oracle, mu_hat, cov_hat, cfg.beta, centered=True)
            risk_true = gaussian_loss_cvar(w_oracle, mean_h, cov_h, cfg.beta, centered=True)
            risk_errs.append(abs(risk_hat - risk_true))
        est_rows.append({
            "window_days": window,
            "mean_error_mu_l2": float(np.mean(mu_errs)),
            "mean_error_cov_op": float(np.mean(cov_errs)),
            "mean_abs_centered_cvar_error": float(np.mean(risk_errs)),
            "scaled_mu_error_sqrtT": float(np.mean(mu_errs) * np.sqrt(window)),
            "scaled_cov_error_sqrtT": float(np.mean(cov_errs) * np.sqrt(window)),
        })
    est_df = pd.DataFrame(est_rows)
    est_df.to_csv(out / "plugin_estimation_convergence.csv", index=False)

    # Correlation comparative statics for a long-only constant-mix book.
    w = np.ones(n) / n
    corr_rows = []
    rho_grid = np.linspace(0.0, 0.75, 31)
    for rho in rho_grid:
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(spec.sigma, spec.sigma) * corr * dt
        corr_rows.append({
            "rho": float(rho),
            "portfolio_std": gaussian_portfolio_moments(w, np.zeros(n), cov)[1],
            "centered_cvar": gaussian_loss_cvar(w, np.zeros(n), cov, cfg.beta, centered=True),
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out / "correlation_cvar_sensitivity.csv", index=False)

    # Analytical derivative checked against a symmetric finite difference for one pair.
    i, j = 0, 1
    h = 1e-5
    base_corr = spec.corr.copy()
    plus = base_corr.copy(); plus[i, j] += h; plus[j, i] += h
    minus = base_corr.copy(); minus[i, j] -= h; minus[j, i] -= h
    cov_plus = np.outer(spec.sigma, spec.sigma) * plus * dt
    cov_minus = np.outer(spec.sigma, spec.sigma) * minus * dt
    fd = (gaussian_loss_cvar(w, np.zeros(n), cov_plus, cfg.beta, centered=True) -
          gaussian_loss_cvar(w, np.zeros(n), cov_minus, cfg.beta, centered=True)) / (2 * h)
    analytic = correlation_cvar_derivative(w, spec.sigma, base_corr, i, j, beta=cfg.beta, horizon_years=dt)

    # Centering identification across arbitrary mean shifts.
    shifts = np.linspace(-0.04, 0.04, 9)
    centered = [gaussian_loss_cvar(w_oracle, mean_h + s, cov_h, cfg.beta, centered=True) for s in shifts]
    uncentered = [gaussian_loss_cvar(w_oracle, mean_h + s, cov_h, cfg.beta, centered=False) for s in shifts]
    center_df = pd.DataFrame({"common_mean_shift": shifts, "centered_cvar": centered, "uncentered_cvar": uncentered})
    center_df.to_csv(out / "centering_identification.csv", index=False)

    _plot_results(out, cvar_df, est_df, corr_df)

    summary = {
        "label": "known_distribution_sequential_diffusion_analysis",
        "oracle_success": bool(oracle["success"]),
        "oracle_exact_cvar": float(oracle["cvar"]),
        "oracle_budget": float(exact_budget),
        "oracle_objective": float(oracle["objective"]),
        "largest_scenario_count": int(cvar_df.iloc[-1]["scenario_count"]),
        "largest_scenario_mean_abs_cvar_error": float(cvar_df.iloc[-1]["mean_abs_cvar_error"]),
        "largest_scenario_mean_weight_l1_distance_to_oracle": float(cvar_df.iloc[-1]["mean_weight_l1_distance_to_oracle"]),
        "correlation_derivative_analytic": float(analytic),
        "correlation_derivative_finite_difference": float(fd),
        "correlation_derivative_abs_error": float(abs(analytic - fd)),
        "centered_cvar_max_shift_difference": float(np.max(centered) - np.min(centered)),
        "uncentered_cvar_shift_slope": float(np.polyfit(shifts, uncentered, 1)[0]),
        "estimation_windows": list(cfg.estimation_windows),
    }
    import json
    (out / "sequential_diffusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
