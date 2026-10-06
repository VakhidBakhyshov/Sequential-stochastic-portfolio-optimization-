"""Known-law and controlled-synthetic diagnostics for Conditional Drawdown-at-Risk."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

from scripts.calculations.drawdown import (
    average_drawdown, drawdown_curve_from_returns, empirical_cdar, empirical_tail_mean,
    portfolio_path_returns,
)


@dataclass(frozen=True)
class CDaRBrownianSpec:
    sigma: float = 0.20
    horizon_years: float = 1.0
    steps: int = 252
    paths: int = 20000
    seed: int = 20260911


def zero_drift_brownian_paths(spec: CDaRBrownianSpec) -> np.ndarray:
    """Simulate arithmetic-return increments dX=sigma*dW."""
    rng = np.random.default_rng(spec.seed)
    dt = spec.horizon_years / spec.steps
    return spec.sigma * np.sqrt(dt) * rng.standard_normal((spec.paths, spec.steps))


def exact_expected_time_average_drawdown_zero_drift_bm(sigma: float, horizon_years: float) -> float:
    """E[T^-1 ∫_0^T (M_t-X_t)dt] for X_t=sigma W_t.

    By Lévy's identity, M_t-W_t has the same marginal law as |W_t|, hence
    E[M_t-W_t]=sqrt(2t/pi). Integrating over t gives the expression below.
    """
    T = float(horizon_years)
    return (2.0 / 3.0) * float(sigma) * np.sqrt(2.0 * T / np.pi)


def continuous_monitoring_grid_average_oracle(sigma: float, horizon_years: float, steps: int) -> float:
    """Average continuous-monitoring drawdown marginal means at discrete observation times.

    This is *not* the expectation of a discretely monitored running maximum; the latter
    is smaller because between-grid peaks are unavailable.  The function is an oracle
    for the continuous process sampled at the grid times only.
    """
    t = np.arange(1, int(steps) + 1, dtype=float) * float(horizon_years) / int(steps)
    return float(sigma) * np.sqrt(2.0 / np.pi) * float(np.mean(np.sqrt(t)))

def brownian_average_drawdown_check(spec: CDaRBrownianSpec) -> dict[str, float]:
    inc = zero_drift_brownian_paths(spec)
    dd = drawdown_curve_from_returns(inc, axis=1)
    # Discrete time-average; omit no points and compare with continuous-time oracle.
    mc = float(dd.mean())
    continuous = exact_expected_time_average_drawdown_zero_drift_bm(spec.sigma, spec.horizon_years)
    return {"mc_average_drawdown": mc, "continuous_exact": continuous, "absolute_error": abs(mc - continuous)}


def cdar_alpha_curve(path_returns: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    return np.asarray([empirical_cdar(path_returns, float(a)) for a in alphas], dtype=float)


def homogeneity_check(path_returns: np.ndarray, alpha: float = 0.95, scales=(0.25, 0.5, 1.0, 1.5)) -> float:
    base = empirical_cdar(path_returns, alpha)
    errs = [abs(empirical_cdar(float(k) * path_returns, alpha) - float(k) * base) for k in scales]
    return float(max(errs))


def convexity_check(asset_paths: np.ndarray, w1: np.ndarray, w2: np.ndarray, theta: float = 0.37, alpha: float = 0.95) -> float:
    wm = theta * w1 + (1.0 - theta) * w2
    lhs = empirical_cdar(portfolio_path_returns(asset_paths, wm), alpha)
    rhs = theta * empirical_cdar(portfolio_path_returns(asset_paths, w1), alpha) + (1.0 - theta) * empirical_cdar(portfolio_path_returns(asset_paths, w2), alpha)
    return float(lhs - rhs)  # should be <= 0 up to numerical noise


def path_order_counterexample() -> dict[str, float]:
    """Two paths with identical terminal return but different CDaR.

    This is the core identification difference from terminal-loss CVaR.
    """
    a = np.array([0.10, -0.20, 0.10], dtype=float)  # terminal 0, deep interim drawdown
    b = np.array([-0.02, 0.01, 0.01], dtype=float)  # terminal 0, shallow drawdown
    return {
        "terminal_a": float(a.sum()), "terminal_b": float(b.sum()),
        "cdar95_a": empirical_cdar(a, 0.95), "cdar95_b": empirical_cdar(b, 0.95),
        "avgdd_a": average_drawdown(a), "avgdd_b": average_drawdown(b),
    }



def exact_terminal_drawdown_quantile_zero_drift_bm(
    sigma: float, horizon_years: float, alpha: float
) -> float:
    """VaR_alpha of terminal drawdown D_T=M_T-X_T for X_t=sigma W_t.

    Lévy's identity gives D_T =d |sigma W_T|, so the drawdown is half-normal
    with scale sigma*sqrt(T).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0 - 1e-12))
    scale = float(sigma) * np.sqrt(float(horizon_years))
    z = float(norm.ppf((1.0 + alpha) / 2.0))
    return scale * z


def exact_terminal_drawdown_cdar_zero_drift_bm(
    sigma: float, horizon_years: float, alpha: float
) -> float:
    """Upper-tail mean of the zero-drift Brownian terminal drawdown.

    If D_T/(sigma*sqrt(T)) is |Z|, Z~N(0,1), then for
    z_alpha=Phi^{-1}((1+alpha)/2),

        E[D_T | D_T >= VaR_alpha(D_T)]
        = sigma*sqrt(T) * phi(z_alpha)/(1-Phi(z_alpha)).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0 - 1e-12))
    scale = float(sigma) * np.sqrt(float(horizon_years))
    z = float(norm.ppf((1.0 + alpha) / 2.0))
    tail = float(1.0 - norm.cdf(z))
    return scale * float(norm.pdf(z)) / max(tail, 1e-15)


def terminal_drawdown_cdar_mc_check(
    spec: CDaRBrownianSpec, alpha: float = 0.95
) -> dict[str, float]:
    """Monte-Carlo check of the exact half-normal terminal-drawdown tail law."""
    inc = zero_drift_brownian_paths(spec)
    terminal_dd = drawdown_curve_from_returns(inc, axis=1)[:, -1]
    mc = float(empirical_tail_mean(terminal_dd, alpha=alpha))
    exact = exact_terminal_drawdown_cdar_zero_drift_bm(spec.sigma, spec.horizon_years, alpha)
    q_mc = float(np.quantile(terminal_dd, alpha))
    q_exact = exact_terminal_drawdown_quantile_zero_drift_bm(spec.sigma, spec.horizon_years, alpha)
    return {
        "alpha": float(alpha),
        "mc_terminal_cdar": mc,
        "exact_terminal_cdar": exact,
        "absolute_cdar_error": abs(mc - exact),
        "mc_terminal_var": q_mc,
        "exact_terminal_var": q_exact,
        "absolute_var_error": abs(q_mc - q_exact),
    }


def risk_geometry_ranking_reversal(alpha: float = 0.95) -> dict[str, float]:
    """Deterministic paths on which terminal-loss CVaR and CDaR rank oppositely."""
    # A recovers fully after a deep underwater episode. B loses modestly and smoothly.
    a = np.array([0.10, -0.20, 0.10], dtype=float)
    b = np.array([-0.05, 0.00, 0.00], dtype=float)
    terminal_loss_a = -float(a.sum())
    terminal_loss_b = -float(b.sum())
    cdar_a = empirical_cdar(a, alpha)
    cdar_b = empirical_cdar(b, alpha)
    return {
        "terminal_loss_a": terminal_loss_a,
        "terminal_loss_b": terminal_loss_b,
        "cdar_a": float(cdar_a),
        "cdar_b": float(cdar_b),
        "terminal_prefers_a": float(terminal_loss_a < terminal_loss_b),
        "cdar_prefers_b": float(cdar_b < cdar_a),
    }
