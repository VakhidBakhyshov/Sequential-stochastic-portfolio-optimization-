"""Closed-form BM/GBM benchmarks used to verify the paper's structural results.

The empirical strategy is not assumed to follow a diffusion.  The purpose of this
module is different: under a data-generating process whose distribution is known,
we can calculate the objects used by the stochastic program exactly and then test
that the code converges to those objects.

Conventions
-----------
* ``mu`` and ``sigma`` are annualized continuously-compounded parameters.
* ``corr`` is the instantaneous Brownian correlation matrix.
* ``dt`` is measured in years (one trading day is normally 1/252).
* CVaR is reported for a *loss* variable, so larger positive values are worse.
* For GBM, the exact simple-return loss is ``L = 1 - exp(Y)`` where ``Y`` is the
  portfolio log return of a continuously rebalanced constant-mix portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm

EPS = 1e-12


@dataclass(frozen=True)
class DiffusionSpec:
    """Parameters of a correlated multi-asset geometric Brownian motion.

    Parameters
    ----------
    mu:
        Annualized vector of arithmetic price drifts in ``dS_i/S_i``.
    sigma:
        Annualized vector of instantaneous volatilities.
    corr:
        Positive-semidefinite correlation matrix of Brownian innovations.
    s0:
        Initial prices.  A scalar is broadcast to all assets.
    trading_days:
        Trading days per year used when converting day counts to year fractions.
    """

    mu: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray
    s0: np.ndarray | float = 100.0
    trading_days: int = 252

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float).reshape(-1)
        sigma = np.asarray(self.sigma, dtype=float).reshape(-1)
        corr = np.asarray(self.corr, dtype=float)
        if mu.size != sigma.size:
            raise ValueError("mu and sigma must have the same length")
        if corr.shape != (mu.size, mu.size):
            raise ValueError("corr must be N x N")
        if np.any(sigma <= 0):
            raise ValueError("all sigma values must be positive")
        if not np.allclose(corr, corr.T, atol=1e-10):
            raise ValueError("corr must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("corr must have ones on the diagonal")
        if np.min(np.linalg.eigvalsh(corr)) < -1e-8:
            raise ValueError("corr must be positive semidefinite")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "corr", corr)
        s0 = np.asarray(self.s0, dtype=float)
        if s0.ndim == 0:
            s0 = np.repeat(float(s0), mu.size)
        if s0.shape != mu.shape or np.any(s0 <= 0):
            raise ValueError("s0 must be positive and broadcastable to N assets")
        object.__setattr__(self, "s0", s0)

    @property
    def n_assets(self) -> int:
        return int(self.mu.size)

    @property
    def covariance(self) -> np.ndarray:
        """Annualized instantaneous covariance matrix ``diag(sigma) corr diag(sigma)``."""
        return np.outer(self.sigma, self.sigma) * self.corr

    @property
    def log_drift(self) -> np.ndarray:
        """Annualized drift of log prices: ``mu - 0.5 sigma^2``."""
        return self.mu - 0.5 * self.sigma**2


@dataclass(frozen=True)
class ArithmeticBrownianSpec:
    """Parameters of a correlated arithmetic Brownian *price* process.

    The benchmark is intentionally diagnostic: arithmetic Brownian prices are
    Gaussian and therefore can become non-positive.  That defect is useful in
    the paper because it gives an exact probability with which to falsify BM as
    a globally admissible price model, while retaining a fully known law.

    ``dS = drift dt + diag(abs_sigma) dW`` with ``dW_i dW_j = corr_ij dt``.
    ``abs_sigma`` is measured in price units per square-root year.
    """

    drift: np.ndarray
    abs_sigma: np.ndarray
    corr: np.ndarray
    s0: np.ndarray | float = 100.0
    trading_days: int = 252

    def __post_init__(self) -> None:
        drift = np.asarray(self.drift, dtype=float).reshape(-1)
        sigma = np.asarray(self.abs_sigma, dtype=float).reshape(-1)
        corr = np.asarray(self.corr, dtype=float)
        if drift.size != sigma.size:
            raise ValueError("drift and abs_sigma must have the same length")
        if corr.shape != (drift.size, drift.size):
            raise ValueError("corr must be N x N")
        if np.any(sigma <= 0):
            raise ValueError("all abs_sigma values must be positive")
        if not np.allclose(corr, corr.T, atol=1e-10):
            raise ValueError("corr must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("corr must have ones on the diagonal")
        if np.min(np.linalg.eigvalsh(corr)) < -1e-8:
            raise ValueError("corr must be positive semidefinite")
        s0 = np.asarray(self.s0, dtype=float)
        if s0.ndim == 0:
            s0 = np.repeat(float(s0), drift.size)
        if s0.shape != drift.shape or np.any(s0 <= 0):
            raise ValueError("s0 must be positive and broadcastable to N assets")
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "abs_sigma", sigma)
        object.__setattr__(self, "corr", corr)
        object.__setattr__(self, "s0", s0)

    @property
    def n_assets(self) -> int:
        return int(self.drift.size)

    @property
    def covariance(self) -> np.ndarray:
        return np.outer(self.abs_sigma, self.abs_sigma) * self.corr


def correlated_bm_price_paths(
    spec: ArithmeticBrownianSpec,
    *,
    n_steps: int,
    dt: float | None = None,
    n_paths: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Simulate exact correlated arithmetic-Brownian price paths."""
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be positive")
    if dt is None:
        dt = 1.0 / spec.trading_days
    if dt <= 0:
        raise ValueError("dt must be positive")
    rng = np.random.default_rng(seed)
    corr_factor = _psd_factor(spec.corr)
    z = rng.standard_normal((n_paths, n_steps, spec.n_assets)) @ corr_factor.T
    inc = spec.drift[None, None, :] * dt + spec.abs_sigma[None, None, :] * np.sqrt(dt) * z
    out = np.empty((n_paths, n_steps + 1, spec.n_assets), dtype=float)
    out[:, 0, :] = spec.s0
    out[:, 1:, :] = spec.s0[None, None, :] + np.cumsum(inc, axis=1)
    return out


def bm_terminal_price_moments(spec: ArithmeticBrownianSpec, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean vector and covariance of terminal arithmetic-BM prices."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    h = horizon_days / spec.trading_days
    return spec.s0 + spec.drift * h, spec.covariance * h


def bm_nonpositive_price_probability(spec: ArithmeticBrownianSpec, horizon_days: int) -> np.ndarray:
    """Exact marginal probability ``P[S_i(T) <= 0]`` under arithmetic BM."""
    mean, cov = bm_terminal_price_moments(spec, horizon_days)
    sd = np.sqrt(np.clip(np.diag(cov), EPS, None))
    return norm.cdf(-mean / sd)


def bm_linear_portfolio_price_moments(
    holdings: Iterable[float], spec: ArithmeticBrownianSpec, horizon_days: int
) -> tuple[float, float]:
    """Exact moments of a fixed-unit linear portfolio ``h' S_T`` under BM."""
    hvec = np.asarray(list(holdings), dtype=float).reshape(-1)
    if hvec.size != spec.n_assets:
        raise ValueError("holdings length must equal n_assets")
    mean, cov = bm_terminal_price_moments(spec, horizon_days)
    return float(hvec @ mean), float(max(hvec @ cov @ hvec, 0.0))


def _psd_factor(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    vals, vecs = np.linalg.eigh((m + m.T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals))


def correlated_gbm_paths(
    spec: DiffusionSpec,
    *,
    n_steps: int,
    dt: float | None = None,
    n_paths: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Simulate exact-discretization correlated GBM price paths.

    Returns
    -------
    ndarray
        Shape ``(n_paths, n_steps + 1, n_assets)``.
    """
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be positive")
    if dt is None:
        dt = 1.0 / spec.trading_days
    if dt <= 0:
        raise ValueError("dt must be positive")

    rng = np.random.default_rng(seed)
    corr_factor = _psd_factor(spec.corr)
    z = rng.standard_normal((n_paths, n_steps, spec.n_assets))
    corr_z = z @ corr_factor.T
    log_inc = spec.log_drift[None, None, :] * dt + spec.sigma[None, None, :] * np.sqrt(dt) * corr_z
    log_paths = np.cumsum(log_inc, axis=1)
    out = np.empty((n_paths, n_steps + 1, spec.n_assets), dtype=float)
    out[:, 0, :] = spec.s0
    out[:, 1:, :] = spec.s0[None, None, :] * np.exp(log_paths)
    return out


def gbm_log_return_moments(spec: DiffusionSpec, horizon_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean vector and covariance matrix of asset log returns over a horizon."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    h = horizon_days / spec.trading_days
    return spec.log_drift * h, spec.covariance * h


def portfolio_log_return_moments(
    weights: Iterable[float],
    spec: DiffusionSpec,
    horizon_days: int,
) -> tuple[float, float]:
    """Exact log-return moments for a continuously rebalanced constant-mix portfolio.

    If ``dV/V = w' mu dt + w' diag(sigma) dW``, then
    ``log(V_T/V_0)`` is normal with mean
    ``(w'mu - 0.5 w'Cov w) T`` and variance ``w'Cov w T``.
    """
    w = np.asarray(list(weights), dtype=float).reshape(-1)
    if w.size != spec.n_assets:
        raise ValueError("weights length must equal n_assets")
    h = horizon_days / spec.trading_days
    variance = float(w @ spec.covariance @ w * h)
    mean = float((w @ spec.mu - 0.5 * (w @ spec.covariance @ w)) * h)
    return mean, max(variance, 0.0)


def normal_loss_var_cvar(mean_return: float, std_return: float, beta: float = 0.95) -> tuple[float, float]:
    """Closed-form VaR and CVaR for loss ``L=-R`` when ``R`` is normal.

    ``VaR_beta(L) = -m + s z_beta`` and
    ``CVaR_beta(L) = -m + s phi(z_beta)/(1-beta)``.
    """
    if not 0 < beta < 1:
        raise ValueError("beta must lie in (0, 1)")
    s = max(float(std_return), 0.0)
    m = float(mean_return)
    z = float(norm.ppf(beta))
    var = -m + s * z
    cvar = -m + s * float(norm.pdf(z)) / (1.0 - beta)
    return float(var), float(cvar)


def centered_normal_cvar(std_return: float, beta: float = 0.95) -> float:
    """CVaR of the centered normal loss ``-(R-E[R])``."""
    return normal_loss_var_cvar(0.0, std_return, beta)[1]


def gbm_simple_loss_var_cvar(log_mean: float, log_std: float, beta: float = 0.95) -> tuple[float, float]:
    """Exact VaR/CVaR for simple-return loss of a lognormal wealth ratio.

    Let ``Y ~ N(m,s^2)`` be the portfolio log return and ``L = 1-exp(Y)`` be the
    simple-return loss.  The worst ``1-beta`` outcomes correspond to the lower
    ``1-beta`` tail of ``Y``.  Therefore

    ``VaR_beta(L) = 1 - exp(m + s z_{1-beta})``

    and

    ``CVaR_beta(L) = 1 - exp(m+s^2/2) Phi(z_{1-beta}-s)/(1-beta)``.
    """
    if not 0 < beta < 1:
        raise ValueError("beta must lie in (0, 1)")
    m = float(log_mean)
    s = max(float(log_std), 0.0)
    q = 1.0 - beta
    zq = float(norm.ppf(q))
    var = 1.0 - np.exp(m + s * zq)
    if s <= EPS:
        cvar = 1.0 - np.exp(m)
    else:
        truncated_mean = np.exp(m + 0.5 * s * s) * norm.cdf(zq - s) / q
        cvar = 1.0 - truncated_mean
    return float(var), float(cvar)


def empirical_cvar_loss(returns: np.ndarray, beta: float = 0.95) -> float:
    """Sample-average CVaR of loss ``-returns`` at confidence ``beta``.

    The implementation matches the Rockafellar--Uryasev epigraph used by the
    benchmark LP.  A fractional boundary observation is used when
    ``(1-beta) * n`` is non-integer.  This also avoids the floating-point
    off-by-one that arises from ``ceil((1-beta)*n)`` at values such as
    ``beta=0.95, n=1000``.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0,1)")
    losses = np.sort(-r)[::-1]
    tail_mass = (1.0 - float(beta)) * losses.size
    # Numerical snapping makes exact intended masses (e.g. 50 for 1000 at
    # 95%) exact rather than 50.00000000000004.
    nearest = round(tail_mass)
    if abs(tail_mass - nearest) <= 1e-10 * max(1.0, abs(tail_mass)):
        tail_mass = float(nearest)
    k = int(np.floor(tail_mass))
    frac = float(tail_mass - k)
    if k <= 0:
        # With a finite empirical distribution and a very deep beta, CVaR is
        # the maximum observed loss.
        return float(losses[0])
    total = float(np.sum(losses[:k]))
    if frac > 0.0 and k < losses.size:
        total += frac * float(losses[k])
    return total / tail_mass


def topk_margin_certificate(scores: np.ndarray, k: int, epsilon: float) -> dict[str, float | bool]:
    """Deterministic top-K stability certificate ``gap > 2 epsilon``."""
    s = np.asarray(scores, dtype=float).reshape(-1)
    if not 1 <= k < s.size:
        raise ValueError("k must satisfy 1 <= k < number of scores")
    order = np.sort(s)[::-1]
    gap = float(order[k - 1] - order[k])
    return {"cutoff_margin": gap, "epsilon": float(epsilon), "stable": bool(gap > 2.0 * epsilon)}


def gaussian_reversal_probability(
    score_i: float,
    score_j: float,
    noise_sd_i: float,
    noise_sd_j: float,
    *,
    noise_corr: float = 0.0,
) -> float:
    """Exact pairwise rank-reversal probability under Gaussian score noise.

    Assumes ``score_i > score_j``.  If estimation errors have correlation ``rho``,
    the noise in ``(j-i)`` has variance
    ``sd_i^2 + sd_j^2 - 2 rho sd_i sd_j``.
    """
    gap = float(score_i - score_j)
    if gap <= 0:
        raise ValueError("score_i must exceed score_j")
    var = noise_sd_i**2 + noise_sd_j**2 - 2.0 * noise_corr * noise_sd_i * noise_sd_j
    if var <= EPS:
        return 0.0
    return float(norm.cdf(-gap / np.sqrt(var)))


def ema_impulse_response(delta: float, alpha: float, horizons: Iterable[int]) -> np.ndarray:
    """Exact effect of a one-date score perturbation after EMA smoothing."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    h = np.asarray(list(horizons), dtype=int)
    if np.any(h < 0):
        raise ValueError("horizons must be nonnegative")
    return float(delta) * alpha * np.power(1.0 - alpha, h)


def partial_execution(previous: np.ndarray, target: np.ndarray, eta: float) -> tuple[np.ndarray, dict[str, float]]:
    """Execute a fraction ``eta`` of the move and return exact turnover identities."""
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie in [0,1]")
    prev = np.asarray(previous, dtype=float).reshape(-1)
    tar = np.asarray(target, dtype=float).reshape(-1)
    if prev.shape != tar.shape:
        raise ValueError("previous and target must have the same shape")
    exe = prev + float(eta) * (tar - prev)
    target_distance = float(np.sum(np.abs(tar - prev)))
    executed_turnover = float(np.sum(np.abs(exe - prev)))
    residual_distance = float(np.sum(np.abs(tar - exe)))
    return exe, {
        "target_distance_l1": target_distance,
        "executed_turnover_l1": executed_turnover,
        "residual_distance_l1": residual_distance,
        "turnover_identity_error": executed_turnover - float(eta) * target_distance,
        "contraction_identity_error": residual_distance - (1.0 - float(eta)) * target_distance,
    }


def solve_return_cvar_lp(
    scenario_returns: np.ndarray,
    expected_returns: np.ndarray,
    previous_weights: np.ndarray,
    *,
    beta: float = 0.95,
    cvar_budget: float | None = None,
    max_weight: float = 0.20,
    turnover_penalty: float = 0.002,
) -> dict[str, np.ndarray | float | bool | str]:
    """Solve the paper's long-only expected-return/CVaR/turnover linear program.

    This small transparent solver is used only by the known-DGP benchmark.  It is
    mathematically the same Rockafellar-Uryasev epigraph representation as the
    production optimizer, but it avoids any empirical-pipeline side effects.
    """
    R = np.asarray(scenario_returns, dtype=float)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    w_prev = np.asarray(previous_weights, dtype=float).reshape(-1)
    if R.ndim != 2 or R.shape[1] != mu.size or mu.size != w_prev.size:
        raise ValueError("scenario_returns, expected_returns and previous_weights have incompatible shapes")
    if not 0 < beta < 1:
        raise ValueError("beta must lie in (0,1)")
    n_s, n = R.shape
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError("max_weight makes the fully-invested long-only set infeasible")

    # Equal-weight reference makes the feasible set nonempty when budget is omitted.
    if cvar_budget is None:
        ew = np.ones(n) / n
        cvar_budget = empirical_cvar_loss(R @ ew, beta=beta)

    # Variables: [w_n, zeta, u_S, p_n, n_n].
    i_w = slice(0, n)
    i_zeta = n
    i_u = slice(n + 1, n + 1 + n_s)
    i_p = slice(n + 1 + n_s, n + 1 + n_s + n)
    i_n = slice(n + 1 + n_s + n, n + 1 + n_s + 2 * n)
    m = n + 1 + n_s + 2 * n

    c = np.zeros(m)
    c[i_w] = -mu
    c[i_p] = turnover_penalty
    c[i_n] = turnover_penalty

    # CVaR budget row.
    A_ub = []
    b_ub = []
    row = np.zeros(m)
    row[i_zeta] = 1.0
    row[i_u] = 1.0 / ((1.0 - beta) * n_s)
    A_ub.append(row)
    b_ub.append(float(cvar_budget))

    # u_s >= -R_s w - zeta  <=>  -R_s w - zeta - u_s <= 0.
    for s in range(n_s):
        row = np.zeros(m)
        row[i_w] = -R[s]
        row[i_zeta] = -1.0
        row[n + 1 + s] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # Full investment and turnover decomposition.
    A_eq = []
    b_eq = []
    row = np.zeros(m)
    row[i_w] = 1.0
    A_eq.append(row)
    b_eq.append(1.0)
    for j in range(n):
        row = np.zeros(m)
        row[j] = 1.0
        row[i_p.start + j] = -1.0
        row[i_n.start + j] = 1.0
        A_eq.append(row)
        b_eq.append(float(w_prev[j]))

    bounds = [(0.0, float(max_weight))] * n
    bounds += [(None, None)]
    bounds += [(0.0, None)] * n_s
    bounds += [(0.0, None)] * (2 * n)

    res = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray(A_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return {
            "success": False,
            "message": str(res.message),
            "weights": np.ones(n) / n,
            "objective": float("nan"),
            "cvar": float("nan"),
            "turnover": float("nan"),
            "cvar_budget": float(cvar_budget),
            "cvar_constraint_slack": float("nan"),
            "cvar_constraint_binding": False,
            "cvar_budget_dual_scipy_min": float("nan"),
            "cvar_budget_shadow_price_max": float("nan"),
        }
    w = res.x[i_w]
    # In this transparent benchmark LP the first A_ub row is the CVaR-budget
    # constraint.  HiGHS reports marginals for the *minimization* problem; the
    # economic shadow price of the manuscript's maximization objective has the
    # opposite sign.  Export both conventions so Proposition 9 can be checked
    # without sign ambiguity.
    try:
        cvar_slack = float(res.ineqlin.residual[0])
        cvar_dual_min = float(res.ineqlin.marginals[0])
        cvar_shadow_max = -cvar_dual_min
    except Exception:
        cvar_slack = cvar_dual_min = cvar_shadow_max = float("nan")
    bind_tol = max(1e-9, 1e-7 * max(1.0, abs(float(cvar_budget))))
    return {
        "success": True,
        "message": str(res.message),
        "weights": w,
        "objective": float(mu @ w - turnover_penalty * np.sum(np.abs(w - w_prev))),
        "cvar": empirical_cvar_loss(R @ w, beta=beta),
        "turnover": float(np.sum(np.abs(w - w_prev))),
        "cvar_budget": float(cvar_budget),
        "cvar_constraint_slack": cvar_slack,
        "cvar_constraint_binding": bool(np.isfinite(cvar_slack) and cvar_slack <= bind_tol),
        "cvar_budget_dual_scipy_min": cvar_dual_min,
        "cvar_budget_shadow_price_max": cvar_shadow_max,
    }
