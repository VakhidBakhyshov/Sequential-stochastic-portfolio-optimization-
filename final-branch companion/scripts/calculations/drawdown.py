"""Drawdown and Conditional Drawdown-at-Risk (CDaR) utilities.

The optimization-facing definition follows Chekhlov, Uryasev and Zabarankin:
portfolio path state is represented in an *additive* return coordinate so that
running peaks and drawdowns are piecewise-linear in portfolio weights. For log-return
inputs this is cumulative log wealth exactly; for simple-return inputs it is the
uncompounded additive path used by the linear CDaR formulation.  This is
what makes the CDaR constrained allocation problem an exact linear program.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def cumulative_uncompounded(returns: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    return np.cumsum(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), axis=axis)


def drawdown_curve_from_returns(returns: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return non-negative drawdown depths from the initial wealth peak (0).

    ``returns`` may be a single path ``(H,)`` or a batch such as ``(S,H)``.
    The initial cumulative return 0 is treated as an admissible peak, which is
    important when a path begins with losses.
    """
    cum = cumulative_uncompounded(returns, axis=axis)
    peak = np.maximum.accumulate(cum, axis=axis)
    peak = np.maximum(peak, 0.0)
    return np.maximum(peak - cum, 0.0)


def empirical_tail_mean(values: np.ndarray, alpha: float = 0.95) -> float:
    """Finite-sample CVaR/upper-tail mean with exact fractional tail mass.

    This avoids the common ``ceil((1-alpha)*n)`` discontinuity and matches the
    Rockafellar-Uryasev sample-average definition for equal-probability atoms.
    """
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    alpha = float(np.clip(alpha, 0.0, 1.0 - EPS))
    if alpha <= 0.0:
        return float(np.mean(x))
    mass = (1.0 - alpha) * x.size
    if mass <= EPS:
        return float(np.max(x))
    sx = np.sort(x)[::-1]  # worst/highest values first
    k = int(np.floor(mass))
    frac = float(mass - k)
    total = float(np.sum(sx[:k])) if k > 0 else 0.0
    if frac > EPS and k < sx.size:
        total += frac * float(sx[k])
    return total / mass


def empirical_cdar(path_returns: np.ndarray, alpha: float = 0.95) -> float:
    """CDaR as the upper-tail mean of drawdown observations.

    For input shape ``(S,H)`` all scenario-time drawdowns are pooled with equal
    mass, which is the sample-average stochastic-program analogue used by the
    CDaR LP in :mod:`scripts.optimizers.cdar`.
    """
    dd = drawdown_curve_from_returns(path_returns, axis=-1)
    return empirical_tail_mean(dd, alpha=alpha)


def max_drawdown_uncompounded(path_returns: np.ndarray) -> float:
    dd = drawdown_curve_from_returns(path_returns, axis=-1)
    return float(np.nanmax(dd)) if dd.size else float("nan")


def average_drawdown(path_returns: np.ndarray) -> float:
    dd = drawdown_curve_from_returns(path_returns, axis=-1)
    return float(np.nanmean(dd)) if dd.size else float("nan")


def portfolio_path_returns(asset_paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Map an ``(S,H,N)`` scenario-path cube to ``(S,H)`` portfolio returns."""
    paths = np.asarray(asset_paths, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if paths.ndim != 3 or paths.shape[2] != w.size:
        raise ValueError(f"asset_paths must be (S,H,N) with N={w.size}; got {paths.shape}")
    return np.einsum("shn,n->sh", paths, w)
