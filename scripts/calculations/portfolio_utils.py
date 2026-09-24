from __future__ import annotations

import numpy as np

EPS = 1e-12


def as_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={x.shape}")
    return np.where(np.isfinite(x), x, np.nan)


def sigmoid_np(x: float) -> float:
    """Small replacement for jax.nn.sigmoid, avoids importing JAX for one scalar."""
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def normalize_long_only(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float | None = None,
    fallback_n: int | None = None,
) -> np.ndarray:
    """
    Long-only postprocessor:
    - removes non-finite and negative weights;
    - sets tiny weights to zero;
    - renormalizes to sum to 1;
    - caps max weights and redistributes the excess.

    Important: min_weight is used as a position threshold, not as a hard lower bound
    for all assets. Otherwise min_weight=0.01 is infeasible when N > 100.
    """
    w = np.asarray(weights, dtype=float).copy()

    if w.size == 0:
        raise ValueError("weights must not be empty")

    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)

    if min_weight > 0:
        w[w < min_weight] = 0.0

    if w.sum() <= EPS:
        w = np.ones(w.size, dtype=float) / w.size
    else:
        w = w / w.sum()

    if max_weight is None:
        return w

    max_weight = float(max_weight)
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")

    active = w > 0
    active_count = int(active.sum())
    if active_count == 0:
        return np.ones(w.size, dtype=float) / w.size

    max_eff = max(max_weight, 1.0 / active_count)

    capped = np.zeros_like(w, dtype=bool)
    for _ in range(w.size + 2):
        over = (w > max_eff) & (~capped)
        if not np.any(over):
            break

        capped[over] = True
        w[over] = max_eff

        remainder = 1.0 - w[capped].sum()
        uncapped = ~capped
        if remainder <= EPS or not np.any(uncapped):
            break

        uncapped_sum = w[uncapped].sum()
        if uncapped_sum <= EPS:
            w[uncapped] = remainder / uncapped.sum()
        else:
            w[uncapped] = w[uncapped] / uncapped_sum * remainder

    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= EPS:
        return np.ones(w.size, dtype=float) / w.size

    return w / total


def tail_cvar_loss(
    scenario_returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.95,
    min_tail_count: int = 5,
) -> float:
    """
    Positive loss CVaR objective. Works even with few scenarios by enforcing
    at least min_tail_count observations in the loss tail.
    """
    r = np.asarray(scenario_returns, dtype=float)
    w = np.asarray(weights, dtype=float)

    portfolio_returns = r @ w
    losses = -portfolio_returns
    losses = losses[np.isfinite(losses)]

    if losses.size == 0:
        return 1e6

    tail_count = max(int(np.ceil((1.0 - alpha) * losses.size)), int(min_tail_count))
    tail_count = min(tail_count, losses.size)

    tail = np.partition(losses, -tail_count)[-tail_count:]
    return float(np.mean(tail))
