"""Pure execution/accounting helpers used by the final publication convention.

Policy-C convention
-------------------
* ``optimizer_anchor`` is a decision/regularisation state (normally the previous target).
* ``executed_state`` is the economic holdings state from which trades, returns and costs start.
* partial execution moves the economic state towards the new target.
* proportional transaction cost is ``rate * gross_traded_notional`` where gross traded
  notional is the L1 change in risky holdings (buys + sells).

Keeping these objects separate prevents the common accounting error where a target that was
never fully held is treated as if it were the economic starting portfolio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def partial_execute(
    executed_state: pd.Series,
    target: pd.Series,
    alpha: float,
) -> pd.Series:
    """Return ``executed_state + alpha * (target - executed_state)`` on a common index."""
    a = float(np.clip(alpha, 0.0, 1.0))
    idx = executed_state.index.union(target.index)
    prev = executed_state.reindex(idx).fillna(0.0).astype(float)
    tar = target.reindex(idx).fillna(0.0).astype(float)
    return prev + a * (tar - prev)


def gross_turnover(previous: pd.Series, current: pd.Series) -> float:
    """L1 traded notional: sum of absolute buys and sells.

    For a fully invested self-financing risky sleeve this is twice the conventional
    one-way turnover ``0.5 * L1``.  The code and manuscript should therefore call this
    *gross/two-way traded notional*, not one-way turnover.
    """
    idx = previous.index.union(current.index)
    p = previous.reindex(idx).fillna(0.0).astype(float)
    c = current.reindex(idx).fillna(0.0).astype(float)
    return float(np.abs(c - p).sum())


def proportional_transaction_cost(
    previous: pd.Series,
    current: pd.Series,
    rate_per_traded_unit: float,
) -> float:
    """Cost fraction charged at ``rate_per_traded_unit`` on each unit actually traded."""
    return float(rate_per_traded_unit) * gross_turnover(previous, current)


def exposure_turnover(exposure: pd.Series) -> pd.Series:
    """Pure scalar exposure-channel turnover ``|k_t-k_{t-1}|``.

    This is useful when the exposure overlay is applied after the composition engine and its
    cost is reported separately.  It is intentionally not claimed to be an additive
    decomposition of full-vector L1 turnover when composition and exposure change together.
    """
    k = pd.Series(exposure, dtype=float).clip(lower=0.0)
    return k.diff().abs().fillna(0.0)
