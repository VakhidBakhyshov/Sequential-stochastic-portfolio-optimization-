"""Regression checks for the final execution / transaction-cost convention.

Run from repository root:
    pytest -q tests/test_execution_convention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.executions.accounting import (
    exposure_turnover,
    gross_turnover,
    partial_execute,
    proportional_transaction_cost,
)


def test_partial_execution_exact_identities():
    prev = pd.Series([0.50, 0.30, 0.20], index=list("ABC"))
    target = pd.Series([0.20, 0.50, 0.30], index=list("ABC"))
    alpha = 0.40

    executed = partial_execute(prev, target, alpha)

    before = float(np.abs(target - prev).sum())
    after = float(np.abs(target - executed).sum())
    traded = gross_turnover(prev, executed)

    assert np.isclose(after, (1.0 - alpha) * before, atol=1e-15)
    assert np.isclose(traded, alpha * before, atol=1e-15)


def test_policy_c_anchor_is_not_economic_starting_state():
    previous_target = pd.Series([0.70, 0.30, 0.00], index=list("ABC"))
    previous_executed = pd.Series([0.50, 0.40, 0.10], index=list("ABC"))
    new_target = pd.Series([0.20, 0.50, 0.30], index=list("ABC"))
    alpha = 0.50

    correct = partial_execute(previous_executed, new_target, alpha)
    wrong = partial_execute(previous_target, new_target, alpha)

    assert not np.allclose(correct.values, wrong.values)
    assert np.isclose(gross_turnover(previous_executed, correct),
                      alpha * gross_turnover(previous_executed, new_target))


def test_cost_is_10bp_per_unit_actually_traded():
    prev = pd.Series([0.60, 0.40], index=["A", "B"])
    cur = pd.Series([0.40, 0.60], index=["A", "B"])
    rate = 0.001  # 10 bp on each unit bought/sold

    # 0.20 sold + 0.20 bought = 0.40 gross traded notional.
    assert np.isclose(gross_turnover(prev, cur), 0.40)
    assert np.isclose(proportional_transaction_cost(prev, cur, rate), 0.0004)
    # Conventional one-way turnover is 0.5 * L1 = 0.20.  Therefore the same cost
    # corresponds to 20 bp per unit of conventional one-way turnover.
    one_way = 0.5 * gross_turnover(prev, cur)
    assert np.isclose(0.0004 / one_way, 0.002)


def test_required_exit_is_traded_and_costed():
    prev = pd.Series({"A": 0.50, "B": 0.30, "EXIT": 0.20})
    target = pd.Series({"A": 0.55, "B": 0.45, "EXIT": 0.00})
    executed = partial_execute(prev, target, 0.50)

    assert np.isclose(executed["EXIT"], 0.10)
    assert gross_turnover(prev, executed) > 0.0
    assert proportional_transaction_cost(prev, executed, 0.001) > 0.0


def test_exposure_turnover_is_separate_scalar_channel():
    k = pd.Series([1.0, 0.8, 0.8, 0.5])
    got = exposure_turnover(k)
    assert np.allclose(got.values, [0.0, 0.2, 0.0, 0.3])
