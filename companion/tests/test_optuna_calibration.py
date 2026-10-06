from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.optimization.analysis import (
    add_policy_scores,
    build_parameter_range_summary,
    choose_policy_trials,
    extract_pareto_front,
)
from scripts.optimization.optuna_calibration import set_dotted_value


def _fake_trials(n: int = 30) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({
        "trial_number": np.arange(n),
        "status": "COMPLETE",
        "param_optimizer.max_weight": 0.06 + 0.10 * x,
        "param_smart_signals.target_monthly_vol": 0.02 + 0.04 * x,
        "annualized_return": 0.07 + 0.18 * x - 0.04 * x**2,
        "annualized_volatility": 0.08 + 0.15 * x,
        "sharpe_ratio": 0.8 + 0.7 * x - 0.5 * x**2,
        "sortino_ratio": 1.0 + 0.8 * x - 0.6 * x**2,
        "max_drawdown": -(0.08 + 0.20 * x),
        "cvar_95": -(0.03 + 0.10 * x),
        "downside_deviation": 0.04 + 0.10 * x,
        "psr": np.clip(0.70 + 0.30 * x, 0, 1),
        "dsr": np.clip(0.55 + 0.40 * x - 0.15 * x**2, 0, 1),
    })


def test_set_dotted_value_handles_list_of_singleton_dicts() -> None:
    cfg = {"model": [{"type": "bayessian"}, {"prior_strength": 63.0}], "optimizer": [{"max_weight": 0.12}]}
    set_dotted_value(cfg, "model.prior_strength", 84.0)
    set_dotted_value(cfg, "optimizer.0.max_weight", 0.10)
    assert cfg["model"][1]["prior_strength"] == 84.0
    assert cfg["optimizer"][0]["max_weight"] == 0.10


def test_policy_selection_returns_distinct_research_questions() -> None:
    trials = add_policy_scores(_fake_trials())
    selected = choose_policy_trials(trials, return_floor_quantile=0.30)
    assert selected["max_return"]["annualized_return"] >= selected["risk_control"]["annualized_return"]
    assert selected["risk_control"]["risk_control_score"] >= trials[trials["annualized_return"] >= trials["annualized_return"].quantile(0.30)]["risk_control_score"].max() - 1e-12


def test_parameter_range_summary_and_pareto_front_are_nonempty() -> None:
    trials = _fake_trials()
    summary = build_parameter_range_summary(trials, max_bins=4)
    pareto = extract_pareto_front(add_policy_scores(trials))
    assert not summary.empty
    assert not pareto.empty
    assert "top_risk_control_share" in summary.columns


def test_precomputed_custom_policy_scores_are_preserved() -> None:
    trials = _fake_trials()
    scored = add_policy_scores(trials, risk_weights={"annualized_return": 10.0, "abs_max_drawdown": 0.0})
    expected = scored.loc[scored["risk_control_score"].idxmax(), "trial_number"]
    selected = choose_policy_trials(scored, return_floor_quantile=0.0)
    assert selected["risk_control"]["trial_number"] == expected


def test_cdar_risk_metric_enters_policy_score_when_requested() -> None:
    trials = _fake_trials()
    trials["cdar_95"] = -(0.02 + 0.12 * np.linspace(0.0, 1.0, len(trials)))
    scored = add_policy_scores(
        trials,
        risk_weights={"abs_cvar_95": 0.0, "abs_cdar_95": 1.0},
    )
    assert "abs_cdar_95" in scored
    assert np.isfinite(scored["risk_control_score"]).all()


def test_nested_optimizer_grid_is_synchronized_with_trial_value() -> None:
    from scripts.optimization.optuna_calibration import sync_nested_optimizer_grid
    cfg = {
        "use_backtest_engine": True,
        "optimizer": [{"turnover_penalty": 0.002, "cdar_budget_mult": 1.0}],
        "optimizer_param_grid": {"turnover_penalty": [0.002], "cdar_budget_mult": [1.0]},
    }
    sync_nested_optimizer_grid(
        cfg,
        {"optimizer.0.turnover_penalty": 0.006, "optimizer.0.cdar_budget_mult": 0.85},
    )
    assert cfg["optimizer_param_grid"]["turnover_penalty"] == [0.006]
    assert cfg["optimizer_param_grid"]["cdar_budget_mult"] == [0.85]
