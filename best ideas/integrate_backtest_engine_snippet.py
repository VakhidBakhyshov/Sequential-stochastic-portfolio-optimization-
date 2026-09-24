"""
Integration snippet for scripts/runs/run.py or scripts/runs/combined_run.py.

Replace the old results_by_fitting_model_alpha(...) call with results_by_nested_backtest_engine(...).
Place imports near the top of run.py / combined_run.py.
"""

from __future__ import annotations

from typing import Any, Type
from dataclasses import replace
import numpy as np
import pandas as pd

from scripts.backtesting.walk_forward_engine import (
    BacktestEngineConfig,
    PortfolioWalkForwardBacktestEngine,
    build_full_weight_series,
    flatten_config,
)
from scripts.optimizers.registry import CVAR_REGISTRY
from scripts.strategies.registry import STRATEGY_REGISTRY
from scripts.models.registry import MODEL_REGISTRY
from scripts.executions.base import BaseExecution


BACKTEST_ENGINE_DEFAULTS = BacktestEngineConfig(
    validation_months=6,
    internal_test_months=2,
    top_k_validation=5,
    select_metric="total_return",      # choose top candidates by return gain
    return_type="log-returns",         # keep equal to execution.return_type
    c_bps=1e-2,                         # keep equal to execution.c_bps if you want high ETF cost
    refit_on_full_history=True,
    include_hold_current=True,
    include_strategies=True,            # lets you compare optimizers with equal/risk/liquidity strategies
    alpha_grid=(1.0, 0.75, 0.50, 0.25),
    rank_penalty_turnover=0.0,
)


OPTIMIZER_PARAM_GRID = {
    # keep the grid small first; then expand after the run is stable
    "type": ["bayessian", "markowitz"],
    "task_type": ["cvar_returns", "expected_returns", "max_sharpe", "min_variance"],
    "max_weight": [0.05, 0.10],
    "min_weight": [0.00, 0.01],
    "turnover_penalty": [0.0, 0.5, 1.0],
    "penalty_type": ["L1"],
    "confidence_level": [0.95, 0.99],
    "is_all_methods": [False],
}


STRATEGY_PARAM_GRID = {
    "type": ["equal_weight", "inverse_volatility", "risk_parity", "liquidity_weighted"],
    "min_weight": [0.00, 0.01],
    "max_weight": [0.10],
}


def results_by_nested_backtest_engine(
    config: dict[str, Any],
    exec_class: Type[BaseExecution],
    portfolios_date: pd.DataFrame,
    mask: pd.Series,
    etfs_list: list[str],
    market_cap_history: pd.DataFrame | np.ndarray,
    historical_returns: pd.DataFrame,
    historical_ewma_returns: pd.DataFrame,
    pred_returns: np.ndarray,
    future_returns: pd.DataFrame,
    w_previous: pd.Series,
    current_balance: float,
) -> tuple[np.ndarray, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Runs nested selection:
    train -> validation top 3-5 -> internal test top 1 -> execute next month.

    Returns:
        execution_result: np.array([new_balance, pnl, cost, return_value])
        w_target: full Series in portfolios_date index
        validation_table: all candidates ranked by validation performance
        test_table: top-k validation candidates ranked by internal-test performance
        split_info: dates/months used in train/validation/test
    """
    model_config = flatten_config(config.get("model"))
    optimizer_config = flatten_config(config.get("optimizer"))
    strategy_config = flatten_config(config.get("strategy"))
    exec_config = flatten_config(config.get("execution"))

    engine_config = replace(
        BACKTEST_ENGINE_DEFAULTS,
        return_type=exec_config.get("return_type", BACKTEST_ENGINE_DEFAULTS.return_type),
        c_bps=float(exec_config.get("c_bps", BACKTEST_ENGINE_DEFAULTS.c_bps)),
    )

    model_type = model_config.get("type")
    model_class = MODEL_REGISTRY.get(model_type) if model_type else None

    engine = PortfolioWalkForwardBacktestEngine(
        engine_config=engine_config,
        optimizer_registry=CVAR_REGISTRY,
        strategy_registry=STRATEGY_REGISTRY,
        model_class=model_class,
        model_config=model_config,
        optimizer_base_config=optimizer_config,
        strategy_base_config=strategy_config,
        optimizer_param_grid=OPTIMIZER_PARAM_GRID,
        strategy_param_grid=STRATEGY_PARAM_GRID,
    )

    selection = engine.select(
        etfs_list=etfs_list,
        historical_returns=historical_returns,
        historical_ewma_returns=historical_ewma_returns,
        market_cap_history=market_cap_history,
        w_previous=pd.Series(w_previous.loc[mask].values, index=etfs_list),
        fallback_pred_returns=pred_returns,
    )

    selected_weights = (
        selection.selected_target_weights_full
        if selection.selected_target_weights_full is not None
        else selection.selected.target_weights
    )
    w_target = build_full_weight_series(
        portfolios_date=portfolios_date,
        mask=mask,
        etfs_list=etfs_list,
        selected_asset_weights=selected_weights,
    )

    results = exec_class(
        exec_config,
        w_previous,
        w_target,
        mask,
        future_returns,
        current_balance,
        alpha=selection.selected.alpha,
    ).execution_process()

    return np.array(results), w_target, selection.validation_table, selection.test_table, selection.split_info


# In your monthly loop, replace:
#
# exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(...)
#
# with:
#
# (new_balance, pnl, cost, return_value), w_target, validation_table, test_table, split_info = results_by_nested_backtest_engine(
#     config=config,
#     exec_class=exec_class,
#     portfolios_date=portfolios[date],
#     mask=mask,
#     etfs_list=etfs_list,
#     market_cap_history=month_market_cap,              # better: pass the full historical market_cap window
#     historical_returns=historical_returns.drop('Date', axis=1) if 'Date' in historical_returns.columns else historical_returns,
#     historical_ewma_returns=historical_ewma_returns.drop('Date', axis=1) if 'Date' in historical_ewma_returns.columns else historical_ewma_returns,
#     pred_returns=pred_returns,
#     future_returns=future_returns,
#     w_previous=w_current,
#     current_balance=current_balance,
# )
#
# Then save validation_table and test_table to results/<folder>/selection_tables/ if needed.
