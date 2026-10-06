from __future__ import annotations

from dataclasses import replace
from typing import Any, Type

import numpy as np
import pandas as pd

from scripts.backtesting.walk_forward_engine import (
    BacktestEngineConfig,
    PortfolioWalkForwardBacktestEngine,
    build_full_weight_series,
    flatten_config,
)
from scripts.executions.base import BaseExecution
from scripts.models.registry import MODEL_REGISTRY
from scripts.optimizers.registry import CVAR_REGISTRY
from scripts.strategies.registry import STRATEGY_REGISTRY


BACKTEST_ENGINE_DEFAULTS = BacktestEngineConfig(
    validation_months=6,
    internal_test_months=2,
    top_k_validation=5,
    select_metric="score",
    return_type="log-returns",
    c_bps=0.0010,
    refit_on_full_history=True,
    include_hold_current=True,
    include_strategies=True,
    alpha_grid=(1.0, 0.75, 0.50, 0.25),
    rank_penalty_turnover=0.02,
)


DEFAULT_OPTIMIZER_PARAM_GRID = {
    # Keep small enough for monthly research runs. Invalid task/type pairs are skipped safely.
    "type": ["bayessian", "markowitz", "black_litterman"],
    "task_type": ["mean_cvar_sharpe", "cvar_returns", "max_sharpe", "min_variance", "black_litterman"],
    "max_weight": [0.08, 0.12],
    "min_weight": [0.00],
    "turnover_penalty": [0.10, 0.50],
    "penalty_type": ["L1"],
    "confidence_level": [0.95, 0.975],
    "risk_aversion": [0.0],
    "is_all_methods": [False],
}



DEFAULT_STRATEGY_PARAM_GRID = {
    "type": [
        "equal_weight",
        "inverse_volatility",
        "risk_parity",
        "liquidity_weighted",
        "momentum_volatility",
        "liquidity_momentum",
        "minimum_correlation",
    ],
    "min_weight": [0.00],
    "max_weight": [0.08, 0.12],
    "top_k": [25, 40],
    "momentum_lookback": [63, 126],
    "liquidity_power": [0.25],
}



def _grid_from_config(config: dict[str, Any], key: str, default: dict[str, list[Any]]) -> dict[str, list[Any]]:
    grid = config.get(key)
    if not grid:
        return default
    return {k: (v if isinstance(v, list) else [v]) for k, v in grid.items()}


def _engine_config_from_config(config: dict[str, Any], exec_config: dict[str, Any]) -> BacktestEngineConfig:
    user_cfg = dict(config.get("backtest_engine", {}) or {})
    base = replace(
        BACKTEST_ENGINE_DEFAULTS,
        return_type=exec_config.get("return_type", BACKTEST_ENGINE_DEFAULTS.return_type),
        c_bps=float(exec_config.get("c_bps", BACKTEST_ENGINE_DEFAULTS.c_bps)),
    )
    allowed = set(BacktestEngineConfig.__dataclass_fields__.keys())
    updates = {k: v for k, v in user_cfg.items() if k in allowed}
    return replace(base, **updates)


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
) -> tuple[np.ndarray, pd.Series, pd.DataFrame, pd.DataFrame, dict[str, Any], float]:
    """Nested train/validation/internal-test selection for one monthly rebalance."""
    model_config = flatten_config(config.get("model"))
    optimizer_config = flatten_config(config.get("optimizer"))
    strategy_config = flatten_config(config.get("strategy"))
    exec_config = flatten_config(config.get("execution"))

    engine_config = _engine_config_from_config(config, exec_config)
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
        optimizer_param_grid=_grid_from_config(config, "optimizer_param_grid", DEFAULT_OPTIMIZER_PARAM_GRID),
        strategy_param_grid=_grid_from_config(config, "strategy_param_grid", DEFAULT_STRATEGY_PARAM_GRID),
    )

    hist_returns = historical_returns.drop(columns=["Date"], errors="ignore")
    hist_ewma = historical_ewma_returns.drop(columns=["Date"], errors="ignore")
    if isinstance(market_cap_history, pd.DataFrame):
        market_cap_input = market_cap_history.drop(columns=["Date"], errors="ignore")
    else:
        market_cap_input = market_cap_history

    selection = engine.select(
        etfs_list=etfs_list,
        historical_returns=hist_returns,
        historical_ewma_returns=hist_ewma,
        market_cap_history=market_cap_input,
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
    ).execution_process(is_dynamic_alpha=True)

    split_info = dict(selection.split_info)
    split_info.update(
        {
            "selected_candidate_id": selection.selected.candidate_id,
            "selected_source": selection.selected.source,
            "selected_method": selection.selected.method,
            "selected_alpha": float(selection.selected.alpha),
            "selected_params": selection.selected.params,
        }
    )

    return (
        np.array(results, dtype=float),
        w_target,
        selection.validation_table,
        selection.test_table,
        split_info,
        float(selection.selected.alpha),
    )
