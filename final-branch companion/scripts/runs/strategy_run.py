import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import copy
import json
import yaml
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Any
from loguru import logger
from beartype import beartype
from datetime import timedelta

from scripts.dataloader.inputs import *
from scripts.dataloader.filter import *

from scripts.calculations.portfolio_utils import sigmoid_np

from scripts.models.registry import *
from scripts.executions.registry import *
from scripts.strategies.registry import *

from scripts.results.results import BaseResults
from scripts.risk_controls.smart_signals import smart_rebalance_diagnostics
from scripts.backtesting.engine_integration import results_by_nested_backtest_engine
from scripts.runs.common_postprocess import (
    candidate_trial_records, save_candidate_trials, save_selection_audit,
    save_risk_matrices, save_run_metadata, generate_research_validation_safely,
    generate_metric_artifacts_safely, generate_signal_research_safely, generate_interactive_report_safely,
)
from scripts.runs.run import (
    read_yaml, _flatten_config,
    _section_from_flat, _safe_drop_date,
    _make_market_cap_frame,
    compare_current_with_previous_weights,
    # results_by_fitting_model_alpha
)


PARAMS = ['balance', 'model', 'strategy', 'execution']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly ETF rebalancing")
    parser.add_argument("--config-path", "-c", type=str, default="strategy.yaml", help="filename for config yaml")
    return parser.parse_args()
    

# @beartype
# def results_by_fitting_model_alpha(
#     config: dict[str, Any],
#     strategy_class: Type[BaseStrategy],
#     exec_class: Type[BaseExecution],
#     portfolios_date: pd.DataFrame,
#     mask: Union[pd.Series, np.ndarray],
#     market_cap: np.ndarray,
#     historical_returns: np.ndarray,
#     pred_returns: np.ndarray,
#     future_returns: pd.DataFrame,
#     w_previous: pd.Series,
#     initial_exp_sum: float,
#     current_balance: float
# ) -> tuple[float, float, np.ndarray, pd.Series]:
    
#     alphas = []
#     info = np.empty((0, 4))
#     case = ["max_alpha", "voted_alpha"]
    
#     exec_config = {k: v for pair in config['execution'] for k, v in pair.items()}
    
#     print()
#     print(f"{strategy_class.__name__=}")
    
#     weights = strategy_class(
#         config,
#         historical_returns,
#         pred_returns,
#         market_cap,
#         w_previous
#     ).get_weights()
    
#     portfolios_date['weights'] = 0.0
#     portfolios_date.loc[mask, 'weights'] = weights
#     w_target = portfolios_date['weights']
    
#     exp_sum = initial_exp_sum
#     # past_month_returns = historical_returns[-60:]
#     past_month_returns = pred_returns
    
#     results = exec_class(
#         exec_config,
#         w_previous,
#         w_target,
#         mask,
#         past_month_returns,
#         current_balance,
#         alpha = 1.0
#     ).execution_process()
    
#     print("1", f"{results=}")
        
#     exp_sum = 0.2 * exp_sum + 0.8 * results[-1]
#     alpha = jax.nn.sigmoid(exp_sum)
#     info = np.vstack((info, results))
#     alphas.append(1.0)
    
#     results = exec_class(
#         exec_config,
#         w_previous,
#         w_target,
#         mask,
#         past_month_returns,
#         current_balance,
#         float(alpha)
#     ).execution_process()
    
#     print("2", f"{results=}")

#     info = np.vstack((info, results))
#     alphas.append(alpha)
        
#     index = np.argsort(info[:, -1])[::-1][0]
    
#     best_index = index%2
#     best_exp_sum = exp_sum
#     best_alpha = float(alphas[index])
#     best_results = info[index]
#     best_weights = weights.copy()
#     best_balance = info[index][1]

#     best_exp_sum, best_alpha, best_results, best_weights = compare_current_with_previous_weights(
#         config,
#         exec_class,
#         mask,
#         historical_returns,
#         pred_returns,
#         w_previous,
#         initial_exp_sum,
#         current_balance,
#         best_balance,
#         best_exp_sum,
#         best_alpha,
#         best_results,
#         best_weights
#     )
    
#     portfolios_date['weights'] = 0.0
#     portfolios_date.loc[mask, 'weights'] = best_weights
#     w_target = portfolios_date['weights']
    
#     results = exec_class(
#         exec_config,
#         w_previous,
#         w_target,
#         mask,
#         future_returns,
#         current_balance,
#         best_alpha
#     ).execution_process()
    
#     print(f"{best_index=}", f"{np.sum(w_target)=}", f"{case[best_index%2]=}")
    
#     return best_exp_sum, best_alpha, np.array(results), w_target


@beartype
def results_by_fitting_model_alpha(
    config: dict[str, Any],
    strategy_class: Type[BaseStrategy],
    exec_class: Type[BaseExecution],
    portfolios_date: pd.DataFrame,
    mask: Union[pd.Series, np.ndarray],
    market_cap: np.ndarray,
    historical_returns: np.ndarray,
    pred_returns: np.ndarray,
    future_returns: pd.DataFrame,
    w_previous: pd.Series,
    initial_exp_sum: float,
    current_balance: float
) -> tuple[float, float, np.ndarray, pd.Series]:
    
    exec_config = _flatten_config(config['execution'])
    # print(f"{strategy_class.__name__=}")
    
    weights = strategy_class(
        config,
        historical_returns,
        pred_returns,
        market_cap,
        w_previous
    ).get_weights()
    
    portfolios_date['weights'] = 0.0
    portfolios_date.loc[mask, 'weights'] = weights
    w_target = portfolios_date['weights']
    
    best_exp_sum = initial_exp_sum
    
    if "fixed_alpha" in exec_config:
        alpha = float(exec_config["fixed_alpha"])
    elif bool(exec_config.get("use_dynamic_alpha", False)):
        alpha = sigmoid_np(initial_exp_sum)
    else:
        alpha = 1.0
    
    # past_month_returns = historical_returns
    past_month_returns = historical_returns[-60:]
    # past_month_returns = pred_returns
    
    exec_processing_class = exec_class(
        exec_config,
        w_previous,
        w_target,
        mask,
        past_month_returns,
        current_balance,
        float(alpha)
    )
    
    results = exec_processing_class.grid_search_execution()
    
    best_exp_sum = 0.2 * initial_exp_sum + 0.8 * results[-1]
    best_alpha = alpha
    best_results = np.array(results, dtype=float)
    best_weights = weights.copy()
    best_balance = best_results[1]
    
    if bool(config.get('is_previous', False)):
        best_exp_sum, best_alpha, best_results, best_weights = compare_current_with_previous_weights(
            config,
            exec_class,
            mask,
            historical_returns,
            pred_returns,
            w_previous,
            initial_exp_sum,
            current_balance,
            best_balance,
            best_exp_sum,
            best_alpha,
            best_results,
            best_weights
        )
    
    portfolios_date['weights'] = 0.0
    portfolios_date.loc[mask, 'weights'] = best_weights
    w_target = portfolios_date['weights']

    smart_cfg = dict(config.get("smart_signals", {}) or {})
    if bool(smart_cfg.get("enabled", False)):
        try:
            etfs_selected = portfolios_date.loc[mask, "Key"].astype(str).tolist() if "Key" in portfolios_date.columns else list(range(int(np.asarray(mask).sum())))
            asset_w = pd.Series(w_target.loc[mask].values, index=etfs_selected, dtype=float)
            _, diag, adjusted_asset_w = smart_rebalance_diagnostics(
                etfs_list=etfs_selected,
                historical_returns=historical_returns,
                pred_returns=pred_returns,
                market_cap_history=market_cap,
                selected_asset_weights=asset_w,
                return_type=exec_config.get("return_type", "log-returns"),
                config=smart_cfg,
            )
            if bool(smart_cfg.get("apply_position_sizing", True)):
                portfolios_date.loc[mask, 'weights'] = adjusted_asset_w.values
                w_target = portfolios_date['weights']
        except Exception as exc:
            print(f"Smart signal sizing skipped: {exc}")
    
    results = exec_class(
        exec_config,
        w_previous,
        w_target,
        mask,
        future_returns,
        current_balance,
        best_alpha
    ).execution_process(is_dynamic_alpha=True)
    
    print(f"{np.sum(w_target)=}")
    
    return best_exp_sum, best_alpha, np.array(results), w_target


@beartype
def model_computation(
    config: dict[str, Any],
    first_business_days: pd.Series,
    market_cap: pd.DataFrame,
    etfs_list: list,
    returns_all: pd.DataFrame,
    ewma_returns_all: pd.DataFrame,
    df_prices: pd.DataFrame,
    portfolios: dict,
    output_folder: str = "markowitz_all"
) -> dict:
    
    current_balance = config['balance']
    window_type = config['window']
    model_config = _flatten_config(config["model"])
    strategy_config = _flatten_config(config["strategy"])
    exec_config = _flatten_config(config["execution"])
   
    model_class = MODEL_REGISTRY[model_config["type"]]
    strategy_class = STRATEGY_REGISTRY[strategy_config["type"]]
    exec_class = EXEC_REGISTRY[exec_config["type"]]
    results_class = BaseResults()
    selection_audit: list[dict[str, Any]] = []
    trial_audit: list[dict[str, Any]] = []
    risk_matrices: list[dict[str, Any]] = []
    
    portfolio_type = config.get('portfolio', 'filtered')
    use_backtest_engine = bool(config.get("use_backtest_engine", True))
    lookback_years = int(config.get("lookback_years", 3))
    
    # dynamic_controller = DynamicParameterController.from_optimizer_config(optimizer_config)
    # dynamic_history: list[dict[str, Any]] = []
    # selection_audit: list[dict[str, Any]] = []
    
    logger.info(
        f"Run start: portfolio={portfolio_type} ,"
        f"model={model_config['type']}, optimizer={strategy_config['type']}, execution={exec_config['type']}"
    )

    start_date = first_business_days.iloc[0] + timedelta(days=365*3)
    initial_date = first_business_days[first_business_days <= start_date].iloc[-1]
    index = first_business_days.index[first_business_days == initial_date].to_numpy()[0]
    
    prev_date = first_business_days[index-1]
    date = initial_date.strftime("%Y-%m-%d")
    mask = portfolios[date]['Value'].apply(lambda x: x == 1)

    portfolios[date]["weights"] = 0.0
    count_etf = len(portfolios[date][mask])
    if count_etf == 0:
        raise ValueError(f"No eligible ETFs on initial date {date}")
    
    initial_weight_type = config.get('initial_weight', 'equal')
    if initial_weight_type == 'equal':
        portfolios[date].loc[mask, "weights"] = np.ones(count_etf) / count_etf
    elif initial_weight_type == 'random':
        rng = np.random.default_rng(int(config.get("seed", 42)))
        w = rng.random(count_etf)
        portfolios[date].loc[mask, "weights"] = w / w.sum()
    elif initial_weight_type == 'dirichlet':
        rng = np.random.default_rng(int(config.get("seed", 42)))
        portfolios[date].loc[mask, "weights"] = rng.dirichlet(np.ones(count_etf))
    else:
        raise ValueError(f"Unknown initial weight type: {initial_weight_type}")
    
    w_current = portfolios[date]["weights"]

    alpha = 1.0
    exp_sum = 0.0
    
    results_class.add_date_results(
        step_date=date,
        etf_list=list(),
        pred_returns=None,
        real_returns=np.array([]),
        model_results=None,
        pnl_results=[current_balance, 0.0, 0.0, 0.0, 0.0]
    )
    
    print()
    
    pbar = tqdm(enumerate(first_business_days.iloc[index:-1]), desc='Portfolio optimization with month weight rebalancing')
    for ind, day in pbar:
        pbar.set_description(f'Portfolio optimization with {ind} month weight rebalancing {day}')
        
        # if ind >= 3:
        #     print(f"Completed {ind} iterations, stopping")
        #     break
        
        date = day.strftime('%Y-%m-%d')
        future_date = first_business_days.iloc[index+ind+1]
        
        if date not in portfolios:
            continue
        
        if portfolio_type == 'filtered':
            etfs_list, mask = get_filtered_etfs(portfolios[date]) # 1 way - use etfs from filtered etfs
        elif portfolio_type == 'sp500_all':
            etfs_list = SP500_VARIANTS
            mask = portfolios[date]['Key'].isin(etfs_list) # 2 way - all etfs that include SPY in their names, somehow related to SP500
        elif portfolio_type == 'sp500_benchmark':
            etfs_list = PURE_SP500_BENCHMARK
            mask = portfolios[date]['Key'].isin(etfs_list) # 3 wat - only SPY, using just benchmark etfs that fully is realted to SP500
        else:
            raise ValueError(f"Unknown portfolio type: {portfolio_type}")
        
        if len(etfs_list) == 0:
            logger.warning(f"No ETFs available for {date}; skipping rebalance.")
            continue
            
        startTime = find_start_time(first_business_days, day, lookback_years=lookback_years)
        historical_returns = get_historical_returns(returns_all, etfs_list, day, startTime, window_type).drop(columns=["Date"], errors="ignore")
        historical_ewma_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime, window_type).drop(columns=["Date"], errors="ignore")
        historical_market_cap = get_historical_returns(market_cap, etfs_list, day, startTime, window_type)
        future_returns = get_first_day_month_returns(returns_all, etfs_list, day, future_date).drop(columns=["Date"], errors="ignore")
        month_market_cap = get_first_day_month_returns(market_cap, etfs_list, prev_date, day).drop(columns=["Date"], errors="ignore")
        
        modelClass = model_class(
            _flatten_config(config['model']),
            etfs_list,
            historical_returns,
            historical_ewma_returns,
            future_returns.values
        )
        
        pred_returns = modelClass.prediction()
        model_metrics = modelClass.evaluate_metrics(pred_returns)
        try:
            snapshot = modelClass.get_risk_matrix_snapshot()
            snapshot["date"] = date
            risk_matrices.append(snapshot)
        except Exception as exc:
            logger.warning(f"Risk-matrix snapshot skipped on {date}: {exc}")
        
        # pred_returns = np.zeros_like(future_prices.values) # predictions only for 1 ETF SPY
        
        selection_info: dict[str, Any] = {"date": date, "engine_used": False}
        try:
            if not use_backtest_engine:
                raise RuntimeError("Nested engine disabled by config.")
            (new_balance, pnl, cost, return_value), w_target, validation_table, test_table, split_info, alpha = results_by_nested_backtest_engine(
                config=config, exec_class=exec_class, portfolios_date=portfolios[date].copy(),
                mask=mask, etfs_list=etfs_list, market_cap_history=historical_market_cap,
                historical_returns=historical_returns, historical_ewma_returns=historical_ewma_returns,
                pred_returns=pred_returns, future_returns=future_returns, w_previous=w_current,
                current_balance=current_balance,
            )
            selection_info.update({"engine_used": True, **split_info})
            selection_info["validation_best"] = validation_table.head(1).to_dict(orient="records")
            selection_info["test_best"] = test_table.head(1).to_dict(orient="records")
            trial_audit.extend(candidate_trial_records(
                rebalance_date=date,
                validation_table=validation_table,
                test_table=test_table,
                selected_candidate_id=split_info.get("selected_candidate_id"),
                attempt_records=split_info.get("candidate_attempt_records"),
            ))
        except Exception as exc:
            logger.warning(f"Nested strategy engine fallback on {date}: {exc}")
            exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(
                config=config, strategy_class=strategy_class, exec_class=exec_class,
                portfolios_date=portfolios[date].copy(), mask=mask, market_cap=month_market_cap.values,
                historical_returns=historical_returns.values, pred_returns=pred_returns,
                future_returns=future_returns, w_previous=w_current, initial_exp_sum=exp_sum,
                current_balance=current_balance,
            )
            selection_info.update({"fallback_reason": str(exc), "selected_alpha": alpha})
        selection_audit.append(selection_info)

        print(alpha, future_date, new_balance, pnl, cost, return_value)
        
        results_class.add_date_results(
            step_date=future_date.strftime('%Y-%m-%d'),
            etf_list=etfs_list,
            pred_returns=np.mean(pred_returns, axis=0),
            real_returns=np.sum(future_returns.values, axis=0),
            model_results=list(model_metrics),
            pnl_results=[new_balance, pnl, cost, return_value, alpha]
        )
        
        alpha_used = float(np.clip(alpha, 0.0, 1.0))
        w_executed = w_current + alpha_used * (w_target - w_current)
        portfolios[date]["target_weights"] = np.asarray(w_target, dtype=float)
        portfolios[date]["executed_weights"] = np.asarray(w_executed, dtype=float)
        portfolios[date]["weights"] = np.asarray(w_executed, dtype=float)
        if risk_matrices:
            risk_matrices[-1]["target_weights"] = {str(k): float(v) for k, v in zip(portfolios[date]["Key"], np.asarray(w_target, dtype=float))}
            risk_matrices[-1]["executed_weights"] = {str(k): float(v) for k, v in zip(portfolios[date]["Key"], np.asarray(w_executed, dtype=float))}
        current_balance = new_balance
        w_current = w_executed.copy()
        prev_date = day
            
    results_class.transform_results_to_df()
    results_class.save_results(output_folder)
    output_path = Path.cwd() / "results" / output_folder
    output_path.mkdir(parents=True, exist_ok=True)
    save_df_dict_to_excel(portfolios, save_path=output_path, filename='weights.xlsx')
    save_selection_audit(output_path, selection_audit)
    save_candidate_trials(output_path, trial_audit)
    save_risk_matrices(output_path, risk_matrices)
    save_run_metadata(output_path, {
        "return_type": exec_config.get("return_type", "log-returns"),
        "scenario_return_type": model_config.get("scenario_return_type", exec_config.get("return_type", "log-returns")),
        "horizon": int(model_config.get("horizon", 21)),
        "strategy": strategy_config,
        "smart_signals_enabled": bool(config.get("smart_signals", {}).get("enabled", False)),
    })
    generate_research_validation_safely(output_path, config=config)
    generate_metric_artifacts_safely(output_path, split_date=config.get("report_split_date"))
    generate_signal_research_safely(output_path, config=config)
    generate_interactive_report_safely(output_path)

    return portfolios


def main() -> None:
    first_business_days = pd.read_excel(OUTPUT / "business_dates.xlsx", index_col=0)['Values']
    sheet_names_str = first_business_days.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()[1:]
    
    df_prices = pd.read_csv(INPUT_PATH / "NewClosePrice.csv")
    df_volumes = pd.read_csv(INPUT_PATH / "Volume.csv")
    
    etf_list = df_prices.columns.drop('Date').sort_values(ascending=True)
    market_cap = _make_market_cap_frame(df_prices, df_volumes)
    # df_volumes = df_volumes[df_prices.columns]
    
    # Used by parsing Adjusting Close Price
    returns_all = pd.read_csv(OUTPUT / "new_etf_returns.csv")
    ewma_returns_all = pd.read_csv(OUTPUT / "new_etf_ewma.csv")
    
    returns_all['Date'] = pd.to_datetime(returns_all['Date'])
    returns_all = returns_all.fillna(0)
    ewma_returns_all['Date'] = pd.to_datetime(ewma_returns_all['Date'])
    
    logger.info('Successful file reading')

    # portfolios = pd.read_excel(OUTPUT / "new_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 600-700 etfs
    portfolios = pd.read_excel(OUTPUT / "last_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price # around 220 etfs

    args = parse_args()
    config_path = Path.cwd()/ "scripts" / "configs" / args.config_path
    config = read_yaml(config_path, PARAMS)

    model_computation(
        config=config,
        first_business_days=first_business_days,
        market_cap=market_cap,
        etfs_list=list(etf_list),
        returns_all=returns_all,
        ewma_returns_all=ewma_returns_all,
        df_prices=df_prices,
        portfolios=portfolios,
        # output_folder="sp500_benchmark_equal_weight"
        # output_folder="sp500_all_equal_weight"
        # output_folder="simple_long"
        # output_folder="risk_parity"
        # output_folder="equal_weight"
        # output_folder="inverse_volatility"
        # output_folder="liquidity_weighted"
        output_folder=config.get("output_folder", "strategy_smart")
    )

    logger.info('SUCCESS')



if __name__ == "__main__":
    main()    
