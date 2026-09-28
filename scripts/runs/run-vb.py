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
from scripts.optimizers.registry import *
from scripts.executions.registry import *

from scripts.results.results import BaseResults
from scripts.optimizers.dynamic_params import DynamicParameterController

from scripts.backtesting.engine_integration import results_by_nested_backtest_engine
from scripts.runs.common_postprocess import (
    candidate_trial_records, save_candidate_trials, save_selection_audit,
    save_risk_matrices, save_run_metadata, generate_research_validation_safely,
    generate_metric_artifacts_safely, generate_signal_research_safely, generate_interactive_report_safely,
    run_postprocessing,
)


PARAMS = ['balance', 'model', 'optimizer', 'execution']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly ETF rebalancing")
    parser.add_argument("--config-path", "-c", type=str, default="bayessian_cvar.yaml", help="filename for config yaml")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="finish validation/metrics/signal research/report from an existing results folder without rerunning optimization",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="optional results subfolder override; defaults to config output_folder",
    )
    return parser.parse_args()


@beartype
def read_yaml(filepath: Path, params: list[str]) -> dict:
    if not filepath.exists():
        raise FileExistsError(f"Unknown filepath: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if set(params) <= set(config.keys()):
        return config
    raise ValueError(f"Missing key in {filepath.name}")


@beartype
def _flatten_config(config_section: list[dict[str, Any]]) -> dict[str, Any]:
    return {k: v for pair in config_section for k, v in pair.items()}


def _section_from_flat(flat: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep compatibility with the old YAML list-of-dicts style."""
    return [dict(flat)]


def _safe_drop_date(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Date"], errors="ignore")


def _make_market_cap_frame(price_df: pd.DataFrame, volume_df: pd.DataFrame) -> pd.DataFrame:
    """Build price * volume while preserving Date as a real datetime column."""
    price_df = price_df.copy()
    volume_df = volume_df.copy()
    date_col = "Date" if "Date" in price_df.columns else price_df.columns[0]
    dates = pd.to_datetime(price_df[date_col], errors="coerce")

    price_num = price_df.drop(columns=[date_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
    volume_num = volume_df.drop(columns=[date_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
    common = [c for c in price_num.columns if c in volume_num.columns]
    market_cap = price_num[common] * volume_num[common]
    market_cap.insert(0, "Date", dates)
    return market_cap
    
    
@beartype
def compare_current_with_previous_weights(
    config: dict[str, Any],
    exec_class: Type[BaseExecution],
    mask: pd.Series,
    historical_returns: np.ndarray,
    pred_returns: np.ndarray,
    w_previous: pd.Series,
    initial_exp_sum: float,
    current_balance: float,
    best_balance: float,
    best_exp_sum: float,
    best_alpha: float,
    best_results: np.ndarray,
    best_weights: np.ndarray  
):
    
    alphas = []
    info = np.empty((0, 4))
    case = ["max_alpha", "voted_alpha"]
    exec_config = _flatten_config(config['execution'])
    
    # past_month_returns = historical_returns
    past_month_returns = historical_returns[-60:]
    # past_month_returns = pred_returns
            
    results = exec_class(
        exec_config,
        w_previous,
        w_previous,
        mask,
        past_month_returns,
        current_balance,
        alpha = 1.0
    ).execution_process()
    
    best_exp_sum = initial_exp_sum
    # exp_sum = 0.2 * best_exp_sum + 0.8 * results[-1]
    # alpha = jax.nn.sigmoid(exp_sum)
    
    if "fixed_alpha" in exec_config:
        alpha = float(exec_config["fixed_alpha"])
    elif bool(exec_config.get("use_dynamic_alpha", False)):
        best_exp_sum = 0.2 * initial_exp_sum + 0.8 * results[-1]
        alpha = sigmoid_np(initial_exp_sum)
    else:
        alpha = 1.0
    
    
    info = np.vstack((info, results))
    alphas.append(1.0)
    
    results = exec_class(
        exec_config,
        w_previous,
        w_previous,
        mask,
        past_month_returns,
        current_balance,
        float(alpha)
    ).execution_process(is_dynamic_alpha=True)

    info = np.vstack((info, results))
    alphas.append(alpha)
    
    index = np.argsort(info[:, 1])[::-1][0]
    
    if info[index][1] >= best_balance:
        best_exp_sum = best_exp_sum
        best_alpha = float(alphas[index])
        best_results = info[index]
        best_weights = w_previous
        best_balance = info[index][1]
        
    return best_exp_sum, best_alpha, best_results, best_weights
    

@beartype
def results_by_fitting_model_alpha(
    config: dict[str, Any],
    optimizer_class: Type[BaseCVaR],
    exec_class: Type[BaseExecution],
    portfolios_date: pd.DataFrame,
    mask: pd.Series,
    market_cap: np.ndarray,
    historical_returns: np.ndarray,
    pred_returns: np.ndarray,
    future_returns: pd.DataFrame,
    w_previous: pd.Series,
    initial_exp_sum: float,
    current_balance: float
) -> tuple[float, float, np.ndarray, pd.Series]:
    
    alphas = []
    info = np.empty((0, 4))
    case = ["max_alpha", "voted_alpha"]
    
    optimizer_config = _flatten_config(config['optimizer'])
    exec_config = _flatten_config(config['execution'])
    
    _, method_weights = optimizer_class(
        optimizer_config,
        market_cap,
        historical_returns,
        pred_returns,
        w_previous[mask].values
    ).get_results()
    
    if len(method_weights) == 0:
        raise ValueError("Optimizer returned no weight vectors")
    
    best_exp_sum = initial_exp_sum
    # alpha = jax.nn.sigmoid(exp_sum)
    
    if "fixed_alpha" in exec_config:
        alpha = float(exec_config["fixed_alpha"])
    elif bool(exec_config.get("use_dynamic_alpha", False)):
        alpha = sigmoid_np(initial_exp_sum)
    else:
        alpha = 1.0
    
    # past_month_returns = historical_returns
    past_month_returns = historical_returns[-60:]
    # past_month_returns = pred_returns
    
    for weights in method_weights:
        portfolios_date['weights'] = 0.0
        portfolios_date.loc[mask, 'weights'] = weights
        w_target = portfolios_date['weights']
        
        exec_processing_class = exec_class(
            exec_config,
            w_previous,
            w_target,
            mask,
            past_month_returns,
            current_balance,
            float(alpha)
        )

        info = np.vstack((info, exec_processing_class.grid_search_execution()))
        alphas.append(alpha)
        
    index = np.argsort(info[:, -1])[::-1][0]
    
    best_index = index
    best_exp_sum = 0.2 * best_exp_sum + 0.8 * info[index][-1]
    best_alpha = float(alphas[index])
    best_results = info[index]
    best_weights = method_weights[index]
    best_balance = info[index][1]
    
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
    
    results = exec_class(
        exec_config,
        w_previous,
        w_target,
        mask,
        future_returns,
        current_balance,
        best_alpha
    ).execution_process(is_dynamic_alpha=True)
    
    print(f"{best_index=}", f"{np.sum(w_target)=}", f"{case[index%2]=}")
    
    return best_exp_sum, best_alpha, np.array(results), w_target


@beartype
def model_computation(
    config: dict[str, Any],
    first_business_days: pd.Series,
    market_cap: pd.DataFrame,
    ewma_returns_all: pd.DataFrame,
    returns_all: pd.DataFrame,
    portfolios: dict,
    output_folder: str = "markowitz_all"
) -> dict:
    
    current_balance = config['balance']
    window_type = config['window']
    model_config = _flatten_config(config["model"])
    optimizer_config = _flatten_config(config["optimizer"])
    exec_config = _flatten_config(config["execution"])
    
    model_class = MODEL_REGISTRY[model_config["type"]]
    optimizer_class = CVAR_REGISTRY[optimizer_config["type"]]
    exec_class = EXEC_REGISTRY[exec_config["type"]]
    results_class = BaseResults()
    
    portfolio_type = config.get('portfolio', 'filtered')
    use_backtest_engine = bool(config.get("use_backtest_engine", True))
    lookback_years = int(config.get("lookback_years", 3))
    
    dynamic_controller = DynamicParameterController.from_optimizer_config(optimizer_config)
    dynamic_history: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    trial_audit: list[dict[str, Any]] = []
    forecast_risk: list[dict[str, Any]] = []   # causal forward posterior-predictive risk
    risk_matrices: list[dict[str, Any]] = []
    
    logger.info(
        f"Run start: portfolio={portfolio_type}"
        f"model={model_config['type']}, optimizer={optimizer_config['type']}, execution={exec_config['type']}"
    )
    
    start_date = first_business_days.iloc[0] + timedelta(days=365 * lookback_years)
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
        pred_returns=np.array([]),
        real_returns=np.array([]),
        model_results=[0.0]*12,
        pnl_results=[current_balance, 0.0, 0.0, 0.0, 0.0]
    )
    
    pbar = tqdm(enumerate(first_business_days.iloc[index:-1]), desc='Portfolio optimization with month weight rebalancing')
    for ind, day in pbar:
        pbar.set_description(f'Portfolio optimization with {ind} month weight rebalancing {day}')
        
        # if ind >= 20:
        #     print(f"Completed {ind} iterations, stopping")
        #     break
        
        date = day.strftime('%Y-%m-%d')
        future_date = first_business_days.iloc[index+ind+1]
        
        if date not in portfolios:
            continue

        if portfolio_type == 'filtered':
            etfs_list, mask = get_filtered_etfs(portfolios[date])
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
        historical_returns = get_historical_returns(returns_all, etfs_list, day, startTime, window_type)
        historical_ewma_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime, window_type) 
        historical_market_cap = get_historical_returns(market_cap, etfs_list, day, startTime, window_type)
        future_returns = get_first_day_month_returns(returns_all, etfs_list, day, future_date).drop(columns=["Date"], errors="ignore")
        month_market_cap = get_first_day_month_returns(market_cap, etfs_list, prev_date, day).drop(columns=["Date"], errors="ignore")
        
        if future_returns.empty or _safe_drop_date(historical_returns).empty:
            logger.warning(f"Empty returns for {date}; skipping rebalance.")
            continue
        
        # dynamic_controller - dynamic changing parameters for optimization
        runtime_config = copy.deepcopy(config)
        dynamic_optimizer_config, dynamic_info = dynamic_controller.next_optimizer_config(
            optimizer_config,
            _safe_drop_date(historical_returns).values,
            _safe_drop_date(historical_market_cap).values,
            date=date,
        )
        runtime_config["optimizer"] = _section_from_flat(dynamic_optimizer_config)
        dynamic_history.append(dynamic_info)
        
        modelClass = model_class(
            _flatten_config(config['model']),
            etfs_list,
            historical_returns,
            historical_ewma_returns,
            future_returns.values
        )
        
        pred_returns = modelClass.prediction()
        # Preserve the ordered path cube separately from YAML/runtime search parameters.
        # CDaR needs (scenario, day, asset) paths; terminal-loss CVaR only needs the
        # scenario fan.  Keeping the cube out of the parameter grid also prevents a
        # large ndarray from contaminating trial IDs and the research ledger.
        try:
            current_scenario_paths = modelClass.get_scenario_paths()
        except Exception:
            current_scenario_paths = None
        # Scenario-distribution diagnostics compare the posterior fan with the realised
        # holding-period vector; scenario rows are not treated as an ordered time path.
        model_metrics = modelClass.evaluate_metrics(pred_returns)
        try:
            snapshot = modelClass.get_risk_matrix_snapshot()
            snapshot["date"] = date
            risk_matrices.append(snapshot)
        except Exception as exc:
            logger.warning(f"Risk-matrix snapshot skipped on {date}: {exc}")
        
        # # Initial variant
        # exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(
        #     config=config,
        #     optimizer_class=optimizer_class,
        #     exec_class=exec_class,
        #     portfolios_date=portfolios[date].copy(),
        #     mask=mask,
        #     market_cap=month_market_cap.values,
        #     historical_returns=_safe_drop_date(historical_returns).values,
        #     pred_returns=pred_returns,
        #     future_returns=future_returns,
        #     w_previous=w_current,
        #     initial_exp_sum=exp_sum,
        #     current_balance=current_balance
        # )
        
        selection_info: dict[str, Any] = {"date": date, "engine_used": False}
        try:
            if not use_backtest_engine:
                raise RuntimeError("Nested engine disabled by config.")

            (new_balance, pnl, cost, return_value), w_target, validation_table, test_table, split_info, alpha = results_by_nested_backtest_engine(
                config=runtime_config,
                exec_class=exec_class,
                portfolios_date=portfolios[date].copy(),
                mask=mask,
                etfs_list=etfs_list,
                market_cap_history=historical_market_cap,
                historical_returns=historical_returns,
                historical_ewma_returns=historical_ewma_returns,
                pred_returns=pred_returns,
                future_returns=future_returns,
                w_previous=w_current,
                current_balance=current_balance,
                scenario_paths=current_scenario_paths,
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
            logger.warning(f"Nested engine fallback on {date}: {exc}")
            fallback_config = copy.deepcopy(runtime_config)
            try:
                _fb_opt = _flatten_config(fallback_config["optimizer"])
                if str(_fb_opt.get("type", "")).lower() == "cdar" or "cdar" in str(_fb_opt.get("task_type", "")).lower():
                    if current_scenario_paths is None:
                        raise RuntimeError("CDaR fallback requires ordered scenario paths from the Bayesian model")
                    _fb_opt["scenario_paths"] = current_scenario_paths
                    fallback_config["optimizer"] = _section_from_flat(_fb_opt)
            except Exception as path_exc:
                if str(_flatten_config(runtime_config["optimizer"]).get("type", "")).lower() == "cdar":
                    raise
                logger.warning(f"Fallback scenario-path preparation skipped on {date}: {path_exc}")
            exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(
                config=fallback_config,
                optimizer_class=optimizer_class,
                exec_class=exec_class,
                portfolios_date=portfolios[date].copy(),
                mask=mask,
                market_cap=month_market_cap.values,
                historical_returns=_safe_drop_date(historical_returns).values,
                pred_returns=pred_returns,
                future_returns=future_returns,
                w_previous=w_current,
                initial_exp_sum=exp_sum,
                current_balance=current_balance,
            )
            selection_info.update({"engine_used": False, "fallback_reason": str(exc), "selected_alpha": alpha})
        
        print(alpha, future_date, new_balance, pnl, cost, return_value)
        # --- clean machine-readable progress line for the notebook (added for last_implementations) ---
        try:
            _n_held = int((np.asarray(w_target, dtype=float) > 1e-4).sum())
        except Exception:
            _n_held = -1
        try:
            _d = future_date.strftime('%Y-%m-%d')
        except Exception:
            _d = str(future_date)
        print(f"PROGRESS\t{_d}\t{len(etfs_list)}\t{_n_held}\t{float(return_value)}\t{float(new_balance)}", flush=True)

        # --- FORWARD model-implied risk of the CHOSEN book (causal: known at the rebalance, before
        #     month t is traded). Posterior-predictive CVaR of w_target over the (S x N) scenario fan.
        #     This is the self-timing signal for the model-CVaR overlay (cvar_target_overlay.py). ---
        try:
            R = np.asarray(pred_returns, dtype=float)             # (S, N) over etfs_list
            w_vec = np.asarray(w_target, dtype=float).reshape(-1)
            if w_vec.shape[0] != R.shape[1]:                      # w_target is full-universe -> mask to etfs_list
                m = np.asarray(mask).astype(bool)
                if m.shape[0] == w_vec.shape[0] and int(m.sum()) == R.shape[1]:
                    w_vec = w_vec[m]
            if R.ndim == 2 and R.shape[1] == w_vec.shape[0] and np.isfinite(w_vec).all():
                # The internal risk signal belongs to the risky-sleeve composition, not to
                # the already-scaled total-wealth exposure.  Normalize first so the signal
                # cannot become circular when a smart/cash overlay is active.
                target_exposure = float(np.sum(w_vec))
                risky_w = w_vec / target_exposure if target_exposure > 1e-12 else w_vec
                port_raw = R @ risky_w
                a_fc = float(optimizer_config.get("forecast_cvar_level", 0.95))
                port_mean = float(np.mean(port_raw))
                port_centered = port_raw - port_mean

                def _cvar_loss(values):
                    losses = -np.asarray(values, dtype=float)
                    var_ = float(np.quantile(losses, a_fc))
                    tail_ = losses[losses >= var_]
                    return float(tail_.mean()) if tail_.size else var_

                cvar_model_raw = _cvar_loss(port_raw)
                cvar_model_centered = _cvar_loss(port_centered)
                # Legacy column now points to the publication-correct centered signal.
                cvar_model = cvar_model_centered
                vol_model = float(np.std(port_centered, ddof=1))

                # Optional path-dependent companion signal.  For CDaR we remove the
                # cross-scenario mean at each day before computing the centered version;
                # this isolates path-shape/covariance risk from a deterministic drift path.
                try:
                    from scripts.calculations.drawdown import empirical_cdar, portfolio_path_returns
                    _paths = np.asarray(current_scenario_paths, dtype=float) if current_scenario_paths is not None else None
                    if _paths is None:
                        raise RuntimeError("no predictive path cube available")
                    _port_paths = portfolio_path_returns(_paths, risky_w)
                    _a_cdar = float(_flatten_config(runtime_config["optimizer"]).get("cdar_confidence_level", a_fc))
                    cdar_model_raw = float(empirical_cdar(_port_paths, alpha=_a_cdar))
                    _paths_centered = _paths - np.mean(_paths, axis=0, keepdims=True)
                    cdar_model_centered = float(empirical_cdar(portfolio_path_returns(_paths_centered, risky_w), alpha=_a_cdar))
                except Exception:
                    cdar_model_raw = cdar_model_centered = float("nan")
            else:
                target_exposure = float(np.sum(w_vec)) if np.ndim(w_vec) else float("nan")
                port_mean = cvar_model_raw = cvar_model_centered = cvar_model = vol_model = float("nan")
                cdar_model_raw = cdar_model_centered = float("nan")
        except Exception:
            target_exposure = float(np.asarray(w_target, dtype=float).sum())
            port_mean = cvar_model_raw = cvar_model_centered = cvar_model = vol_model = float("nan")
            cdar_model_raw = cdar_model_centered = float("nan")
        forecast_risk.append({
            "date": _d,
            "cvar_model": cvar_model,
            "cvar_model_centered": cvar_model_centered,
            "cvar_model_raw": cvar_model_raw,
            "cdar_model": cdar_model_centered,
            "cdar_model_centered": cdar_model_centered,
            "cdar_model_raw": cdar_model_raw,
            "portfolio_scenario_mean": port_mean,
            "vol_model": vol_model,
            "forecast_cvar_centered": True,
            "target_risky_exposure": target_exposure,
            "backward_vol_timer": selection_info.get("backward_vol_timer"),
            "forward_cvar_timer": selection_info.get("forward_cvar_timer"),
            "forward_cdar_timer": selection_info.get("forward_cdar_timer"),
            "forward_risk_timer": selection_info.get("forward_risk_timer"),
            "forward_risk_measure": selection_info.get("forward_risk_measure"),
            "conviction_timer": selection_info.get("conviction_timer"),
            "overlay_fraction": selection_info.get("overlay_fraction"),
        })
        selection_audit.append(selection_info)

        dynamic_controller.update_after_realized_return(float(return_value))
        
        results_class.add_date_results(
            step_date=future_date.strftime('%Y-%m-%d'),
            etf_list=etfs_list,
            pred_returns=np.mean(pred_returns, axis=0),   # per-ETF expected return (mean over S scenarios)
            real_returns=np.sum(future_returns.values, axis=0),
            model_results=list(model_metrics),
            pnl_results=[new_balance, pnl, cost, return_value, alpha],
        )
        
        # Carry the weights that were actually executed, not the unexecuted target.
        # This fixes next-month turnover/cost accounting when alpha < 1.
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
    
    if dynamic_history:
        pd.DataFrame(dynamic_history).to_csv(output_path / "dynamic_parameter_history.csv", index=False)
    if forecast_risk:
        pd.DataFrame(forecast_risk).to_csv(output_path / "forecast_risk.csv", index=False)
    save_selection_audit(output_path, selection_audit)
    save_candidate_trials(output_path, trial_audit)
    save_risk_matrices(output_path, risk_matrices)
    # The risk snapshots are among the largest objects in a long walk-forward run.
    # They are no longer needed in memory once persisted; release them before the
    # validation/Plotly post-processing stage.
    risk_matrices.clear()
    try:
        current_scenario_paths = None
        pred_returns = None
        modelClass = None
    except Exception:
        pass
    import gc
    gc.collect()

    save_run_metadata(output_path, {
        "return_type": exec_config.get("return_type", "log-returns"),
        "scenario_return_type": model_config.get("scenario_return_type", exec_config.get("return_type", "log-returns")),
        "horizon": int(model_config.get("horizon", 21)),
        "optimizer": optimizer_config,
        "smart_signals_enabled": bool(config.get("smart_signals", {}).get("enabled", False)),
        "note": "Numerical results are valid only when generated from point-in-time input data.",
    })
    # Heavy validation/report generation is intentionally deferred until this
    # function returns.  That lets Python release the model, scenario, portfolio,
    # and audit working sets before post-processing starts.
    return portfolios


def build_portfolios_from_matrix(matrix_path, sheet_names_str, returns_cols, market_cap=None, cap_n=None):
    """Build the per-rebalance `portfolios` dict {date_str -> DataFrame[Key,Value,weights]} from an
    ELIGIBILITY MATRIX csv (rows=month-end dates, cols=tickers, values 0/1). Date mapping is by
    (year, month): a matrix row for month M is used for the pipeline's rebalance in month M (the
    matrices are already forward-dated -> causal). Eligible tickers are intersected with the available
    returns columns. If cap_n is set, keep the top-cap_n eligible names by trailing dollar-volume."""
    mtx = pd.read_csv(matrix_path)
    dcol = mtx.columns[0]
    mtx[dcol] = pd.to_datetime(mtx[dcol])
    mtx = mtx.set_index(dcol)
    ret_set = set(returns_cols)
    # CONSISTENT full-universe row set (same Keys, same order, EVERY month) so cross-month weight
    # vectors stay aligned (mirrors the original last_filtered_weights format). Value flags monthly
    # eligibility; the optimizer/data only ever touch the eligible (Value==1) subset.
    all_tickers = [t for t in mtx.columns if t in ret_set]
    ym_to_row = {(d.year, d.month): mtx.loc[d] for d in mtx.index}
    mc = None
    if cap_n is not None and market_cap is not None:
        mc = market_cap.copy()
        if 'Date' in mc.columns:
            mc['Date'] = pd.to_datetime(mc['Date']); mc = mc.set_index('Date')
        mc = mc.apply(pd.to_numeric, errors='coerce')
    portfolios = {}
    for ds in sheet_names_str:
        d = pd.to_datetime(ds); key = (d.year, d.month)
        value = pd.Series(0, index=all_tickers, dtype=int)
        if key in ym_to_row:
            row = ym_to_row[key]
            elig = [t for t in all_tickers if int(row.get(t, 0)) == 1]
            if cap_n is not None and len(elig) > int(cap_n) and mc is not None:
                win = mc.loc[mc.index <= d].tail(21)
                cols = [t for t in elig if t in win.columns]
                if cols:
                    liq = win[cols].mean(axis=0).fillna(0.0)
                    elig = liq.sort_values(ascending=False).head(int(cap_n)).index.tolist()
            if elig:
                value.loc[elig] = 1
        portfolios[ds] = pd.DataFrame({'Key': all_tickers, 'Value': value.values})
    return portfolios


def main() -> None:
    args = parse_args()
    config_path = Path.cwd() / "scripts" / "configs" / args.config_path
    config = read_yaml(config_path, PARAMS)
    output_folder = args.result_folder or config.get("output_folder", "advanced_bayesian_portfolio")
    output_path = Path.cwd() / "results" / output_folder

    # Recovery path for a completed optimization whose post-processing was killed
    # by the OS.  Crucially, do this before loading the large ETF input matrices.
    if args.postprocess_only:
        run_postprocessing(
            output_path,
            config=config,
            split_date=config.get("report_split_date"),
        )
        logger.info(f"SUCCESS (postprocess-only): {output_path}")
        return

    first_business_days = pd.read_excel(OUTPUT / "business_dates.xlsx", index_col=0)["Values"]
    sheet_names_str = first_business_days.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()[1:]

    df_prices = pd.read_csv(INPUT_PATH / "NewClosePrice.csv")
    df_volumes = pd.read_csv(INPUT_PATH / "Volume.csv")
    market_cap = _make_market_cap_frame(df_prices, df_volumes)

    # BEST - Used by parsing Adjusting Close Price
    returns_all = pd.read_csv(OUTPUT / "new_etf_returns.csv")
    ewma_returns_all = pd.read_csv(OUTPUT / "new_etf_ewma.csv")

    returns_all['Date'] = pd.to_datetime(returns_all['Date'])
    returns_all = returns_all.fillna(0)
    ewma_returns_all['Date'] = pd.to_datetime(ewma_returns_all['Date'])

    logger.info('Successful file reading')

    # ---- universe source: eligibility MATRIX (new) or the multi-sheet filtered xlsx (default) ----
    pmatrix = config.get("portfolio_matrix")
    if pmatrix:
        ret_cols = [c for c in returns_all.columns if c != "Date"]
        portfolios = build_portfolios_from_matrix(
            OUTPUT / pmatrix, sheet_names_str, ret_cols,
            market_cap=market_cap, cap_n=config.get("universe_cap"),
        )
        _ne = [len(v) for v in portfolios.values() if len(v)]
        logger.info(f"universe matrix {pmatrix}: {len(_ne)} active months, "
                    f"eligible/month min={min(_ne) if _ne else 0} max={max(_ne) if _ne else 0} cap={config.get('universe_cap')}")
    else:
        portfolios = pd.read_excel(OUTPUT / "last_filtered_weights.xlsx", sheet_name=sheet_names_str)

    model_computation(
        config=config,
        first_business_days=first_business_days,
        market_cap=market_cap,
        ewma_returns_all=ewma_returns_all,
        returns_all=returns_all,
        portfolios=portfolios,
        output_folder=output_folder,
    )

    # The numerical run and core artifacts are now on disk.  Drop all large input
    # and portfolio objects before importing/allocating validation + Plotly state.
    del portfolios, returns_all, ewma_returns_all, market_cap, df_prices, df_volumes, first_business_days
    import gc
    gc.collect()

    run_postprocessing(
        output_path,
        config=config,
        split_date=config.get("report_split_date"),
    )

    logger.info('SUCCESS')



if __name__ == "__main__":
    main()    
