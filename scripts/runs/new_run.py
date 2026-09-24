import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import yaml
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Any
from loguru import logger
from beartype import beartype
from datetime import timedelta

from scripts.dataloader.inputs import *
from scripts.dataloader.filter import *
from scripts.calculations.matrix import *

from scripts.models.registry import *
from scripts.optimizers.registry import *
from scripts.executions.registry import *

from scripts.results.results import BaseResults


PARAMS = [
    'balance',
    'model',
    'optimizer',
    'execution'
]


@beartype
def read_yaml(filepath: Path, params: list[str]) -> dict:
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if set(params) <= set(config.keys()):
        return config
    else:
        raise ValueError(f"Missing key in {filepath.name}")
    
    
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
    exp_sum = initial_exp_sum
    
    # past_month_returns = historical_returns[-60:]
    past_month_returns = pred_returns
    
    exec_config = {k: v for pair in config['execution'] for k, v in pair.items()}
            
    results = exec_class(
        exec_config,
        w_previous,
        w_previous,
        mask,
        past_month_returns,
        current_balance,
        alpha = 1.0
    ).execution_process()
    
    exp_sum = 0.2 * exp_sum + 0.8 * results[-1]
    alpha = jax.nn.sigmoid(exp_sum)
    info = np.vstack((info, results))
    alphas.append(1.0)
    
    results = exec_class(
        exec_config,
        w_previous,
        w_previous,
        mask,
        past_month_returns,
        current_balance,
        alpha=float(alpha)
    ).execution_process()

    info = np.vstack((info, results))
    alphas.append(alpha)
    
    index = np.argsort(info[:, 1])[::-1][0]
    
    if info[index][1] >= best_balance:
        best_exp_sum = exp_sum
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
    all_future_returns: pd.DataFrame,
    future_returns: pd.DataFrame,
    w_previous: pd.Series,
    initial_exp_sum: float,
    current_balance: float
) -> tuple[float, float, np.ndarray, pd.Series]:
    
    alphas = []
    info = np.empty((0, 4))
    case = ["max_alpha", "voted_alpha"]
    
    optimizer_config = {k: v for pair in config['optimizer'] for k, v in pair.items()}
    exec_config = {k: v for pair in config['execution'] for k, v in pair.items()}
    
    _, method_weights = optimizer_class(
        optimizer_config,
        market_cap,
        historical_returns,
        pred_returns,
        w_previous[mask].values
    ).get_results()
    
    print(f"{len(method_weights)=}")
    
    exp_sum = initial_exp_sum
    # past_month_returns = historical_returns[-60:]
    past_month_returns = pred_returns
    
    for weights in method_weights:
        portfolios_date['weights'] = 0.0
        portfolios_date.loc[mask, 'weights'] = weights
        w_target = portfolios_date['weights']
        
        results = exec_class(
            exec_config,
            w_previous,
            w_target,
            mask,
            past_month_returns, #future_returns,
            current_balance,
            alpha = 1.0
        ).execution_process()
        
        exp_sum = 0.2 * exp_sum + 0.8 * results[-1]
        alpha = jax.nn.sigmoid(exp_sum)
        info = np.vstack((info, results))
        alphas.append(1.0)
        
        results = exec_class(
            exec_config,
            w_previous,
            w_target,
            mask,
            past_month_returns, #future_returns,
            current_balance,
            alpha=float(alpha)
        ).execution_process()

        info = np.vstack((info, results))
        alphas.append(alpha)
        
    index = np.argsort(info[:, -1])[::-1][0]
    
    best_index = index%2
    best_exp_sum = exp_sum
    best_alpha = float(alphas[index])
    best_results = info[index]
    best_weights = method_weights[index//2]
    best_balance = info[index][1]
    
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
        all_future_returns, # future_returns,
        current_balance,
        best_alpha,
        False
    ).execution_process()
    
    # portfolios_date['weights'] = 0.0
    # portfolios_date.loc[mask, 'weights'] = best_weights
    # w_target = portfolios_date['weights']
    
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
    model_class = MODEL_REGISTRY[config['model'][0]['type']]
    optimizer_class = CVAR_REGISTRY[config['optimizer'][0]['type']]
    exec_class = EXEC_REGISTRY[config['execution'][0]['type']]
    results_class = BaseResults()
    
    print(config['model'][0]['type'], config['optimizer'][0]['type'], config['execution'][0]['type'])

    start_date = first_business_days.iloc[0] + timedelta(days=365*3)
    initial_date = first_business_days[first_business_days <= start_date].iloc[-1]
    index = first_business_days.index[first_business_days == initial_date].to_numpy()[0]
    
    prev_date = first_business_days[index-1]
    date = initial_date.strftime("%Y-%m-%d")
    mask = portfolios[date]['Value'].apply(lambda x: x == 1)

    portfolios[date]["weights"] = 0.0
    count_etf = len(portfolios[date][mask])
    
    portfolios[date].loc[mask, "weights"] = np.ones(count_etf) / count_etf # 1 way
    # portfolios[date].loc[mask, "weights"] = np.array([1/count_etf]*count_etf) # 1 way
    
    # w = np.random.random(count_etf)
    # portfolios[date].loc[mask, "weights"] = w / w.sum() # 2 way
    
    # portfolios[date].loc[mask, "weights"] = np.random.dirichlet(np.ones(count_etf)) # 3 way
    
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
        # if ind >= 20:
        #     print(f"Completed {ind} iterations, stopping")
        #     break
        
        pbar.set_description(f'Portfolio optimization with {ind} month weight rebalancing {day}')
        
        date = day.strftime('%Y-%m-%d')
        future_date = first_business_days.iloc[index+ind+1]
            
        if date in portfolios:
            etfs_list, mask = get_filtered_etfs(portfolios[date]) # 1 way - use etfs from filtered etfs
            # etfs_list, mask = SP500_VARIANTS, portfolios[date]['Key'].isin(SP500_VARIANTS) # 2 way - all etfs that include SPY in their names, somehow related to SP500
            # etfs_list, mask = PURE_SP500_BENCHMARK, portfolios[date]['Key'].isin(PURE_SP500_BENCHMARK) # 3 wat - only SPY, using just benchmark etfs that fully is realted to SP500
            
            startTime = find_start_time(first_business_days, day)
            historical_returns = get_historical_returns(returns_all, etfs_list, day, startTime, window_type)
            historical_ewma_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime, window_type) 
            all_future_returns = get_first_day_month_returns(returns_all, None, day, future_date).drop('Date', axis=1)
            future_returns = get_first_day_month_returns(returns_all, etfs_list, day, future_date).drop('Date', axis=1)
            month_market_cap = get_first_day_month_returns(market_cap, etfs_list, prev_date, day).drop('Date', axis=1)
            
            modelClass = model_class(
                {k: v for pair in config['model'] for k, v in pair.items()},
                etfs_list,
                historical_returns,
                historical_ewma_returns,
                future_returns.values
            )
            
            pred_returns = modelClass.prediction()
            model_metrics = modelClass.evaluate_metrics(pred_returns)
            
            print(f"{type(pred_returns)=}")
            
            exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(
                config,
                optimizer_class,
                exec_class,
                portfolios[date],
                mask,
                month_market_cap.values,
                historical_returns.drop('Date', axis=1).values,
                pred_returns,
                all_future_returns,
                future_returns,
                w_current,
                exp_sum,
                current_balance
            )
            
            print(alpha, future_date, new_balance, pnl, cost, return_value)
            
            results_class.add_date_results(
                step_date=future_date.strftime('%Y-%m-%d'),
                etf_list=etfs_list,
                pred_returns=np.sum(pred_returns, axis=0),
                real_returns=np.sum(future_returns.values, axis=0),
                model_results=list(model_metrics),
                pnl_results=[new_balance, pnl, cost, return_value, alpha]
            )
            
            print(f"{w_current.shape=}", f"{w_target.shape=}")
            
            portfolios[date]['weights'] = w_target
            current_balance = new_balance
            w_current = w_target
            prev_date = day
            
    results_class.transform_results_to_df()
    results_class.save_results(output_folder)
    save_df_dict_to_excel(portfolios, save_path=Path.cwd() / "results" / output_folder, filename='weights.xlsx')

    return portfolios


def main():
        
    first_business_days = pd.read_excel(OUTPUT / "business_dates.xlsx", index_col=0)
    first_business_days = first_business_days['Values']
    
    sheet_names_str = first_business_days.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()[1:]
    
    df_prices = pd.read_csv(INPUT_PATH / "NewClosePrice.csv")
    df_volumes = pd.read_csv(INPUT_PATH / "Volume.csv")
    df_volumes = df_volumes[df_prices.columns]
    
    prices = df_prices.apply(pd.to_numeric, errors="coerce")
    volumes = df_volumes.apply(pd.to_numeric, errors="coerce")
    market_cap = prices * volumes
    
    # portfolios = pd.read_excel(OUTPUT / "new_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 600-700 etfs
    # portfolios = pd.read_excel(OUTPUT / "last_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 220 etfs
    portfolios = pd.read_excel(OUTPUT / "final_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 220 etfs (portfolios + prices same columns)
    
    # returns_all = pd.read_excel(OUTPUT / "etf_returns.xlsx")
    # ewma_returns_all = pd.read_excel(OUTPUT / "etf_ewma.xlsx") # my version - log-returns prices then EMA
    # # ewma_returns_all = pd.read_excel(OUTPUT / "NEW_ewma.xlsx") # Slava asked - EMA of prices then log-returns
    
    # Used by parsing Adjusting Close Price
    returns_all = pd.read_csv(OUTPUT / "new_etf_returns.csv", index_col=0).drop(['Year', 'Month'], axis=1)
    ewma_returns_all = pd.read_csv(OUTPUT / "new_etf_ewma.csv", index_col=0).drop(['Year', 'Month'], axis=1)
    
    returns_all['Date'] = pd.to_datetime(returns_all['Date'])
    returns_all = returns_all.fillna(0)
    ewma_returns_all['Date'] = pd.to_datetime(ewma_returns_all['Date'])
    
    logger.info('Successful file reading')

    model_computation(
        read_yaml(Path.cwd()/"scripts/configs/test.yaml", PARAMS),
        first_business_days,
        market_cap,
        ewma_returns_all,
        returns_all,
        portfolios,
        # output_folder="sp500_benchmark_bayessian"
        # output_folder="sp500_all_bayessian"
        # output_folder="new_bayessian_all_2"
        # output_folder="new_markowitz_all"
        # output_folder="new_black_litterman"
        # output_folder="bayessian_deviation_from_target"
        # output_folder="markowitz_max_decorrelation"
        output_folder="horseshoe_posterior_alpha_95_bayessian_all"
        # output_folder="gaussian_posterior_alpha_99_bayessian_all"
        # output_folder="t_student_posterior_alpha_99_bayessian_all"
        # output_folder="empirical_bayes_posterior_alpha_99_bayessian_all"
        # output_folder="spike_slab_posterior_alpha_99_bayessian_all"
    )

    logger.info('SUCCESS')



if __name__ == "__main__":
    main()    
