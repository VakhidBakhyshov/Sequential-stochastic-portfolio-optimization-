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

from scripts.models.registry import *
from scripts.strategies.registry import *
from scripts.executions.registry import *

from scripts.runs.run import read_yaml, compare_current_with_previous_weights

from scripts.results.results import BaseResults


INPUT_PATH = Path.cwd() / "datasets" / "csv"


PARAMS = [
    'balance',
    'model',
    'strategy',
    'execution'
]   
    

@beartype
def results_by_fitting_model_alpha(
    config: dict[str, Any],
    exec_class: Type[BaseExecution],
    portfolios_date: pd.DataFrame,
    mask: Union[pd.Series, np.ndarray],
    market_cap: np.ndarray,
    historical_returns: np.ndarray,
    historical_prices: np.ndarray,
    pred_returns: np.ndarray,
    future_prices: pd.DataFrame,
    future_returns: pd.DataFrame,
    w_previous: pd.Series,
    initial_exp_sum: float,
    current_balance: float
) -> tuple[float, float, np.ndarray, pd.Series]:
    
    alphas = []
    info = np.empty((0, 4))
    case = ["max_alpha", "voted_alpha"]
    
    exec_config = {k: v for pair in config['execution'] for k, v in pair.items()}
    
    best_balance = -100
    past_month_returns = historical_returns[-60:]
    # past_month_returns = pred_returns
    
    alphas = []
    strategies_weights = []
    info = np.empty((0, 4))
    
    for strategy_class in STRATEGY_REGISTRY.values():
        weights = strategy_class(
            config,
            historical_returns,
            pred_returns,
            market_cap,
            w_previous
        ).get_weights()
        
        exp_sum = initial_exp_sum
        
        portfolios_date['weights'] = 0.0
        portfolios_date.loc[mask, 'weights'] = weights
        w_target = portfolios_date['weights']
        
        results = exec_class(
            exec_config,
            w_previous,
            w_target,
            mask,
            past_month_returns,
            current_balance,
            alpha = 1.0
        ).execution_process()
        
        # print("1", f"{results=}")
            
        exp_sum = 0.2 * exp_sum + 0.8 * results[-1]
        alpha = jax.nn.sigmoid(exp_sum)
        info = np.vstack((info, results))
        strategies_weights.append(weights)
        alphas.append(1.0)
        
        results = exec_class(
            exec_config,
            w_previous,
            w_target,
            mask,
            past_month_returns,
            current_balance,
            float(alpha)
        ).execution_process()
        
        # print("2", f"{results=}")

        info = np.vstack((info, results))
        strategies_weights.append(weights)
        alphas.append(alpha)
            
    index = np.argsort(info[:, -1])[::-1][0]
    best_index = index%2

    best_exp_sum, best_alpha, best_results, best_weights = compare_current_with_previous_weights(
        config=config, exec_class=exec_class, mask=mask,
        historical_returns=historical_returns, pred_returns=pred_returns, w_previous=w_previous,
        initial_exp_sum=initial_exp_sum, current_balance=current_balance,
        best_balance=info[index][1],
        best_exp_sum=exp_sum,
        best_alpha=float(alphas[index]),
        best_results=info[index],
        best_weights=strategies_weights[index]
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
    ).execution_process()
    
    print(f"{best_index=}", f"{np.sum(w_target)=}", f"{case[best_index%2]=}")
    
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
   
    model_class = MODEL_REGISTRY[config['model'][0]['type']]
    exec_class = EXEC_REGISTRY[config['execution'][0]['type']]
    results_class = BaseResults()
    
    print(config['strategy'][0]['type'], config['execution'][0]['type'])

    start_date = first_business_days.iloc[0] + timedelta(days=365*3)
    initial_date = first_business_days[first_business_days <= start_date].iloc[-1]
    index = first_business_days.index[first_business_days == initial_date].to_numpy()[0]
    
    prev_date = first_business_days[index-1]
    date = initial_date.strftime("%Y-%m-%d")
    mask = portfolios[date]['Value'].apply(lambda x: x == 1)

    portfolios[date]["weights"] = 0.0
    count_etf = len(portfolios[date][mask])
    
    # portfolios[date].loc[mask, "weights"] = np.array([1/count_etf]*count_etf) # 1 way
    
    # w = np.random.random(count_etf)
    # w /= w.sum()
    # portfolios[date].loc[mask, "weights"] = w # 2 way
    
    portfolios[date].loc[mask, "weights"] = np.random.dirichlet(np.ones(count_etf)) # 3 way
    
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
        # if ind >= 3:
        #     print(f"Completed {ind} iterations, stopping")
        #     break
        
        pbar.set_description(f'Portfolio optimization with {ind} month weight rebalancing {day}')
        
        date = day.strftime('%Y-%m-%d')
        future_date = first_business_days.iloc[index+ind+1]
            
        if date in portfolios:
            etfs_list, mask = get_filtered_etfs(portfolios[date])
            startTime = find_start_time(first_business_days, day)
            historical_prices = get_first_day_month_returns(df_prices, etfs_list, prev_date, day).drop('Date', axis=1)
            future_prices = get_first_day_month_returns(df_prices, etfs_list, day, future_date).drop('Date', axis=1)
            
            historical_returns = get_historical_returns(returns_all, etfs_list, day, startTime, window_type).drop('Date', axis=1)
            historical_ewma_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime, window_type).drop('Date', axis=1)
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
            
            exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(
                config,
                exec_class,
                portfolios[date],
                mask,
                month_market_cap.values,
                historical_returns.values,
                historical_prices.values,
                pred_returns,
                future_prices,
                future_returns,
                w_current,
                exp_sum,
                current_balance
            )
            
            print(alpha, future_date, new_balance, pnl, cost, return_value)
            
            results_class.add_date_results(
                step_date=future_date.strftime('%Y-%m-%d'),
                etf_list=etfs_list,
                pred_returns=None,
                real_returns=np.sum(future_prices.values, axis=0),
                model_results=None,
                pnl_results=[new_balance, pnl, cost, return_value, alpha]
            )
            
            portfolios[date]['weights'] = w_target.values
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
    
    df_prices = pd.read_csv(INPUT_PATH / "NewClosePrice.csv")
    df_volumes = pd.read_csv(INPUT_PATH / "Volume.csv")
    df_volumes = df_volumes[df_prices.columns]
    
    etf_list = df_prices.columns.drop('Date').sort_values(ascending=True)

    prices = df_prices.apply(pd.to_numeric, errors="coerce").drop('Date', axis=1).fillna(0)
    volumes = df_volumes.apply(pd.to_numeric, errors="coerce").drop('Date', axis=1).fillna(0)
    market_cap = prices * volumes
    market_cap = market_cap
    market_cap['Date'] = df_prices['Date']
    
    # Used by parsing Adjusting Close Price
    returns_all = pd.read_csv(OUTPUT / "new_etf_returns.csv")
    ewma_returns_all = pd.read_csv(OUTPUT / "new_etf_ewma.csv")
    
    returns_all['Date'] = pd.to_datetime(returns_all['Date'])
    returns_all = returns_all.fillna(0)
    ewma_returns_all['Date'] = pd.to_datetime(ewma_returns_all['Date'])
    
    logger.info('Successful file reading')

    first_business_days = pd.read_excel(OUTPUT / "business_dates.xlsx", index_col=0)
    first_business_days = first_business_days['Values']
    
    sheet_names_str = first_business_days.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()[1:]
    
    # portfolios = pd.read_excel(OUTPUT / "new_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 600-700 etfs
    portfolios = pd.read_excel(OUTPUT / "last_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price # around 220 etfs

    model_computation(
        read_yaml(Path.cwd()/"scripts/configs/strategy.yaml", PARAMS),
        first_business_days,
        market_cap,
        list(etf_list),
        returns_all,
        ewma_returns_all,
        df_prices,
        portfolios,
        output_folder="combined_strategy_run"
    )

    logger.info('SUCCESS')



if __name__ == "__main__":
    main()    
