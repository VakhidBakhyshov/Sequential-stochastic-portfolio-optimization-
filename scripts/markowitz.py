import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import numpy as np
import pandas as pd

np.random.seed(42)

from scipy.optimize import minimize
from tqdm import tqdm
from loguru import logger
from beartype import beartype

from scripts.bayessian_model import *


@beartype
def get_meanReturns_covMatrix(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.mean(returns, axis=0), np.cov(returns, rowvar=False)


@beartype
def portfolio_perfomance(weights: np.ndarray, meanReturns: np.ndarray, covMatrix: np.ndarray) -> tuple[float, float]:
    weights = weights.reshape(-1, 1)
    expectedReturns = np.dot(weights.T, meanReturns.reshape(-1, 1))
    variance = np.dot(weights.T, np.dot(covMatrix, weights))
    return expectedReturns.flatten()[0], variance.flatten()[0]


@beartype
def function_negative_sharpe(weights: np.ndarray, meanReturns: np.ndarray, covMatrix: np.ndarray) -> float:
    expectedReturns, variance = portfolio_perfomance(weights, meanReturns, covMatrix)
    variance = np.sqrt(max(variance, 1e-10))
    return -expectedReturns / variance


@beartype
def maximize_sharpe(meanReturns: np.ndarray, covMatrix: np.ndarray, bound = (0, None), max_iter: int = 1000) -> tuple[float, np.ndarray]:
    args = (meanReturns, covMatrix)
    count_etf = covMatrix.shape[0]
    bounds = [bound] * count_etf
    weights = np.array([1/count_etf]*count_etf)
    constraints = {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}
    # result = minimize(fun=function_negative_sharpe, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': max_iter})
    result = minimize(fun=function_negative_sharpe, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = np.where(result.x >= 1e-3, result.x, 0)
    return result.fun, weights


@beartype
def function_variance(weights: np.ndarray, meanReturns: np.ndarray, covMatrix: np.ndarray) -> float:
    return portfolio_perfomance(weights, meanReturns, covMatrix)[1]


@beartype
def minimize_variance(meanReturns: np.ndarray, covMatrix: np.ndarray, bound = (0, None), max_iter: int = 1000) -> tuple[float, np.ndarray]:
    args = (meanReturns, covMatrix)
    count_etf = covMatrix.shape[0]
    bounds = [bound] * count_etf
    weights = np.array([1/count_etf]*count_etf)
    constraints = {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}
    # result = minimize(fun=function_variance, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': max_iter})
    result = minimize(fun=function_variance, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = np.where(result.x >= 1e-3, result.x, 0)
    return result.fun, weights


@beartype
def best_fitting_model(
    meanReturns: np.ndarray,
    covMatrix: np.ndarray,
    mask: pd.Series,
    portfolios_date: pd.DataFrame,
    future_returns: pd.DataFrame,
    w_current: pd.Series,
    exp_sum: float,
    current_balance: float,
    optimization_method: list[Literal["max_sharpe", "min_var"]] = ["max_sharpe", "min_var"]
) -> tuple[float, float, np.ndarray, pd.Series]:

    case, alphas = [], []
    info = np.empty((0, 4))
    best_weights = np.empty((0, portfolios_date.shape[0]))
    
    for type in optimization_method:
        if type == "max_sharpe":
            _, weights = maximize_sharpe(meanReturns, covMatrix)
        elif type == "min_var":
            _, weights = minimize_variance(meanReturns, covMatrix)
        else:
            raise ValueError(f"Unknown type: {type}")
            
        case.append(type)
        case.append(f"{type} reduce alpha")
    
        portfolios_date['weights'] = 0.0
        portfolios_date.loc[mask, 'weights'] = weights
        w_target = portfolios_date['weights']
        
        results = execution_process(w_current, w_target, mask, future_returns, current_balance, 1.0)
        alphas.append(1.0)
        info = np.vstack((info, results))
        best_weights = np.vstack((best_weights, w_target.to_numpy()))
        
        new_exp_sum = 0.2 * exp_sum + 0.8 * results[-1]
        alpha = jax.nn.sigmoid(exp_sum)
        alphas.append(alpha)
        info = np.vstack((info, execution_process(w_current, w_target, mask, future_returns, current_balance, float(alpha))))
        best_weights = np.vstack((best_weights, w_target.to_numpy()))
        
    index = np.argsort(info[:, -1])[::-1][0]
    print(f"case = {case[index]}")
    
    return new_exp_sum, float(alphas[index]), info[index], pd.Series(best_weights[index])


# @beartype
# def function_mean(weights: np.ndarray, meanReturns: np.ndarray, covMatrix: np.ndarray) -> float:
#     return portfolio_perfomance(weights, meanReturns, covMatrix)[0]


# @beartype
# def efficient_optimization(meanReturns: np.ndarray, covMatrix: np.ndarray, targetReturn: float, count_strategies: int, bound = (1e-2, 0.5)) -> tuple[float, np.ndarray]:
#     args = (meanReturns, covMatrix)
#     bounds = (bound for _ in range(count_strategies))
#     weights = np.array([1/count_strategies]*count_strategies)
#     constraints = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x:  function_mean(x, meanReturns, covMatrix) - targetReturn})
#     result = minimize(fun=function_variance, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
#     return result.fun, result.x

        
# @beartype
# def efficient_frontier(min_variance_returns: float, max_sharpe_returns: float, meanReturns: pd.Series, covMatrix: pd.DataFrame, count_strategies: int) -> tuple[np.ndarray, list]:
#     results = []
#     # target_returns = np.linspace(min_variance_returns, max_sharpe_returns, 20)
#     target_returns = np.linspace(max_sharpe_returns, min_variance_returns, 20)
#     for targetReturn in target_returns:
#         results.append(efficient_optimization(meanReturns, covMatrix, targetReturn, count_strategies)[0])
#     return target_returns, results
    

@beartype
def markowitz_model_computation(
    first_business_days: pd.Series,
    ewma_returns_all: pd.DataFrame,
    returns_all: pd.DataFrame,
    portfolios: dict,
    optimization_method: list[Literal["max_sharpe", "min_var"]] = ["max_sharpe", "min_var"],
    initial_balance: float = 1000.0
) -> tuple[dict, list]:

    start_date = first_business_days.iloc[0] + timedelta(days=365*3)
    initial_date = first_business_days[first_business_days <= start_date].iloc[-1]
    index = first_business_days.index[first_business_days == initial_date].to_numpy()[0]
    
    date = initial_date.strftime("%Y-%m-%d")
    mask = portfolios[date]['Value'].apply(lambda x: x == 1)

    portfolios[date]["weights"] = 0.0
    count_etf = len(portfolios[date][mask])
    portfolios[date].loc[mask, "weights"] = np.array([1/count_etf]*count_etf)
    w_current = portfolios[date]["weights"]

    exp_sum = 0.0
    pnl_info = []
    current_balance = initial_balance
    pnl_info.append([date, current_balance, 0.0, 0.0, 0.0])
    
    pbar = tqdm(enumerate(first_business_days.iloc[index:-1]), desc='Portfolio optimization with month weight rebalancing')
    for ind, day in pbar:
        # if ind >= 20:
        #     print(f"Completed {ind} iterations, stopping")
        #     break
        
        date = day.strftime('%Y-%m-%d')
        future_date = first_business_days.iloc[index+ind+1]
        
        pbar.set_description(f'Portfolio optimization with {ind} month weight rebalancing {day}')
            
        if date in portfolios:
            etfs_list, mask = get_filtered_etfs(portfolios[date])
            startTime = find_start_time(first_business_days, day)
            
            # historical_returns = get_historical_returns(returns_all, etfs_list, day, startTime)
            historical_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime)
            
            meanReturns, covMatrix = get_meanReturns_covMatrix(historical_returns.drop('Date', axis=1).values)
            future_returns = get_first_day_month_returns(returns_all, etfs_list, day, future_date).drop('Date', axis=1)
            
            exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = best_fitting_model(
                meanReturns, covMatrix,
                mask, portfolios[date],
                future_returns, w_current,
                exp_sum, current_balance,
                optimization_method
            )
            
            print(alpha, future_date, new_balance, pnl, cost, return_value)
            pnl_info.append([future_date.strftime('%Y-%m-%d'), new_balance, pnl, cost, return_value])
            
            current_balance = new_balance
            w_current = w_target

    df = pd.DataFrame(np.array(pnl_info, dtype=object), columns=['Date', 'Balance', 'PnL', 'Cost', 'Returns'])
    df.to_csv(OUTPUT / "pnl-2.csv", index=False)
    logger.info(f'pnls info successfully saved to {OUTPUT / "pnl-2.csv"}')
    
    save_df_dict_to_excel(portfolios, save_path=OUTPUT, filename='weights_rebalance-2.xlsx')

    return portfolios, pnl_info


def main():
        
    first_business_days = pd.read_excel(OUTPUT / "business_dates.xlsx", index_col=0)
    first_business_days = first_business_days['Values']
    
    sheet_names_str = first_business_days.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()[1:]
    
    # portfolios = pd.read_excel(OUTPUT / "new_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 600-700 etfs
    portfolios = pd.read_excel(OUTPUT / "last_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price # around 220 etfs
    
    # Used by parsing Adjusting Close Price
    returns_all = pd.read_csv(OUTPUT / "new_etf_returns.csv")
    ewma_returns_all = pd.read_csv(OUTPUT / "new_etf_ewma.csv")
    
    returns_all['Date'] = pd.to_datetime(returns_all['Date'])
    ewma_returns_all['Date'] = pd.to_datetime(ewma_returns_all['Date'])
    
    returns_all = returns_all.fillna(0)
    
    logger.info('Successful file reading')

    markowitz_model_computation(
        first_business_days,
        ewma_returns_all,
        returns_all,
        portfolios
    )

    logger.info('SUCCESS')



if __name__ == "__main__":
    main()
