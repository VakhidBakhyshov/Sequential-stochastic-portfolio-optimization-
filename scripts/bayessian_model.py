import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import numpy as np
import pandas as pd

np.random.seed(42)

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from beartype import beartype
from datetime import timedelta
from typing import Union, Literal
from scipy.optimize import minimize
from scipy.stats import gaussian_kde, spearmanr


from scripts.past.parser import save_df_dict_to_excel

INPUT_PATH = Path.cwd() / "datasets" / "csv"
OUTPUT = INPUT_PATH.parent / "excel"


@beartype
def get_filtered_etfs(portfolios_date: pd.DataFrame) -> tuple[list, pd.Series]:
    mask = portfolios_date['Value'].apply(lambda x: x == 1)
    filtered_etfs = portfolios_date[mask]
    return filtered_etfs['Key'].tolist(), mask


@beartype
def find_start_time(dates: pd.Series, day: pd.Timestamp) -> pd.Timestamp:
    target = day - pd.DateOffset(years=1)
    idx = (dates - target).abs().idxmin() # closest available date
    return dates.loc[idx]


@beartype
def get_first_day_month_returns(
    returns_df: pd.DataFrame,
    filtered_etfs: Union[list, None],
    startTime: Union[str, pd.Timestamp, None],
    endTime: Union[str, pd.Timestamp, None]
) -> pd.DataFrame:
    
    if startTime and endTime:
        month_return = returns_df[(returns_df['Date'] >= startTime) & (returns_df['Date'] < endTime)]
    
    elif startTime:
        month_return = returns_df[returns_df['Date'] >= startTime]
        
    elif endTime:
        month_return = returns_df[returns_df['Date'] < endTime]
        
    else:
        raise ValueError(f'Two dates startTime and endTime both None, but we need that one of them will be at least not None')
    
    if isinstance(filtered_etfs, list):
        return month_return[['Date']+filtered_etfs]
    else:
        return month_return


# version where day - startTime is around 3 years
@beartype
def get_historical_returns(ewma_returns: pd.DataFrame, etfs_list: list[str], day: pd.Timestamp, startTime: pd.Timestamp) -> pd.DataFrame:
    # startTime = find_start_time(business_days, day)
    return get_first_day_month_returns(ewma_returns, etfs_list, startTime, day)


def get_returns_with_distribution(
    returns_etf: pd.Series,
    mu_i: float,
    n_samples: int,
    mean_return_type: Literal["sampled", "historical"] = "sampled",
    dist: Literal["normal", "t-student", "laplace", "lognormal"] = "normal",
    df: int = 5  # degrees of freedom for t-distribution
) -> tuple[float, float, np.ndarray]:
    
    sigma_i = returns_etf.std()
    
    if mean_return_type == "historical": # 1 way - calculate posterior_mean_variance from historical returns
        mean_return = np.mean(returns_etf)
        r_i = returns_etf
        
    elif mean_return_type == "sampled": # 2 way - calculate posterior_mean_variance from sampled returns
        if dist == "normal":
            r_i = np.random.normal(mu_i, sigma_i, n_samples)

        elif dist == "t-student":
            r_i = mu_i + sigma_i * np.random.standard_t(df, size=n_samples)

        elif dist == "laplace":
            r_i = np.random.laplace(mu_i, sigma_i / np.sqrt(2), n_samples)

        elif dist == "lognormal":
            r_i = np.random.lognormal(mean=mu_i, sigma=sigma_i, size=n_samples)

        else:
            raise ValueError(f"Unsupported distribution: {dist}")

        mean_return = np.mean(r_i)
        
    else:
        raise ValueError(f"Unsupported mean return type: {mean_return_type}")

    return sigma_i**2, mean_return, r_i


def sample_posterior(
    returns: np.ndarray,
    n_samples: int,
    posterior_type: Literal["gaussian", "t_student", "laplace", "horseshoe", "spike_slab"],
    prior_mean: float,
    prior_var: float,
) -> np.ndarray:

    sample_mean = np.mean(returns)
    sample_var = np.var(returns)

    if posterior_type == "gaussian":
        tau_post = 1 / (1 / prior_var + n_samples / sample_var)
        mu_post = tau_post * (prior_mean / prior_var + n_samples * sample_mean / sample_var)
        return np.random.normal(mu_post, np.sqrt(tau_post), size=n_samples)

    elif posterior_type == "t_student":
        df = 5
        scale = np.sqrt(sample_var)
        return sample_mean + scale * np.random.standard_t(df, size=n_samples)

    elif posterior_type == "laplace":
        b = np.sqrt(sample_var / 2)
        return np.random.laplace(sample_mean, b, size=n_samples)

    elif posterior_type == "horseshoe":
        tau = np.abs(np.random.standard_cauchy())  # global shrinkage
        lam = np.abs(np.random.standard_cauchy())  # local shrinkage
        shrinkage = tau * lam

        mu_post = sample_mean * shrinkage
        sigma_post = np.sqrt(sample_var) * shrinkage

        return np.random.normal(mu_post, sigma_post + 1e-6, size=n_samples)

    elif posterior_type == "spike_slab":
        p = 0.5  # probability of being "active"
        is_active = np.random.binomial(1, p)

        if is_active:
            return np.random.normal(sample_mean, np.sqrt(sample_var), size=n_samples)
        else:
            return np.zeros(n_samples)

    else:
        raise ValueError(f"Unknown posterior type: {posterior_type}")


@beartype
def get_return_scenatios_by_posteriors(
    etfs_list: list,
    historical_returns: pd.DataFrame,
    historical_ewma_returns: pd.DataFrame,
    n_samples: int,
    mean_return_type: Literal["sampled", "historical"] = "sampled",
    dist: Literal["normal", "t-student", "laplace", "lognormal"] = "normal",
    posterior_type: Literal["gaussian", "t_student", "laplace", "horseshoe", "spike_slab"] = "gaussian"
) -> np.ndarray:
    
    @beartype
    def calculate_prior_mean_variance(historical_returns: pd.DataFrame) -> tuple[float, float]:
        historical_returns = historical_returns[historical_returns.columns[1:]]
        prior_mean_0, squared_prior_variance_0 = historical_returns.mean().mean(), (historical_returns.std() ** 2).mean()    
        return prior_mean_0, squared_prior_variance_0

    # n_samples = len(historical_returns)
    posterior_draws = np.empty((n_samples, len(etfs_list)))

    prior_mean_0, squared_prior_variance_0 = calculate_prior_mean_variance(historical_ewma_returns)
    mu_i = np.random.normal(prior_mean_0, np.sqrt(squared_prior_variance_0), len(etfs_list))
    
    for ind, etf in enumerate(etfs_list):
        # squared_sigma_i, mean_return, r_i = get_returns_with_distribution(historical_returns[etf], mu_i[ind], n_samples, mean_return_type, dist)
        # squared_tau_post, mu_post_i = calculate_posterior_mean_variance(prior_mean_0, squared_prior_variance_0, squared_sigma_i, mean_return, n_samples)
        # posterior_draws[:, ind] = np.random.normal(mu_post_i, np.sqrt(squared_tau_post), size=n_samples)
        
        r_i = get_returns_with_distribution(historical_returns[etf], mu_i[ind], n_samples, mean_return_type, dist)[-1]
        posterior_draws[:, ind] = sample_posterior(r_i, n_samples, posterior_type, prior_mean_0, squared_prior_variance_0)

    return posterior_draws


@beartype
def make_positive_semifinite_matrix(cov_matrix: np.ndarray, calculation_type: Literal["fast", "long"] = "fast") -> np.ndarray:
    @beartype
    def B(theta: np.ndarray):
        cosine_B = np.cos(theta)
        cosine_B_shifted = np.c_[cosine_B, np.ones(cosine_B.shape[0])]
        sin_B = np.sin(theta)
        sin_B_shifted = np.c_[np.ones(sin_B.shape[0]), sin_B]
        sin_B_shifted_cumprod = np.cumprod(sin_B_shifted, axis=-1)
        return cosine_B_shifted * sin_B_shifted_cumprod
    
    @beartype
    def reshape_theta_to_appropriate_size(theta: np.ndarray, n: int):
        theta_size = theta.shape[0]
        rows_dim = int(n)
        cols_dim = int(theta_size / rows_dim)
        return theta.reshape(rows_dim, cols_dim)
    
    @beartype
    def l2_norm(theta: np.ndarray, C: np.ndarray):
        theta = reshape_theta_to_appropriate_size(theta=theta, n=C.shape[0])
        B_ = B(theta)
        C_approx = np.dot(B_, B_.T)
        return np.linalg.norm(C_approx - C, ord = 'fro')
    
    @beartype
    def eigenvalue_clipping(matrix: np.ndarray, delta: float = 1e-8, jitter: float = 1e-6) -> np.ndarray:
        # eigenvalues, eigenvectors = np.linalg.eig(matrix)
        # positive_eigenvalues = np.where(eigenvalues < 0, delta, eigenvalues)
        # Q, Λ = eigenvectors, np.diag(positive_eigenvalues)
        # return np.dot(np.dot(Q, Λ), Q.T)
    
        corr_matrix = (matrix + matrix.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        # eigenvalues = np.clip(eigenvalues, delta, None)
        eigenvalues = np.where(eigenvalues < delta, delta, eigenvalues)
        corr_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), eigenvectors.T)
        corr_matrix += jitter * np.eye(corr_matrix.shape[0])
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        return corr_matrix

    
    @beartype
    def check_matrix_positive_semifinite(matrix: np.ndarray) -> np.bool:
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues >= 0)
    
    if check_matrix_positive_semifinite(cov_matrix):
        return cov_matrix
    
    else:
        if calculation_type == "fast":
            new_matrix = eigenvalue_clipping(cov_matrix)
        
        elif calculation_type == "long":
            n = cov_matrix.shape[0]
            theta = np.random.rand(n * (n - 1))
            res = minimize(fun=l2_norm, x0 =theta, args=cov_matrix, method="BFGS", options={'maxiter': 100})
            theta_opt = res.x
            theta_opt_reshaped = reshape_theta_to_appropriate_size(theta=theta_opt, n=n)
            new_matrix = np.dot(B(theta_opt_reshaped), B(theta_opt_reshaped).T)
        
        else:
            raise ValueError(f"Unsupported type: {calculation_type}")
        
        # print(check_matrix_positive_semifinite(new_matrix))
        return new_matrix


@beartype
def calculate_multivariate_shock_matrix(cov_matrix: np.ndarray, n_samples: int) -> np.ndarray:
    def safe_cholesky(matrix, max_tries=5):
        jitter = 1e-8
        for i in range(max_tries):
            try:
                return np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                matrix = matrix + jitter * np.eye(matrix.shape[0])
                jitter *= 10
        raise np.linalg.LinAlgError("Matrix not PD even after jitter.")

    # L = np.linalg.cholesky(cov_matrix)
    L = safe_cholesky(cov_matrix)
    n_assets = cov_matrix.shape[0]
    Z = np.random.normal(loc=0, scale=1, size=(n_samples, n_assets))
    epsilon = np.dot(Z, L.T)
    return epsilon


@beartype
def evaluate_metrics(return_scenarios: np.ndarray, future_returns: np.ndarray) -> tuple[float, float, float, float, float]:
    waic_var, waic_lppd = [], []
    rank_ics, mae_list = [], []
    coverage_hits, log_pred_scores = [], []

    print(return_scenarios.shape, future_returns.shape)

    n_samples, n_assets = return_scenarios.shape

    for t in range(n_samples):
        y_true = future_returns[t]           # (n_assets,)
        y_draws = return_scenarios[t]                # using all draws as predictive dist

        # --- Log predictive density (via KDE approximation) ---
        log_probs = []
        pointwise_loglik = []

        for i in range(n_assets):
            if y_draws.ndim == 1:
                samples = y_draws
            else:
                samples = y_draws[:, i]

            kde = gaussian_kde(samples)
            log_p = np.log(kde.evaluate(y_true[i])[0] + 1e-12)

            log_probs.append(log_p)
            pointwise_loglik.append(log_p)

        log_pred_scores.append(np.mean(log_probs))

        # --- WAIC components ---
        pointwise_loglik = np.array(pointwise_loglik)
        waic_lppd.append(np.sum(pointwise_loglik))
        waic_var.append(np.sum(np.var(pointwise_loglik)))

        # --- Rank IC (Spearman across assets) ---
        rho, _ = spearmanr(y_true, return_scenarios.mean(axis=0))
        rank_ics.append(rho if not np.isnan(rho) else 0.0)

        # --- MAE ---
        mae = np.mean(np.abs(return_scenarios.mean(axis=0) - y_true))
        mae_list.append(mae)

        # --- 95% credible interval coverage ---
        lower = np.percentile(return_scenarios, 2.5, axis=0)
        upper = np.percentile(return_scenarios, 97.5, axis=0)

        hits = (y_true >= lower) & (y_true <= upper)
        coverage_hits.extend(hits.astype(int))

    log_score = float(np.mean(log_pred_scores))
    waic = float(-2 * (np.sum(waic_lppd) - np.sum(waic_var)))
    rank_ic = float(np.mean(rank_ics))
    mae = float(np.mean(mae_list))
    coverage = float(np.mean(coverage_hits))

    return log_score, waic, rank_ic, mae, coverage


@beartype
def minimize_cvar(weights: np.ndarray, return_scenarios: np.ndarray, w_previous: np.ndarray, alpha: float = 0.99, turnover_penalty: float = 0.75) -> float:
    portfolio_returns = np.dot(return_scenarios, weights)
    s_samples = portfolio_returns.shape[0]
    sorted_losses = np.sort(-portfolio_returns)[::-1]
    var_index = int(s_samples * (1 - alpha))
    var = sorted_losses[var_index]
    cvar_loss = sorted_losses[:var_index+1].mean()
        
    # turnover = np.sum((weights - w_previous)**2) # l2-penalty more heavy condition
    turnover = np.sum(np.abs(weights - w_previous)) # l1-penalty
    
    transaction_cost = turnover_penalty * turnover
    return cvar_loss - transaction_cost


@beartype
def cvar_optimization(return_scenarios: np.ndarray, w_previous: np.ndarray, bound: tuple = (0, None), max_iter: int = 1000) -> tuple[float, np.ndarray]:
    args = (return_scenarios, w_previous)
    count_etf = return_scenarios.shape[1]
    bounds = [bound] * count_etf
    weights = np.array([1/count_etf]*count_etf)
    constraints = {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}
    # result = minimize(fun=minimize_cvar, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': max_iter})
    result = minimize(fun=minimize_cvar, x0 = weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = np.where(result.x >= 1e-4, result.x, 0)
    return result.fun, weights


@beartype
def execution_process(w_current: pd.Series, w_target: pd.Series, mask: pd.Series, future_returns: pd.DataFrame, current_balance: float, alpha: float) -> tuple[float, float, float, float]:
    @beartype
    def calculate_exec_weights_turnover_transaction_costs(w_current: pd.Series, w_target: pd.Series, alpha: float = 1.0) -> tuple[float, float, pd.Series]:
        w_exec = w_current + alpha * (w_target - w_current)
        turnover = abs(w_exec - w_current).sum()
        c_bps = 0.001
        cost = c_bps * turnover
        return turnover, cost, w_exec
    
    @beartype
    def calculate_pnl(w_exec: pd.Series, future_returns: pd.DataFrame, cost: float) -> float:
        simple_returns = np.exp(future_returns.copy()) - 1 # we have log-returns that's why
        # simple_returns = future_returns.copy()
        daily_returns = np.dot(simple_returns.values, w_exec.values)
        monthly_return = np.prod(1 + daily_returns) - 1
        return monthly_return - cost

    @beartype
    def calculate_portfolio_value_return_value(current_balance: float, pnl: float) -> tuple[float, float]:
        new_balance = current_balance * (1 + pnl)
        return_value = (new_balance / current_balance) - 1
        return new_balance, return_value

    _, cost, w_exec = calculate_exec_weights_turnover_transaction_costs(w_current, w_target, alpha)
    pnl = calculate_pnl(w_exec[mask], future_returns, cost)
    new_balance, return_value = calculate_portfolio_value_return_value(current_balance, pnl)
    return new_balance, pnl, cost, return_value


@beartype
def results_by_fitting_model_alpha(return_scenarios: np.ndarray,
    mask: pd.Series,
    portfolios_date: pd.DataFrame,
    future_returns: pd.DataFrame,
    w_current: pd.Series,
    exp_sum: float,
    current_balance: float
) -> tuple[float, float, np.ndarray, pd.Series]:
    
    alphas = []
    info = np.empty((0, 4))
    case = ["max_alpha", "voted_alpha"]
    
    _, weights = cvar_optimization(return_scenarios, w_current[mask].values)
    portfolios_date['weights'] = 0.0
    portfolios_date.loc[mask, 'weights'] = weights
    w_target = portfolios_date['weights']
    
    results = execution_process(w_current, w_target, mask, future_returns, current_balance, 1.0)
    exp_sum = 0.2 * exp_sum + 0.8 * results[-1]
    alpha = jax.nn.sigmoid(exp_sum)
    info = np.vstack((info, results))
    alphas.append(1.0)

    info = execution_process(w_current, w_target, mask, future_returns, current_balance, float(alpha))
    info = np.vstack((info, results))
    alphas.append(alpha)
    
    index = np.argsort(info[:, -1])[::-1][0]
    print(f"sum weights = {np.sum(weights)}, case = {case[index]}")
    
    return exp_sum, float(alphas[index]), info[index], w_target
    

@beartype
def bayessian_model_computation(
    first_business_days: pd.Series,
    ewma_returns_all: pd.DataFrame,
    returns_all: pd.DataFrame,
    portfolios: dict,
    calculation_type: Literal["fast", "long"] = "fast",
    mean_return_type: Literal["sampled", "historical"] = "sampled",
    dist: Literal["normal", "t-student", "laplace", "lognormal"] = "t-student",
    posterior_type: Literal["gaussian", "t_student", "laplace", "horseshoe", "spike_slab"] = "horseshoe",
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

    alpha = 1.0
    exp_sum = 0.0
    current_balance = initial_balance
    
    pnl_info = []
    pnl_info.append([date, current_balance, 0.0, 0.0, 0.0])
    
    model_metrics = []
    model_metrics.append([date, 0, 0, 0, 0, 0])
    
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
            historical_returns = get_historical_returns(returns_all, etfs_list, day, startTime)
            # historical_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime)
            
            historical_ewma_returns = get_historical_returns(ewma_returns_all, etfs_list, day, startTime) 
            
            future_returns = get_first_day_month_returns(returns_all, etfs_list, day, future_date).drop('Date', axis=1)
            # n_samples = len(historical_returns)
            n_samples = len(future_returns)
            posterior_draws = get_return_scenatios_by_posteriors(etfs_list, historical_returns, historical_ewma_returns, n_samples, mean_return_type, dist, posterior_type)
            
            cov_matrix = np.cov(historical_ewma_returns[etfs_list].values, rowvar=False) 
            stds = np.sqrt(np.diag(cov_matrix))
            
            D_inr_sqrt = np.diag(1 / stds)
            corr_matrix = np.dot(np.dot(D_inr_sqrt, cov_matrix), D_inr_sqrt)
            corr_matrix = make_positive_semifinite_matrix(corr_matrix, calculation_type)
            
            D_inr_sqrt = np.diag(stds)
            cov_matrix = np.dot(np.dot(D_inr_sqrt, corr_matrix), D_inr_sqrt)
            epsilon = calculate_multivariate_shock_matrix(cov_matrix, n_samples)       
            return_scenarios = posterior_draws + epsilon
            
            log_score, waic, rank_ic, mae, coverage = evaluate_metrics(return_scenarios, future_returns.values)
            # print(log_score, waic, rank_ic, mae, coverage)
            model_metrics.append([future_date.strftime('%Y-%m-%d'), log_score, waic, rank_ic, mae, coverage])
            
            exp_sum, alpha, (new_balance, pnl, cost, return_value), w_target = results_by_fitting_model_alpha(
                return_scenarios,
                mask,
                portfolios[date],
                future_returns,
                w_current,
                exp_sum,
                current_balance
            )
            
            print(alpha, future_date, new_balance, pnl, cost, return_value)
            pnl_info.append([future_date.strftime('%Y-%m-%d'), new_balance, pnl, cost, return_value])
            
            current_balance = new_balance
            w_current = w_target

    df = pd.DataFrame(np.array(pnl_info, dtype=object), columns=['Date', 'Balance', 'PnL', 'Cost', 'Returns'])
    df.to_csv(OUTPUT / "pnl-1.csv", index=False)
    logger.info(f'pnls info successfully saved to {OUTPUT / "pnl-1.csv"}')
    
    df_metrics = pd.DataFrame(np.array(model_metrics, dtype=object), columns=['Date', 'log_score', 'waic', 'rank_ic', 'mae', "coverage_95"])
    df_metrics.to_csv(OUTPUT / "model_metrics.csv", index=False)
    logger.info(f'pnls info successfully saved to {OUTPUT / "model_metrics.csv"}')
    
    save_df_dict_to_excel(portfolios, save_path=OUTPUT, filename='weights_rebalance-1.xlsx')

    return portfolios, pnl_info


def main():
        
    first_business_days = pd.read_excel(OUTPUT / "business_dates.xlsx", index_col=0)
    first_business_days = first_business_days['Values']
    
    sheet_names_str = first_business_days.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()[1:]
    
    # portfolios = pd.read_excel(OUTPUT / "filtered_weights.xlsx", sheet_name=sheet_names_str) # Used incorrect old version ClosePrice
    # portfolios = pd.read_excel(OUTPUT / "new_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price around 600-700 etfs
    portfolios = pd.read_excel(OUTPUT / "last_filtered_weights.xlsx", sheet_name=sheet_names_str) # Used Adjusted close price # around 220 etfs
    
    # returns_all = pd.read_excel(OUTPUT / "etf_returns.xlsx")
    # ewma_returns_all = pd.read_excel(OUTPUT / "etf_ewma.xlsx") # my version - log-returns prices then EMA
    # # ewma_returns_all = pd.read_excel(OUTPUT / "NEW_ewma.xlsx") # Slava asked - EMA of prices then log-returns
    
    # Used by parsing Adjusting Close Price
    returns_all = pd.read_csv(OUTPUT / "new_etf_returns.csv")
    ewma_returns_all = pd.read_csv(OUTPUT / "new_etf_ewma.csv")
    
    returns_all['Date'] = pd.to_datetime(returns_all['Date'])
    ewma_returns_all['Date'] = pd.to_datetime(ewma_returns_all['Date'])
    
    returns_all = returns_all.fillna(0)
    
    logger.info('Successful file reading')

    bayessian_model_computation(
        first_business_days,
        ewma_returns_all,
        returns_all,
        portfolios
    )

    logger.info('SUCCESS')



if __name__ == "__main__":
    main()    
