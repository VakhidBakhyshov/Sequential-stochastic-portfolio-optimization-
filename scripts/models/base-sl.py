import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from typing import Any
from beartype import beartype
from scipy.stats import gaussian_kde, spearmanr


class BaseModel():
    def __init__(
        self,
        config: dict[str, Any],
        etfs_list: list,
        historical_returns: pd.DataFrame,
        historical_ewma_returns: pd.DataFrame,
        future_returns: np.ndarray
    ):
        
        self.config = config
        self.etfs_list = etfs_list
        self.returns = historical_returns
        self.ewma_returns = historical_ewma_returns
        self.future_returns = np.asarray(future_returns, dtype=float)
        
    @beartype
    def prediction(self):
        pass

    @beartype
    def evaluate_metrics(self, pred_returns: np.ndarray) -> tuple:
        self.waic_var, self.waic_lppd = [], []
        self.rank_ics, self.mae_list = [], []
        self.coverage_hits, self.log_pred_scores = [], []
        
        self.sign_precisions = []
        self.r_squareds = []
        self.rmse_list = []
        self.predicted_means = []
        self.true_means = []
        self.sign_precision_true_positives = []

        n_samples, n_assets = pred_returns.shape
        
        # print(f"{self.future_returns.shape=}", f"{pred_returns.shape}")

        for t in range(n_samples):
            y_true = self.future_returns[t]           # (n_assets,)
            y_draws = pred_returns[t]        # using all draws as predictive dist
            pred_mean = pred_returns.mean(axis=0)

            # Log predictive density
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

            self.log_pred_scores.append(np.mean(log_probs))

            # WAIC
            pointwise_loglik = np.array(pointwise_loglik)
            self.waic_lppd.append(np.sum(pointwise_loglik))
            self.waic_var.append(np.sum(np.var(pointwise_loglik)))

            # Rank IC
            rho, _ = spearmanr(y_true, pred_returns.mean(axis=0))
            self.rank_ics.append(rho if not np.isnan(rho) else 0.0)

            # MAE
            mae = np.mean(np.abs(pred_returns.mean(axis=0) - y_true))
            self.mae_list.append(mae)

            # 95% credible interval coverage
            lower = np.percentile(pred_returns, 2.5, axis=0)
            upper = np.percentile(pred_returns, 97.5, axis=0)

            hits = (y_true >= lower) & (y_true <= upper)
            self.coverage_hits.extend(hits.astype(int))
            
            # Sign precision
            pred_sign = np.sign(y_draws)
            true_sign = np.sign(y_true)
            sign_precision = (pred_sign == true_sign)
            self.sign_precisions.append(np.mean(sign_precision))
            
            # sign_precision_true_positives
            true_positives = np.mean(true_sign > 0)
            self.sign_precision_true_positives.append(np.mean(true_positives))
            
            # R-sqaured
            pred_mean = np.mean(y_draws)
            true_mean = np.mean(y_true)
            ss_res = np.sum((y_true - pred_mean)**2)
            ss_tot = np.sum((y_true - true_mean)**2)
            r_2 = 1 - ss_res/ss_tot if abs(ss_tot) <= 1e-5 else 1 - 1e+5 * ss_res
            self.r_squareds.append(r_2)
            
            # RMSE
            rmse = np.sqrt(np.mean((y_true - pred_mean)**2))
            self.rmse_list.append(rmse)
            
            # pred_means, true_means
            self.predicted_means.append(pred_mean)
            self.true_means.append(true_mean)

        log_score = np.mean(self.log_pred_scores)
        waic = -2 * (np.sum(self.waic_lppd) - np.sum(self.waic_var))
        rank_ic = np.mean(self.rank_ics)
        mae = np.mean(self.mae_list)
        coverage = np.mean(self.coverage_hits)
        
        sign_precision = np.mean(self.sign_precisions)
        avg_true_positives = np.mean(self.sign_precision_true_positives)
        r_squared = np.mean(self.r_squareds)
        rmse = np.mean(self.rmse_list)
        
        avg_predicted_mean = np.mean(self.predicted_means)
        avg_true_mean = np.mean(self.true_means)
        self.mean_deviation = avg_predicted_mean - avg_true_mean
        
        return log_score, waic, rank_ic, mae, coverage, sign_precision, avg_true_positives, r_squared, rmse, avg_predicted_mean, avg_true_mean, self.mean_deviation
