import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from beartype import beartype
from typing import Any, Literal, Optional

from scripts.optimizers.base import BaseCVaR


class BlackLitterman(BaseCVaR):
    def __init__(
        self,
        config: dict[str, Any],
        market_cap: np.ndarray,
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        w_previous: np.ndarray,
        bound: tuple = (0, None)
    ):
        
        self.config = config
        
        super().__init__(self.config, market_cap, historical_returns, pred_returns, w_previous, bound)
        
        self.all_methods = ["black_litterman"]
        self.tau = float(self.config.get("tau", 0.05))
        self.risk_free_rate = float(self.config.get("risk_free_rate", 0.02)) / 252.0
        self.count_etf = self.pred_returns.shape [1]
        
        # # 1 way - considering last month for delta calculation as fraction of mean reaturns last month to var
        # self.last_month = self.historical_returns[-self.market_cap.shape[0]+1:]
        # self.excess_return = np.sum(self.last_month, axis=1) - self.risk_free_rate
        # self.delta = np.mean(self.excess_return)/np.var(self.excess_return, ddof=1)
        
        # # 2 way - considering all month (all market portfolio)
        # total_market_cap = np.sum(self.market_cap, axis=0)
        # w_market = total_market_cap / np.sum(total_market_cap)
        # market_returns = np.dot(self.historical_returns, w_market) - self.risk_free_rate
        # self.delta = np.mean(market_returns) / np.var(market_returns, ddof=1)
        
        self.cov_matrix = self.remove_noise_cov_matrix()
        self.delta = self._estimate_risk_aversion()
        args = self._calculate_posterior()

        if self.is_all_methods:
            print("combined_methods")
            for method in self.all_methods:
                self.optimizer(
                    args=args,
                    count_etf=self.count_etf,
                    method_type=method
                )
            
        else:
            print("single_methods")
            self.optimizer(
                args=args,
                count_etf=self.count_etf,
                method_type=self.config.get("task_type", "black_litterman")
            )
            
        self.get_results()
        
    
    # # Past
    # def _calculate_equilibrium_returns(self) -> np.ndarray:        
    #     total_market_cap = np.sum(self.market_cap, axis=0)
    #     w_market = total_market_cap / np.sum(total_market_cap)
    #     equilibrium_returns = self.delta * np.dot(self.cov_matrix, w_market)
    #     return equilibrium_returns
    
    
    # @beartyp
    # # Past
    # def _construct_investor_views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     def row_matrix(row, type: Literal["1", "2", "3"]):
    #         if type == "1":
    #             row[17] = 1
    #             row[62] = -1
    #             view_returns = 0.03
    #         elif type == "2":
    #             row[:50] = 1/50
    #             view_returns = 0.04
    #         elif type == "3":
    #             row[50:100] = 1/50
    #             view_returns = 0.02
            
    #         return row, view_returns
            
    #     p = np.zeros((3, self.count_etf))
    #     q = np.zeros((3))
        
    #     for i in range(p.shape[0]):
    #         p[i], q[i] = row_matrix(p[i], str(i+1))        

    #     omega = self.tau * np.dot(p, np.dot(self.cov_matrix, p.T))

    #     return p, q, omega
        
        
    def _estimate_risk_aversion(self) -> float:
        if self.market_cap.ndim == 2 and self.market_cap.shape[1] == self.count_etf:
            latest_liquidity = np.nanmean(np.maximum(self.market_cap, 0.0), axis=0)
        else:
            latest_liquidity = np.ones(self.count_etf)
        if latest_liquidity.sum() <= 1e-12:
            w_market = np.ones(self.count_etf) / self.count_etf
        else:
            w_market = latest_liquidity / latest_liquidity.sum()
        market_returns = self.historical_returns @ w_market - self.risk_free_rate
        var = np.nanvar(market_returns, ddof=1)
        if not np.isfinite(var) or var <= 1e-12:
            return float(self.config.get("risk_aversion_delta", 2.5))
        delta = np.nanmean(market_returns) / var
        if not np.isfinite(delta) or delta <= 0:
            delta = float(self.config.get("risk_aversion_delta", 2.5))
        return float(np.clip(delta, 0.1, 25.0))
    
    
    def _calculate_equilibrium_returns(self) -> np.ndarray:
        if self.market_cap.ndim == 2 and self.market_cap.shape[1] == self.count_etf:
            total_market_cap = np.nanmean(np.maximum(self.market_cap, 0.0), axis=0)
        else:
            total_market_cap = np.ones(self.count_etf)
        if total_market_cap.sum() <= 1e-12:
            w_market = np.ones(self.count_etf) / self.count_etf
        else:
            w_market = total_market_cap / total_market_cap.sum()
        return self.delta * (self.cov_matrix @ w_market)


    @beartype
    def _construct_investor_views(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Robust signal-derived views instead of hard-coded positional views.
        With no ticker metadata available here, views are based only on information
        known at the rebalance date: historical momentum and volatility.
        """
        n = self.count_etf
        views = []
        q = []
        min_group = int(self.config.get("view_group_min", 3))
        k = max(min_group, n // 10)
        k = min(k, max(1, n // 2))

        if n >= 2:
            trailing = np.nan_to_num(np.nansum(self.historical_returns[-63:], axis=0), nan=0.0)
            order = np.argsort(trailing)
            losers = order[:k]
            winners = order[-k:]
            row = np.zeros(n)
            row[winners] = 1.0 / len(winners)
            row[losers] = -1.0 / len(losers)
            views.append(row)
            q.append(float(self.config.get("momentum_view_return", 0.01)))

        if n >= 4:
            vol = np.nanstd(self.historical_returns[-126:], axis=0, ddof=1)
            vol = np.nan_to_num(vol, nan=np.nanmedian(vol) if np.any(np.isfinite(vol)) else 0.0)
            order = np.argsort(vol)
            low_vol = order[:k]
            high_vol = order[-k:]
            row = np.zeros(n)
            row[low_vol] = 1.0 / len(low_vol)
            row[high_vol] = -1.0 / len(high_vol)
            views.append(row)
            q.append(float(self.config.get("low_vol_view_return", 0.0025)))

        if not views:
            p = np.zeros((1, n))
            p[0, 0] = 1.0
            q_arr = np.array([0.0])
        else:
            p = np.vstack(views)
            q_arr = np.asarray(q, dtype=float)

        omega = self.tau * (p @ self.cov_matrix @ p.T)
        omega = (omega + omega.T) / 2.0
        omega += float(self.config.get("omega_jitter", 1e-6)) * np.eye(omega.shape[0])
        return p, q_arr, omega
    
    
    @beartype
    def _calculate_posterior(self) -> tuple[np.ndarray, np.ndarray]:
        self.equilibrium = self._calculate_equilibrium_returns()
        self.matrix_p, self.vector_q, self.omega = self._construct_investor_views()
        
        omega_inv = np.linalg.inv(self.omega)
        tau_sigma_inv = np.linalg.pinv(self.tau * self.cov_matrix)
        posterior_cov_inv = np.linalg.inv(tau_sigma_inv + np.dot(self.matrix_p.T, np.dot(omega_inv, self.matrix_p)))
        
        self.posterior_mean = np.dot(
            posterior_cov_inv,
            np.dot(tau_sigma_inv, self.equilibrium) + np.dot(self.matrix_p.T, np.dot(omega_inv, self.vector_q))
        )
        
        self.posterior_cov = self.cov_matrix + posterior_cov_inv
        self.posterior_cov = (self.posterior_cov + self.posterior_cov.T) / 2.0
        return self.posterior_mean, self.posterior_cov


    @beartype
    def function_to_optimize(
        self,
        weights: np.ndarray, posterior_excess: np.ndarray, posterior_cov: np.ndarray,
        method_type: Literal["black_litterman"]
    ) -> float:
            
        if method_type == "black_litterman":
            utility = np.dot(weights.T, posterior_excess) - 0.5 * self.delta * np.dot(weights.T, np.dot(posterior_cov, weights))
            return float(-utility + self.calculate_turnover_penalty(weights))
        raise ValueError(f"Unknown type: {method_type}")
