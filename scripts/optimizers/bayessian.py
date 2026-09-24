
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from beartype import beartype
from typing import Any, Literal

from scripts.optimizers.base import BaseCVaR
from scripts.calculations.portfolio_utils import tail_cvar_loss


class BayessianCVaR(BaseCVaR):
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
        
        super().__init__(config, market_cap, historical_returns, pred_returns, w_previous, bound)
        
        self.all_methods = ["cvar_returns", "mean_cvar_sharpe", "expected_returns", "deviation_from_target", "bayesian_sharpe"]

        self.alpha_level = self.config.get("confidence_level", 0.99)
        
        self.cov_matrix = self.remove_noise_cov_matrix()
        args=(self.pred_returns, self.w_previous)
        # args=(self.historical_returns, self.w_previous)
        
        if self.is_all_methods:
            print("combined_methods")
            for method in self.all_methods:
                self.optimizer(
                    args=args,
                    count_etf=self.pred_returns.shape[1],
                    method_type=method
                )
            
        else:
            print("single_methods")
            self.optimizer(
                args=args,
                count_etf=self.pred_returns.shape[1],
                method_type=self.config.get("task_type", "cvar_returns")
            )
            
        self.get_results()
        
  
    @beartype
    def function_to_optimize(
        self,
        weights: np.ndarray, pred_returns: np.ndarray, w_previous: np.ndarray,
        method_type: Literal["cvar_returns", "return_cvar_constraint", "mean_cvar_sharpe", "expected_returns", "deviation_from_target", "bayesian_sharpe"]
    ) -> float:
        
        weights = np.asarray(weights, dtype=float)
        pred_returns = np.asarray(pred_returns, dtype=float)
        w_previous = np.asarray(w_previous, dtype=float)
        
        penalty = self.calculate_turnover_penalty(weights)
        
        risk_free_monthly = self.config.get("risk_free_rate", 0.02) / 12
        risk_aversion = float(self.config.get("risk_aversion", 0.0))
        
        portfolio_returns = np.dot(pred_returns, weights)
        mean_ret = np.mean(portfolio_returns)
        vol = np.std(portfolio_returns, ddof=1) #+ 1e-8
        vol = max(vol, 1e-8)
        
        # losses = -portfolio_returns
        # losses = losses[np.isfinite(losses)]
        # var_threshold = np.quantile(losses, self.alpha_level)
        # cvar_loss = losses[losses >= var_threshold].mean()
        
        cvar_loss = tail_cvar_loss(
            pred_returns,
            weights,
            alpha=self.alpha_level,
            min_tail_count=int(self.config.get("min_tail_count", 5)),
        )
        
        sharpe = (mean_ret - risk_free_monthly) / vol
        
        if method_type == "cvar_returns":
            optimization_func = cvar_loss

        elif method_type == "return_cvar_constraint":
            # MAX expected return s.t. CVaR <= budget. The actual constrained solve is the LP in
            # base._solve_return_cvar_lp; this branch only reports a value (lower = more return).
            optimization_func = -mean_ret

        elif method_type == "mean_cvar_sharpe":
            return_weight = self.config.get("return_weight", 1.0)
            cvar_weight = self.config.get("cvar_weight", 1.0)
            sharpe_weight = self.config.get("sharpe_weight", 1.0)
            vol_weight = self.config.get("vol_weight", 0.0)

            score = (
                sharpe_weight * sharpe
                + return_weight * mean_ret
                - cvar_weight * cvar_loss
                - vol_weight * vol
            )
            optimization_func = -score
            
        elif method_type == "expected_returns":
            # expected_returns = np.mean(self.pred_returns, axis=0)
            # optimization_func = -np.dot(expected_returns.T, weights)
            
            # expected_returns = np.nanmean(pred_returns, axis=0)
            # variance_penalty = float(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            # optimization_func = -float(np.dot(expected_returns, weights)) + risk_aversion * variance_penalty
            
            variance_penalty = float(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            optimization_func = -mean_ret + risk_aversion * variance_penalty
            
        elif method_type == "deviation_from_target":
            target_step = float(self.config.get("target_step", 0.05))
            target = w_previous + target_step
            
            # diff = target - weights # 1 way
            # diff = weights - w_previous  # 2 way
            diff = weights - target
            
            optimization_func = np.dot(
                diff.T,
                np.dot(self.cov_matrix, diff)
            )
            
        elif method_type == "bayesian_sharpe":  
            optimization_func = -sharpe
            
        else:
            raise ValueError(f"Unknown type: {method_type}")
            
        return float(optimization_func + penalty)
