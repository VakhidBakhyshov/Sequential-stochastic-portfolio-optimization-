import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from beartype import beartype
from typing import Any, Literal

from scripts.optimizers.base import BaseCVaR


class MarkowitzCVaR(BaseCVaR):
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
        
        self.all_methods = ["max_sharpe", "min_variance", "max_return_min_vol", "max_diversification", "max_decorrelation"]
        
        self.meanReturns = np.mean(self.pred_returns, axis=0)
        self.cov_matrix = self.remove_noise_cov_matrix()
        self.corr_marix = self.get_correlation_matrix()
        
        args = (self.meanReturns, self.cov_matrix, self.corr_marix)

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
                method_type=self.config.get("task_type", "max_sharpe")
            )
            
        self.get_results()
    
    
    @beartype
    def get_correlation_matrix(self) -> np.ndarray:
        stds = np.sqrt(np.diag(self.cov_matrix))
        stds_safe = np.where(np.abs(stds) < 1e-10, 1e-10, stds)
        D_inr_sqrt = np.diag(1 / stds_safe)
        return np.dot(np.dot(D_inr_sqrt, self.cov_matrix), D_inr_sqrt)
    
    
    @beartype
    def portfolio_perfomance(self, weights: np.ndarray, meanReturns: np.ndarray, covMatrix: np.ndarray) -> tuple[float, float]:
        weights = weights.reshape(-1, 1)
        expectedReturns = np.dot(weights.T, meanReturns.reshape(-1, 1))
        variance = np.dot(weights.T, np.dot(covMatrix, weights))
        return expectedReturns.flatten()[0], variance.flatten()[0]


    @beartype
    def function_to_optimize(
        self,
        weights: np.ndarray, meanReturns: np.ndarray, covMatrix: np.ndarray, corr_matrix: np.ndarray,
        method_type: Literal["max_sharpe", "min_variance", "max_return_min_vol", "max_diversification", "max_decorrelation"]
    ) -> float:
        
        risk_free_rate = float(self.config.get("risk_free_rate", 0.02)) / 12.0 # Monthly risk-free rate
        expectedReturns, variance = self.portfolio_perfomance(weights, meanReturns, covMatrix)
        
        volatility = np.sqrt(max(variance, 1e-6))
        # volatility = np.sqrt(max(variance, 1e-10))
        
        penalty = self.calculate_turnover_penalty(weights)
            
        if method_type == "max_sharpe":
            sharpe = (expectedReturns-risk_free_rate) / volatility
            optimization_func = -sharpe
        
        elif method_type == "min_variance":
            optimization_func = variance
        
        elif method_type == "max_return_min_vol":
            turnover_penalty = 0.5
            objective = expectedReturns - turnover_penalty * variance
            optimization_func = -objective
        
        elif method_type == "max_diversification":
            weighted_vol = np.dot(weights.T, np.std(self.pred_returns, axis=0, ddof=1))
            diversification_ratio = weighted_vol / volatility
            optimization_func = -diversification_ratio
        
        elif method_type ==  "max_decorrelation":
            # corr_matrix = self.get_correlation_matrix(covMatrix)
            corr_risk = np.dot(weights.T, np.dot(corr_matrix, weights))
            optimization_func = corr_risk
            
        else:
            raise ValueError(f"Unknown type: {method_type}")
        
        return optimization_func + penalty
