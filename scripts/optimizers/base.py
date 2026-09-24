import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from beartype import beartype
from typing import Any, Callable
from scipy.optimize import minimize

from scripts.calculations.portfolio_utils import normalize_long_only

class BaseCVaR:
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
        self.method = self.config.get("method", "SLSQP")
        self.min_weight = self.config.get("min_weight", 0.01)
        self.max_weight = self.config.get("max_weight", 0.1)
        self.is_all_methods = self.config.get("is_all_methods", True)
        self.max_iter = self.config.get('max_iter', None)
        
        print(f"{self.is_all_methods=}, {self.min_weight=}, {self.max_weight=}")
        
        self.constraint_max_weight = bool(self.config.get("constraint_max_weight", True))
        self.turnover_penalty = self.config.get("turnover_penalty", 1.0)
        self.penalty_type = self.config.get("penalty_type", "L1")
        
        self.market_cap = np.asarray(market_cap, dtype=float)
        self.historical_returns = np.asarray(historical_returns, dtype=float)
        self.pred_returns = np.asarray(pred_returns, dtype=float)
        self.w_previous = np.asarray(w_previous, dtype=float)
        
        if self.constraint_max_weight:
            self.bound = (0.0, self.max_weight) # mutilpe weights with constraint of max_weight
        else:
            self.bound = (0.0, None) # case where i'm getting 1 etf in portfolio with the best perfomance (no constraint on max_weight)
        
        self.all_methods: list[str] = []
        self.optimal_values: list[float] = []
        self.optimal_weights: list[np.ndarray] = []
        
        
    # @beartype
    # def remove_noise_cov_matrix(self):
    #     cov_matrix = np.cov(self.historical_returns, rowvar=False)
    #     eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
    #     q = self.historical_returns.shape[1] / self.historical_returns.shape[0]
    #     sigma2 = np.mean(np.diag(cov_matrix))
    #     lam_plus = sigma2 * (1 + 1/np.sqrt(q))**2
        
    #     noise_mask = eigvals < lam_plus
    #     n_noise = noise_mask.sum()
    #     if n_noise > 0:
    #         eigvals[noise_mask] = eigvals[noise_mask].mean()
    #     cov_clean = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))
    #     return cov_clean
    
    
    @beartype
    def remove_noise_cov_matrix(self) -> np.ndarray:
        """
        Marchenko-Pastur clipping with the correct upper edge when q=N/T.
        Previous code used (1 + 1/sqrt(q))^2, which over-clipped when T>N.
        """
        x = np.asarray(self.historical_returns, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            return np.eye(max(1, x.shape[1] if x.ndim == 2 else 1)) * 1e-6

        cov_matrix = np.cov(x, rowvar=False)
        cov_matrix = np.atleast_2d(cov_matrix)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.clip(eigvals, 1e-12, None)

        q = x.shape[1] / max(x.shape[0], 1)  # N/T
        q = max(q, 1e-8)
        sigma2 = float(np.mean(np.diag(cov_matrix)))
        sigma2 = max(sigma2, 1e-12)
        lam_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

        noise_mask = eigvals < lam_plus
        if np.any(noise_mask):
            eigvals[noise_mask] = float(np.mean(eigvals[noise_mask]))

        cov_clean = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov_clean = (cov_clean + cov_clean.T) / 2.0
        cov_clean += 1e-10 * np.eye(cov_clean.shape[0])
        return cov_clean
        
        
    def calculate_turnover_penalty(self, weights: np.ndarray) -> float:
        # if self.penalty_type == "L1":
        #     turnover = np.sum(np.abs(weights - self.w_previous))
        # elif self.penalty_type == "L2":
        #     turnover = np.sum((weights - self.w_previous)**2)
        
        weights = np.asarray(weights, dtype=float)
        prev = np.asarray(self.w_previous, dtype=float)
        if prev.shape != weights.shape:
            prev = np.resize(prev, weights.shape)

        if self.penalty_type == "L1":
            turnover = np.sum(np.abs(weights - prev))
        elif self.penalty_type == "L2":
            turnover = np.sum((weights - prev) ** 2)
        else:
            raise ValueError(f'Unknown penalty type {self.penalty_type}')
        return float(self.turnover_penalty * turnover)
        
    
    def function_to_optimize(self, *args, **kwargs) -> Callable:
        raise NotImplementedError("Child class must implement function_to_optimize")
    
    
    def _postprocess_weights(self, weights: np.ndarray) -> np.ndarray:
        # # 1 method - simple
        # weights = np.where(weights >= self.min_weight, weights, 0) # constraint for min_weight - if weight < min_weight => weight = 0
        # # weights /= np.sum(weights) # we are normalizing weights after constraint on min_weight
        # return weights
        
        # 2 method - updated + complicated
        return normalize_long_only(
            weights,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            fallback_n=weights.size,
        )
    
    @beartype
    def optimizer(self, args: tuple, count_etf: int, method_type: str) -> None:
        bounds = [self.bound] * count_etf
        x0 = np.ones(count_etf, dtype=float) / count_etf
        
        # for optimizer method COBYLA
        if self.method == "COBYLA":
            constraints = [
                {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)},        # sum(x) <= 1
                {'type': 'ineq', 'fun': lambda x: np.sum(x) - 0.99}      # sum(x) >= 0.99
            ]
        # for optimizer method SLSQP
        else:
            constraints = [{'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}]
        
        objective = lambda w: self.function_to_optimize(w, *args, method_type)
        options = {'maxiter': self.max_iter} if self.max_iter is not None else None
        
        result = minimize(
            fun=objective,
            x0=x0,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        
        if not result.success or not np.all(np.isfinite(result.x)):
            weights = x0
            res = float(objective(weights))
        else:
            weights = result.x
            res = float(result.fun)
        
        weights = self._postprocess_weights(weights)
        self.optimal_values.append(res)
        self.optimal_weights.append(weights)
        
        print(f"{method_type=}", f"{np.sum(weights)=:.6f}", f"active={np.sum(weights > 0)}", f"{weights.shape=}")
        
        
    @beartype
    def get_results(self) -> tuple[list[float], list[np.ndarray]]:
        return self.optimal_values, self.optimal_weights
    
    
    @beartype
    def get_method_names(self) -> list:
        return self.all_methods
