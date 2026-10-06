import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from typing import Any, Callable
from beartype import beartype
from scipy.optimize import minimize

from scripts.calculations.portfolio_utils import normalize_long_only


class BaseStrategy:
    def __init__(
        self,
        config: dict[str, Any],
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        market_cap: np.ndarray,
        w_previous: np.ndarray
    ):
  
        self.config = config
        self.min_weight = self.config.get("min_weight", 0.01)
        self.max_weight = self.config.get("max_weight", 0.1)
        
        self.historical_returns = historical_returns
        self.pred_returns = pred_returns
        self.market_cap = market_cap
        self.w_previous = w_previous
        self.weights = np.zeros(self.historical_returns.shape[1], dtype=float)
        
    
    # def function_to_optimize(self, *args, **kwargs) -> Callable:
    #     raise NotImplementedError("Child class must implement function_to_optimize")
    
    
    @beartype
    def clip_weights(self):
        self.weights = np.where(self.weights <= self.min_weight, 0, self.weights)
        self.weights = np.where(self.weights >= self.max_weight, self.max_weight, self.weights)
        
        # self.weights = np.where(self.weights <= self.min_weight, 0, self.weights) # constraint for min_weight - if weight < min_weight => weight = 0
        # self.weights = np.where(self.weights >= self.max_weight, self.max_weight, self.weights)
       
        # 2 method - updated + complicated
        self.weights = normalize_long_only(
            self.weights,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            fallback_n=self.weights.size,
        )
    
    
    @beartype
    def get_weights(self) -> np.ndarray:
        self.clip_weights()
        return self.weights
