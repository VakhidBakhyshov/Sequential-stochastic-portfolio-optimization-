import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from typing import Any
from beartype import beartype

from scripts.strategies.base import BaseStrategy


class SimpleLongStrategy(BaseStrategy):
    def __init__(
        self,
        config: dict[str, Any],
        historical_returns: np.ndarray,
        pred_returns: np.ndarray,
        market_cap: np.ndarray,
        w_previous: np.ndarray
    ):
        
        self.config = config
        super().__init__(self.config, historical_returns, pred_returns, market_cap, w_previous)
        
        self.optimizer()
        self.clip_weights()
    
    
    @beartype
    def optimizer(self):
        mask = ~np.isnan(self.historical_returns)
        first_idx = mask.argmax(axis=0)
        
        self.month_price_diff = np.diag(self.historical_returns[first_idx, :] - self.historical_returns[-1, :]) # 1 way
        # self.month_price_diff = self.pred_returns # 2 way
        weight_sum = np.sum(np.abs(self.month_price_diff))
        
        # print(f"{self.month_price_diff=}")
        
        if weight_sum > 0:
            self.long_weights = np.minimum(0, self.month_price_diff) / weight_sum
            # self.short_weights = np.maximum(0, self.month_price_diff) / weight_sum
            self.short_weights = np.zeros_like(self.month_price_diff)
            self.weights = self.long_weights + self.short_weights
            # print(f"{self.weights=}")
        else:
            self.weights = np.ones_like(self.month_price_diff) / len(self.weights)
           
        print(f"{np.sum(self.weights)=}")
