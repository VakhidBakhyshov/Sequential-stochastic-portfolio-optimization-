import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Any
from beartype import beartype
from scipy.optimize import minimize

from scripts.strategies.base import BaseStrategy


class RiskParityStrategy(BaseStrategy):
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
    
    
    # @beartype
    # def optimizer(self):
    #     self.weights = np.maximum(0, self.historical_returns[:, -1] - self.historical_returns[:, 0])
    #     self.weights = self.weights / np.sum(self.weights)

    
    @beartype
    def optimizer(self):
        x = np.nan_to_num(self.historical_returns, nan=0.0)
        cov = np.cov(x, rowvar=False)
        n = cov.shape[0]
        x0 = np.ones(n) / n

        def risk_contributions(w: np.ndarray) -> np.ndarray:
            port_vol = np.dot(w, np.dot(cov, w))
            port_vol = np.sqrt(max(port_vol, 1e-12))
            mrc = np.dot(cov, w) / port_vol
            return w * mrc

        def objective(w: np.ndarray) -> float:
            rc = risk_contributions(w)
            target = np.ones_like(rc) * (rc.sum() / len(rc))
            return float(((rc - target) ** 2).sum())

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bnds = [(0.0, self.max_weight) for _ in range(n)]
        res = minimize(objective, x0=x0, method="SLSQP", bounds=bnds, constraints=cons)
        w = np.clip(res.x, self.min_weight, self.max_weight)
        self.weights = w / max(w.sum(), 1e-12)
