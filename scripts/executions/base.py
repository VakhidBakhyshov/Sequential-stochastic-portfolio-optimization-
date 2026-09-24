import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from typing import Any, Union, Literal
from beartype import beartype

class BaseExecution():
    def __init__(
        self,
        config: dict[str, Any],
        w_current: pd.Series,
        w_target: pd.Series,
        mask: Union[pd.Series, np.ndarray, None],
        future_returns: Union[np.ndarray, pd.DataFrame],
        current_balance: float,
        alpha: float
    ):
    
        self.config = config
        self.mask = mask
        self.w_target = w_target.copy()
        self.w_current = w_current.copy()
        self.future_returns = future_returns
        
        if isinstance(self.future_returns, pd.DataFrame):
            self.future_returns = self.future_returns.values
            
        self.future_returns = np.nan_to_num(self.future_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.current_balance = current_balance
        self.alpha = alpha
        self.type = self.config.get("return_type", "log-returns")
        self.is_past = self.config.get("is_past", False)
        self.c_bps = float(self.config.get("c_bps", 1e-4))
        
        
    @beartype
    def calculate_exec_weights_turnover_transaction_costs(self, alpha: float) -> tuple[float, float, pd.Series]:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        w_exec = self.w_current + alpha * (self.w_target - self.w_current)
        turnover = float(abs(w_exec - self.w_current).sum())
        cost = float(self.c_bps * turnover)
        return float(turnover), cost, w_exec
        
    
    @beartype
    def calculate_pnl(self, w_exec: pd.Series, cost: float) -> float:
        if self.type == "log-returns":
            simple_returns = np.exp(self.future_returns) - 1.0
        elif self.type == "returns":
            simple_returns = self.future_returns
            
        daily_returns = np.dot(simple_returns, w_exec.values)
        daily_returns = np.clip(daily_returns, -0.99, 10.0)
        monthly_return = np.prod(1.0 + daily_returns) - 1.0
        return monthly_return - cost
    
    
    @beartype
    def calculate_portfolio_value_return_value(self, pnl: float) -> tuple[float, float]:
        new_balance = self.current_balance * (1.0 + pnl)
        return_value = (new_balance / self.current_balance) - 1.0
        return new_balance, return_value
    
    
    @beartype   
    def execution_process(self, alpha: float = 1.0, is_dynamic_alpha: bool = False) -> tuple[float, float, float, float]:
        if is_dynamic_alpha:
            alpha = self.alpha
            
        _, cost, w_exec = self.calculate_exec_weights_turnover_transaction_costs(alpha)
        
        if self.is_past:
            pnl = self.calculate_pnl(w_exec[self.mask], cost)   # for run.py
        else:
            pnl = self.calculate_pnl(w_exec, cost)            # for strategy_run.py + new_run2.py
        
        new_balance, return_value = self.calculate_portfolio_value_return_value(pnl)
        return new_balance, pnl, cost, return_value
    
    
    @beartype
    def grid_search_execution(self) -> tuple:
        results = np.empty((0, 4))
        alphas = np.linspace(0.1, 1, 10)
        for alpha in alphas:
            results = np.vstack((results, self.execution_process(alpha)))
         
        alphas = np.append(alphas, self.alpha)  
        results = np.vstack((results, self.execution_process(is_dynamic_alpha=True)))
        
        index = np.argsort(results[:, -1])[::-1][0]
        print(f"ORACLE alpha grid_search best {alphas[index]=},{results[index]=}")
        return tuple(results[index])
