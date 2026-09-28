import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from typing import Any, Union, Literal
from beartype import beartype

from scripts.executions.accounting import partial_execute, gross_turnover, proportional_transaction_cost

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
        self.future_returns_is_full_state = False
        if isinstance(self.future_returns, pd.DataFrame):
            # run.py passes a full-universe frame in final-policy mode.  Preserve the
            # dimensionality flag before converting to ndarray so holdings that have just
            # left the feasible set still earn their realised return until liquidated.
            self.future_returns_is_full_state = self.future_returns.shape[1] == len(self.w_current)
            self.future_returns = self.future_returns.values

        self.future_returns = np.nan_to_num(self.future_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.current_balance = current_balance
        self.alpha = alpha
        self.type = self.config.get("return_type", "log-returns")
        self.is_past = self.config.get("is_past", False)
        self.c_bps = float(self.config.get("c_bps", 1e-3))  # default 10 bp per gross traded unit
        
        
    @beartype
    def calculate_exec_weights_turnover_transaction_costs(self, alpha: float) -> tuple[float, float, pd.Series]:
        w_exec = partial_execute(self.w_current, self.w_target, alpha)
        turnover = gross_turnover(self.w_current, w_exec)
        cost = proportional_transaction_cost(self.w_current, w_exec, self.c_bps)
        return float(turnover), float(cost), w_exec
        
    
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
        
        if self.is_past and not self.future_returns_is_full_state:
            # Legacy/diagnostic path: return frame contains only currently eligible names.
            pnl = self.calculate_pnl(w_exec[self.mask], cost)
        else:
            # Final publication path: return frame is aligned to the full portfolio state.
            # This books returns on partially liquidated holdings even after they leave the
            # current feasible set.
            pnl = self.calculate_pnl(w_exec, cost)
        
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
        print(f"IN-SAMPLE execution grid: alpha={alphas[index]}, result={results[index]}. IN-SAMPLE alpha grid-search best")
        return tuple(results[index])
