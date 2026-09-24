import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

from typing import Any, Callable, Union
from beartype import beartype
from scipy.optimize import minimize


class BaseFeature:
    def __init__(
        self,
        config: dict[str, Any],
        input_data: Union[pd.DataFrame, np.ndarray],
    ):
  
        self.config = config
        
        self.feature_name = self.config.get("name")
        self.data = input_data
        
        self.feature = []

    
    def function_to_optimize(self, *args, **kwargs) -> Callable:
        raise NotImplementedError("Child class must implement function_to_optimize")
    
    
    @beartype
    def optimizer(self, args: tuple, count_etf: int, method_type: str) -> None:
        bounds = [self.bound] * count_etf
        weights = np.array([1/count_etf]*count_etf)
        constraints = {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1}
        
        print(f"{method_type=}")
        
        func_to_optimize = lambda w: self.function_to_optimize(w, *args, method_type)
        
        if self.max_iter is None:
            result = minimize(
                fun=func_to_optimize,
                x0 = weights,
                # args=args,
                method=self.method,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iter}
            )
        
        else:
            result = minimize(
                fun=func_to_optimize,
                x0 = weights,
                # args=args,
                method=self.method,
                bounds=bounds,
                constraints=constraints
            )
        
        self.optimal_values.append(result.fun)
        # self.optimal_weights.append(result.x)
        self.optimal_weights.append(np.where(result.x >= self.min_weight, result.x, 0))
        
        
    @beartype
    def get_results(self) -> tuple[list[float], list[np.ndarray]]:
        return self.optimal_values, self.optimal_weights
    
    
    @beartype
    def get_method_names(self) -> list:
        return self.all_methods
