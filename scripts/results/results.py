import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from beartype import beartype
from typing import Literal, Union


OUTPUT = Path.cwd() / "results"


class BaseResults():
    def __init__(self):
        
        self.metrics = {
            "etf": {},
            "preds": {},
            "real": {},
            "model": {},
            "pnl": {}
        }

        
    @beartype
    def add_step_results(
        self,
        step_date: str,
        results: np.ndarray,
        type_results: Literal["etf", "preds", "real", "model", "pnl"]
    ) -> None:
        
        self.metrics[type_results].update({step_date: results})

    
    @beartype
    def add_date_results(
        self,
        step_date: str,
        etf_list: list,
        pred_returns: Union[np.ndarray, None],
        real_returns: np.ndarray,
        model_results: Union[list, None],
        pnl_results: list
    ) -> None:

        self.add_step_results(step_date, results=np.array(etf_list), type_results="etf")
        self.add_step_results(step_date, results=real_returns, type_results="real")
        self.add_step_results(step_date, results=np.array(pnl_results), type_results="pnl")
        
        if pred_returns is not None:
            self.add_step_results(step_date, results=pred_returns, type_results="preds")
        
        if model_results is not None:
            self.add_step_results(step_date, results=np.array(model_results), type_results="model")
        
        
    @beartype
    def get_results(self, type_results: Union[Literal["etf", "preds", "real", "model", "pnl"], None]) -> dict:
        if type_results is not None:
            return self.metrics[type_results]
        else:
            return self.metrics
        
        
    @beartype
    def transform_results_to_df(self) -> None:
        columns = np.unique(np.concatenate(list(self.metrics["etf"].values())))

        self.preds_df = pd.DataFrame({}, columns=columns, index=self.metrics["etf"].keys())
        self.real_df = pd.DataFrame({}, columns=columns, index=self.metrics["etf"].keys())
        
        self.model_df = pd.DataFrame(
            {},
            columns=[
                'log_score', 'predictive_deviance_proxy', 'rank_ic', 'mae', "coverage_95",
                'sign_precision', 'avg_true_positives', 'r_squared', 'rmse',
                'avg_predicted_mean', 'avg_true_mean', 'mean_deviation'
            ],
            index=self.metrics["etf"].keys()
        )
        
        self.pnl_df = pd.DataFrame(
            {},
            columns=['Balance', 'PnL', 'Cost', 'Returns', 'Alpha'],
            index=self.metrics["etf"].keys()
        )

        for date, columns in self.metrics["etf"].items():
            if date in self.metrics["preds"]:
                self.preds_df.loc[self.preds_df.index.isin([date]), columns]= self.metrics["preds"][date]
            
            self.real_df.loc[self.real_df.index.isin([date]), columns]= self.metrics["real"][date]
            
            if date in self.metrics["model"]:
                self.model_df.loc[self.model_df.index.isin([date]), self.model_df.columns]= self.metrics["model"][date]
            
            self.pnl_df.loc[self.pnl_df.index.isin([date]), self.pnl_df.columns]= self.metrics["pnl"][date]
        

    @beartype
    def save_results(self, name: str) -> None:
        folder = OUTPUT / name
        folder.mkdir(parents=True, exist_ok=True)
        
        if not self.preds_df.empty:
            filename = folder / "preds.csv"
            self.preds_df.to_csv(filename, index=True)
            logger.info(f"preds_df info successfully saved to {filename}")
            
        if not self.real_df.empty:
            filename = folder / "real.csv"
            self.real_df.to_csv(filename, index=True)
            logger.info(f"real_df info successfully saved to {filename}")
            
        if not self.model_df.empty:
            filename = folder / "model.csv"
            self.model_df.to_csv(filename, index=True)
            logger.info(f"model_df info successfully saved to {filename}")
            
        if not self.pnl_df.empty:
            filename = folder / "pnl.csv"
            self.pnl_df.to_csv(filename, index=True)
            logger.info(f"pnl_df info successfully saved to {filename}")
