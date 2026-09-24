import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from tqdm import tqdm
from loguru import logger
from beartype import beartype
from typing import Union, Literal
from scripts.dataloader.inputs import *


@beartype
def get_filtered_etfs(portfolios_date: pd.DataFrame) -> tuple[list, pd.Series]:
    mask = portfolios_date['Value'].apply(lambda x: x == 1)
    filtered_etfs = portfolios_date[mask]
    return filtered_etfs['Key'].tolist(), mask


@beartype
def find_start_time(dates: pd.Series, day: pd.Timestamp, lookback_years: int = 3) -> pd.Timestamp:
    target = day - pd.DateOffset(years=int(lookback_years))
    idx = (dates - target).abs().idxmin() # closest available date
    return dates.loc[idx]


@beartype
def get_first_day_month_returns(
    returns_df: pd.DataFrame,
    filtered_etfs: Union[list, None],
    startTime: Union[str, pd.Timestamp, None],
    endTime: Union[str, pd.Timestamp, None]
) -> pd.DataFrame:
    returns_df = returns_df.copy()
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    
    if startTime is not None and endTime is not None:
        month_return = returns_df[(returns_df['Date'] >= startTime) & (returns_df['Date'] < endTime)]
    elif startTime is not None:
        month_return = returns_df[returns_df['Date'] >= startTime]
    elif endTime is not None:
        month_return = returns_df[returns_df['Date'] < endTime]
    else:
        raise ValueError("At least one of startTime or endTime must be provided.")
    
    if isinstance(filtered_etfs, list):
        return month_return[['Date']+filtered_etfs]
    else:
        return month_return


# version where day - startTime is around 1 or 3 years
@beartype
def get_historical_returns(
    ewma_returns: pd.DataFrame, etfs_list: Union[list, None],
    endTime: pd.Timestamp, startTime: pd.Timestamp,
    window_type: Literal["rolling", "expanding"]
) -> pd.DataFrame:
    
    # startTime = find_start_time(business_days, day)
    if window_type == "rolling":
        return get_first_day_month_returns(ewma_returns, etfs_list, startTime, endTime)
    elif window_type == "expanding":
        return get_first_day_month_returns(ewma_returns, etfs_list, None, endTime)
    else:
        raise ValueError(f"Unknown type: {window_type}")


@beartype
def save_df_dict_to_excel(
    df_stats: Union[dict[str, pd.DataFrame], dict[pd.Timestamp, dict[str, int]]],
    save_path: Union[Path, None] = None,
    filename: Union[str, None] = None
) -> None:

    if save_path and filename:
        writer = pd.ExcelWriter(save_path / filename, engine='openpyxl')
        pbar = tqdm(df_stats.items(), desc='Saving to excel df with multiple keys')
        for ticker, df in pbar:
            pbar.set_description(f'Saving to excel df of key {ticker}')
            if isinstance(df, dict):
                ticker = ticker.strftime("%Y-%m-%d")
                df = pd.DataFrame(list(df.items()), columns=['Key', 'Value'])
                
            df.to_excel(writer, sheet_name=ticker, index=False)
        
        writer.close()
        logger.info(f'df successfully saved to {save_path} / {filename}')
