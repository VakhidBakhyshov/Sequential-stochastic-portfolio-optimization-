import numpy as np
import pandas as pd
from beartype import beartype
from typing import Union, Literal

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.past.parser import *


WINDOW = 30


@beartype
def estimate_midPrice(
    df: pd.DataFrame,
    method: Literal['Average OHLC', 'Average HL'] = 'Average HL'
) -> pd.DataFrame:

    if set(['MaxPrice', 'MinPrice', 'OpenPrice', 'ClosePrice']) <= set(df.columns):
        if method == 'Average HL':
            df['MidPrice'] = 0.5 * (df['MaxPrice'] + df['MinPrice'])
        
        else:
            df['MidPrice'] = 0.25 * (df['MaxPrice'] + df['MinPrice'] + df['OpenPrice'] + df['ClosePrice'])
            
    return df
    
    
@beartype
def estimate_BidAsk_spread(
    df: Union[dict[str, pd.DataFrame], pd.DataFrame],
    method: Literal['High-Low-Ratio', 'High-Low-Corwin'] = 'High-Low-Corwin'
) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
    
    def calculate_spread(df: pd.DataFrame, method: Literal['High-Low-Ratio', 'High-Low-Corwin']) -> pd.DataFrame:
        df = estimate_midPrice(df)
        if set(['MaxPrice', 'MinPrice', 'MidPrice']) <= set(df.columns):
            low, high = df['MinPrice'], df['MaxPrice']
            
            if method == 'High-Low-Ratio':
                df['Spread'] = (high - low) / df['MidPrice']
            
            elif method == 'High-Low-Corwin':
                high_1, high_2 = high, high.shift(1)
                low_1, low_2 = low, low.shift(1)
                
                const = 3 - 2 * np.sqrt(2)
                beta = (np.log(high_1/low_1)**2 + np.log(high_2/low_2)**2).rolling(2).sum()
                gamma = np.log(np.maximum(high_1, high_2) / np.minimum(low_1, low_2))**2
                
                alpha = (np.sqrt(2) - 1) * np.sqrt(beta) / const - np.sqrt(gamma / const)
                alpha = np.clip(alpha, a_min=0, a_max=None)
                df['Spread'] = 2 * (np.exp(alpha) - 1) / (np.exp(alpha) + 1)
            
            else:
                return df
           
        return df
        
    if isinstance(df, dict):
        for ticker in df.keys():
            df[ticker] = calculate_spread(df[ticker], method)
        
    elif isinstance(df, pd.DataFrame):
        df = calculate_spread(df, method)
        
    else:
        raise TypeError(f'Wrong type of df {type(df)}')
    
    return df


@beartype
def estimate_BidAsk_Prices(df: Union[dict[str, pd.DataFrame], pd.DataFrame]) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
    def calculate_bid_ask(df: pd.DataFrame) -> pd.DataFrame:
        if set(['MidPrice', 'Spread']) <= set(df.columns):
            spread, mid = df['Spread'], df['MidPrice']
            df[f'AskPrice'] = mid + 0.5 * spread
            df[f'BidPrice'] = mid - 0.5 * spread
        
        return df
    
    if isinstance(df, dict):
        for ticker in df.keys():
            df[ticker] = calculate_bid_ask(df[ticker])
        return df
        
    elif isinstance(df, pd.DataFrame):
        return calculate_bid_ask(df)
        
    else:
        raise TypeError(f'Wrong type of df {type(df)}')
    
    
@beartype
def returns_metric_calculation(df: Union[dict[str, pd.DataFrame], pd.DataFrame]) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
    def calculate_returns(df: pd.DataFrame, type: Literal['returns', 'log-returns'] = 'log-returns') -> pd.DataFrame:
        if 'ClosePrice' in df.columns:
            if type == 'returns':
                df[type] = df['ClosePrice'].pct_change()
            else:
                df[type] = np.log(df['ClosePrice'] / df['ClosePrice'].shift(1))
                
            df.loc[0, type] = 0
        
        return df
    
    def calculate_EWMA(df: pd.DataFrame, type: Literal['returns', 'log-returns'] = 'log-returns') -> pd.DataFrame:
        if type in df.columns:
            # df[f'EMA{WINDOW}'] = df[type].ewm(halflife=WINDOW, adjust=True).mean()
            df[f'EMA{WINDOW}'] = df[type].ewm(span=WINDOW, adjust=True).mean()

        return df
    
    if isinstance(df, dict):
        for ticker in df.keys():
            df[ticker] = calculate_returns(df[ticker])
            df[ticker] = calculate_EWMA(df[ticker])
        return df
        
    elif isinstance(df, pd.DataFrame):
        df = calculate_returns(df)
        df = calculate_EWMA(df[ticker])
        
    else:
        raise TypeError(f'Wrong type of df {type(df)}')
    
    
@beartype
def get_etfs_returns_ewma(
    df: dict[str, pd.DataFrame],
    save_path: Union[Path, None] = None,
    type: Literal['returns', 'log-returns', f'EMA{WINDOW}'] = 'log-returns'
) ->  pd.DataFrame:

    returns_df = pd.DataFrame()
    for ticker in df.keys():
        df_ticker = df[ticker]
        if set(['Date', type]) <= set(df_ticker.columns):
            df_ticker = df_ticker[['Date', type]]
            df_ticker = df_ticker.rename(columns={type: ticker})
        
        if returns_df.empty:
            returns_df = df_ticker
        else:
            returns_df = pd.merge(returns_df, df_ticker, on='Date')
        
    if save_path:
        if type == 'log-returns': returns_df.to_excel(save_path / "etf_returns.xlsx", index=False)
        elif type == f'EMA{WINDOW}': returns_df.to_excel(save_path / "etf_ewma.xlsx", index=False)
    
    return returns_df


@beartype
def calculate(df_stats: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    df_stats = cosmetic_changes(df_stats)
    # df_stats = estimate_BidAsk_spread(df_stats)
    # df_stats = estimate_BidAsk_Prices(df_stats)
    df_stats = returns_metric_calculation(df_stats)
    # save_df_dict_to_excel(df_stats, INPUT_PATH.parent / "excel", "etf_nasdaq.xlsx")
    return df_stats
