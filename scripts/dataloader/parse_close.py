import os
import argparse
import numpy as np
import pandas as pd

from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from beartype import beartype
from typing import Union, Literal


WINDOW = 30
INPUT_PATH = Path.cwd() / "datasets" / "csv"


@beartype
def cosmetic_changes(df: pd.DataFrame) -> pd.DataFrame:
    def replace(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns[1:]:
            df[col] = df[col].mask(df[col] == 'No data', np.nan)
            
            # df[col] = df[col].replace(0, np.nan)
            
            if df[col].dtype == 'object':
                if df[col].astype(str).str.contains(',').any():
                    df[col] = df[col].str.replace(',', '.').astype(float)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if col == 'Date' or col.lower() == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        return df
    
    return replace(df)


@beartype
def filter_business_dates(df: pd.DataFrame) -> pd.DataFrame:
    def sort_dates(df: pd.DataFrame) -> pd.DataFrame:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            return df.reset_index(drop=True)

        return df
    
    def get_only_business_dates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Removing weekends (that's Saturday and Sunday, saving only dates from Monday to Friday inclusive) and removing holidays
        """
        if 'Date' in df.columns:
            all_business_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
            df = df[df['Date'].isin(all_business_dates)]
        
        return df.reset_index(drop=True)
    

    df = sort_dates(df)
    df = get_only_business_dates(df)
        
    return df


@beartype
def filling_values(df: pd.DataFrame) -> pd.DataFrame:    
    def fill_na_values(df: pd.Series, window=20) -> pd.Series:
        # Special condition that check if all values of series is numbers, not strings
        if not np.issubdtype(df.dtype, np.number):
            return df
        
        non_na_idx = df.index[~df.isna()]
        if len(non_na_idx) == 0:
            return df

        start = non_na_idx[0]
        # start = df.index[~df.isna()][0]
        end = len(df)
        
        df = df.copy()
        for i in range(start, end):
            if pd.isna(df.iloc[i]):
                df.iloc[i] = df.iloc[start:i].mean() if i - start < window else df.iloc[i-window-1:i].mean()
                
        return df
    
    for i in df.columns[1:]:
        df[i] = fill_na_values(df[i])            
    return df



@beartype
def returns_metric_calculation(
    df: pd.DataFrame,
    type: Literal['returns', 'log-returns'] = 'log-returns',
    save_path: Union[Path, None] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    def calculate_returns(df: pd.DataFrame, type: Literal['returns', 'log-returns'] = 'log-returns') -> pd.DataFrame:
        returns_df = df.copy()
        
        if type == 'returns':
            for col in df.columns[1:]:
                returns_df[col] = df[col].pct_change()
                returns_df.loc[0, col] = 0
        else:
            for col in df.columns[1:]:
                returns_df[col] = np.log(df[col] / df[col].shift(1))
                returns_df.loc[0, col] = 0
        
        return returns_df
    
    def calculate_EWMA(df: pd.DataFrame) -> pd.DataFrame:
        ema_returns_df = df.copy()
        for col in df.columns[1:]:
            # ema_returns_df[col] = df[col].ewm(span=WINDOW, adjust=False).mean()
            ema_returns_df[col] = df[col].ewm(halflife=WINDOW, adjust=False).mean()

        return ema_returns_df
        
    
    returns_df = calculate_returns(df, type)
    ema_returns_df = calculate_EWMA(returns_df)
    
    if save_path:
        returns_df.to_csv(save_path / "new_etf_returns.csv")
        ema_returns_df.to_csv(save_path / "new_etf_ewma.csv")
        
    return returns_df, ema_returns_df


@beartype
def get_first_business_days_each_month(df: pd.DataFrame) -> pd.Series:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # return df['Date'][df['Date'].dt.is_month_end] # Wrong idea because we are missing first days of month due to events (holidays and etc)
    return df.groupby(['Year', 'Month'])['Date'].min().reset_index()['Date']
        
        
@beartype
def save_series_to_excel(series: pd.Series, save_path: Union[Path, None] = None, filename: Union[str, None] = None) -> None:
    if not series.empty and save_path and filename:
        series.to_excel(save_path / filename)


@beartype
def parse() -> None:
    save_path = INPUT_PATH.parent / "excel"
    try:
        df = pd.read_csv(INPUT_PATH / "NewClosePrice.csv")
        df = cosmetic_changes(df)
        df = filter_business_dates(df)
        
        first_business_days = get_first_business_days_each_month(df)
        first_business_days.to_frame(name='Values').to_excel(save_path / 'business_dates.xlsx')
        
        df = filling_values(df)
        df.to_csv(save_path / "new.csv")
        
        returns_metric_calculation(df, 'log-returns', save_path)  
        logger.info("SUCCESS")
    
    except Exception as e:
        logger.error(f"Error in parse function: {e}")
    
        

if __name__ == "__main__":
    parse()
