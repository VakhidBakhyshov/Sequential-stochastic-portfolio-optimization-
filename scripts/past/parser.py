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
from typing import Union


INPUT_PATH = Path.cwd() / "datasets" / "csv"

NAMES = [
    'Meta',
    'OpenPrice',
    # 'ClosePrice',
    'NewClosePrice',
    'MinPrice',
    'MaxPrice',
    'Volume',
    'AUM',
    'Dividends',
]


@beartype
def get_dict_by_parsing_csv() -> dict[str, pd.DataFrame]:
    df_dict = {}
    for idx, name in enumerate(NAMES):
        if os.path.exists(INPUT_PATH / f'{name}.csv'):
            df_dict.update({name: pd.read_csv(INPUT_PATH / f'{name}.csv')})
            
            if idx == 0:
                df_dict[name] = df_dict[name].drop_duplicates(subset=['Тикер'], keep='first')
            
        else:
            raise FileNotFoundError(f"Not found file {INPUT_PATH / f'{name}.csv'}")

    return df_dict


@beartype
def get_etf_codes(df: pd.DataFrame, name: str) -> list:
    """
    we know that name == 'ISIN' a code, but using general function for further using
    """
    try:
        return sorted(df[name].astype('str'))
    except:
        raise TypeError(f"Column name {name} doesn't exist in this Dataframe or maybe dataframe is empty")


@beartype
def cosmetic_changes(df: Union[dict[str, pd.DataFrame], pd.DataFrame]) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
    def replace(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns[1:]:
            df[col] = df[col].mask(df[col] == 'No data', np.nan)
            
            if df[col].dtype == 'object':
                if df[col].astype(str).str.contains(',').any():
                    df[col] = df[col].str.replace(',', '.').astype(float)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if col == 'Date' or col.lower() == 'date':
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        return df
        
    if isinstance(df, dict):
        for code in df.keys():
            if code != NAMES[0]:
                df[code] = replace(df[code])
            else:
                if 'Date' in df[code].columns:
                    df[code]['Date'] = pd.to_datetime(df[code]['Date'], errors='coerce')
    
    elif isinstance(df, pd.DataFrame):
        df = replace(df)
    
    else:
        raise TypeError("DataFrame is empty or column 'Date' doesn't exist in DataFrame")
        
    return df


@beartype
def filter_business_dates(df: Union[dict[str, pd.DataFrame], pd.DataFrame]) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
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
    
    if isinstance(df, dict):
        for code in df.keys():
            df[code] = sort_dates(df[code])
            df[code] = get_only_business_dates(df[code])
        
    elif isinstance(df, pd.DataFrame):
        df = sort_dates(df)
        df = get_only_business_dates(df)
        
    else:
        raise TypeError(f'Wrong type of df {type(df)}')
    
    return df


@beartype
def filling_values(df: Union[dict[str, pd.DataFrame]]) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:    
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
    
    def fill_df(df: pd.DataFrame) -> pd.DataFrame:
        for i in df.columns[1:]:
            df[i] = fill_na_values(df[i])            
        return df
    
    if isinstance(df, dict):
        for code in df.keys():
            df[code] = fill_df(df[code])
        return df
                
    elif isinstance(df, pd.DataFrame):
        return fill_df(df)
    
    else:
        raise TypeError("DataFrame is empty or column 'Date' doesn't exist in DataFrame")


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
        
        
@beartype
def save_series_to_excel(series: pd.Series, save_path: Union[Path, None] = None, filename: Union[str, None] = None) -> None:
    if not series.empty and save_path and filename:
        series.to_excel(save_path / filename)


@beartype
def get_merge_statistics(
    df_dict: dict[str, pd.DataFrame],
    tickers: list[str]
    # save_path: Union[Path, None] = None,
    # filename: Union[str, None] = "etf_nasdaq.xlsx"
) -> dict[str, pd.DataFrame]:

    df_stats = {ticker: pd.DataFrame() for ticker in tickers}
    
    pbar = tqdm(NAMES, desc='Merging Process')
    for name in pbar:
        pbar.set_description(f'Merging Process on {name}')
        df = df_dict[name]
        
        if set(tickers) <= set(df.columns[1:]):
            df = df.rename(columns={col: f'{name} {col}' for col in df.columns[1:]})
            
            for ticker in tickers:
                df_stats[ticker] = pd.merge(df_stats[ticker], df[['Date', f'{name} {ticker}']], on='Date') if not df_stats[ticker].empty else df[['Date', f'{name} {ticker}']]
                df_stats[ticker] = df_stats[ticker].rename(columns={f'{name} {ticker}': name})

        # else:
        #     logger.info(f'For Statistics skip dict key {name}')

    df_stats = filling_values(df_stats)
    # save_df_dict_to_excel(df_stats, save_path, filename=filename)
        
    return df_stats


@beartype
def parse() -> tuple[list[str], pd.DataFrame, dict[str, pd.DataFrame]]:
    save_path = INPUT_PATH.parent / "excel"
    try:
        df_dict = get_dict_by_parsing_csv()
        df_dict = cosmetic_changes(df_dict)
        df_dict = filter_business_dates(df_dict)

        df_summary = df_dict[NAMES[0]]
        tickers = get_etf_codes(df_summary, 'Тикер')
        
        df_stats = get_merge_statistics(df_dict, tickers)
        
        save_series_to_excel(df_stats[tickers[0]]['Date'], save_path, "business_dates.xlsx")
        save_series_to_excel(pd.Series(tickers), save_path, "tickers.xlsx")
        
        logger.info("SUCCESS")
        return tickers, df_summary, df_stats
    
    except Exception as e:
        logger.error(f"Error in parse function: {e}")
        return [], pd.DataFrame(), {}
    
        

if __name__ == "__main__":
    parse()
