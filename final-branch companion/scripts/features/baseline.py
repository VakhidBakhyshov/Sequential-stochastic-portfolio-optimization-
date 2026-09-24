import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from pathlib import Path

from scripts.dataloader.filter import *
from scripts.dataloader.parse_close import *

pd.set_option("future.no_silent_downcasting", True)

@beartype
def read_parse_csv(filepath: Path, certain_cols: Union[list, None]):
    df = pd.read_csv(filepath)
    df = df.replace('No data', np.nan).sort_values(by=['Date'])
    if certain_cols is None:
        return df, df.columns
    return df[certain_cols], certain_cols


@beartype
def melt_transpose_df(df: pd.DataFrame, value_name: str, id_vars: list[str] = ['Date'], var_name: str = 'asset'):
    df = df.melt(
        id_vars=id_vars,
        var_name=var_name,
        value_name=value_name
    )
    
    df[value_name] = df[value_name].astype(str)
    df[value_name] = df[value_name].str.replace(",", ".", regex=False)
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    
    return df


@beartype
def filling_values_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Sort to ensure proper order
    df = df.sort_values(['asset', 'Date'])
    
    # Process all columns at once within each group
    for col in numeric_cols:
        # Backfill leading NaNs within each asset group
        df[col] = df.groupby('asset')[col].bfill()
        
        # Fill remaining with expanding mean within each group
        df[col] = df.groupby('asset')[col].transform(
            lambda x: x.fillna(x.expanding(min_periods=1).mean())
        )
    
    return df


@beartype
def filling_values_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Sort to ensure proper order
    df = df.sort_values(['asset', 'Date'])
    
    # Process all columns at once within each group
    for col in numeric_cols:
        # Backfill leading NaNs within each asset group
        df[col] = df.groupby('asset')[col].bfill()
        
        # Fill remaining with expanding mean within each group
        df[col] = df.groupby('asset')[col].transform(
            lambda x: x.fillna(x.expanding(min_periods=1).mean())
        )
    
    return df


@beartype
def basic_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    df_merged['lag_open'] = df_merged.groupby('asset')['open'].shift(1)
    df_merged['lag_close'] = df_merged.groupby('asset')['close'].shift(1)
    df_merged.iloc[0, df_merged.columns.get_loc('lag_open')] = 0.0
    df_merged.iloc[0, df_merged.columns.get_loc('lag_close')] = 0.0
    return df_merged


def main():
    filepath = Path.cwd() / "datasets" / "csv"
    df_close, cols = read_parse_csv(filepath / "NewClosePrice.csv", None)
    df_open = read_parse_csv(filepath / "OpenPrice.csv", None)[0]
    df_high = read_parse_csv(filepath / "MaxPrice.csv", None)[0]
    df_low = read_parse_csv(filepath / "MinPrice.csv", None)[0]
    df_volume = read_parse_csv(filepath / "Volume.csv", None)[0]

    df_open_long = melt_transpose_df(df_open, value_name='open')
    df_high_long = melt_transpose_df(df_high, value_name='high')
    df_low_long = melt_transpose_df(df_low, value_name='low')
    df_close_long = melt_transpose_df(df_close, value_name='close')
    df_volume_long = melt_transpose_df(df_volume, value_name='volume')

    df_merged = (
        df_open_long
        .merge(df_high_long, on=["Date", "asset"])
        .merge(df_low_long, on=["Date", "asset"])
        .merge(df_close_long, on=["Date", "asset"])
        .merge(df_volume_long, on=["Date", "asset"])
    ).reset_index(drop=True)

    df_merged = filling_values_fast(df_merged)
    df_merged = basic_features(df_merged)
    df_merged.to_csv(filepath / "features.csv")
    logger.info(f'Features successfully saved to {filepath / "features.csv"}')
    return df_merged



if __name__ == "__main__":
    main()
