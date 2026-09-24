import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from beartype import beartype
from datetime import timedelta
from typing import Union, Literal

from scripts.past.parser import *
from scripts.past.calculate import *
from scripts.past.plot import *


@beartype
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", "-sd", help="skip collect and parse datas", action='store_true')
    return parser.parse_args()


@beartype
def get_first_business_days_each_month(df: pd.DataFrame) -> pd.Series:
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # return df['Date'][df['Date'].dt.is_month_end] # Wrong idea because we are missing first days of month due to events (holidays and etc)
    return df.groupby(['Year', 'Month'])['Date'].min().reset_index()['Date']


@beartype
def portfolio_rebalanced(tickers: list[str], flags: list[int]) -> dict:
    if len(tickers) == len(flags):
        portfolio = dict({ticker: is_taken for ticker, is_taken in zip(tickers, flags)})
        return portfolio

    else:
        raise ValueError("Size of tickers != size of flags for each ticker")


@beartype
def initial_portfolio(first_business_days: Union[list, pd.Series], tickers: list[str]) -> dict:
    return dict({day: portfolio_rebalanced(tickers, [0] * len(tickers)) for day in first_business_days})


@beartype
def get_rebalanced_tickers(portfolios: dict[pd.Timestamp, dict[str, int]]) -> dict[pd.Timestamp, dict[str, int]]:
    final_portfolio = {}
    for date, portfolio in portfolios.copy().items():
        filtered_portfolio = {k: v for k, v in portfolio.items() if v == 1}
        if len(filtered_portfolio) > 0:
            final_portfolio[date] = filtered_portfolio
            
    return final_portfolio


@beartype
def liquidity_filter(
    portfolios: dict[pd.Timestamp, dict[str, int]],
    first_business_days: pd.Series,
    df_dict: dict[str, pd.DataFrame],
    bps: int = 10**5
) -> dict[pd.Timestamp, dict[str, int]]:
    
    def get_df_last_60_days(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
        month_end_date = pd.to_datetime(date)
        start = month_end_date - timedelta(days=60)
        
        if 'Date' in df.columns:
            index = df['Date'].index[(df['Date'] >= start) & (df['Date'] < month_end_date)]
            # mask = ~df.iloc[index]['Volume'].isna() # mask for removing nan-values
            # return df.iloc[index][mask]
            return df.iloc[index].dropna(how='all')
        
        else:
            raise KeyError('Column Date not found in dataframe')


    def get_AUM_criteria(aum_series: pd.Series, threshold: int = 100) -> np.bool:
        aum_series = pd.to_numeric(aum_series, errors='coerce')
        non_zero_aum = aum_series[aum_series > 0]
        if len(non_zero_aum) == 0:
            return False
        
        return non_zero_aum.iloc[-1] >= threshold * 10 * bps
        
        
    def get_nonan_mean_median(series: pd.Series, type: Literal['mean', 'median']) -> np.number:
        series = pd.to_numeric(series, errors='coerce')
        if type == 'mean':   
            return series[~series.isna()].mean()
        return series[~series.isna()].median()


    def ADDV_AUM_spread_criterias(df: pd.DataFrame) -> int:
        df = get_df_last_60_days(df, date)
        if df.empty:
            return 0
        
        if set(['Volume', 'AUM', 'Spread']) <= set(df.columns):
            volume_mean = get_nonan_mean_median(df['Volume'], type='mean')
            spread_median = get_nonan_mean_median(df['Spread'], type='median')
            # spread_flag = (bps * spread_median >= 10) & (bps * spread_median <= 30)
            # spread_flag = (bps * spread_median <= 300)
            spread_flag = True
            aum_flag = get_AUM_criteria(df['AUM'])
            # aum_flag = False
            volume_flag = (volume_mean >= 5 * 10 * bps)
            return int(volume_flag & aum_flag & spread_flag)
            # return int((volume_flag | aum_flag) & spread_flag)
          
        # return 0 
    
    pbar = tqdm(first_business_days, desc=f'Portfolio rebalancing')
    for date in pbar:
        pbar.set_description(f'Portfolio Rebalancing on {date}')
        for ticker in df_dict.keys():
            portfolios[date][ticker] = ADDV_AUM_spread_criterias(df_dict[ticker])
    
    return portfolios


def get_rebalanced_portfolios(
    df_stats: dict[str, pd.DataFrame],
    tickers: list[str],
    save_path: Union[Path, None] = None
) -> tuple[pd.Series, dict[pd.Timestamp, dict[str, int]], dict[pd.Timestamp, dict[str, int]]]:

    first_business_days = get_first_business_days_each_month(df_stats[tickers[0]])
    portfolios = initial_portfolio(first_business_days, tickers)
    
    portfolios = liquidity_filter(portfolios, first_business_days, df_stats)
    portfolio_heatmap(portfolios, save_path.parent)
    
    final_portfolio = get_rebalanced_tickers(portfolios)
    save_df_dict_to_excel(portfolios, save_path=save_path, filename='portfolio_all_etfs.xlsx')
    save_df_dict_to_excel(final_portfolio, save_path=save_path, filename='portfolio_rebalance.xlsx')
    
    if save_path:
        first_business_days.to_frame(name='Values').to_excel(save_path / 'business_dates.xlsx')
        logger.info(f'business days saved to {save_path / "business_dates.xlsx"}')
    
    return first_business_days[1:], portfolios, final_portfolio


@beartype
def get_first_day_month_returns(
    returns_df: pd.DataFrame,
    filtered_etfs: Union[list, None],
    startTime: Union[str, pd.Timestamp, None],
    endTime: Union[str, pd.Timestamp, None]
) -> pd.DataFrame:
    
    if startTime and endTime:
        month_return = returns_df[(returns_df['Date'] >= startTime) & (returns_df['Date'] < endTime)]
    
    elif startTime:
        month_return = returns_df[returns_df['Date'] >= startTime]
        
    elif endTime:
        month_return = returns_df[returns_df['Date'] < endTime]
        
    else:
        raise ValueError(f'Two dates startTime and endTime both None, but we need that one of them will be at least not None')
    
    if isinstance(filtered_etfs, list):
        return month_return[['Date']+filtered_etfs]
    else:
        return month_return


@beartype
def get_rebalanced_returns_dict(
    first_business_days: pd.Series,
    returns_df: pd.DataFrame,
    ewma_returns_df: pd.DataFrame,
    portfolios: dict[pd.Timestamp, dict[str, int]],
    save_path: Union[Path, None] = None
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    
    startTime = None
    returns_dict = {}
    ewma_cov_returns_dict = {}
    
    for rebalance_day in first_business_days:
        if rebalance_day in portfolios:
            returns_rebalanced = get_first_day_month_returns(returns_df, sorted(portfolios[rebalance_day].keys()), startTime, endTime=rebalance_day)
            returns_dict.update({rebalance_day.strftime("%Y-%m-%d"): returns_rebalanced})
            
            ewma_returns_rebalanced = get_first_day_month_returns(ewma_returns_df, sorted(portfolios[rebalance_day].keys()), startTime, endTime=rebalance_day)
            ewma_cov_returns_dict.update({rebalance_day.strftime("%Y-%m-%d"): ewma_returns_rebalanced[ewma_returns_rebalanced.columns[1:]].cov()})
        
        startTime = rebalance_day

    save_df_dict_to_excel(returns_dict, save_path=save_path, filename='returns_rebalance.xlsx')
    save_df_dict_to_excel(ewma_cov_returns_dict, save_path=save_path, filename='ewma_cov_returns_rebalance.xlsx')

    return returns_dict, ewma_cov_returns_dict


@beartype
def main():
    args = parse_args()
    output = INPUT_PATH.parent / "excel"
    
    if args.skip_data:
        summary = pd.read_csv(INPUT_PATH.parent / "csv" / "Meta.csv", index_col=0)
        tickers = pd.read_excel(output / "tickers.xlsx", index_col=0)
        tickers = tickers.iloc[:, 0].tolist()
        
        df_stats = pd.read_excel(output / "etf_nasdaq.xlsx", sheet_name=tickers)
    
    else:
        print('1')
        tickers, summary, df_stats = parse()
        df_stats = calculate(df_stats)

    returns_df = get_etfs_returns_ewma(df_stats, output)
    ewma_returns_df = get_etfs_returns_ewma(df_stats, output, type=f'EMA{WINDOW}')

    # first_business_days, _, final_portfolio = get_rebalanced_portfolios(df_stats, tickers, output)
    # plot_pie_diagram(final_portfolio, summary, output.parent.parent / "plots")
    
    # get_rebalanced_returns_dict(first_business_days, returns_df, ewma_returns_df, final_portfolio, output)
        
    logger.info("SUCCESS")
    


if __name__ == "__main__":
    main()
