import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar

from scripts.dataloader.filter import *

pd.set_option('future.no_silent_downcasting', True)



def get_sheet(filename, sheetname=None):
    if sheetname is not None:
        df = pd.read_excel(filename, sheet_name=sheetname)
    else:
        try:
            df = pd.read_csv(filename, sep=',', decimal=',', quotechar='"', engine='python')
        except:
            df = pd.read_excel(filename)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.replace({"No data": np.nan, "Cbonds authorization is required": np.nan})#.infer_objects(copy=False)
    df = df.infer_objects(copy=False)
    df = df.sort_index()
    df = df.astype(float)
    return df


def main():
    asset_info = pd.read_csv(INPUT_PATH / "Meta.csv")
    asset_info.columns = ["ISIN", "Ticker", "ETF & Funds", "Provider", "Object", "Sector", "Geography", "Total Expense Ratio", "Exchange", "Replication Method"]
    asset_info = asset_info.replace({"No data": np.nan}).dropna(subset=["Ticker"]).set_index("Ticker")

    # df_close = get_sheet(INPUT_PATH / "ClosePrice.csv")
    df_close = get_sheet(INPUT_PATH / "NewClosePrice.csv")
    cols = df_close.columns
    
    df_open = get_sheet(INPUT_PATH / "OpenPrice.csv")
    df_vol = get_sheet(INPUT_PATH / "Volume.csv")
    
    # important adding, columns in files ClosePrice.csv, OpenPrice.csv, Volume.csv are equal, but in NewClosePrice.csv columns are less
    df_open, df_vol = df_open[cols], df_vol[cols]

    df_amo = (df_close + df_open) / 2 * df_vol
    
    # df_amo_monthly = df_amo.resample('BMS').sum()
    us_bd = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())
    df_amo_monthly = df_amo.resample(us_bd).sum()

    min_months = 36 # 36 month with at least 1 trading day in each
    df_eligibility = df_amo_monthly.rolling(min_months+1, min_periods=min_months+1).sum().gt(min_months).astype(int)

    # amount_threshold = 1e10 # Let's pick $10 bil threshold
    # amount_threshold = 1e-4 # around 600-700 etfs
    amount_threshold = 0.5e-4 # around 220 etfs
    rolling_window = 36 # Have to take some rolling mean window to smooth out, otherwise sample too unstable

    # We could use the rolling window of the percentage (mean of divisions), but taking the percentage of rolling means makes more sense (division of means)
    df_amo_rolling = df_amo_monthly.rolling(window=rolling_window).mean()
    df_amo_rolling = df_amo_rolling.div(df_amo_rolling.sum(axis=1), axis=0)

    eligibility_matrix = (df_amo_rolling.shift(1) > amount_threshold).astype(int) * df_eligibility
    
    # eligibility_matrix = pd.read_csv(INPUT_PATH / "Eligibility.csv")
    
    weights_rebalance = eligibility_matrix.sort_index(axis=1)
    weights_rebalance_dict = weights_rebalance.to_dict('index')
    
    # save_df_dict_to_excel(weights_rebalance_dict, save_path=INPUT_PATH.parent / "excel", filename='filtered_weights.xlsx')
    # save_df_dict_to_excel(weights_rebalance_dict, save_path=INPUT_PATH.parent / "excel", filename='new_filtered_weights.xlsx')
    
    # save_df_dict_to_excel(weights_rebalance_dict, save_path=INPUT_PATH.parent / "excel", filename='last_filtered_weights.xlsx')
    save_df_dict_to_excel(weights_rebalance_dict, save_path=INPUT_PATH.parent / "excel", filename='final_filtered_weights.xlsx')


if __name__ == "__main__":
    main()  
