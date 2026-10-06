"""Point-in-time liquidity screen of the eligible set (Section 2.2 of the paper).

For every fund the daily traded value is the mid price times share volume, V = (P_open + P_close) / 2 * Q.
Traded value is aggregated by calendar month (first US business day as the month label), the 36-month
rolling mean of each fund's share of aggregate traded value is compared with a fixed threshold, and a
fund is eligible in month m when its lagged share exceeds the threshold and it has traded in each of the
preceding 36 months. The result is written as one sheet per rebalance date with columns Key (ticker)
and Value (1 eligible, 0 not).

Inputs (datasets/csv): Meta.csv, NewClosePrice.csv, OpenPrice.csv, Volume.csv.
Output (datasets/excel): the eligibility workbook named by OUTPUT_NAME.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar

from scripts.dataloader.filter import *

pd.set_option('future.no_silent_downcasting', True)

OUTPUT_NAME = "last_filtered_weights.xlsx"
AMOUNT_THRESHOLD = 0.5e-4     # share of aggregate traded value
ROLLING_WINDOW = 36           # months
MIN_MONTHS = 36               # months with at least one trading day


def get_sheet(filename, sheetname=None):
    if sheetname is not None:
        df = pd.read_excel(filename, sheet_name=sheetname)
    else:
        try:
            df = pd.read_csv(filename, sep=',', decimal=',', quotechar='"', engine='python')
        except Exception:
            df = pd.read_excel(filename)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.replace({"No data": np.nan, "Cbonds authorization is required": np.nan})
    df = df.infer_objects(copy=False)
    df = df.sort_index()
    df = df.astype(float)
    return df


def main():
    asset_info = pd.read_csv(INPUT_PATH / "Meta.csv")
    asset_info = asset_info.replace({"No data": np.nan}).dropna(subset=["Ticker"]).set_index("Ticker")

    df_close = get_sheet(INPUT_PATH / "NewClosePrice.csv")
    df_open = get_sheet(INPUT_PATH / "OpenPrice.csv")
    df_vol = get_sheet(INPUT_PATH / "Volume.csv")

    # the open-price and volume files carry the full vendor ticker set; funds without an adjusted-close
    # series stay in the workbook with zero eligibility
    cols = df_open.columns
    df_close = df_close.reindex(columns=cols)
    df_vol = df_vol[cols]

    df_amo = (df_close + df_open) / 2 * df_vol

    us_bd = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())
    df_amo_monthly = df_amo.resample(us_bd).sum()

    df_eligibility = df_amo_monthly.rolling(MIN_MONTHS + 1, min_periods=MIN_MONTHS + 1).sum().gt(MIN_MONTHS).astype(int)

    # share of the 36-month rolling mean of traded value, lagged one month
    df_amo_rolling = df_amo_monthly.rolling(window=ROLLING_WINDOW).mean()
    df_amo_rolling = df_amo_rolling.div(df_amo_rolling.sum(axis=1), axis=0)

    eligibility_matrix = (df_amo_rolling.shift(1) > AMOUNT_THRESHOLD).astype(int) * df_eligibility

    weights_rebalance = eligibility_matrix.sort_index(axis=1)
    weights_rebalance_dict = weights_rebalance.to_dict('index')

    save_df_dict_to_excel(weights_rebalance_dict, save_path=INPUT_PATH.parent / "excel", filename=OUTPUT_NAME)


if __name__ == "__main__":
    main()
