"""Multi-metric score filter (not used by the paper; the electronic companion's universe ablation).

Starts from the liquidity screen of scripts/dataloader/liquidity_screen.py and keeps, in every month, the
80 eligible funds with the highest cross-sectional score built from 12-month and 3-month momentum, relative
strength against SPY, liquidity share, 3-month volatility and 6-month drawdown. Run from the package root:

    python extensions/universe_comparison/score_filter.py

Inputs (datasets/csv): Meta.csv, NewClosePrice.csv, OpenPrice.csv, Volume.csv.
Output (datasets/excel): score_filtered_weights.xlsx.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd

from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar

from scripts.dataloader.filter import *
from scripts.dataloader.liquidity_screen import get_sheet, AMOUNT_THRESHOLD, ROLLING_WINDOW, MIN_MONTHS

pd.set_option('future.no_silent_downcasting', True)

OUTPUT_NAME = "score_filtered_weights.xlsx"
TOP_N = 80


def main():
    asset_info = pd.read_csv(INPUT_PATH / "Meta.csv")
    asset_info = asset_info.replace({"No data": np.nan}).dropna(subset=["Ticker"]).set_index("Ticker")

    df_close = get_sheet(INPUT_PATH / "NewClosePrice.csv")
    cols = df_close.columns

    df_open = get_sheet(INPUT_PATH / "OpenPrice.csv")
    df_vol = get_sheet(INPUT_PATH / "Volume.csv")
    df_open, df_vol = df_open[cols], df_vol[cols]

    df_amo = (df_close + df_open) / 2 * df_vol

    us_bd = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())
    rebalance_dates = pd.date_range(start=df_close.index.min(), end=df_close.index.max(), freq=us_bd)

    df_amo_monthly = df_amo.resample(us_bd).sum()
    df_amo_monthly = df_amo_monthly.reindex(rebalance_dates)

    df_eligibility = df_amo_monthly.rolling(MIN_MONTHS + 1, min_periods=MIN_MONTHS + 1).sum().gt(MIN_MONTHS).astype(int)

    df_amo_rolling = df_amo_monthly.rolling(window=ROLLING_WINDOW).mean()
    df_amo_rolling = df_amo_rolling.div(df_amo_rolling.sum(axis=1), axis=0)

    eligibility_matrix = (df_amo_rolling.shift(1) > AMOUNT_THRESHOLD).astype(int) * df_eligibility

    returns_daily = df_close.pct_change(fill_method=None)

    mom_12m = df_close.pct_change(252, fill_method=None).shift(1)
    mom_3m = df_close.pct_change(63, fill_method=None).shift(1)
    vol_3m = returns_daily.rolling(63).std().shift(1)

    rolling_max_6m = df_close.rolling(126).max()
    drawdown_6m = (df_close / rolling_max_6m - 1.0).shift(1)

    if "SPY" in df_close.columns:
        spy_mom_6m = df_close["SPY"].pct_change(126, fill_method=None).shift(1)
        rel_strength = df_close.pct_change(126, fill_method=None).shift(1).sub(spy_mom_6m, axis=0)
    else:
        rel_strength = mom_3m * 0.0

    def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean(axis=1)
        std = df.std(axis=1).replace(0.0, np.nan)
        return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)

    liq_score = zscore_cross_section(df_amo_rolling)
    mom12_score = zscore_cross_section(mom_12m)
    mom3_score = zscore_cross_section(mom_3m)
    rel_score = zscore_cross_section(rel_strength)
    vol_score = zscore_cross_section(vol_3m)
    dd_score = zscore_cross_section(drawdown_6m.abs())

    score = (
        0.35 * mom12_score
        + 0.25 * mom3_score
        + 0.20 * rel_score
        + 0.15 * liq_score
        - 0.30 * vol_score
        - 0.15 * dd_score
    )

    score = score.where(eligibility_matrix.astype(bool))

    top_score_matrix = score.rank(axis=1, ascending=False) <= TOP_N

    eligibility_matrix = (
        eligibility_matrix.fillna(0).astype(bool)
        & top_score_matrix.fillna(False)
    ).astype(int)

    eligibility_matrix = eligibility_matrix.reindex(rebalance_dates).fillna(0).astype(int)

    weights_rebalance = eligibility_matrix.sort_index(axis=1)
    weights_rebalance_dict = {
        date: {ticker: int(value) for ticker, value in row.items()}
        for date, row in weights_rebalance.to_dict("index").items()
    }

    save_df_dict_to_excel(weights_rebalance_dict, save_path=INPUT_PATH.parent / "excel", filename=OUTPUT_NAME)


if __name__ == "__main__":
    main()
