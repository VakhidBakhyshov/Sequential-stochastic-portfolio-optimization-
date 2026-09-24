"""Rolling-score robustness variant of the monthly ETF universe filter.

``monthly_etf_filtration_2.py`` is the primary EMA policy.  This module keeps the
same causal metric engine but smooths each ETF's cross-sectional score by a
rolling average across prior rebalance months.  It is intended for robustness
checks, not as a duplicate implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.dataloader.monthly_etf_filtration_2 import (
    FilterConfig, infer_first_trading_days, read_business_dates, read_market_csv,
    resolve_path, run_monthly_filtration, save_outputs,
)


def apply_rolling_score_policy(scores: pd.DataFrame, top_n: int, rolling_score_months: int = 3) -> pd.DataFrame:
    out = scores.copy()
    out["rebalance_date"] = pd.to_datetime(out["rebalance_date"])
    out = out.sort_values(["etf", "rebalance_date"])
    out["raw_aggregate_score"] = pd.to_numeric(out["aggregate_score"], errors="coerce")
    out["rolling_aggregate_score"] = (
        out.groupby("etf", group_keys=False)["raw_aggregate_score"]
        .transform(lambda x: x.rolling(max(1, int(rolling_score_months)), min_periods=1).mean())
    )
    out["aggregate_score"] = out["rolling_aggregate_score"]
    out["rank"] = np.nan
    out["selected"] = False
    for _, idx in out.groupby("rebalance_date").groups.items():
        g = out.loc[idx]
        eligible_col = "eligible_after_filters" if "eligible_after_filters" in g else "passed_base_filter"
        eligible = g.loc[g[eligible_col].fillna(False)].sort_values(
            ["rolling_aggregate_score", "price_change", "log_dollar_volume"],
            ascending=[False, False, False],
        ).head(top_n)
        out.loc[eligible.index, "rank"] = np.arange(1, len(eligible) + 1)
        out.loc[eligible.index, "selected"] = True
    return out.sort_values(["rebalance_date", "etf"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monthly ETF filtration with rolling score averaging.")
    p.add_argument("--data-root", type=Path, default=Path.cwd())
    p.add_argument("--prices", type=Path, default=Path("datasets/csv/NewClosePrice.csv"))
    p.add_argument("--volumes", type=Path, default=Path("datasets/csv/Volume.csv"))
    p.add_argument("--business-dates", type=Path, default=Path("datasets/excel/business_dates.xlsx"))
    p.add_argument("--output-dir", type=Path, default=Path("datasets/monthly_filtration_rolling"))
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--spy-filter", choices=["none", "return", "smart", "strict"], default="smart")
    p.add_argument("--include-benchmark", action="store_true")
    p.add_argument("--min-observations", type=int, default=10)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--min-total-volume", type=float, default=0.0)
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--rolling-score-months", type=int, default=3)
    p.add_argument("--include-rebalance-day", action="store_true")
    p.add_argument("--do-not-skip-first-business-date", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args(); root = a.data_root.resolve()
    prices = read_market_csv(resolve_path(a.prices, root)); volumes = read_market_csv(resolve_path(a.volumes, root))
    bpath = resolve_path(a.business_dates, root)
    dates = read_business_dates(bpath, skip_first=not a.do_not_skip_first_business_date) if bpath.exists() else infer_first_trading_days(prices)
    cfg = FilterConfig(
        top_n=a.top_n, benchmark=a.benchmark, spy_filter=a.spy_filter,
        exclude_benchmark=not a.include_benchmark, min_observations=a.min_observations,
        min_price=a.min_price, min_total_volume=a.min_total_volume,
        include_rebalance_day=a.include_rebalance_day,
        skip_first_business_date=not a.do_not_skip_first_business_date,
        lookback_days=a.lookback_days, score_ema_alpha=1.0,
    )
    raw = run_monthly_filtration(prices, volumes, dates, cfg)
    scores = apply_rolling_score_policy(raw, a.top_n, a.rolling_score_months)
    save_outputs(scores, resolve_path(a.output_dir, root), a.top_n)
    print(f"Processed {scores['rebalance_date'].nunique()} rebalance months with a {a.rolling_score_months}-month rolling score.")


if __name__ == "__main__":
    main()
