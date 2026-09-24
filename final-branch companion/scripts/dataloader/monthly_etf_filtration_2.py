"""
Monthly ETF filtration / universe selection script.

Goal
----
For each monthly rebalance date, calculate ETF-level metrics using a rolling lookback window, 
aggregate those metrics into one score (smoothed with EMA to reduce turnover), 
sort ETFs by that score, and save the top N ETFs for every month.

Typical usage from the project root:
    python monthly_etf_filtration.py \
        --top-n 100 \
        --benchmark SPY \
        --spy-filter smart \
        --lookback-days 90 \
        --score-ema-alpha 0.5 \
        --output-dir results/monthly_filtration
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HIGHER_IS_BETTER_WEIGHTS: Dict[str, float] = {
    # Momentum / return quality
    "price_change": 0.20,
    "excess_return_vs_benchmark": 0.18,
    "risk_adjusted_return": 0.14,
    "risk_adjusted_excess_return": 0.14,
    "positive_days_ratio": 0.04,

    # Liquidity / tradability
    "log_total_volume": 0.08,
    "log_dollar_volume": 0.10,
    "volume_change": 0.05,

    # Relative strength versus benchmark
    "momentum_ratio_vs_benchmark": 0.07,
}

LOWER_IS_BETTER_WEIGHTS: Dict[str, float] = {
    # Risk penalties
    "volatility": 0.08,
    "downside_volatility": 0.05,
    "drawdown_abs": 0.07,

    # Liquidity penalty; lower Amihud illiquidity is better
    "amihud_illiquidity": 0.04,
}


@dataclass(frozen=True)
class FilterConfig:
    top_n: int = 100
    benchmark: str = "SPY"
    spy_filter: str = "smart"  # none, return, smart, strict
    exclude_benchmark: bool = True
    min_observations: int = 10
    min_price: float = 1.0
    min_total_volume: float = 0.0
    include_rebalance_day: bool = False
    skip_first_business_date: bool = True
    
    # Inertia / Turnover Management
    lookback_days: int = 90          # Rolling window size (90 days = ~3 months)
    score_ema_alpha: float = 0.5     # 1.0 = completely replace score each month, 0.1 = very slow change


# -----------------------------------------------------------------------------
# Loading utilities
# -----------------------------------------------------------------------------

def resolve_path(path: Path | str, data_root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else data_root / path


def find_date_column(df: pd.DataFrame) -> str:
    candidates = ["Date", "date", "Datetime", "datetime", "timestamp", "Time", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    return df.columns[0]


def read_market_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input file is empty: {path}")

    date_col = find_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")
    df = df.set_index(date_col)
    df.index.name = "Date"

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")
    return df


def read_business_dates(path: Path, skip_first: bool = True) -> pd.DatetimeIndex:
    raw = pd.read_excel(path, index_col=0)
    if raw.empty:
        raise ValueError(f"Business date file is empty: {path}")

    if "Values" in raw.columns:
        values = raw["Values"]
    else:
        values = raw.iloc[:, 0]

    dates = pd.to_datetime(values, errors="coerce").dropna().sort_values()
    if skip_first and len(dates) > 1:
        dates = dates.iloc[1:]
    return pd.DatetimeIndex(dates.unique())


def infer_first_trading_days(prices: pd.DataFrame) -> pd.DatetimeIndex:
    monthly = prices.groupby(prices.index.to_period("M")).apply(lambda x: x.index.min())
    return pd.DatetimeIndex(monthly.to_list()).sort_values()


def align_price_volume_columns(prices: pd.DataFrame, volumes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    common = [col for col in prices.columns if col in volumes.columns]
    if not common:
        raise ValueError("No common ETF columns found between prices and volumes.")
    return prices[common].copy(), volumes[common].copy(), common


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_divide(num: pd.Series, den: pd.Series | float, fill_value: float = np.nan) -> pd.Series:
    result = num / den
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill_value)


def calculate_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    cumulative = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return drawdown.min(axis=0)


def calculate_beta_and_corr(returns: pd.DataFrame, benchmark: str) -> Tuple[pd.Series, pd.Series]:
    beta = pd.Series(np.nan, index=returns.columns, dtype=float)
    corr = pd.Series(np.nan, index=returns.columns, dtype=float)

    if benchmark not in returns.columns:
        return beta, corr

    benchmark_returns = returns[benchmark]
    with np.errstate(invalid='ignore', divide='ignore'):
        for col in returns.columns:
            pair = pd.concat([returns[col], benchmark_returns], axis=1).dropna()
            if len(pair) < 3:
                continue
            pair_benchmark_var = pair.iloc[:, 1].var(ddof=1)
            if not np.isfinite(pair_benchmark_var) or pair_benchmark_var <= 0:
                continue
            beta[col] = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair_benchmark_var
            corr[col] = pair.iloc[:, 0].corr(pair.iloc[:, 1])

    return beta, corr


def calculate_month_metrics(
    month_prices: pd.DataFrame,
    month_volumes: pd.DataFrame,
    previous_total_volume: Optional[pd.Series],
    rebalance_date: pd.Timestamp,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    
    # Count actual observations before any fill. Backward filling would create
    # artificial pre-inception history and introduces a subtle look-ahead bias.
    raw_prices = month_prices.copy()
    observations = raw_prices.notna().sum(axis=0)
    month_prices = raw_prices.ffill()
    month_volumes = month_volumes.fillna(0.0)

    if len(month_prices) < 2:
        raise ValueError(f"Not enough price observations for {rebalance_date.date()}")

    start_price = raw_prices.apply(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan)
    end_price = raw_prices.apply(lambda x: x.dropna().iloc[-1] if x.notna().any() else np.nan)
    price_diff = end_price - start_price
    price_change = safe_divide(price_diff, start_price, fill_value=np.nan)

    daily_returns = month_prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    total_volume = month_volumes.sum(axis=0)
    avg_price = month_prices.mean(axis=0)
    dollar_volume = (month_prices * month_volumes).sum(axis=0)

    if previous_total_volume is None:
        volume_change = pd.Series(0.0, index=month_prices.columns)
    else:
        previous_total_volume = previous_total_volume.reindex(month_prices.columns).fillna(0.0)
        volume_change = (total_volume - previous_total_volume) / previous_total_volume.replace(0.0, np.nan)
        volume_change = volume_change.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    daily_volatility = daily_returns.std(axis=0, skipna=True)
    volatility = daily_volatility * math.sqrt(252)
    mean_daily_return = daily_returns.mean(axis=0, skipna=True)
    risk_adjusted_return = mean_daily_return / daily_volatility.replace(0.0, np.nan) * math.sqrt(252)
    risk_adjusted_return = risk_adjusted_return.replace([np.inf, -np.inf], np.nan)

    downside_returns = daily_returns.where(daily_returns < 0.0, 0.0)
    downside_volatility = downside_returns.std(axis=0, skipna=True) * math.sqrt(252)

    positive_days_ratio = (daily_returns > 0.0).sum(axis=0) / daily_returns.notna().sum(axis=0).replace(0, np.nan)
    max_drawdown = calculate_max_drawdown(daily_returns)
    drawdown_abs = -max_drawdown

    beta_to_benchmark, corr_to_benchmark = calculate_beta_and_corr(daily_returns, benchmark)

    daily_dollar_volume = (month_prices * month_volumes).replace(0.0, np.nan)
    amihud_illiquidity = (daily_returns.abs() / daily_dollar_volume).mean(axis=0, skipna=True)

    metrics = pd.DataFrame(
        {
            "rebalance_date": rebalance_date,
            "window_start": window_start,
            "window_end": window_end,
            "etf": month_prices.columns,
            "observations": observations,
            "start_price": start_price,
            "end_price": end_price,
            "price_diff": price_diff,
            "price_change": price_change,
            "total_volume": total_volume,
            "avg_price": avg_price,
            "dollar_volume": dollar_volume,
            "volume_change": volume_change,
            "volatility": volatility,
            "downside_volatility": downside_volatility,
            "risk_adjusted_return": risk_adjusted_return,
            "positive_days_ratio": positive_days_ratio,
            "max_drawdown": max_drawdown,
            "drawdown_abs": drawdown_abs,
            "beta_to_benchmark": beta_to_benchmark,
            "corr_to_benchmark": corr_to_benchmark,
            "amihud_illiquidity": amihud_illiquidity,
        }
    ).reset_index(drop=True)

    metrics["log_total_volume"] = np.log1p(metrics["total_volume"].clip(lower=0.0))
    metrics["log_dollar_volume"] = np.log1p(metrics["dollar_volume"].clip(lower=0.0))

    if benchmark in set(metrics["etf"]):
        b = metrics.loc[metrics["etf"] == benchmark].iloc[0]
        metrics["benchmark_price_change"] = b["price_change"]
        metrics["excess_return_vs_benchmark"] = metrics["price_change"] - b["price_change"]
        metrics["momentum_ratio_vs_benchmark"] = (1.0 + metrics["price_change"]) / (1.0 + b["price_change"]) - 1.0
        metrics["volatility_ratio_vs_benchmark"] = metrics["volatility"] / b["volatility"] if b["volatility"] > 0 else np.nan
        metrics["risk_adjusted_return_benchmark"] = b["risk_adjusted_return"]
        metrics["risk_adjusted_excess_return"] = metrics["excess_return_vs_benchmark"] / metrics["volatility"].replace(0.0, np.nan)
        metrics["total_volume_ratio_vs_benchmark"] = metrics["total_volume"] / b["total_volume"] if b["total_volume"] > 0 else np.nan
        metrics["dollar_volume_ratio_vs_benchmark"] = metrics["dollar_volume"] / b["dollar_volume"] if b["dollar_volume"] > 0 else np.nan
    else:
        for col in ["benchmark_price_change", "excess_return_vs_benchmark", "momentum_ratio_vs_benchmark", 
                    "volatility_ratio_vs_benchmark", "risk_adjusted_return_benchmark", "risk_adjusted_excess_return", 
                    "total_volume_ratio_vs_benchmark", "dollar_volume_ratio_vs_benchmark"]:
            metrics[col] = np.nan

    return metrics


# -----------------------------------------------------------------------------
# Filtering and scoring
# -----------------------------------------------------------------------------

def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)

    fill = x.median(skipna=True)
    x = x.fillna(fill)

    lo, hi = x.quantile(0.01), x.quantile(0.99)
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        x = x.clip(lower=lo, upper=hi)

    # pandas assigns rank 1 to the smallest value when ascending=True.
    # Because the aggregate score is sorted in descending order, desirable
    # observations must receive the largest percentile: use ascending=True for
    # higher-is-better metrics and ascending=False for lower-is-better metrics.
    return x.rank(method="average", pct=True, ascending=higher_is_better)


def add_aggregate_score(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    score = pd.Series(0.0, index=out.index, dtype=float)
    used_weight = 0.0

    for col, weight in HIGHER_IS_BETTER_WEIGHTS.items():
        if col in out.columns:
            score += weight * percentile_score(out[col], higher_is_better=True)
            used_weight += weight

    for col, weight in LOWER_IS_BETTER_WEIGHTS.items():
        if col in out.columns:
            score += weight * percentile_score(out[col], higher_is_better=False)
            used_weight += weight

    if used_weight <= 0:
        raise ValueError("No scoring metrics were available.")

    out["aggregate_score"] = score / used_weight
    return out


def apply_base_filters(metrics: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    out = metrics.copy()
    out["passed_base_filter"] = True

    out.loc[out["observations"] < cfg.min_observations, "passed_base_filter"] = False
    out.loc[out["end_price"] < cfg.min_price, "passed_base_filter"] = False
    out.loc[out["total_volume"] < cfg.min_total_volume, "passed_base_filter"] = False

    if cfg.exclude_benchmark and cfg.benchmark in set(out["etf"]):
        out.loc[out["etf"] == cfg.benchmark, "passed_base_filter"] = False

    return out


def apply_benchmark_filter(metrics: pd.DataFrame, cfg: FilterConfig) -> pd.DataFrame:
    out = metrics.copy()
    out["passed_benchmark_filter"] = True

    mode = cfg.spy_filter.lower()
    if mode == "none" or cfg.benchmark not in set(out["etf"]):
        return out

    b = out.loc[out["etf"] == cfg.benchmark].iloc[0]

    if mode == "return":
        condition = out["price_change"] > b["price_change"]
    elif mode == "smart":
        condition = (
            (out["price_change"] > b["price_change"]) &
            (out["risk_adjusted_return"] > b["risk_adjusted_return"]) &
            (out["volatility"] <= 1.5 * b["volatility"])
        )
    elif mode == "strict":
        condition = (
            (out["price_change"] > b["price_change"]) &
            (out["risk_adjusted_return"] > b["risk_adjusted_return"]) &
            (out["positive_days_ratio"] > b["positive_days_ratio"]) &
            (out["total_volume"] > b["total_volume"]) &
            (out["volume_change"] > b["volume_change"]) &
            (out["volatility"] < b["volatility"]) &
            (out["drawdown_abs"] < b["drawdown_abs"])
        )
    else:
        raise ValueError("spy_filter must be one of: none, return, smart, strict")

    out.loc[~condition.fillna(False), "passed_benchmark_filter"] = False
    return out


def select_top_etfs(month_metrics: pd.DataFrame, cfg: FilterConfig, previous_scores: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    scored = add_aggregate_score(month_metrics)
    
    # ---------------------------------------------------------
    # Apply Exponential Moving Average (EMA) smoothing to score
    # ---------------------------------------------------------
    if previous_scores is not None and cfg.score_ema_alpha < 1.0:
        scored = scored.set_index("etf")
        aligned_prev = previous_scores.reindex(scored.index).fillna(scored["aggregate_score"])
        scored["aggregate_score"] = (
            cfg.score_ema_alpha * scored["aggregate_score"] + 
            (1.0 - cfg.score_ema_alpha) * aligned_prev
        )
        scored = scored.reset_index()

    current_scores = scored.set_index("etf")["aggregate_score"]

    scored = apply_base_filters(scored, cfg)
    scored = apply_benchmark_filter(scored, cfg)

    scored["eligible_after_filters"] = scored["passed_base_filter"] & scored["passed_benchmark_filter"]
    scored["selected"] = False
    scored["rank"] = np.nan

    selected = scored.loc[scored["eligible_after_filters"]].sort_values(
        by=["aggregate_score", "price_change", "log_dollar_volume"],
        ascending=[False, False, False],
    )

    top_idx = selected.head(cfg.top_n).index
    scored.loc[top_idx, "rank"] = np.arange(1, len(top_idx) + 1)
    scored.loc[top_idx, "selected"] = True
    return scored, current_scores


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def get_month_window(
    prices_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    include_end: bool,
) -> np.ndarray:
    if include_end:
        return (prices_index >= start) & (prices_index <= end)
    return (prices_index >= start) & (prices_index < end)


def run_monthly_filtration(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    cfg: FilterConfig,
) -> pd.DataFrame:
    prices, volumes, _ = align_price_volume_columns(prices, volumes)

    rebalance_dates = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))).sort_values()
    all_months: List[pd.DataFrame] = []

    previous_total_volume: Optional[pd.Series] = None
    previous_scores: Optional[pd.Series] = None

    for rebalance_date in rebalance_dates:
        # Use lookback_days to define the evaluation window (Solution 1)
        window_start = rebalance_date - pd.Timedelta(days=cfg.lookback_days)
        window_end = rebalance_date
        
        mask = get_month_window(prices.index, window_start, window_end, cfg.include_rebalance_day)
        month_prices = prices.loc[mask]
        month_volumes = volumes.reindex(prices.index).loc[mask]

        if len(month_prices) < cfg.min_observations:
            continue

        month_metrics = calculate_month_metrics(
            month_prices=month_prices,
            month_volumes=month_volumes,
            previous_total_volume=previous_total_volume,
            rebalance_date=rebalance_date,
            window_start=window_start,
            window_end=window_end,
            benchmark=cfg.benchmark,
        )
        
        # Calculate scores, apply smoothing (Solution 2), and rank
        month_scored, current_scores = select_top_etfs(month_metrics, cfg, previous_scores)
        all_months.append(month_scored)

        # Save states for the next month's loop
        previous_total_volume = month_metrics.set_index("etf")["total_volume"]
        previous_scores = current_scores

    if not all_months:
        raise ValueError("No monthly windows were processed. Check your dates and min_observations.")

    return pd.concat(all_months, ignore_index=True)


def save_outputs(scores: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = scores.copy()
    for col in ["rebalance_date", "window_start", "window_end"]:
        scores[col] = pd.to_datetime(scores[col]).dt.strftime("%Y-%m-%d")

    scores.to_csv(output_dir / "monthly_etf_scores.csv", index=False)

    top_long = scores.loc[scores["rank"].notna()].copy()
    top_long["rank"] = top_long["rank"].astype(int)
    top_long = top_long.sort_values(["rebalance_date", "rank"])
    top_long.to_csv(output_dir / "top_etfs_long.csv", index=False)

    rows = []
    by_month: Dict[str, List[str]] = {}
    for date, group in top_long.groupby("rebalance_date"):
        etfs = group.sort_values("rank")["etf"].head(top_n).tolist()
        by_month[date] = etfs
        row = {"rebalance_date": date}
        row.update({f"etf_{i + 1}": etf for i, etf in enumerate(etfs)})
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "top_etfs_wide.csv", index=False)

    with open(output_dir / "top_etfs_by_month.json", "w", encoding="utf-8") as f:
        json.dump(by_month, f, indent=2)

    # Universe stability diagnostics: symmetric-difference turnover and overlap.
    turnover_rows = []
    previous: set[str] | None = None
    for date in sorted(by_month):
        current = set(by_month[date])
        if previous is not None:
            union = previous | current
            turnover_rows.append({
                "rebalance_date": date,
                "universe_turnover_half_l1": len(previous ^ current) / max(2 * top_n, 1),
                "jaccard_overlap": len(previous & current) / max(len(union), 1),
                "entered": len(current - previous),
                "exited": len(previous - current),
            })
        previous = current
    pd.DataFrame(turnover_rows).to_csv(output_dir / "universe_turnover.csv", index=False)

    if yaml is not None:
        with open(output_dir / "top_etfs_by_month.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(by_month, f, sort_keys=True, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly ETF filtration and top-N ETF selector.")
    parser.add_argument("--data-root", type=Path, default=Path.cwd(), help="Project root. Relative input paths are resolved from here.")
    parser.add_argument("--prices", type=Path, default=Path("datasets/csv/NewClosePrice.csv"), help="Close price CSV path.")
    parser.add_argument("--volumes", type=Path, default=Path("datasets/csv/Volume.csv"), help="Volume CSV path.")
    parser.add_argument("--business-dates", type=Path, default=Path("datasets/excel/business_dates.xlsx"), help="Excel file with monthly rebalance dates.")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/monthly_filtration"), help="Directory for output files.")
    parser.add_argument("--top-n", type=int, default=100, help="Number of ETFs to select per month.")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker for relative metrics.")
    parser.add_argument("--spy-filter", choices=["none", "return", "smart", "strict"], default="smart", help="Hard benchmark filter mode.")
    parser.add_argument("--include-benchmark", action="store_true", help="Allow benchmark ETF itself to be selected.")
    parser.add_argument("--min-observations", type=int, default=10, help="Minimum price observations required in the month.")
    parser.add_argument("--min-price", type=float, default=1.0, help="Minimum ending ETF price.")
    parser.add_argument("--min-total-volume", type=float, default=0.0, help="Minimum monthly total volume.")
    
    # New Arguments for smoothing/inertia
    parser.add_argument("--lookback-days", type=int, default=90, help="Lookback window in days (e.g., 90 for 3 months).")
    parser.add_argument("--score-ema-alpha", type=float, default=0.5, help="Smoothing factor for scores (0.0 to 1.0). 1.0 = no smoothing.")
    
    parser.add_argument("--include-rebalance-day", action="store_true", help="Include rebalance day in metric window. Default avoids look-ahead.")
    parser.add_argument("--do-not-skip-first-business-date", action="store_true", help="Use first row from business_dates.xlsx instead of skipping it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()

    price_path = resolve_path(args.prices, data_root)
    volume_path = resolve_path(args.volumes, data_root)
    business_dates_path = resolve_path(args.business_dates, data_root)
    output_dir = resolve_path(args.output_dir, data_root)

    prices = read_market_csv(price_path)
    volumes = read_market_csv(volume_path)

    if business_dates_path.exists():
        rebalance_dates = read_business_dates(
            business_dates_path,
            skip_first=not args.do_not_skip_first_business_date,
        )
    else:
        rebalance_dates = infer_first_trading_days(prices)

    cfg = FilterConfig(
        top_n=args.top_n,
        benchmark=args.benchmark,
        spy_filter=args.spy_filter,
        exclude_benchmark=not args.include_benchmark,
        min_observations=args.min_observations,
        min_price=args.min_price,
        min_total_volume=args.min_total_volume,
        include_rebalance_day=args.include_rebalance_day,
        skip_first_business_date=not args.do_not_skip_first_business_date,
        lookback_days=args.lookback_days,
        score_ema_alpha=args.score_ema_alpha,
    )

    scores = run_monthly_filtration(prices, volumes, rebalance_dates, cfg)
    save_outputs(scores, output_dir, top_n=cfg.top_n)

    selected_count = int(scores["rank"].notna().sum())
    month_count = scores["rebalance_date"].nunique()
    print(f"Processed {month_count} monthly windows using a {cfg.lookback_days}-day lookback and EMA alpha of {cfg.score_ema_alpha}.")
    print(f"Saved {selected_count} selected ETF rows.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
