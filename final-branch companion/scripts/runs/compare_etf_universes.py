"""Causal ETF-universe ablation with a deliberately simple equal-weight strategy.

This script answers one narrow research question:

    Does the richer monthly ETF universe improve the *action set itself* relative
    to the legacy ``new_filter_data.py`` universe, before a Bayesian model or
    optimizer can confound the comparison?

The comparison uses the same project market files and rebalance calendar as
``scripts/runs/run.py``.  It evaluates each universe with a monthly equal-weight,
fully-invested buy-and-hold portfolio, charges proportional turnover costs, and
exports matrices that can be fed back into ``run.py`` for a second-stage
full-pipeline ablation.

Primary arms
------------
1. ``basic_legacy``
   Legacy score/eligibility logic, including the historical sufficiency test as
   implemented in ``new_filter_data.py``.  This arm is retained for provenance;
   its history test is not a true count of 36 positive months.
2. ``basic_repaired``
   Same legacy score, but with a causal 36-completed-month positive-traded-value
   history requirement.  This is the stronger baseline and should be the
   publication comparison.
3. ``advanced_matched``
   ``monthly_etf_filtration_2.py`` with the same top-N breadth as the basic arm,
   no hard SPY gate, and SPY allowed.  This isolates the richer scoring system.
4. ``advanced_native``
   The richer filter with its requested/native top-N, SPY filter, and benchmark
   exclusion settings.

The code does not force a winner.  It reports performance, downside risk,
universe quality/stability, and paired block-bootstrap evidence.  A claim that
one universe is "better" is warranted only if the realized outputs support it.

Typical project-root command
----------------------------
python scripts/runs/compare_etf_universes.py \
    --output-dir results/universe_comparison \
    --basic-top-n 80 \
    --advanced-native-top-n 100 \
    --advanced-native-spy-filter smart \
    --cost-bps 10 \
    --bootstrap-reps 5000

The default input paths mirror ``scripts/runs/run.py``:
    datasets/csv/NewClosePrice.csv
    datasets/csv/OpenPrice.csv
    datasets/csv/Volume.csv
    datasets/excel/business_dates.xlsx
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.dataloader.monthly_etf_filtration_2 import (
    FilterConfig,
    calculate_month_metrics,
    infer_first_trading_days,
    read_business_dates,
    read_market_csv,
    run_monthly_filtration,
)

try:
    from scripts.results.stats import metric_summary
except Exception:  # keep the universe audit runnable even if optional stats deps fail
    metric_summary = None


EPS = 1e-12
PERIODS_PER_YEAR = 12

# Keep overlapping basic arms visually distinguishable.  In many months the
# legacy and repaired screens select exactly the same ETFs, so a solid line
# drawn second would otherwise hide the first one completely.
PLOT_STYLES = {
    "basic_legacy": {
        "color": "tab:blue",
        "linestyle": "--",
        "linewidth": 2.2,
        "marker": "o",
        "markersize": 3.0,
        "markevery": 3,
        "zorder": 4,
    },
    "basic_repaired": {
        "color": "tab:orange",
        "linestyle": "-",
        "linewidth": 2.0,
        "zorder": 3,
    },
    "advanced_matched": {"color": "tab:green", "linewidth": 2.0, "zorder": 2},
    "advanced_native": {"color": "tab:red", "linewidth": 2.0, "zorder": 2},
}


@dataclass
class BacktestOutput:
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    selected_count: pd.Series
    valid_count: pd.Series
    coverage: pd.Series


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare legacy and advanced ETF universes using equal-weight portfolios.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--prices", type=Path, default=Path("datasets/csv/NewClosePrice.csv"))
    p.add_argument("--open-prices", type=Path, default=Path("datasets/csv/OpenPrice.csv"))
    p.add_argument("--volumes", type=Path, default=Path("datasets/csv/Volume.csv"))
    p.add_argument("--business-dates", type=Path, default=Path("datasets/excel/business_dates.xlsx"))
    p.add_argument("--output-dir", type=Path, default=Path("results/universe_comparison"))
    p.add_argument("--export-run-matrices-dir", type=Path, default=Path("datasets/excel/universe_comparison"))

    p.add_argument("--basic-top-n", type=int, default=80)
    p.add_argument("--legacy-liquidity-threshold", type=float, default=0.5e-4)
    p.add_argument("--legacy-liquidity-window-months", type=int, default=36)
    p.add_argument("--legacy-history-months", type=int, default=36)

    p.add_argument("--advanced-lookback-days", type=int, default=90)
    p.add_argument("--advanced-score-ema-alpha", type=float, default=0.5)
    p.add_argument("--advanced-min-observations", type=int, default=10)
    p.add_argument("--advanced-min-price", type=float, default=1.0)
    p.add_argument("--advanced-min-total-volume", type=float, default=0.0)
    p.add_argument("--advanced-native-top-n", type=int, default=100)
    p.add_argument("--advanced-native-spy-filter", choices=["none", "return", "smart", "strict"], default="smart")
    p.add_argument("--benchmark", type=str, default="SPY")

    p.add_argument("--warmup-years", type=int, default=3, help="Match run.py's default 3-year warmup before evaluation.")
    p.add_argument("--risk-free-rate", type=float, default=0.02)
    p.add_argument("--cost-bps", type=float, default=10.0, help="Cost per unit one-way turnover, in basis points.")
    p.add_argument("--charge-initial-trade", action="store_true")
    p.add_argument("--bootstrap-reps", type=int, default=5000)
    p.add_argument("--bootstrap-block-length", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rolling-window", type=int, default=12)
    return p.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def _shared_columns(*frames: pd.DataFrame) -> list[str]:
    cols = set(frames[0].columns)
    for f in frames[1:]:
        cols &= set(f.columns)
    return [c for c in frames[0].columns if c in cols]


def _completed_month_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily values to calendar months and index by Period[M]."""
    out = df.groupby(df.index.to_period("M")).sum(min_count=1)
    out.index.name = "month"
    return out


def build_basic_universes(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    *,
    top_n: int,
    benchmark: str,
    liquidity_threshold: float,
    liquidity_window_months: int,
    history_months: int,
) -> tuple[dict[str, dict[pd.Timestamp, set[str]]], pd.DataFrame]:
    """Rebuild the legacy ``new_filter_data.py`` score on the shared calendar.

    Two eligibility variants are returned.  ``basic_legacy`` keeps the original
    sum-based history condition for provenance.  ``basic_repaired`` replaces it
    by a causal count of positive traded-value months.
    """
    # OpenPrice is only an optional refinement of the traded-value proxy.  Some
    # project snapshots end much earlier than NewClosePrice/Volume, and some
    # newer ETFs are absent from OpenPrice altogether.  Restricting `common` to
    # all three files therefore truncates both basic arms and also (in main)
    # unfairly removes assets from the advanced arms.
    common = _shared_columns(close, volume)
    close = close[common].sort_index()
    open_ = open_.reindex(index=close.index, columns=common)
    volume = volume[common].reindex(close.index)

    valid_close = close.where(close > 0.0)
    valid_open = open_.where(open_ > 0.0)
    midpoint = (valid_close + valid_open) / 2.0
    traded_price = midpoint.combine_first(valid_close).combine_first(valid_open)

    fallback_count = int((valid_close.notna() & valid_open.isna()).to_numpy().sum())
    if fallback_count:
        warnings.warn(
            f"OpenPrice is unavailable for {fallback_count:,} valid close-price "
            "observations; using close price as the traded-value proxy for those "
            "observations so the basic universes are not truncated.",
            UserWarning,
            stacklevel=2,
        )

    traded_value = traded_price * volume
    monthly_amount = _completed_month_table(traded_value)

    # Legacy liquidity share: percentage of 36-month rolling mean, lagged one
    # month so the month being entered does not provide its own liquidity.
    rolling_amount = monthly_amount.rolling(
        liquidity_window_months, min_periods=liquidity_window_months
    ).mean()
    liquidity_share = rolling_amount.div(rolling_amount.sum(axis=1), axis=0)
    lagged_liquidity_share = liquidity_share.shift(1)
    monthly_liq_z = _zscore_cross_section(liquidity_share).shift(1)

    # As-implemented history condition in new_filter_data.py.  It is preserved
    # as a provenance arm but should not be used as the strongest publication
    # baseline because it compares a traded-value sum with the integer 36.
    legacy_history = (
        monthly_amount.rolling(history_months + 1, min_periods=history_months + 1)
        .sum()
        .gt(history_months)
    )

    # Causal repair: exactly H completed months with positive traded value.
    completed_positive = monthly_amount.shift(1).gt(0.0).astype(int)
    repaired_history = (
        completed_positive.rolling(history_months, min_periods=history_months)
        .sum()
        .eq(history_months)
    )

    legacy_eligible = legacy_history & (lagged_liquidity_share > liquidity_threshold)
    repaired_eligible = repaired_history & (lagged_liquidity_share > liquidity_threshold)

    returns_daily = close.pct_change(fill_method=None)
    mom_12m = close.pct_change(252, fill_method=None).shift(1)
    mom_3m = close.pct_change(63, fill_method=None).shift(1)
    vol_3m = returns_daily.rolling(63).std().shift(1)
    rolling_max_6m = close.rolling(126).max()
    drawdown_6m = (close / rolling_max_6m - 1.0).shift(1)

    if benchmark in close.columns:
        spy_mom_6m = close[benchmark].pct_change(126, fill_method=None).shift(1)
        rel_strength = close.pct_change(126, fill_method=None).shift(1).sub(spy_mom_6m, axis=0)
    else:
        rel_strength = mom_3m * 0.0

    mom12_z = _zscore_cross_section(mom_12m)
    mom3_z = _zscore_cross_section(mom_3m)
    rel_z = _zscore_cross_section(rel_strength)
    vol_z = _zscore_cross_section(vol_3m)
    dd_z = _zscore_cross_section(drawdown_6m.abs())

    dates = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))).sort_values()
    universes = {"basic_legacy": {}, "basic_repaired": {}}
    audit_rows: list[dict] = []

    for d in dates:
        period = d.to_period("M")
        if period not in lagged_liquidity_share.index:
            universes["basic_legacy"][d] = set()
            universes["basic_repaired"][d] = set()
            continue

        # Daily signal values are shifted one trading observation, so the row on
        # the rebalance date itself contains information only through t-1.
        available_signal_dates = mom12_z.index[mom12_z.index <= d]
        if len(available_signal_dates) == 0:
            universes["basic_legacy"][d] = set()
            universes["basic_repaired"][d] = set()
            continue
        sd = available_signal_dates[-1]

        liq_z = monthly_liq_z.loc[period].reindex(common).fillna(0.0)
        score = (
            0.35 * mom12_z.loc[sd].reindex(common).fillna(0.0)
            + 0.25 * mom3_z.loc[sd].reindex(common).fillna(0.0)
            + 0.20 * rel_z.loc[sd].reindex(common).fillna(0.0)
            + 0.15 * liq_z
            - 0.30 * vol_z.loc[sd].reindex(common).fillna(0.0)
            - 0.15 * dd_z.loc[sd].reindex(common).fillna(0.0)
        )

        for arm, elig_table in [
            ("basic_legacy", legacy_eligible),
            ("basic_repaired", repaired_eligible),
        ]:
            elig = elig_table.loc[period].reindex(common).fillna(False).astype(bool)
            ranked = score.where(elig).dropna().sort_values(ascending=False)
            selected = set(ranked.head(top_n).index)
            universes[arm][d] = selected
            audit_rows.append({
                "date": d,
                "arm": arm,
                "signal_date": sd,
                "eligible_before_top_n": int(elig.sum()),
                "selected_count": len(selected),
                "causal_history_rule": arm == "basic_repaired",
                "note": (
                    "Legacy sum-based history condition from new_filter_data.py"
                    if arm == "basic_legacy"
                    else f"Causal {history_months}-completed-month positive traded-value requirement"
                ),
            })

    return universes, pd.DataFrame(audit_rows)


def advanced_scores_to_universe(scores: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    selected = scores.loc[scores["selected"].fillna(False)].copy()
    selected["rebalance_date"] = pd.to_datetime(selected["rebalance_date"])
    by_date = {
        pd.Timestamp(d): set(g["etf"].astype(str))
        for d, g in selected.groupby("rebalance_date")
    }
    all_dates = pd.to_datetime(scores["rebalance_date"].unique())
    return {pd.Timestamp(d): by_date.get(pd.Timestamp(d), set()) for d in all_dates}


def build_advanced_universes(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    rebalance_dates: Iterable[pd.Timestamp],
    *,
    basic_top_n: int,
    native_top_n: int,
    native_spy_filter: str,
    benchmark: str,
    lookback_days: int,
    score_ema_alpha: float,
    min_observations: int,
    min_price: float,
    min_total_volume: float,
) -> tuple[dict[str, dict[pd.Timestamp, set[str]]], dict[str, pd.DataFrame]]:
    # Matched arm: same breadth and benchmark treatment as the legacy screen so
    # any result is not mechanically caused by N or an SPY hard gate.
    matched_cfg = FilterConfig(
        top_n=basic_top_n,
        benchmark=benchmark,
        spy_filter="none",
        exclude_benchmark=False,
        min_observations=min_observations,
        min_price=min_price,
        min_total_volume=min_total_volume,
        include_rebalance_day=False,
        skip_first_business_date=False,
        lookback_days=lookback_days,
        score_ema_alpha=score_ema_alpha,
    )
    native_cfg = FilterConfig(
        top_n=native_top_n,
        benchmark=benchmark,
        spy_filter=native_spy_filter,
        exclude_benchmark=True,
        min_observations=min_observations,
        min_price=min_price,
        min_total_volume=min_total_volume,
        include_rebalance_day=False,
        skip_first_business_date=False,
        lookback_days=lookback_days,
        score_ema_alpha=score_ema_alpha,
    )

    matched_scores = run_monthly_filtration(close, volume, rebalance_dates, matched_cfg)
    native_scores = run_monthly_filtration(close, volume, rebalance_dates, native_cfg)
    return (
        {
            "advanced_matched": advanced_scores_to_universe(matched_scores),
            "advanced_native": advanced_scores_to_universe(native_scores),
        },
        {"advanced_matched": matched_scores, "advanced_native": native_scores},
    )


def _asset_period_returns(close: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tickers: set[str]) -> pd.Series:
    cols = [c for c in tickers if c in close.columns]
    if not cols or start not in close.index or end not in close.index:
        return pd.Series(dtype=float)
    p0 = pd.to_numeric(close.loc[start, cols], errors="coerce")
    p1 = pd.to_numeric(close.loc[end, cols], errors="coerce")
    valid = p0.notna() & p1.notna() & (p0 > 0.0) & (p1 > 0.0)
    if not valid.any():
        return pd.Series(dtype=float)
    return (p1[valid] / p0[valid] - 1.0).astype(float)


def equal_weight_backtest(
    close: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    universe: dict[pd.Timestamp, set[str]],
    *,
    cost_bps: float,
    charge_initial_trade: bool,
) -> BacktestOutput:
    net: dict[pd.Timestamp, float] = {}
    gross: dict[pd.Timestamp, float] = {}
    turnover: dict[pd.Timestamp, float] = {}
    costs: dict[pd.Timestamp, float] = {}
    selected_count: dict[pd.Timestamp, float] = {}
    valid_count: dict[pd.Timestamp, float] = {}
    coverage: dict[pd.Timestamp, float] = {}

    pretrade_weights: pd.Series | None = None
    cost_rate = float(cost_bps) / 10000.0

    for i in range(len(rebalance_dates) - 1):
        d = pd.Timestamp(rebalance_dates[i])
        nxt = pd.Timestamp(rebalance_dates[i + 1])
        selected = set(universe.get(d, set()))
        selected_count[d] = len(selected)
        asset_r = _asset_period_returns(close, d, nxt, selected)
        valid_count[d] = len(asset_r)
        coverage[d] = len(asset_r) / max(len(selected), 1)
        if asset_r.empty:
            continue

        target = pd.Series(1.0 / len(asset_r), index=asset_r.index, dtype=float)
        if pretrade_weights is None:
            turn = 1.0 if charge_initial_trade else 0.0
        else:
            union = pretrade_weights.index.union(target.index)
            turn = 0.5 * float(
                (target.reindex(union, fill_value=0.0) - pretrade_weights.reindex(union, fill_value=0.0))
                .abs().sum()
            )

        gross_r = float((target * asset_r).sum())
        cost = cost_rate * turn
        net_r = (1.0 - cost) * (1.0 + gross_r) - 1.0

        # Weights drift during the month because we rebalance only at month start.
        ending_value = target * (1.0 + asset_r)
        denom = float(ending_value.sum())
        pretrade_weights = ending_value / denom if denom > EPS else target.copy()

        gross[d] = gross_r
        net[d] = net_r
        turnover[d] = turn
        costs[d] = cost

    def s(x: dict[pd.Timestamp, float]) -> pd.Series:
        return pd.Series(x, dtype=float).sort_index()

    return BacktestOutput(
        returns=s(net),
        gross_returns=s(gross),
        turnover=s(turnover),
        costs=s(costs),
        selected_count=s(selected_count),
        valid_count=s(valid_count),
        coverage=s(coverage),
    )


def _simple_metrics(returns: pd.Series, rf: float) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return {}
    n = len(r)
    years = n / PERIODS_PER_YEAR
    gross = float(np.prod(1.0 + np.clip(r.to_numpy(), -0.999999, None)))
    ann = gross ** (1.0 / years) - 1.0 if years > 0 and gross > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)) if n > 1 else np.nan
    excess = r - rf / PERIODS_PER_YEAR
    sharpe = float(excess.mean() / excess.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)) if n > 1 and excess.std(ddof=1) > EPS else np.nan
    downside = r.clip(upper=0.0)
    down_dev = float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(PERIODS_PER_YEAR))
    sortino = float((ann - rf) / down_dev) if down_dev > EPS else np.nan
    wealth = (1.0 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    maxdd = float(dd.min())
    q = float(r.quantile(0.05))
    cvar = float(r[r <= q].mean()) if (r <= q).any() else q
    calmar = float(ann / abs(maxdd)) if abs(maxdd) > EPS else np.nan
    return {
        "Cumulative Return": gross - 1.0,
        "Annualized Return": ann,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": maxdd,
        "CVaR (95%)": cvar,
        "Calmar Ratio": calmar,
        "N Periods": float(n),
    }


def performance_summary(
    name: str,
    out: BacktestOutput,
    *,
    risk_free_rate: float,
    benchmark_returns: pd.Series | None,
) -> dict[str, float | str]:
    r = out.returns.dropna()
    if metric_summary is not None and not r.empty:
        balance = 1000.0 * (1.0 + r).cumprod()
        pnl = pd.DataFrame({"Returns": r, "Balance": balance}, index=r.index)
        try:
            m = metric_summary(
                pnl,
                risk_free_rate=risk_free_rate,
                periods_per_year=PERIODS_PER_YEAR,
                benchmark_returns=benchmark_returns,
                num_trials=1,
                trial_sharpe_std=0.0,
            )
        except Exception:
            m = _simple_metrics(r, risk_free_rate)
    else:
        m = _simple_metrics(r, risk_free_rate)

    m.update({
        "arm": name,
        "Average Monthly Turnover": float(out.turnover.reindex(r.index).mean()) if not r.empty else np.nan,
        "Annualized One-Way Turnover": float(out.turnover.reindex(r.index).mean() * 12.0) if not r.empty else np.nan,
        "Cumulative Transaction Cost": float(out.costs.reindex(r.index).sum()) if not r.empty else np.nan,
        "Average Selected Count": float(out.selected_count.reindex(r.index).mean()) if not r.empty else np.nan,
        "Average Valid Count": float(out.valid_count.reindex(r.index).mean()) if not r.empty else np.nan,
        "Average Price Coverage": float(out.coverage.reindex(r.index).mean()) if not r.empty else np.nan,
    })
    return m


def benchmark_returns(close: pd.DataFrame, dates: pd.DatetimeIndex, benchmark: str) -> pd.Series:
    if benchmark not in close.columns:
        return pd.Series(dtype=float)
    out = {}
    for i in range(len(dates) - 1):
        d, nxt = pd.Timestamp(dates[i]), pd.Timestamp(dates[i + 1])
        if d in close.index and nxt in close.index:
            p0, p1 = close.at[d, benchmark], close.at[nxt, benchmark]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                out[d] = float(p1 / p0 - 1.0)
    return pd.Series(out, dtype=float).sort_index()


def universe_stability(universes: dict[str, dict[pd.Timestamp, set[str]]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for arm, u in universes.items():
        prev: set[str] | None = None
        for d in dates:
            cur = set(u.get(pd.Timestamp(d), set()))
            if prev is not None:
                union = prev | cur
                inter = prev & cur
                rows.append({
                    "date": d,
                    "arm": arm,
                    "selected_count": len(cur),
                    "replacement_fraction": len(prev ^ cur) / max(2 * max(len(prev), len(cur), 1), 1),
                    "jaccard_overlap": len(inter) / max(len(union), 1),
                    "entered": len(cur - prev),
                    "exited": len(prev - cur),
                })
            prev = cur
    return pd.DataFrame(rows)


def cross_universe_overlap(universes: dict[str, dict[pd.Timestamp, set[str]]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    arms = list(universes)
    rows = []
    for d in dates:
        for i, a in enumerate(arms):
            for b in arms[i + 1 :]:
                sa = set(universes[a].get(pd.Timestamp(d), set()))
                sb = set(universes[b].get(pd.Timestamp(d), set()))
                union = sa | sb
                rows.append({
                    "date": d,
                    "arm_a": a,
                    "arm_b": b,
                    "intersection": len(sa & sb),
                    "union": len(union),
                    "jaccard": len(sa & sb) / max(len(union), 1),
                })
    return pd.DataFrame(rows)


def compute_common_quality_table(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    dates: pd.DatetimeIndex,
    universes: dict[str, dict[pd.Timestamp, set[str]]],
    *,
    benchmark: str,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prev_volume: pd.Series | None = None
    for d in dates:
        start = d - pd.Timedelta(days=lookback_days)
        mask = (close.index >= start) & (close.index < d)
        p = close.loc[mask]
        v = volume.reindex(close.index).loc[mask]
        if len(p) < 3:
            continue
        metrics = calculate_month_metrics(p, v, prev_volume, d, start, d, benchmark=benchmark)
        prev_volume = metrics.set_index("etf")["total_volume"]
        metrics = metrics.set_index("etf")
        for arm, u in universes.items():
            tickers = [t for t in u.get(pd.Timestamp(d), set()) if t in metrics.index]
            if not tickers:
                continue
            sub = metrics.loc[tickers]
            rows.append({
                "date": d,
                "arm": arm,
                "n_assets": len(sub),
                "median_observations_90d": float(sub["observations"].median()),
                "median_dollar_volume": float(sub["dollar_volume"].median()),
                "median_volatility": float(sub["volatility"].median()),
                "median_downside_volatility": float(sub["downside_volatility"].median()),
                "median_drawdown_abs": float(sub["drawdown_abs"].median()),
                "median_risk_adjusted_return": float(sub["risk_adjusted_return"].median()),
                "median_amihud_illiquidity": float(sub["amihud_illiquidity"].median()),
                "median_abs_corr_to_benchmark": float(sub["corr_to_benchmark"].abs().median()),
            })
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        return monthly, pd.DataFrame()
    summary = monthly.groupby("arm").agg(
        months=("date", "count"),
        avg_assets=("n_assets", "mean"),
        median_observations_90d=("median_observations_90d", "median"),
        median_dollar_volume=("median_dollar_volume", "median"),
        median_volatility=("median_volatility", "median"),
        median_downside_volatility=("median_downside_volatility", "median"),
        median_drawdown_abs=("median_drawdown_abs", "median"),
        median_risk_adjusted_return=("median_risk_adjusted_return", "median"),
        median_amihud_illiquidity=("median_amihud_illiquidity", "median"),
        median_abs_corr_to_benchmark=("median_abs_corr_to_benchmark", "median"),
    ).reset_index()
    return monthly, summary


def _metric_vector(r: np.ndarray, rf: float) -> dict[str, float]:
    s = pd.Series(r)
    return _simple_metrics(s, rf)


def _circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    idx: list[int] = []
    L = max(1, min(int(block_length), n))
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend(((start + np.arange(L)) % n).tolist())
    return np.asarray(idx[:n], dtype=int)


def paired_block_bootstrap(
    basic: pd.Series,
    advanced: pd.Series,
    *,
    reps: int,
    block_length: int,
    risk_free_rate: float,
    seed: int,
    label: str,
) -> pd.DataFrame:
    pair = pd.concat([basic.rename("basic"), advanced.rename("advanced")], axis=1).dropna()
    if len(pair) < 12 or reps <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    metrics = [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "CVaR (95%)",
        "Calmar Ratio",
    ]
    deltas = {m: [] for m in metrics}
    b = pair["basic"].to_numpy(dtype=float)
    a = pair["advanced"].to_numpy(dtype=float)
    for _ in range(int(reps)):
        idx = _circular_block_indices(len(pair), block_length, rng)
        mb = _metric_vector(b[idx], risk_free_rate)
        ma = _metric_vector(a[idx], risk_free_rate)
        for m in metrics:
            deltas[m].append(float(ma.get(m, np.nan) - mb.get(m, np.nan)))

    rows = []
    lower_is_better = {"Annualized Volatility"}
    for m, vals in deltas.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        prob = float(np.mean(arr < 0.0)) if m in lower_is_better else float(np.mean(arr > 0.0))
        rows.append({
            "comparison": label,
            "metric": m,
            "delta_advanced_minus_basic_mean": float(np.mean(arr)),
            "delta_median": float(np.median(arr)),
            "ci_2.5pct": float(np.quantile(arr, 0.025)),
            "ci_97.5pct": float(np.quantile(arr, 0.975)),
            "probability_advanced_better": prob,
            "n_bootstrap": len(arr),
            "block_length_months": block_length,
        })
    return pd.DataFrame(rows)


def membership_matrix(universe: dict[pd.Timestamp, set[str]], all_tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for d in dates:
        selected = universe.get(pd.Timestamp(d), set())
        row = {"Date": d}
        row.update({t: int(t in selected) for t in all_tickers})
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_sharpe(r: pd.Series, window: int, rf: float) -> pd.Series:
    excess = r - rf / PERIODS_PER_YEAR
    return excess.rolling(window).mean() / excess.rolling(window).std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)


def _plot_arm_series(series: pd.Series, arm: str) -> None:
    """Plot one arm with a stable style, including visible overlap handling."""
    series.plot(label=arm, **PLOT_STYLES.get(arm, {}))


def save_plots(
    outputs: dict[str, BacktestOutput],
    stability: pd.DataFrame,
    quality_monthly: pd.DataFrame,
    output_dir: Path,
    *,
    rolling_window: int,
    risk_free_rate: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    for arm, out in outputs.items():
        r = out.returns.dropna()
        if not r.empty:
            _plot_arm_series((1.0 + r).cumprod(), arm)
    plt.title("Equal-weight cumulative wealth by ETF universe")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "01_cumulative_wealth.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for arm, out in outputs.items():
        r = out.returns.dropna()
        if not r.empty:
            wealth = (1.0 + r).cumprod()
            dd = wealth / wealth.cummax() - 1.0
            _plot_arm_series(dd, arm)
    plt.title("Equal-weight underwater / drawdown paths")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "02_drawdowns.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for arm, out in outputs.items():
        rs = rolling_sharpe(out.returns, rolling_window, risk_free_rate)
        _plot_arm_series(rs, arm)
    plt.axhline(0.0, linewidth=0.8)
    plt.title(f"Rolling {rolling_window}-month Sharpe ratio")
    plt.ylabel("Annualized Sharpe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "03_rolling_sharpe.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for arm, out in outputs.items():
        rv = out.returns.rolling(rolling_window).std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
        _plot_arm_series(rv, arm)
    plt.title(f"Rolling {rolling_window}-month annualized volatility")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "04_rolling_volatility.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for arm, out in outputs.items():
        _plot_arm_series(out.turnover, arm)
    plt.title("One-way portfolio turnover from equal-weight rebalancing")
    plt.ylabel("Turnover")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "05_portfolio_turnover.png", dpi=180)
    plt.close()

    if not stability.empty:
        plt.figure(figsize=(11, 6))
        for arm, g in stability.groupby("arm"):
            _plot_arm_series(g.set_index("date")["jaccard_overlap"], arm)
        plt.title("Month-to-month ETF-universe Jaccard overlap")
        plt.ylabel("Jaccard overlap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "06_universe_jaccard.png", dpi=180)
        plt.close()

    if not quality_monthly.empty:
        for i, metric in enumerate([
            "median_dollar_volume",
            "median_volatility",
            "median_downside_volatility",
            "median_drawdown_abs",
            "median_risk_adjusted_return",
            "median_amihud_illiquidity",
        ], start=7):
            pivot = quality_monthly.pivot(index="date", columns="arm", values=metric)
            plt.figure(figsize=(11, 6))
            for col in pivot.columns:
                _plot_arm_series(pivot[col], col)
            plt.title(metric.replace("_", " ").title())
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{i:02d}_{metric}.png", dpi=180)
            plt.close()


def build_scorecard(
    summary: pd.DataFrame,
    quality: pd.DataFrame,
    stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    basic_arm: str,
    advanced_arm: str,
) -> pd.DataFrame:
    if basic_arm not in set(summary["arm"]) or advanced_arm not in set(summary["arm"]):
        return pd.DataFrame()
    s = summary.set_index("arm")
    q = quality.set_index("arm") if not quality.empty else pd.DataFrame()
    stab = stability.groupby("arm").mean(numeric_only=True) if not stability.empty else pd.DataFrame()
    rows = []

    def add(name: str, b: float, a: float, higher: bool, source: str):
        rows.append({
            "criterion": name,
            "source": source,
            "basic_value": b,
            "advanced_value": a,
            "higher_is_better": higher,
            "advanced_wins": bool(a > b) if higher else bool(a < b),
            "advanced_minus_basic": a - b,
        })

    for m, higher in [
        ("Annualized Return", True),
        ("Sharpe Ratio", True),
        ("Sortino Ratio", True),
        ("Annualized Volatility", False),
        ("Max Drawdown", True),
        ("CVaR (95%)", True),
        ("Calmar Ratio", True),
        ("Annualized One-Way Turnover", False),
    ]:
        if m in s.columns:
            add(m, float(s.at[basic_arm, m]), float(s.at[advanced_arm, m]), higher, "equal_weight_backtest")

    if not q.empty and basic_arm in q.index and advanced_arm in q.index:
        for m, higher in [
            ("median_dollar_volume", True),
            ("median_volatility", False),
            ("median_downside_volatility", False),
            ("median_drawdown_abs", False),
            ("median_risk_adjusted_return", True),
            ("median_amihud_illiquidity", False),
        ]:
            add(m, float(q.at[basic_arm, m]), float(q.at[advanced_arm, m]), higher, "selected_asset_quality")

    if not stab.empty and basic_arm in stab.index and advanced_arm in stab.index:
        add(
            "Mean Universe Jaccard Overlap",
            float(stab.at[basic_arm, "jaccard_overlap"]),
            float(stab.at[advanced_arm, "jaccard_overlap"]),
            True,
            "universe_stability",
        )

    if not bootstrap.empty:
        bsub = bootstrap.loc[bootstrap["comparison"] == f"{advanced_arm}_vs_{basic_arm}"]
        for _, r in bsub.iterrows():
            rows.append({
                "criterion": f"Bootstrap P(advanced better): {r['metric']}",
                "source": "paired_block_bootstrap",
                "basic_value": np.nan,
                "advanced_value": float(r["probability_advanced_better"]),
                "higher_is_better": True,
                "advanced_wins": bool(r["probability_advanced_better"] >= 0.90),
                "advanced_minus_basic": np.nan,
            })

    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    scorecard: pd.DataFrame,
    bootstrap: pd.DataFrame,
    *,
    primary_basic: str,
    primary_advanced: str,
) -> None:
    lines = [
        "# ETF Universe Comparison — Equal-Weight Ablation",
        "",
        "## Purpose",
        "This experiment isolates the ETF-universe decision by holding the portfolio rule fixed at monthly equal weight. "
        "The Bayesian model, CVaR optimizer, dynamic parameters and exposure overlay are deliberately absent from this first-stage test.",
        "",
        f"Primary publication comparison: **{primary_advanced} vs {primary_basic}**.",
        "The legacy-as-implemented arm is retained only for provenance because its old history condition is not a literal count of 36 complete months.",
        "",
        "## Interpretation rule",
        "Do not write that the advanced universe is superior simply because one return statistic is larger. Prefer a joint statement supported by "
        "equal-weight Sharpe/Sortino, drawdown/CVaR, selected-asset quality, universe stability, transaction costs, and paired block-bootstrap evidence.",
        "",
        "## Summary metrics",
        summary.to_markdown(index=False) if not summary.empty else "No summary metrics were generated.",
        "",
        "## Scorecard",
        scorecard.to_markdown(index=False) if not scorecard.empty else "No scorecard was generated.",
        "",
        "## Paired block bootstrap",
        bootstrap.to_markdown(index=False) if not bootstrap.empty else "Bootstrap unavailable or too few paired months.",
        "",
        "## Second-stage full framework test",
        "The CSV eligibility matrices in `run_matrices/` can be copied to `datasets/excel/universe_comparison/` and referenced by `portfolio_matrix` in an otherwise identical run.py YAML. "
        "That second stage answers whether the universe effect survives the Bayesian/CVaR optimizer; it must not replace this equal-weight ablation.",
    ]
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = resolve(root, args.output_dir)
    run_matrix_dir = resolve(root, args.export_run_matrices_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_matrix_dir.mkdir(parents=True, exist_ok=True)

    close = read_market_csv(resolve(root, args.prices))
    open_ = read_market_csv(resolve(root, args.open_prices))
    volume = read_market_csv(resolve(root, args.volumes))
    # The advanced filter needs close and volume only.  Keep their full shared
    # action set; the basic filter handles missing OpenPrice observations with
    # a close-price fallback when estimating traded value.
    common = _shared_columns(close, volume)
    close = close[common]
    volume = volume[common]
    open_ = open_.reindex(index=close.index, columns=common)

    business_path = resolve(root, args.business_dates)
    if business_path.exists():
        rebalance_dates = read_business_dates(business_path, skip_first=False)
    else:
        rebalance_dates = infer_first_trading_days(close)
    rebalance_dates = pd.DatetimeIndex([d for d in rebalance_dates if d in close.index]).sort_values().unique()
    if len(rebalance_dates) < 14:
        raise ValueError("Too few rebalance dates for a monthly comparison.")

    basic_u, basic_audit = build_basic_universes(
        close, open_, volume, rebalance_dates,
        top_n=args.basic_top_n,
        benchmark=args.benchmark,
        liquidity_threshold=args.legacy_liquidity_threshold,
        liquidity_window_months=args.legacy_liquidity_window_months,
        history_months=args.legacy_history_months,
    )
    advanced_u, advanced_scores = build_advanced_universes(
        close, volume, rebalance_dates,
        basic_top_n=args.basic_top_n,
        native_top_n=args.advanced_native_top_n,
        native_spy_filter=args.advanced_native_spy_filter,
        benchmark=args.benchmark,
        lookback_days=args.advanced_lookback_days,
        score_ema_alpha=args.advanced_score_ema_alpha,
        min_observations=args.advanced_min_observations,
        min_price=args.advanced_min_price,
        min_total_volume=args.advanced_min_total_volume,
    )
    universes = {**basic_u, **advanced_u}

    # Match run.py: do not evaluate performance until a 3-year (default) warmup
    # has elapsed, while still allowing the filters to build their lagged state.
    warmup_target = rebalance_dates[0] + pd.DateOffset(years=args.warmup_years)
    eval_dates = rebalance_dates[rebalance_dates >= warmup_target]
    if len(eval_dates) < 13:
        raise ValueError("Warmup leaves fewer than 12 holding periods. Reduce --warmup-years or provide more data.")

    outputs = {
        arm: equal_weight_backtest(
            close, eval_dates, u,
            cost_bps=args.cost_bps,
            charge_initial_trade=args.charge_initial_trade,
        )
        for arm, u in universes.items()
    }
    spy = benchmark_returns(close, eval_dates, args.benchmark)

    summary_rows = [
        performance_summary(
            arm, out,
            risk_free_rate=args.risk_free_rate,
            benchmark_returns=spy,
        )
        for arm, out in outputs.items()
    ]
    summary = pd.DataFrame(summary_rows)
    first_cols = ["arm", "Cumulative Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "CVaR (95%)", "Calmar Ratio", "Annualized One-Way Turnover"]
    summary = summary[[c for c in first_cols if c in summary.columns] + [c for c in summary.columns if c not in first_cols]]

    monthly = pd.concat(
        [
            pd.DataFrame({
                "date": out.returns.index,
                "arm": arm,
                "net_return": out.returns.values,
                "gross_return": out.gross_returns.reindex(out.returns.index).values,
                "turnover": out.turnover.reindex(out.returns.index).values,
                "cost": out.costs.reindex(out.returns.index).values,
                "selected_count": out.selected_count.reindex(out.returns.index).values,
                "valid_count": out.valid_count.reindex(out.returns.index).values,
                "price_coverage": out.coverage.reindex(out.returns.index).values,
            })
            for arm, out in outputs.items()
        ],
        ignore_index=True,
    )

    stability = universe_stability(universes, eval_dates[:-1])
    overlap = cross_universe_overlap(universes, eval_dates[:-1])
    quality_monthly, quality_summary = compute_common_quality_table(
        close, volume, eval_dates[:-1], universes,
        benchmark=args.benchmark,
        lookback_days=args.advanced_lookback_days,
    )

    bootstrap_tables = []
    for basic_arm, advanced_arm in [
        ("basic_repaired", "advanced_matched"),
        ("basic_legacy", "advanced_native"),
    ]:
        if basic_arm in outputs and advanced_arm in outputs:
            bootstrap_tables.append(
                paired_block_bootstrap(
                    outputs[basic_arm].returns,
                    outputs[advanced_arm].returns,
                    reps=args.bootstrap_reps,
                    block_length=args.bootstrap_block_length,
                    risk_free_rate=args.risk_free_rate,
                    seed=args.seed,
                    label=f"{advanced_arm}_vs_{basic_arm}",
                )
            )
    bootstrap = pd.concat([x for x in bootstrap_tables if not x.empty], ignore_index=True) if any(not x.empty for x in bootstrap_tables) else pd.DataFrame()

    scorecard = build_scorecard(
        summary, quality_summary, stability, bootstrap,
        basic_arm="basic_repaired",
        advanced_arm="advanced_matched",
    )

    # Save tabular results.
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    monthly.to_csv(output_dir / "monthly_equal_weight_returns.csv", index=False)
    stability.to_csv(output_dir / "universe_stability.csv", index=False)
    overlap.to_csv(output_dir / "cross_universe_overlap.csv", index=False)
    quality_monthly.to_csv(output_dir / "universe_quality_monthly.csv", index=False)
    quality_summary.to_csv(output_dir / "universe_quality_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_block_bootstrap.csv", index=False)
    scorecard.to_csv(output_dir / "advanced_vs_basic_scorecard.csv", index=False)
    basic_audit.to_csv(output_dir / "basic_filter_audit.csv", index=False)
    advanced_scores["advanced_matched"].to_csv(output_dir / "advanced_matched_scores.csv", index=False)
    advanced_scores["advanced_native"].to_csv(output_dir / "advanced_native_scores.csv", index=False)

    # Matrices are saved both with the results and in datasets/excel so run.py
    # can consume them through config['portfolio_matrix'] without code changes.
    all_tickers = list(close.columns)
    local_matrix_dir = output_dir / "run_matrices"
    local_matrix_dir.mkdir(parents=True, exist_ok=True)
    for arm, u in universes.items():
        matrix = membership_matrix(u, all_tickers, rebalance_dates)
        matrix.to_csv(local_matrix_dir / f"{arm}_matrix.csv", index=False)
        matrix.to_csv(run_matrix_dir / f"{arm}_matrix.csv", index=False)

    # Compact run.py config patch examples, not full configs: merge these keys
    # into the SAME base YAML to keep every other model/optimizer setting fixed.
    patches = {
        arm: {
            "portfolio": "filtered",
            "portfolio_matrix": f"universe_comparison/{arm}_matrix.csv",
            "universe_cap": None,
            "output_folder": f"universe_ablation_{arm}",
        }
        for arm in universes
    }
    (output_dir / "run_py_config_patches.json").write_text(json.dumps(patches, indent=2), encoding="utf-8")

    save_plots(
        outputs, stability, quality_monthly, output_dir / "plots",
        rolling_window=args.rolling_window,
        risk_free_rate=args.risk_free_rate,
    )
    write_report(
        output_dir, summary, scorecard, bootstrap,
        primary_basic="basic_repaired",
        primary_advanced="advanced_matched",
    )

    manifest = {
        "evaluation_start": str(eval_dates[0].date()),
        "evaluation_end_rebalance": str(eval_dates[-1].date()),
        "holding_periods_available": int(len(eval_dates) - 1),
        "cost_bps": args.cost_bps,
        "warmup_years": args.warmup_years,
        "basic_top_n": args.basic_top_n,
        "advanced_native_top_n": args.advanced_native_top_n,
        "advanced_native_spy_filter": args.advanced_native_spy_filter,
        "advanced_lookback_days": args.advanced_lookback_days,
        "advanced_score_ema_alpha": args.advanced_score_ema_alpha,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_block_length": args.bootstrap_block_length,
        "primary_comparison": "advanced_matched_vs_basic_repaired",
        "note": "Equal-weight ablation isolates universe selection; full run.py optimizer test is second-stage evidence.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nETF universe comparison complete.")
    print(f"Results: {output_dir}")
    print(f"run.py matrices: {run_matrix_dir}")
    if not summary.empty:
        cols = [c for c in ["arm", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown", "CVaR (95%)", "Annualized One-Way Turnover"] if c in summary.columns]
        print(summary[cols].to_string(index=False))
    if not scorecard.empty:
        wins = int(scorecard["advanced_wins"].sum())
        print(f"\nPrimary scorecard: advanced_matched wins {wins}/{len(scorecard)} listed criteria. See advanced_vs_basic_scorecard.csv; do not use the count alone as a significance test.")


if __name__ == "__main__":
    main()
