"""Robust performance statistics and publication figures for portfolio backtests.

The module keeps the original public API and adds:

* Probabilistic and Deflated Sharpe Ratios using sampling-frequency inference;
* annual/monthly benchmark comparisons;
* false-strategy theorem density heatmap;
* train/test and rolling diagnostics suitable for the interactive report.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    from beartype import beartype
except ImportError:  # reporting must not fail because an optional runtime type checker is absent
    def beartype(obj):
        return obj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from scripts.calculations.drawdown import empirical_tail_mean
from scripts.validation.false_strategy import (
    deflated_sharpe_ratio as _deflated_sharpe_ratio,
    expected_max_sharpe as _expected_max_sharpe,
    exact_expected_max_sharpe_gaussian as _exact_expected_max_sharpe_gaussian,
    false_strategy_density_surface,
    probabilistic_sharpe_ratio as _probabilistic_sharpe_ratio,
)

EPS = 1e-12
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _prepare_pnl(pnls: pd.DataFrame) -> pd.DataFrame:
    df = pnls.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass
    for c in ["Balance", "PnL", "Cost", "Returns", "Alpha"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Returns" not in df:
        if "Balance" not in df:
            raise ValueError("pnl must contain Returns or Balance")
        df["Returns"] = df["Balance"].pct_change().fillna(0.0)
    if "Balance" not in df:
        df["Balance"] = 1000.0 * (1.0 + df["Returns"].fillna(0.0)).cumprod()
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Returns"])


def _returns(pnls: pd.DataFrame) -> pd.Series:
    return _prepare_pnl(pnls)["Returns"].astype(float)


def _safe_ann_return(r: pd.Series, periods_per_year: int) -> float:
    if r.empty:
        return np.nan
    gross = float(np.prod(1.0 + np.clip(r.to_numpy(dtype=float), -0.999999, None)))
    years = len(r) / periods_per_year
    return gross ** (1.0 / years) - 1.0 if years > 0 and gross > 0 else -1.0


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 12,
    risk_free_rate: float = 0.0,
    adjust_serial_correlation: bool = True,
) -> float:
    """Bailey--López de Prado PSR; benchmark Sharpe is annualized."""
    return float(_probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=benchmark_sharpe,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        adjust_serial_correlation=adjust_serial_correlation,
    ))


def expected_max_sharpe(
    num_trials: int | float,
    sharpe_std: float = 1.0,
    sharpe_mean: float = 0.0,
) -> float:
    return float(_expected_max_sharpe(num_trials, sharpe_mean=sharpe_mean, sharpe_std=sharpe_std))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    num_trials: int | float = 1,
    sharpe_std: float = 1.0,
    periods_per_year: int = 12,
    sharpe_mean: float = 0.0,
    risk_free_rate: float = 0.0,
    adjust_serial_correlation: bool = True,
) -> float:
    return float(_deflated_sharpe_ratio(
        returns,
        num_trials=num_trials,
        trial_sharpe_mean=sharpe_mean,
        trial_sharpe_std=sharpe_std,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        adjust_serial_correlation=adjust_serial_correlation,
    ))


def metric_summary(
    pnls: pd.DataFrame,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 12,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    num_trials: int | float = 1,
    trial_sharpe_mean: float = 0.0,
    trial_sharpe_std: float = 0.0,
) -> dict[str, float]:
    df = _prepare_pnl(pnls)
    r = df["Returns"].astype(float)
    n = len(r)
    if n == 0:
        return {}
    ann = _safe_ann_return(r, periods_per_year)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    excess = r - risk_free_rate / periods_per_year
    ex_sd = float(excess.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(excess.mean() / ex_sd * np.sqrt(periods_per_year)) if ex_sd > EPS else np.nan
    downside = np.minimum(r.to_numpy(dtype=float), 0.0)
    down_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))
    sortino = float((ann - risk_free_rate) / down_dev) if down_dev > EPS else np.nan

    balance = df["Balance"].astype(float)
    dd = balance / balance.cummax() - 1.0
    maxdd = float(dd.min())
    drawdown_depth = (-dd).clip(lower=0.0)
    cdar95 = float(empirical_tail_mean(drawdown_depth.to_numpy(dtype=float), alpha=0.95))
    underwater = dd[dd < 0]
    avgdd = float(underwater.mean()) if not underwater.empty else 0.0
    pain = float(dd.abs().mean())
    calmar = float(ann / abs(maxdd)) if abs(maxdd) > EPS else np.nan
    pain_ratio = float((ann - risk_free_rate) / pain) if pain > EPS else np.nan

    var95 = float(np.quantile(r, 0.05))
    tail = r[r <= var95]
    cvar95 = float(tail.mean()) if not tail.empty else var95
    wins, losses = r[r > 0], r[r < 0]
    win_rate = float((r > 0).mean())
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = float(losses.mean()) if not losses.empty else 0.0
    expectancy = float(win_rate * avg_win + (1.0 - win_rate) * avg_loss)
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > EPS else np.inf
    gains = float(r[r > 0].sum())
    loss_sum = float(-r[r < 0].sum())
    omega = float(gains / loss_sum) if loss_sum > EPS else np.inf

    beta = treynor = information_ratio = tracking_error = np.nan
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.Series):
            b = pd.to_numeric(benchmark_returns.copy(), errors="coerce")
            if not isinstance(b.index, pd.DatetimeIndex) and isinstance(r.index, pd.DatetimeIndex):
                n = min(len(b), len(r))
                b = pd.Series(b.to_numpy(dtype=float)[-n:], index=r.index[-n:])
        else:
            b_values = np.asarray(benchmark_returns, dtype=float).reshape(-1)
            n = min(len(b_values), len(r))
            b = pd.Series(b_values[-n:], index=r.index[-n:])
        joined = pd.concat([r.rename("strategy"), b.rename("benchmark")], axis=1).dropna()
        if len(joined) > 2 and float(joined["benchmark"].var(ddof=1)) > EPS:
            beta = float(joined.cov().loc["strategy", "benchmark"] / joined["benchmark"].var(ddof=1))
            treynor = float((ann - risk_free_rate) / beta) if abs(beta) > EPS else np.nan
            active = joined["strategy"] - joined["benchmark"]
            tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
            information_ratio = float(active.mean() / active.std(ddof=1) * np.sqrt(periods_per_year)) if active.std(ddof=1) > EPS else np.nan

    psr = probabilistic_sharpe_ratio(
        r,
        benchmark_sharpe=0.0,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    dsr = deflated_sharpe_ratio(
        r,
        num_trials=num_trials,
        sharpe_std=trial_sharpe_std,
        sharpe_mean=trial_sharpe_mean,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    dsr_benchmark = expected_max_sharpe(num_trials, sharpe_std=trial_sharpe_std, sharpe_mean=trial_sharpe_mean)

    return {
        "Cumulative Return": float(np.prod(1.0 + r) - 1.0),
        "Annualized Return": ann,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": maxdd,
        "Average Drawdown": avgdd,
        "Calmar Ratio": calmar,
        "VaR (95%)": var95,
        "CVaR (95%)": cvar95,
        "CDaR (95%)": cdar95,
        "Downside Deviation": down_dev,
        "Win Rate": win_rate,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Expectancy": expectancy,
        "Profit Factor": profit_factor,
        "Omega Ratio": omega,
        "Pain Ratio": pain_ratio,
        "Mean Absolute Deviation": float(np.mean(np.abs(r - r.mean()))),
        "Skewness": float(skew(r, bias=False)) if n >= 3 else np.nan,
        "Excess Kurtosis": float(kurtosis(r, fisher=True, bias=False)) if n >= 4 else np.nan,
        "Beta": beta,
        "Treynor Ratio": treynor,
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio,
        "PSR": psr,
        "DSR": dsr,
        "DSR Benchmark E[max SR]": dsr_benchmark,
        "Number of Trials": float(num_trials),
        "Trial Sharpe Mean": float(trial_sharpe_mean),
        "Trial Sharpe Std": float(trial_sharpe_std),
        "N Periods": float(n),
    }


def calculate_metrics(pnls: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.DataFrame:
    m = metric_summary(pnls, risk_free_rate=risk_free_rate)
    return pd.DataFrame({"Metric": list(m), "Value": [f"{v:.6f}" if np.isfinite(v) else str(v) for v in m.values()]})


def calculate_extended_metrics(pnls: pd.DataFrame, risk_free_rate: float = 0.02, periods_per_year: int = 12) -> pd.DataFrame:
    m = metric_summary(pnls, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
    return pd.DataFrame({"Metric": list(m), "Value": [f"{v:.6f}" if np.isfinite(v) else str(v) for v in m.values()]})


def split_pnl_train_test(pnls: pd.DataFrame, train_fraction: float = 0.70, split_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _prepare_pnl(pnls)
    if split_date is not None and isinstance(df.index, pd.DatetimeIndex):
        d = pd.Timestamp(split_date)
        return df.loc[df.index <= d].copy(), df.loc[df.index > d].copy()
    if len(df) <= 1:
        return df.copy(), df.iloc[0:0].copy()
    k = max(1, min(len(df) - 1, int(round(len(df) * float(train_fraction)))))
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def calculate_train_test_metrics(
    pnls: pd.DataFrame,
    train_fraction: float = 0.70,
    split_date: str | None = None,
    risk_free_rate: float = 0.02,
    num_trials: int | float = 1,
    trial_sharpe_mean: float = 0.0,
    trial_sharpe_std: float = 0.0,
) -> pd.DataFrame:
    overall = _prepare_pnl(pnls)
    train, test = split_pnl_train_test(overall, train_fraction, split_date)
    rows: list[dict[str, Any]] = []
    for label, df in [("overall", overall), ("train", train), ("test", test)]:
        if df.empty:
            continue
        record: dict[str, Any] = {
            "sample": label, "start": str(df.index.min()), "end": str(df.index.max()), "n_periods": len(df)
        }
        record.update(metric_summary(
            df,
            risk_free_rate=risk_free_rate,
            num_trials=num_trials,
            trial_sharpe_mean=trial_sharpe_mean,
            trial_sharpe_std=trial_sharpe_std,
        ))
        rows.append(record)
    return pd.DataFrame(rows)


def rolling_sharpe_ratio(returns: pd.Series, window: int = 12, risk_free_rate: float = 0.02, periods_per_year: int = 12) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce")
    excess = r - risk_free_rate / periods_per_year
    return (excess.rolling(window).mean() / excess.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)).rename(f"rolling_sharpe_{window}")


def rolling_sortino_ratio(returns: pd.Series, window: int = 12, risk_free_rate: float = 0.02, periods_per_year: int = 12) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce")
    downside = r.clip(upper=0.0)
    down_dev = downside.pow(2).rolling(window).mean().pow(0.5) * np.sqrt(periods_per_year)
    ann = r.rolling(window).mean() * periods_per_year
    return ((ann - risk_free_rate) / down_dev.replace(0.0, np.nan)).rename(f"rolling_sortino_{window}")


def monthly_return_matrix(returns: pd.Series | np.ndarray) -> pd.DataFrame:
    r = pd.Series(returns).copy() if not isinstance(returns, pd.Series) else returns.copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index, errors="coerce")
    r = pd.to_numeric(r, errors="coerce").dropna()
    table = pd.DataFrame({"return": r, "year": r.index.year, "month": r.index.month})
    matrix = table.pivot_table(index="month", columns="year", values="return", aggfunc=lambda x: float(np.prod(1.0 + x) - 1.0))
    matrix = matrix.reindex(range(1, 13))
    matrix.index = MONTH_NAMES
    return matrix


def annual_return_table(
    strategy_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    benchmark_name: str = "SPY",
) -> pd.DataFrame:
    s = pd.Series(strategy_returns).copy() if not isinstance(strategy_returns, pd.Series) else strategy_returns.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = pd.to_numeric(s, errors="coerce").dropna()
    out = s.groupby(s.index.year).apply(lambda x: float(np.prod(1.0 + x) - 1.0)).rename("Strategy").to_frame()
    if benchmark_returns is not None:
        b = pd.Series(benchmark_returns).copy() if not isinstance(benchmark_returns, pd.Series) else benchmark_returns.copy()
        if not isinstance(b.index, pd.DatetimeIndex):
            b.index = pd.to_datetime(b.index, errors="coerce")
        b = pd.to_numeric(b, errors="coerce").dropna()
        out[benchmark_name] = b.groupby(b.index.year).apply(lambda x: float(np.prod(1.0 + x) - 1.0))
    out.index.name = "Year"
    return out.sort_index()


def plot_monthly_returns_heatmap(returns: pd.Series | np.ndarray, figsize: tuple[float, float] = (12, 7)):
    matrix = monthly_return_matrix(returns)
    fig, ax = plt.subplots(figsize=figsize)
    values = matrix.to_numpy(dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    vmax = float(np.quantile(finite, 0.95)) if finite.size else 0.1
    vmax = max(vmax, 0.01)
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=[str(c) for c in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    ax.set_title("Strategy Monthly Returns")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1%}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Return")
    fig.tight_layout()
    return fig, ax


def plot_annual_returns_bars(
    strategy_returns: pd.Series | np.ndarray,
    benchmark_returns: pd.Series | np.ndarray | None = None,
    benchmark_name: str = "SPY",
    figsize: tuple[float, float] = (12, 6),
):
    table = annual_return_table(strategy_returns, benchmark_returns, benchmark_name)
    fig, ax = plt.subplots(figsize=figsize)
    table.plot(kind="bar", ax=ax)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(f"Annual Returns: Strategy vs {benchmark_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_false_strategy_heatmap(
    *,
    sharpe_std: float = 1.0,
    sharpe_mean: float = 0.0,
    figsize: tuple[float, float] = (13, 7),
):
    surface = false_strategy_density_surface(sharpe_mean=sharpe_mean, sharpe_std=sharpe_std)
    pivot = surface.pivot(index="max_sharpe", columns="number_of_trials", values="relative_density")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.pcolormesh(x, y, pivot.to_numpy(dtype=float), shading="auto", cmap="inferno")
    ax.set_xscale("log")
    expected = surface.groupby("number_of_trials")["expected_max_sharpe"].first().reindex(x)
    ax.plot(x, expected.to_numpy(dtype=float), linestyle="--", linewidth=2.0, label="E[max(SR)] FST approximation")
    # Sparse exact Gaussian order-statistic reference: this validates the ridge
    # without turning the whole raster generation into an integration benchmark.
    if len(x):
        sample_idx = np.unique(np.linspace(0, len(x) - 1, min(14, len(x))).astype(int))
        x_exact = x[sample_idx]
        y_exact = np.array([
            _exact_expected_max_sharpe_gaussian(k, sharpe_mean=sharpe_mean, sharpe_std=sharpe_std)
            for k in x_exact
        ])
        ax.plot(x_exact, y_exact, marker="o", markersize=3.0, linewidth=1.0, label="Exact iid-Gaussian E[max(SR)]")
    ax.set_xlabel("Number of trials K")
    ax.set_ylabel("Maximum Sharpe among K null strategies")
    qualifier = "standardized pedagogical null" if abs(sharpe_mean) < 1e-12 and abs(sharpe_std - 1.0) < 1e-12 else "research-calibrated null"
    ax.set_title(
        f"False-strategy maximum under iid Gaussian no-skill Sharpe estimates\n"
        f"{qualifier}: mean(SR)={sharpe_mean:g}, std(SR)={sharpe_std:g} (not a Uniform distribution)"
    )
    ax.legend()
    fig.colorbar(image, ax=ax, label="Relative conditional density f(max SR | K)")
    fig.tight_layout()
    return fig, ax


def _trial_statistics_from_folder(folder: Path) -> tuple[int, float, float]:
    summary_path = folder / "research_validation_summary.csv"
    if summary_path.exists():
        try:
            row = pd.read_csv(summary_path).iloc[0]
            return (
                max(int(round(float(row.get("effective_number_of_trials", 1)))), 1),
                float(row.get("trial_sharpe_mean", 0.0)),
                float(row.get("trial_sharpe_std", 0.0)),
            )
        except Exception:
            pass
    audit = folder / "trial_audit.csv"
    if audit.exists():
        try:
            a = pd.read_csv(audit)
            if "recipe_id" in a.columns:
                return max(int(a["recipe_id"].nunique()), 1), 0.0, 0.0
            return max(int(a.get("candidate_id", pd.Series(dtype=str)).nunique()), 1), 0.0, 0.0
        except Exception:
            pass
    return 1, 0.0, 0.0


def export_metric_artifacts(
    result_folder: str | Path,
    split_date: str | None = None,
    train_fraction: float = 0.70,
    benchmark_ticker: str = "SPY",
) -> tuple[Path, Path]:
    folder = Path(result_folder)
    pnl_path = folder / "pnl.csv"
    if not pnl_path.exists():
        raise FileNotFoundError(pnl_path)
    pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
    num_trials, trial_mean, trial_std = _trial_statistics_from_folder(folder)
    tt = calculate_train_test_metrics(
        pnl,
        train_fraction=train_fraction,
        split_date=split_date,
        num_trials=num_trials,
        trial_sharpe_mean=trial_mean,
        trial_sharpe_std=trial_std,
    )
    tt_path = folder / "train_test_metrics.csv"
    tt.to_csv(tt_path, index=False)
    r = _prepare_pnl(pnl)["Returns"]
    rolling = pd.concat([
        rolling_sharpe_ratio(r, 6), rolling_sharpe_ratio(r, 12), rolling_sharpe_ratio(r, 24),
        rolling_sortino_ratio(r, 6), rolling_sortino_ratio(r, 12), rolling_sortino_ratio(r, 24),
    ], axis=1)
    rolling.index.name = "Date"
    rolling_path = folder / "rolling_metrics.csv"
    rolling.to_csv(rolling_path)

    monthly_return_matrix(r).to_csv(folder / "monthly_returns_matrix.csv")
    benchmark = None
    real_path = folder / "real.csv"
    if real_path.exists():
        real = pd.read_csv(real_path, index_col=0, parse_dates=True)
        if benchmark_ticker in real.columns:
            benchmark = pd.to_numeric(real[benchmark_ticker], errors="coerce")
    annual_return_table(r, benchmark, benchmark_ticker).to_csv(folder / "annual_returns_comparison.csv")

    trial_k, trial_mean, trial_std = _trial_statistics_from_folder(folder)
    calibrated_std = trial_std if np.isfinite(trial_std) and trial_std > 1e-12 else 1.0
    figures = [
        (plot_monthly_returns_heatmap(r)[0], folder / "monthly_returns_heatmap.png"),
        (plot_annual_returns_bars(r, benchmark, benchmark_ticker)[0], folder / "annual_returns_vs_spy.png"),
        (plot_false_strategy_heatmap()[0], folder / "false_strategy_heatmap_standardized.png"),
        (plot_false_strategy_heatmap(sharpe_mean=trial_mean, sharpe_std=calibrated_std)[0], folder / "false_strategy_heatmap_calibrated.png"),
    ]
    # Backward-compatible filename now points to the calibrated research null.
    figures.append((plot_false_strategy_heatmap(sharpe_mean=trial_mean, sharpe_std=calibrated_std)[0], folder / "false_strategy_heatmap.png"))
    for fig, path in figures:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return tt_path, rolling_path


def plot_metrics_table(metrics_df: pd.DataFrame, figsize=(12, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Strategy Performance Metrics")
    fig.tight_layout()
    return fig, ax


def plot_strategy_charts(pnls: pd.DataFrame):
    df = _prepare_pnl(pnls)
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    axes[0].plot(df.index, df.get("Cost", pd.Series(0.0, index=df.index)))
    axes[0].set_title("Transaction cost")
    axes[1].plot(df.index, df["Returns"])
    axes[1].set_title("Monthly return")
    dd = df["Balance"] / df["Balance"].cummax() - 1.0
    axes[2].plot(df.index, dd)
    axes[2].fill_between(df.index, dd, 0, alpha=0.25)
    axes[2].set_title("Drawdown")
    axes[3].plot(df.index, df["Balance"])
    axes[3].set_title("Portfolio balance")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, list(axes)


def plot_train_test_performance(pnls: pd.DataFrame, train_fraction: float = 0.70, split_date: str | None = None, figsize=(14, 7)):
    df = _prepare_pnl(pnls)
    _, test = split_pnl_train_test(df, train_fraction, split_date)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df["Balance"], label="Strategy balance")
    if not test.empty:
        ax.axvline(test.index.min(), linestyle="--", label="Train/test split")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Strategy performance with train/test split")
    return fig, ax


def plot_strategy_with_metrics_separate(path_to_pnl: Path, risk_free_rate: float = 0.02):
    pnl = pd.read_csv(path_to_pnl, index_col=0, parse_dates=True)
    metrics = calculate_extended_metrics(pnl, risk_free_rate=risk_free_rate)
    return plot_strategy_charts(pnl), plot_metrics_table(metrics)



@beartype
def save_figure(
    fig: plt.Figure,
    output: Path,
    filename: str,
    dpi: int = 300,
) -> Path:
    """Save one matplotlib figure and close it."""
    output.mkdir(parents=True, exist_ok=True)

    path = output / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {path}")
    return path


@beartype
def save_plots_to_files(
    charts_fig: plt.Figure,
    table_fig: plt.Figure,
    output: Path,
    name: str = "",
) -> tuple[Path, Path]:
    """Save the main strategy chart and metrics table."""
    output.mkdir(parents=True, exist_ok=True)

    suffix = f"_{name}" if name else ""

    charts_path = output / f"charts{suffix}.png"
    table_path = output / f"table{suffix}.png"

    charts_fig.savefig(
        charts_path,
        dpi=300,
        bbox_inches="tight",
    )
    table_fig.savefig(
        table_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(charts_fig)
    plt.close(table_fig)

    print(f"Charts saved to: {charts_path}")
    print(f"Table saved to: {table_path}")

    return charts_path, table_path


@beartype
def main() -> None:
    folder_name = "advanced_smart_bayesian"
    folder_name = "main_combined"
    folder_name = "main_cvartarget"
    folder_name = "main_dyn_off"
    folder_name = "main_dyn_strong"
    folder_name = "main_voltarget"

    project_root = Path.cwd()
    inputs = project_root / "results" / folder_name
    output = project_root / "plots" / "charts" / folder_name

    output.mkdir(parents=True, exist_ok=True)

    pnl_path = inputs / "pnl.csv"

    if not inputs.exists():
        raise FileNotFoundError(
            f"Results folder does not exist: {inputs}"
        )

    if not pnl_path.exists():
        raise FileNotFoundError(
            f"PnL file does not exist: {pnl_path}"
        )

    print(f"Reading PnL from: {pnl_path}")
    print(f"Saving figures to: {output}")

    # ------------------------------------------------------------------
    # 1. Read and prepare PnL
    # ------------------------------------------------------------------
    pnl = pd.read_csv(
        pnl_path,
        index_col=0,
        parse_dates=True,
    )
    pnl = _prepare_pnl(pnl)
    returns = pnl["Returns"]

    # ------------------------------------------------------------------
    # 2. Strategy charts: cost, returns, drawdown, balance
    # ------------------------------------------------------------------
    charts_fig, _ = plot_strategy_charts(pnl)

    metrics_df = calculate_extended_metrics(
        pnl,
        risk_free_rate=0.02,
        periods_per_year=12,
    )
    table_fig, _ = plot_metrics_table(metrics_df)

    save_plots_to_files(
        charts_fig=charts_fig,
        table_fig=table_fig,
        output=output,
        name=folder_name,
    )

    # Also save metrics as CSV.
    metrics_path = output / "performance_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    # ------------------------------------------------------------------
    # 3. Monthly-return heatmap
    # ------------------------------------------------------------------
    monthly_fig, _ = plot_monthly_returns_heatmap(returns)

    save_figure(
        monthly_fig,
        output,
        "monthly_returns_heatmap.png",
    )

    monthly_matrix_path = output / "monthly_returns_matrix.csv"
    monthly_return_matrix(returns).to_csv(monthly_matrix_path)
    print(f"Saved: {monthly_matrix_path}")

    # ------------------------------------------------------------------
    # 4. Benchmark data
    # ------------------------------------------------------------------
    benchmark_returns = None
    real_path = inputs / "real.csv"
    benchmark_ticker = "SPY"

    if real_path.exists():
        real = pd.read_csv(
            real_path,
            index_col=0,
            parse_dates=True,
        )

        if benchmark_ticker in real.columns:
            benchmark_returns = pd.to_numeric(
                real[benchmark_ticker],
                errors="coerce",
            )

            # Important:
            # If real.csv contains benchmark prices rather than returns,
            # transform prices into returns.
            benchmark_returns = benchmark_returns.pct_change()

            print(
                f"Using {benchmark_ticker} benchmark from: "
                f"{real_path}"
            )
        else:
            print(
                f"Warning: {benchmark_ticker} is not present "
                f"in {real_path}"
            )
    else:
        print(
            f"Warning: benchmark file does not exist: {real_path}"
        )

    # ------------------------------------------------------------------
    # 5. Annual strategy-versus-benchmark chart
    # ------------------------------------------------------------------
    annual_fig, _ = plot_annual_returns_bars(
        strategy_returns=returns,
        benchmark_returns=benchmark_returns,
        benchmark_name=benchmark_ticker,
    )

    save_figure(
        annual_fig,
        output,
        "annual_returns_vs_spy.png",
    )

    annual_table_path = output / "annual_returns_comparison.csv"
    annual_return_table(
        strategy_returns=returns,
        benchmark_returns=benchmark_returns,
        benchmark_name=benchmark_ticker,
    ).to_csv(annual_table_path)

    print(f"Saved: {annual_table_path}")

    # ------------------------------------------------------------------
    # 6. False-strategy theorem density heatmap
    # ------------------------------------------------------------------
    num_trials, trial_mean, trial_std = (
        _trial_statistics_from_folder(inputs)
    )

    # A zero standard deviation gives an uninformative density surface.
    heatmap_sharpe_std = (
        trial_std if trial_std > EPS else 1.0
    )

    false_strategy_fig, _ = plot_false_strategy_heatmap(
        sharpe_std=heatmap_sharpe_std,
        sharpe_mean=trial_mean,
    )

    save_figure(
        false_strategy_fig,
        output,
        "false_strategy_heatmap.png",
    )

    # ------------------------------------------------------------------
    # 7. Train/test performance chart
    # ------------------------------------------------------------------
    train_test_fig, _ = plot_train_test_performance(
        pnl,
        train_fraction=0.70,
        split_date=None,
    )

    save_figure(
        train_test_fig,
        output,
        "train_test_performance.png",
    )

    # ------------------------------------------------------------------
    # 8. Train/test metrics
    # ------------------------------------------------------------------
    train_test_metrics = calculate_train_test_metrics(
        pnl,
        train_fraction=0.70,
        split_date=None,
        risk_free_rate=0.02,
        num_trials=num_trials,
        trial_sharpe_mean=trial_mean,
        trial_sharpe_std=trial_std,
    )

    train_test_path = output / "train_test_metrics.csv"
    train_test_metrics.to_csv(train_test_path, index=False)
    print(f"Saved: {train_test_path}")

    # ------------------------------------------------------------------
    # 9. Rolling Sharpe and Sortino values
    # ------------------------------------------------------------------
    rolling_metrics = pd.concat(
        [
            rolling_sharpe_ratio(
                returns,
                window=6,
                risk_free_rate=0.02,
                periods_per_year=12,
            ),
            rolling_sharpe_ratio(
                returns,
                window=12,
                risk_free_rate=0.02,
                periods_per_year=12,
            ),
            rolling_sharpe_ratio(
                returns,
                window=24,
                risk_free_rate=0.02,
                periods_per_year=12,
            ),
            rolling_sortino_ratio(
                returns,
                window=6,
                risk_free_rate=0.02,
                periods_per_year=12,
            ),
            rolling_sortino_ratio(
                returns,
                window=12,
                risk_free_rate=0.02,
                periods_per_year=12,
            ),
            rolling_sortino_ratio(
                returns,
                window=24,
                risk_free_rate=0.02,
                periods_per_year=12,
            ),
        ],
        axis=1,
    )

    rolling_metrics.index.name = "Date"

    rolling_path = output / "rolling_metrics.csv"
    rolling_metrics.to_csv(rolling_path)
    print(f"Saved: {rolling_path}")

    # ------------------------------------------------------------------
    # 10. Plot rolling metrics
    # ------------------------------------------------------------------
    rolling_fig, rolling_ax = plt.subplots(
        figsize=(15, 8)
    )

    for column in rolling_metrics.columns:
        rolling_ax.plot(
            rolling_metrics.index,
            rolling_metrics[column],
            label=column,
        )

    rolling_ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )
    rolling_ax.set_title(
        "Rolling Sharpe and Sortino Ratios"
    )
    rolling_ax.set_xlabel("Date")
    rolling_ax.set_ylabel("Ratio")
    rolling_ax.grid(alpha=0.25)
    rolling_ax.legend(
        loc="best",
        ncol=2,
    )
    rolling_fig.tight_layout()

    save_figure(
        rolling_fig,
        output,
        "rolling_sharpe_sortino.png",
    )

    print("\nAll metric artifacts were generated successfully.")
    print(f"Output folder: {output.resolve()}")


if __name__ == "__main__":
    main()
