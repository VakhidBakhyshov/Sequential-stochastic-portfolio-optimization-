import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union
from scipy.stats import norm, skew, kurtosis, anderson, shapiro
from beartype import beartype
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


@beartype
def calculate_metrics(pnls, risk_free_rate=0.02):
    returns = pnls['Returns'].values
    
    # Calculate number of trading periods per year (assuming monthly data)
    periods_per_year = 12
    n_years = len(returns) / periods_per_year
    total_return = (pnls['Balance'].iloc[-1] / pnls['Balance'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (1/n_years) - 1
    
    annualized_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    # Risk-adjusted returns
    excess_returns = returns - risk_free_rate/periods_per_year
    sharpe_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns, ddof=1) if np.std(excess_returns, ddof=1) > 0 else 0
    
    # Downside Deviation
    target_return = 0
    downside_returns = np.minimum(returns- target_return, 0)
    downside_returns_squared = downside_returns ** 2
    
    # Sortino ratio
    downside_deviation_annual = np.sqrt(np.mean(downside_returns_squared)) * np.sqrt(periods_per_year)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation_annual if downside_deviation_annual > 0 else 0
    
    # better approarch (instead of using returns, uses directly balance)
    balance = pnls['Balance'].values
    running_max = np.maximum.accumulate(balance)
    drawdown = (balance - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # cumulative = (1 + returns).cumprod()
    # running_max = np.maximum.accumulate(cumulative)
    # drawdown = (cumulative - running_max) / running_max
    # max_drawdown = drawdown.min()
    
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # threshold = risk_free_rate
    threshold = 0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
    
    skewness = skew(returns)
    kurt = kurtosis(returns, fisher=True)
    
    # Normality tests
    anderson_stat, anderson_critical, anderson_significance = anderson(returns)
    shapiro_stat, shapiro_p = shapiro(returns)
    
    # Mean Absolute Deviation + Interquartile Range
    mad = np.mean(np.abs(returns - np.mean(returns)))
    iqr = np.percentile(returns, 75) - np.percentile(returns, 25)
    
    # Pain Ratio (return over average drawdown)
    pain_index = np.mean(np.abs(drawdown))
    pain_ratio = annualized_return / pain_index if pain_index > 0 else 0
    
    # Treynor Ratio (requires beta, simplified version using market proxy)
    # Assuming S&P 500 as market proxy - you might want to pass actual market returns
    treynor_ratio = (annualized_return - risk_free_rate) / 1.0  # Simplified, assuming beta=1
    
    metrics = {
        'Metric': [
            'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio',
            'Treynor Ratio', 'Calmar Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)',
            'Downside Deviation', 'Omega Ratio', 'Skewness', 'Kurtosis',
            'Anderson-Darling Stat', 'Shapiro-Wilk Stat', 'Shapiro-Wilk p',
            'Mean Absolute Deviation', 'IQR', 'Pain Ratio'
        ],
        'Value': [
            f'{annualized_return:.4f}',
            f'{annualized_vol:.4f}',
            f'{sharpe_ratio:.4f}',
            f'{sortino_ratio:.4f}',
            f'{treynor_ratio:.4f}',
            f'{calmar_ratio:.4f}',
            f'{max_drawdown:.4f}',
            f'{var_95:.4f}',
            f'{cvar_95:.4f}',
            f'{downside_deviation_annual:.4f}',
            f'{omega_ratio:.4f}',
            f'{skewness:.4f}',
            f'{kurt:.4f}',
            f'{anderson_stat:.4f}',
            f'{shapiro_stat:.4f}',
            f'{shapiro_p:.4f}',
            f'{mad:.4f}',
            f'{iqr:.4f}',
            f'{pain_ratio:.4f}'
        ]
    }
    
    return pd.DataFrame(metrics)


@beartype
def plot_metrics_table(metrics_df, figsize=(12, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=metrics_df.values,
                     colLabels=metrics_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
        else:
            if j == 0:  # Metric names
                cell.set_facecolor('#e6e6e6')
                cell.set_text_props(weight='bold')
            else:  # Values
                # Color code based on metric type and value
                metric_name = metrics_df.iloc[i-1, 0]
                value = float(metrics_df.iloc[i-1, 1])
                
                if 'Return' in metric_name or 'Ratio' in metric_name:
                    if value > 0:
                        cell.set_facecolor('#d4edda')  # Light green for positive
                    else:
                        cell.set_facecolor('#f8d7da')  # Light red for negative
                elif 'Drawdown' in metric_name or 'VaR' in metric_name:
                    if value < 0:
                        cell.set_facecolor('#d4edda')  # Light green for good (negative drawdown)
                    else:
                        cell.set_facecolor('#f8d7da')  # Light red for bad
                else:
                    cell.set_facecolor('#f8f9fa')  # Light gray for neutral
    
    plt.title('Strategy Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.close()
    
    return fig, ax


@beartype
def plot_strategy_charts(pnls):    
    pnls['Date'] = pd.to_datetime(pnls['Date'])
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(20, 4*4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, :])  # Cost plot
    ax2 = fig.add_subplot(gs[1, :])  # Returns plot
    ax3 = fig.add_subplot(gs[2, :])  # Drawdown plot
    ax4 = fig.add_subplot(gs[3, :])  # Balance plot

    ax1.plot(pnls['Date'], pnls['Cost'], label='Cost', color='red', linewidth=1)
    ax1.fill_between(pnls['Date'], 0, pnls['Cost'], alpha=0.3, color='red')
    ax1.set_ylabel('Cost')

    ax11 = ax1.twinx()
    ax11.plot(pnls['Date'], pnls['Cost'].cumsum(), label='Cumulative Cost', color='purple', linewidth=2)
    ax11.set_ylabel('Cumulative Cost')

    # ax1.set_xlabel('Date')
    ax1.set_title('Transaction Costs Over Time')
    ax1.grid(True, alpha=0.3)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax11.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    returns = pnls['Returns']
    cumsum_returns = returns.cumsum()
    ax2.plot(pnls['Date'], returns, label='Period Returns', color='blue', alpha=0.7, linewidth=1)
    ax2.fill_between(pnls['Date'], 0, returns, alpha=0.3, color='green', where=(returns >= 0))
    ax2.fill_between(pnls['Date'], 0, returns, alpha=0.3, color='red', where=(returns < 0))
    ax2.set_ylabel('Returns')
    
    ax22 = ax2.twinx()
    ax22.plot(pnls['Date'], cumsum_returns, label='Cumulative Returns', color='blue', linewidth=2)
    # ax22.fill_between(pnls['Date'], 0, cumsum_returns, alpha=0.3, color='green', where=(cumsum_returns >= 0))
    # ax22.fill_between(pnls['Date'], 0, cumsum_returns, alpha=0.3, color='red', where=(cumsum_returns < 0))
    ax2.set_ylabel('Cumulative Returns')
    
    ax2.set_title('Returns Over Time')
    ax2.grid(True, alpha=0.3)
    
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax22.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    running_peak = np.maximum.accumulate(pnls['Balance'])
    max_drawdown = (pnls['Balance'] - running_peak) / running_peak * 100
    ax3.fill_between(pnls['Date'], 0, max_drawdown, alpha=0.5, color='red', where=(max_drawdown < 0))
    ax3.plot(pnls['Date'], max_drawdown, label=f'Max DD: {max_drawdown.min():.1f}%', color='darkred', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Strategy Drawdown')

    ax4.plot(pnls['Date'], pnls['Balance'], label='Strategy Balance', color='purple', linewidth=2)
    ax4.fill_between(pnls['Date'], pnls['Balance'].iloc[0], pnls['Balance'], 
                     alpha=0.3, color='purple', where=(pnls['Balance'] >= pnls['Balance'].iloc[0]))
    ax4.fill_between(pnls['Date'], pnls['Balance'].iloc[0], pnls['Balance'], 
                     alpha=0.3, color='orange', where=(pnls['Balance'] < pnls['Balance'].iloc[0]))
    ax4.set_ylabel('Balance')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Portfolio Balance Over Time')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle('Strategy Performance Analysis - Charts', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.close()
    
    return fig, [ax1, ax2, ax3, ax4]


@beartype
def plot_strategy_with_metrics_separate(path_to_pnl: Path, risk_free_rate=0.02):
    pnls = pd.read_csv(path_to_pnl, index_col=0)
    pnls['Date'] = pd.to_datetime(pnls.index).strftime("%Y-%m-%d")
    
    metrics_df = calculate_metrics(pnls, risk_free_rate)
    charts_fig, _ = plot_strategy_charts(pnls)
    table_fig, _ = plot_metrics_table(metrics_df)
    
    return charts_fig, table_fig, metrics_df


@beartype
def save_plots_to_files(charts_fig, table_fig, output: Path, name: str = ""):
    charts_fig.savefig(output / f"charts_{name}.png", dpi=300, bbox_inches='tight')
    table_fig.savefig(output / f"table_{name}.png", dpi=300, bbox_inches='tight')
    print(f"Charts saved to: {output / f'charts_{name}.png'}")
    print(f"Table saved to: {output / f'table_{name}.png'}")
    
 
def main():
    # folder_name = "new_bayessian_all_best"
    # folder_name = "new_markowitz_all_best"
    # folder_name = "new_black_litterman"
    
    # folder_name = "new_combined_bayessian_markowitz_black_litterman"
    # folder_name = "new_combined_bayessian_markowitz"
    # folder_name = "equal_weight"
    
    # folder_name = "sp500_all_equal_weight"
    # folder_name = "sp500_benchmark_equal_weight"
    # folder_name = "sp500_all_bayessian"
    # folder_name = "sp500_benchmark_bayessian"
    
    # folder_name = "t_student_posterior_alpha_99_bayessian_all"
    
    # folder_name = "bayessian_cvar_returns_no_previous"
    # folder_name = "advanced_bayessian_walk_forward_no_max"
    # folder_name = "new_liquidity_weighted"
    
    folder_name = "advanced_smart_bayesian"
    
    inputs = Path.cwd() / "results" / folder_name
    charts_fig, table_fig, metrics = plot_strategy_with_metrics_separate(path_to_pnl=inputs / "pnl.csv")
    output = Path.cwd() / "plots" / "charts" / folder_name
    output.mkdir(parents=True, exist_ok=True)
    save_plots_to_files(charts_fig, table_fig, output, name="")



if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Added research metrics and train/test helpers for parameter-stability analysis
# -----------------------------------------------------------------------------

def _safe_returns_from_pnl(pnls: pd.DataFrame) -> pd.Series:
    if 'Returns' not in pnls.columns:
        raise ValueError("pnl DataFrame must contain a 'Returns' column")
    return pd.to_numeric(pnls['Returns'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()


def calculate_extended_metrics(pnls: pd.DataFrame, risk_free_rate: float = 0.02, periods_per_year: int = 12) -> pd.DataFrame:
    """Extended metrics requested for the strategy study.

    Adds: win rate, average drawdown, expectancy, rolling-compatible Sharpe/Sortino,
    and profit factor. The output keeps the same two-column schema as calculate_metrics().
    """
    base = calculate_metrics(pnls, risk_free_rate=risk_free_rate)
    returns = _safe_returns_from_pnl(pnls)
    balance = pd.to_numeric(pnls['Balance'], errors='coerce') if 'Balance' in pnls.columns else (1.0 + returns).cumprod()
    balance = balance.reindex(returns.index).ffill().fillna(method='bfill') if hasattr(balance, 'reindex') else pd.Series(balance)
    dd = balance / balance.cummax() - 1.0
    underwater = dd[dd < 0]

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) else np.nan
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-12 else np.inf
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss if np.isfinite(win_rate) else np.nan
    avg_drawdown = underwater.mean() if len(underwater) else 0.0

    extra = pd.DataFrame({
        'Metric': ['Win Rate', 'Average Drawdown', 'Expectancy', 'Profit Factor', 'Average Win', 'Average Loss'],
        'Value': [f'{win_rate:.4f}', f'{avg_drawdown:.4f}', f'{expectancy:.4f}', f'{profit_factor:.4f}', f'{avg_win:.4f}', f'{avg_loss:.4f}']
    })
    return pd.concat([base, extra], ignore_index=True)


def split_pnl_train_test(pnls: pd.DataFrame, train_fraction: float = 0.70, split_date: Union[str, None] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal train/test split for realized strategy PnL.

    Use split_date for a fixed calendar split, otherwise use train_fraction.
    """
    df = pnls.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    if split_date is not None and isinstance(df.index, pd.DatetimeIndex):
        d = pd.to_datetime(split_date)
        return df.loc[df.index <= d].copy(), df.loc[df.index > d].copy()
    n = len(df)
    k = max(1, min(n - 1, int(round(n * float(train_fraction))))) if n > 2 else n
    return df.iloc[:k].copy(), df.iloc[k:].copy()


def calculate_train_test_metrics(pnls: pd.DataFrame, train_fraction: float = 0.70, split_date: Union[str, None] = None, risk_free_rate: float = 0.02) -> pd.DataFrame:
    train, test = split_pnl_train_test(pnls, train_fraction=train_fraction, split_date=split_date)
    rows = []
    for label, df in [('overall', pnls), ('train', train), ('test', test)]:
        if df.empty:
            continue
        m = calculate_extended_metrics(df, risk_free_rate=risk_free_rate)
        record = {'sample': label, 'start': str(df.index.min()), 'end': str(df.index.max()), 'n_periods': len(df)}
        record.update(dict(zip(m['Metric'], m['Value'])))
        rows.append(record)
    return pd.DataFrame(rows)


def rolling_sharpe_ratio(returns: pd.Series, window: int = 12, risk_free_rate: float = 0.02, periods_per_year: int = 12) -> pd.Series:
    r = pd.to_numeric(returns, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    idx = []
    for i in range(window, len(r) + 1):
        x = r.iloc[i-window:i] - risk_free_rate / periods_per_year
        sd = x.std(ddof=1)
        out.append(x.mean() / sd * np.sqrt(periods_per_year) if sd > 1e-12 else np.nan)
        idx.append(r.index[i-1])
    return pd.Series(out, index=idx, name=f'rolling_sharpe_{window}')


def rolling_sortino_ratio(returns: pd.Series, window: int = 12, risk_free_rate: float = 0.02, periods_per_year: int = 12) -> pd.Series:
    r = pd.to_numeric(returns, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    idx = []
    for i in range(window, len(r) + 1):
        x = r.iloc[i-window:i]
        downside_dev = np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)) * np.sqrt(periods_per_year)
        ann_ret = x.mean() * periods_per_year
        out.append((ann_ret - risk_free_rate) / downside_dev if downside_dev > 1e-12 else np.nan)
        idx.append(r.index[i-1])
    return pd.Series(out, index=idx, name=f'rolling_sortino_{window}')


def plot_train_test_performance(pnls: pd.DataFrame, train_fraction: float = 0.70, split_date: Union[str, None] = None, figsize=(14, 7)):
    """Matplotlib plot that marks train/test split while preserving the old stats.py style."""
    train, test = split_pnl_train_test(pnls, train_fraction=train_fraction, split_date=split_date)
    fig, ax = plt.subplots(figsize=figsize)
    b = pd.to_numeric(pnls['Balance'], errors='coerce') if 'Balance' in pnls.columns else (1 + _safe_returns_from_pnl(pnls)).cumprod()
    ax.plot(b.index, b.values, label='Strategy balance')
    if not test.empty:
        ax.axvline(test.index.min(), linestyle='--', linewidth=1.2, label='Train/test split')
    ax.set_title('Strategy performance with train/test split')
    ax.set_xlabel('Date')
    ax.set_ylabel('Balance')
    ax.grid(alpha=0.25)
    ax.legend()
    return fig, ax
