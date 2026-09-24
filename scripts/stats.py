import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, skew, kurtosis, anderson, shapiro
from beartype import beartype
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

from scripts.bayessian_model import OUTPUT


@beartype
def calculate_metrics(pnls, risk_free_rate=0.02):
    returns = pnls['Returns'].values
    dates = pd.to_datetime(pnls['Date'])
    
    # Calculate number of trading periods per year (assuming monthly data)
    periods_per_year = 12
    
    total_return = (pnls['Balance'].iloc[-1] / pnls['Balance'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    annualized_vol = np.std(returns) * np.sqrt(periods_per_year)
    
    # Risk-adjusted returns
    excess_returns = returns - risk_free_rate/periods_per_year
    sharpe_ratio = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    
    # Downside Deviation
    target_return = 0
    downside_returns_squared = np.minimum(returns - target_return, 0) ** 2
    downside_deviation_annual = np.sqrt(np.mean(downside_returns_squared)) * np.sqrt(periods_per_year)
    
    threshold = 0
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else np.inf
    
    skewness = skew(returns)
    kurt = kurtosis(returns)
    
    # Normality tests
    anderson_stat, anderson_critical, anderson_significance = anderson(returns)
    shapiro_stat, shapiro_p = shapiro(returns)
    
    # Mean Absolute Deviation
    mad = np.mean(np.abs(returns - np.mean(returns)))
    
    # Interquartile Range
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
    pnls = pd.read_csv(path_to_pnl)
    
    metrics_df = calculate_metrics(pnls, risk_free_rate)
    charts_fig, _ = plot_strategy_charts(pnls)
    table_fig, _ = plot_metrics_table(metrics_df)
    
    return charts_fig, table_fig, metrics_df


@beartype
def save_plots_to_files(charts_fig, table_fig, output: Path, name: str = "bayessian_best"):
    charts_fig.savefig(output / f"charts_{name}.png", dpi=300, bbox_inches='tight')
    table_fig.savefig(output / f"table_{name}.png", dpi=300, bbox_inches='tight')
    print(f"Charts saved to: {output / f'charts_{name}.png'}")
    print(f"Table saved to: {output / f'table_{name}.png'}")
    
 
def main():
    charts_fig, table_fig, metrics = plot_strategy_with_metrics_separate(path_to_pnl=OUTPUT / "pnl-1.csv")
    save_plots_to_files(charts_fig, table_fig, OUTPUT.parent.parent / "plots" / "charts")



if __name__ == "__main__":
    main()
