import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    # better approach (instead of using returns, uses directly balance)
    balance = pnls['Balance'].values
    running_max = np.maximum.accumulate(balance)
    drawdown = (balance - running_max) / running_max
    max_drawdown = drawdown.min()
    
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
    treynor_ratio = (annualized_return - risk_free_rate) / 1.0  # Simplified, assuming beta=1
    
    metrics_dict = {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Treynor Ratio': treynor_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Downside Deviation': downside_deviation_annual,
        'Omega Ratio': omega_ratio,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Anderson-Darling Stat': anderson_stat,
        'Shapiro-Wilk Stat': shapiro_stat,
        'Shapiro-Wilk p': shapiro_p,
        'Mean Absolute Deviation': mad,
        'IQR': iqr,
        'Pain Ratio': pain_ratio
    }
    
    return metrics_dict


@beartype
def plot_metrics_table(metrics_df, figsize=(12, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Format values to 4 decimal places for display
    display_df = metrics_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
    
    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     rowLabels=display_df.index,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Adjust column widths
    table.auto_set_column_width(col=list(range(len(display_df.columns) + 1)))
    
    # Color header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white', fontsize=10)
        elif j == 0:  # Row labels (metric names)
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold', fontsize=9)
        else:
            # Color code based on metric type and value
            metric_name = display_df.index[i-1]
            value = metrics_df.iloc[i-1, j-1]
            
            if 'Return' in metric_name or 'Ratio' in metric_name or 'Omega' in metric_name:
                if value > 0:
                    cell.set_facecolor('#d4edda')  # Light green for positive
                else:
                    cell.set_facecolor('#f8d7da')  # Light red for negative
            elif 'Drawdown' in metric_name or 'VaR' in metric_name or 'Downside' in metric_name:
                if value < 0:
                    cell.set_facecolor('#d4edda')  # Light green for good (negative drawdown)
                else:
                    cell.set_facecolor('#f8d7da')  # Light red for bad
            else:
                cell.set_facecolor('#f8f9fa')  # Light gray for neutral
    
    plt.title('Strategy Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig, ax


@beartype
def plot_comparative_charts(all_pnls_dict, figsize=(24, 16)):
    """Plot comparative charts for multiple strategies"""
    
    # Create figure with GridSpec - bigger overall figure
    fig = plt.figure(figsize=figsize)  # Increased from (20, 12) to (24, 16)
    
    # Adjust GridSpec with more height ratios and increased hspace for spacing
    gs = fig.add_gridspec(4, 1, 
                          height_ratios=[1, 1, 1, 1], 
                          hspace=0.5,  # Increased from 0.4 to 0.5 for more space between subplots
                          top=0.94,    # Leave room for suptitle
                          bottom=0.06,  # Leave room at bottom
                          left=0.08,    # More space for ylabels
                          right=0.95)   # More space on right
    
    ax1 = fig.add_subplot(gs[0, :])  # Cumulative Cost
    ax2 = fig.add_subplot(gs[1, :])  # Cumulative Returns
    ax3 = fig.add_subplot(gs[2, :])  # Drawdown
    ax4 = fig.add_subplot(gs[3, :])  # Balance
    
    # Color palette for different strategies
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_pnls_dict)))
    
    for (label, pnls), color in zip(all_pnls_dict.items(), colors):
        pnls_copy = pnls.copy()
        pnls_copy['Date'] = pd.to_datetime(pnls_copy['Date'])
        
        # Plot cumulative cost
        ax1.plot(pnls_copy['Date'], pnls_copy['Cost'].cumsum(), 
                label=label, color=color, linewidth=2.5, alpha=0.8)  # Slightly thicker lines
        
        # Plot cumulative returns
        cumulative_returns = (1 + pnls_copy['Returns']).cumprod() - 1
        ax2.plot(pnls_copy['Date'], cumulative_returns, 
                label=label, color=color, linewidth=2.5, alpha=0.8)
        
        # Plot drawdown
        running_peak = np.maximum.accumulate(pnls_copy['Balance'])
        max_drawdown = (pnls_copy['Balance'] - running_peak) / running_peak * 100
        ax3.plot(pnls_copy['Date'], max_drawdown, 
                label=label, color=color, linewidth=2.5, alpha=0.8)
        
        # Plot actual balance (NOT normalized)
        ax4.plot(pnls_copy['Date'], pnls_copy['Balance'], 
                label=label, color=color, linewidth=2.5, alpha=0.8)
    
    # Styling for all subplots - increased font sizes
    for ax, title, ylabel in [
        (ax1, 'Cumulative Transaction Costs Comparison', 'Cumulative Cost'),
        (ax2, 'Cumulative Returns Comparison', 'Cumulative Returns'),
        (ax3, 'Drawdown Comparison', 'Drawdown (%)'),
        (ax4, 'Portfolio Balance Comparison', 'Balance ($)')
    ]:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)  # Increased fontsize and added pad
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)  # Increased fontsize and labelpad
        ax.legend(loc='best', fontsize=11, framealpha=0.9)  # Increased legend fontsize
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)  # Styled grid
        ax.tick_params(axis='both', labelsize=10)  # Increased tick label size
        ax.tick_params(axis='x', rotation=45)
    
    # Main title with more spacing
    fig.suptitle('Strategy Performance Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add extra spacing between subplots by adjusting subplot parameters
    plt.subplots_adjust(hspace=0.5)  # This is redundant with GridSpec hspace but ensures spacing
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
    
    return fig, [ax1, ax2, ax3, ax4]


@beartype
def compare_multiple_strategies(folder_names: list, base_path: Path = None):
    """Compare multiple strategies from different folders"""
    
    if base_path is None:
        base_path = Path.cwd() / "results"
    
    all_pnls = {}
    all_metrics = {}
    
    # Load data from all folders
    for folder_name in folder_names:
        input_path = base_path / folder_name / "pnl.csv"
        if input_path.exists():
            pnls = pd.read_csv(input_path, index_col=0)
            pnls['Date'] = pd.to_datetime(pnls.index).strftime("%Y-%m-%d")
            all_pnls[folder_name] = pnls
            all_metrics[folder_name] = calculate_metrics(pnls)
        else:
            print(f"Warning: PnL file not found for {folder_name} at {input_path}")
    
    if not all_pnls:
        raise ValueError("No valid PnL files found")
    
    # Create metrics DataFrame (rows as metrics, columns as folder names)
    metrics_df = pd.DataFrame(all_metrics)
    
    # Generate comparative charts
    charts_fig, _ = plot_comparative_charts(all_pnls)
    table_fig, _ = plot_metrics_table(metrics_df)
    
    return charts_fig, table_fig, metrics_df


@beartype
def save_plots_to_files(charts_fig, table_fig, output: Path, name: str = "comparison"):
    output.mkdir(parents=True, exist_ok=True)
    charts_fig.savefig(output / f"charts_{name}.png", dpi=300, bbox_inches='tight')
    table_fig.savefig(output / f"table_{name}.png", dpi=300, bbox_inches='tight')
    print(f"Charts saved to: {output / f'charts_{name}.png'}")
    print(f"Table saved to: {output / f'table_{name}.png'}")


@beartype
def save_metrics_to_csv(metrics_df: pd.DataFrame, output: Path, name: str = "comparison"):
    """Save metrics to CSV file"""
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"metrics_{name}.csv"
    metrics_df.to_csv(csv_path)
    print(f"Metrics saved to: {csv_path}")


def main():
    
    folder_name = "comparison_0"
    folder_names = [
        "bayessian_cvar_returns",
        "bayessian_expected_returns",
        "bayessian_deviation_from_target",
        "new_bayessian_all",
    ]
    
    folder_name = "comparison_1"
    folder_names = [
        "bayessian_cvar_returns_best",
        "bayessian_expected_returns_best",
        "bayessian_deviation_from_target_best",
        "new_bayessian_all_best",
    ]
    
    folder_name = "comparison_2"
    folder_names = [
        "markowitz_max_sharpe",
        "markowitz_min_variance",
        "markowitz_max_return_min_vol",
        "markowitz_max_diversification",
        "markowitz_max_decorrelation",
    ]
    
    folder_name = "comparison_3"
    folder_names = [
        "horseshoe_posterior_alpha_95_bayessian_all",
        "horseshoe_posterior_alpha_99_bayessian_all",
        "gaussian_posterior_alpha_99_bayessian_all",
        "spike_slab_posterior_alpha_99_bayessian_all",
        "t_student_posterior_alpha_99_bayessian_all",
        "empirical_bayes_posterior_alpha_99_bayessian_all",
    ]
    
    # folder_name = "comparison_4"
    # folder_names = [
    #     "combined_bayessian_markowitz_black_litterman",
    #     "combined_bayessian_markowitz",
    #     "combined_bayessian_black_litterman",
    #     "combined_markowitz_black_litterman",
    # ]
    
    folder_name = "comparison_5"
    folder_names = [
        "simple_long",
        "risk_parity",
        "equal_weight",
        "inverse_volatility",
        "liquidity_weighted",
        "sp500_all_equal_weight",
        "sp500_benchmark_equal_weight",
    ]
    
    folder_name = "comparison_6"
    folder_names = [
        "new_bayessian_all",
        "new_bayessian_all_best",
        "new_markowitz_all",
        "sp500_all_bayessian",
        "sp500_benchmark_bayessian",
        "sp500_all_equal_weight",
        "sp500_benchmark_equal_weight",
    ]
    
    charts_fig, table_fig, metrics_df = compare_multiple_strategies(folder_names)
    output = Path.cwd() / "plots" / "charts" / folder_name
    save_plots_to_files(charts_fig, table_fig, output, name="all_strategies")
    
    # save_metrics_to_csv(metrics_df, output, name="all_strategies")
    # print("PERFORMANCE METRICS COMPARISON")
    # print(metrics_df.round(4))
    # plt.show()


if __name__ == "__main__":
    main()
