import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_model_results(df, save_path='./bayesian_plots'):
    df['Date'] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    fig, axes = plt.subplots(4, 3, figsize=(30, 20))
    fig.suptitle('Model Performance Metrics Over Time', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['Date'], df['log_score'], marker='o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Log Score (Higher is Better)', fontsize=12)
    axes[0, 0].set_ylabel('Log Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['Date'], df['waic'], marker='s', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_title('WAIC (Lower is Better)', fontsize=12)
    axes[0, 1].set_ylabel('WAIC')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(df['Date'], df['rank_ic'], marker='^', linewidth=2, markersize=6, color='green')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('Rank IC (Positive is Better)', fontsize=12)
    axes[0, 2].set_ylabel('Rank IC')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['Date'], df['mae'], marker='v', linewidth=2, markersize=6, color='red')
    axes[1, 0].set_title('MAE (Lower is Better)', fontsize=12)
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['Date'], df['coverage_95'], marker='D', linewidth=2, markersize=6, color='purple')
    axes[1, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Ideal 95%')
    axes[1, 1].fill_between(df['Date'], 0.95-0.05, 0.95+0.05, alpha=0.2, color='gray')
    axes[1, 1].set_title('95% Credible Interval Coverage', fontsize=12)
    axes[1, 1].set_ylabel('Coverage')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Sign Precision over time
    axes[1, 2].plot(df['Date'], df['sign_precision'], marker='p', linewidth=2, markersize=6, color='teal')
    axes[1, 2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random 50%')
    axes[1, 2].axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect 100%')
    axes[1, 2].set_title('Sign Precision Over Time', fontsize=12)
    axes[1, 2].set_ylabel('Sign Precision')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # R-squared over time
    axes[2, 0].plot(df['Date'], df['r_squared'], marker='h', linewidth=2, markersize=6, color='brown')
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    axes[2, 0].fill_between(df['Date'], 0, 1, alpha=0.1, color='green', label='Good Range')
    axes[2, 0].set_title('R-squared Over Time', fontsize=12)
    axes[2, 0].set_ylabel('R-squared')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # RMSE over time
    axes[2, 1].plot(df['Date'], df['rmse'], marker='8', linewidth=2, markersize=6, color='darkred')
    axes[2, 1].set_title('RMSE Over Time (Lower is Better)', fontsize=12)
    axes[2, 1].set_ylabel('RMSE')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Predicted Mean over time
    axes[2, 2].plot(df['Date'], df['avg_predicted_mean'], marker='*', linewidth=2, markersize=8, color='navy', label='Predicted')
    axes[2, 2].plot(df['Date'], df['avg_true_mean'], marker='o', linewidth=2, markersize=6, color='darkgreen', label='True', alpha=0.7)
    axes[2, 2].set_title('Predicted vs True Mean Over Time', fontsize=12)
    axes[2, 2].set_ylabel('Mean Value')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # Mean Deviation over time
    axes[3, 0].plot(df['Date'], df['mean_deviation'], marker='d', linewidth=2, markersize=6, color='darkorange')
    axes[3, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Perfect Alignment')
    axes[3, 0].fill_between(df['Date'], df['mean_deviation'], 0, alpha=0.2, color='orange')
    axes[3, 0].set_title('Mean Deviation Over Time', fontsize=12)
    axes[3, 0].set_ylabel('Deviation')
    axes[3, 0].legend()
    axes[3, 0].grid(True, alpha=0.3)
    
    # Avg True Positives over time
    axes[3, 1].plot(df['Date'], df['avg_true_positives'], marker='X', linewidth=2, markersize=6, color='indigo')
    axes[3, 1].set_title('Average True Positives Over Time', fontsize=12)
    axes[3, 1].set_ylabel('Avg True Positives')
    axes[3, 1].grid(True, alpha=0.3)
    
    # Summary statistics table (moved to last position)
    axes[3, 2].axis('tight')
    axes[3, 2].axis('off')
    
    summary_stats = pd.DataFrame({
        'Metric': ['Log Score', 'WAIC', 'Rank IC', 'MAE', 'Coverage 95%'],
        'Mean': [df['log_score'].mean(), df['waic'].mean(), df['rank_ic'].mean(), 
                 df['mae'].mean(), df['coverage_95'].mean()],
        'Std': [df['log_score'].std(), df['waic'].std(), df['rank_ic'].std(),
                df['mae'].std(), df['coverage_95'].std()],
        'Min': [df['log_score'].min(), df['waic'].min(), df['rank_ic'].min(),
                df['mae'].min(), df['coverage_95'].min()],
        'Max': [df['log_score'].max(), df['waic'].max(), df['rank_ic'].max(),
                df['mae'].max(), df['coverage_95'].max()]
    })
    
    table = axes[3, 2].table(cellText=summary_stats.round(4).values,
                              colLabels=summary_stats.columns,
                              cellLoc='center',
                              loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[2, 1].set_title('Summary Statistics', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'time_series_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics_for_corr = ['log_score', 'waic', 'rank_ic', 'mae', 'coverage_95']
    corr_matrix = df[metrics_for_corr].corr()
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(metrics_for_corr)))
    ax.set_yticks(np.arange(len(metrics_for_corr)))
    ax.set_xticklabels(metrics_for_corr, rotation=45, ha='right')
    ax.set_yticklabels(metrics_for_corr)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add correlation values in cells
    for i in range(len(metrics_for_corr)):
        for j in range(len(metrics_for_corr)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", 
                          color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    ax.set_title('Correlation Between Different Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Bayesian Model Metrics', fontsize=16, fontweight='bold')
    
    metrics_plots = [
        ('log_score', 'Log Score Distribution', axes[0, 0]),
        ('waic', 'WAIC Distribution', axes[0, 1]),
        ('rank_ic', 'Rank IC Distribution', axes[0, 2]),
        ('mae', 'MAE Distribution', axes[1, 0]),
        ('coverage_95', '95% Coverage Distribution', axes[1, 1])
    ]
    
    for metric, title, ax in metrics_plots:
        ax.hist(df[metric], bins=10, edgecolor='black', alpha=0.7)
        ax.axvline(df[metric].mean(), color='r', linestyle='--', label=f'Mean: {df[metric].mean():.4f}')
        ax.axvline(df[metric].median(), color='g', linestyle='--', label=f'Median: {df[metric].median():.4f}')
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Box plot for all metrics (normalized)
    axes[1, 2].axis('off')  # Hide the last subplot
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Normalized boxplot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize metrics for comparison
    normalized_df = df[metrics_for_corr].copy()
    for col in normalized_df.columns:
        if col != 'coverage_95':  # Keep coverage as is since it's already bounded
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
    
    # For WAIC and MAE (lower is better), invert so all metrics align as "higher is better"
    normalized_df['waic'] = 1 - normalized_df['waic']
    normalized_df['mae'] = 1 - normalized_df['mae']
    
    normalized_df.boxplot(ax=ax)
    ax.set_title('Normalized Metrics Comparison (All Aligned: Higher is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Score')
    ax.set_xlabel('Metrics')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'normalized_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Performance radar chart (latest vs average)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Get latest metrics and average metrics
    latest = df.iloc[-1][metrics_for_corr].values
    avg = df[metrics_for_corr].mean().values
    
    # For WAIC and MAE (lower is better), invert for radar chart
    latest_inv = latest.copy()
    avg_inv = avg.copy()
    for i, metric in enumerate(metrics_for_corr):
        if metric in ['waic', 'mae']:
            latest_inv[i] = 1 / (latest[i] + 1e-10)  # Avoid division by zero
            avg_inv[i] = 1 / (avg[i] + 1e-10)
        elif metric == 'coverage_95':
            # For coverage, we want to be close to 0.95
            latest_inv[i] = 1 - abs(latest[i] - 0.95)
            avg_inv[i] = 1 - abs(avg[i] - 0.95)
    
    # Normalize for radar
    latest_norm = (latest_inv - latest_inv.min()) / (latest_inv.max() - latest_inv.min())
    avg_norm = (avg_inv - avg_inv.min()) / (avg_inv.max() - avg_inv.min())
    
    angles = np.linspace(0, 2*np.pi, len(metrics_for_corr), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    latest_norm = np.append(latest_norm, latest_norm[0])
    avg_norm = np.append(avg_norm, avg_norm[0])
    
    ax.plot(angles, latest_norm, 'o-', linewidth=2, label='Latest Performance')
    ax.fill(angles, latest_norm, alpha=0.25)
    ax.plot(angles, avg_norm, 'o-', linewidth=2, label='Average Performance')
    ax.fill(angles, avg_norm, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_corr)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart (Latest vs Average)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots have been saved to: {save_path}")
    print(f"Generated plots: time_series_metrics.png, correlation_heatmap.png, distribution_plots.png, normalized_comparison.png, radar_chart.png")



def main():
    # folder_name = "bayessian_all"
    # folder_name = "markowitz_all"
    folder_name = "combined"
    inputs = Path.cwd() / "results" / folder_name / 'model.csv'
    df = pd.read_csv(inputs, index_col=0)
    output = Path.cwd() / "plots" / "model-metrics" / folder_name
    output.mkdir(parents=True, exist_ok=True)
    plot_model_results(df, save_path=output)



if __name__ == "__main__":
    main()
