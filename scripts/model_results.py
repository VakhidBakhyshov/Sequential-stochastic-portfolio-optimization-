import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scripts.bayessian_model import OUTPUT


def plot_model_results(df, save_path='./bayesian_plots'):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
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
    
    axes[1, 0].plot(df['Date'], df['rank_ic'], marker='^', linewidth=2, markersize=6, color='green')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Rank IC (Positive is Better)', fontsize=12)
    axes[1, 0].set_ylabel('Rank IC')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['Date'], df['mae'], marker='v', linewidth=2, markersize=6, color='red')
    axes[1, 1].set_title('MAE (Lower is Better)', fontsize=12)
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].plot(df['Date'], df['coverage_95'], marker='D', linewidth=2, markersize=6, color='purple')
    axes[2, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='Ideal 95%')
    axes[2, 0].fill_between(df['Date'], 0.95-0.05, 0.95+0.05, alpha=0.2, color='gray')
    axes[2, 0].set_title('95% Credible Interval Coverage', fontsize=12)
    axes[2, 0].set_ylabel('Coverage')
    axes[2, 0].set_ylim([0, 1])
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].axis('tight')
    axes[2, 1].axis('off')
    
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
    
    table = axes[2, 1].table(cellText=summary_stats.round(4).values,
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
    df = pd.read_csv(OUTPUT / 'model_metrics.csv')
    plot_model_results(df, save_path=OUTPUT.parent.parent / "plots" / "model-metrics")



if __name__ == "__main__":
    main()
