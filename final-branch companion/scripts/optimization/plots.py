"""Publication-oriented plots for Optuna calibration results."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from scripts.optimization.analysis import add_policy_scores, build_parameter_range_summary


def _ensure_scores(trials: pd.DataFrame) -> pd.DataFrame:
    return trials.copy() if {"return_policy_score", "risk_control_score"}.issubset(trials.columns) else add_policy_scores(trials)


def save_parameter_effect_heatmap(trials: pd.DataFrame, output_path: str | Path) -> Path:
    summary = build_parameter_range_summary(trials)
    if summary.empty:
        raise ValueError("No parameter ranges are available for plotting.")
    metric_names = [
        "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "cvar_95", "cdar_95", "psr", "dsr", "risk_control_score",
    ]
    cols = [f"median_{m}" for m in metric_names if f"median_{m}" in summary.columns]
    matrix = summary.set_index(summary["parameter"] + " = " + summary["range"])[cols].copy()
    # Standardize each metric across parameter ranges so unlike units are comparable.
    matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0).replace(0.0, np.nan)
    # Convert every column to a common "higher is more favorable" interpretation.
    for risk_metric in ("median_annualized_volatility", "median_downside_deviation", "median_turnover"):
        if risk_metric in matrix.columns:
            matrix[risk_metric] *= -1.0
    matrix = matrix.fillna(0.0).clip(-3.0, 3.0)
    labels = [c.removeprefix("median_").replace("_", " ") for c in cols]

    height = max(6.0, min(24.0, 0.32 * len(matrix) + 2.5))
    fig, ax = plt.subplots(figsize=(14, height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index, fontsize=8)
    ax.set_title("Parameter-range effect on strategy metrics (robust standardized medians)")
    ax.set_xlabel("Performance metric")
    ax.set_ylabel("Calibrated parameter range")
    fig.colorbar(image, ax=ax, label="Standardized favorable effect (green = better)")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_metric_distribution_shift(trials: pd.DataFrame, output_path: str | Path, top_fraction: float = 0.20) -> Path:
    df = _ensure_scores(trials)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("COMPLETE")]
    cutoff = float(df["risk_control_score"].quantile(1.0 - top_fraction))
    df["group"] = np.where(df["risk_control_score"] >= cutoff, "Best calibrated range", "All other trials")
    metrics = [c for c in ["annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "max_drawdown", "cvar_95", "cdar_95", "dsr"] if c in df]
    normalized = []
    for metric in metrics:
        x = pd.to_numeric(df[metric], errors="coerce")
        lo, hi = float(x.quantile(0.05)), float(x.quantile(0.95))
        scale = max(hi - lo, 1e-12)
        for value, group in zip(((x - lo) / scale).clip(-0.25, 1.25), df["group"]):
            normalized.append({"metric": metric.replace("_", " "), "normalized_value": value, "group": group})
    long = pd.DataFrame(normalized).dropna()
    positions = np.arange(len(metrics), dtype=float)
    width = 0.32
    fig, ax = plt.subplots(figsize=(14, 7))
    for offset, group in [(-width / 2, "All other trials"), (width / 2, "Best calibrated range")]:
        arrays = [long[(long["metric"] == m.replace("_", " ")) & (long["group"] == group)]["normalized_value"].to_numpy() for m in metrics]
        bp = ax.boxplot(arrays, positions=positions + offset, widths=width * 0.9, patch_artist=True, showfliers=False)
        for box in bp["boxes"]:
            box.set_alpha(0.55)
        # Add a single dummy line for a clean legend.
        ax.plot([], [], linewidth=8, alpha=0.55, label=group)
    ax.set_xticks(positions, labels=[m.replace("_", " ") for m in metrics], rotation=30, ha="right")
    ax.set_ylabel("Within-metric normalized value")
    ax.set_title("Metric distribution shift after parameter calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_return_risk_frontier(trials: pd.DataFrame, output_path: str | Path) -> Path:
    df = _ensure_scores(trials)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("COMPLETE")]
    x = pd.to_numeric(df["annualized_volatility"], errors="coerce")
    y = pd.to_numeric(df["annualized_return"], errors="coerce")
    c = pd.to_numeric(df["risk_control_score"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(x, y, c=c, cmap="viridis", alpha=0.75)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    ax.set_title("Return-risk calibration frontier")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Risk-control score")
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def save_numeric_parameter_pair_heatmaps(
    trials: pd.DataFrame,
    output_directory: str | Path,
    *,
    target: str = "risk_control_score",
    max_parameters: int = 4,
) -> list[Path]:
    df = _ensure_scores(trials)
    params = []
    for col in [c for c in df if c.startswith("param_")]:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() == df[col].notna().sum() and x.nunique() >= 3:
            association = abs(float(x.corr(pd.to_numeric(df[target], errors="coerce"))))
            params.append((col, association if np.isfinite(association) else 0.0))
    params = [p for p, _ in sorted(params, key=lambda z: z[1], reverse=True)[:max_parameters]]
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(len(params)):
        for j in range(i + 1, len(params)):
            p1, p2 = params[i], params[j]
            work = df[[p1, p2, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if work.empty:
                continue
            work["x_bin"] = pd.qcut(work[p1], q=min(6, work[p1].nunique()), duplicates="drop")
            work["y_bin"] = pd.qcut(work[p2], q=min(6, work[p2].nunique()), duplicates="drop")
            pivot = work.pivot_table(index="y_bin", columns="x_bin", values=target, aggfunc="median", observed=False)
            fig, ax = plt.subplots(figsize=(10, 7))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
            ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(x) for x in pivot.columns], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)), labels=[str(y) for y in pivot.index])
            ax.set_xlabel(p1.removeprefix("param_"))
            ax.set_ylabel(p2.removeprefix("param_"))
            ax.set_title(f"Median {target.replace('_', ' ')} by parameter ranges")
            fig.colorbar(image, ax=ax, label=target.replace("_", " "))
            fig.tight_layout()
            path = out_dir / f"pair_heatmap__{p1.removeprefix('param_').replace('.', '_')}__{p2.removeprefix('param_').replace('.', '_')}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths
