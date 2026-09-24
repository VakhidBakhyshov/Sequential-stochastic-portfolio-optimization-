"""Analysis utilities for return-maximizing and risk-controlled calibration.

The functions in this module are deliberately independent of Optuna so that a
completed study can be re-analysed from ``trial_metrics.csv`` without rerunning
expensive portfolio simulations.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

EPS = 1e-12

RETURN_COLUMNS = (
    "annualized_return",
    "sharpe_ratio",
    "sortino_ratio",
    "psr",
    "dsr",
)
RISK_COLUMNS = (
    "annualized_volatility",
    "abs_max_drawdown",
    "abs_cvar_95",
    "abs_cdar_95",
    "downside_deviation",
    "turnover",
)


def _robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    median = float(x.median()) if x.notna().any() else 0.0
    mad = float((x - median).abs().median()) if x.notna().any() else 0.0
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(x.std(ddof=0))
    if not np.isfinite(scale) or scale <= EPS:
        return pd.Series(0.0, index=x.index)
    return ((x - median) / scale).clip(-6.0, 6.0).fillna(0.0)


def add_policy_scores(
    trials: pd.DataFrame,
    *,
    return_weights: dict[str, float] | None = None,
    risk_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add comparable return and risk-control utility scores.

    Robust cross-sectional z-scores prevent one metric's units from dominating.
    Higher is better for both returned scores. Risk metrics therefore enter the
    risk-control utility with a minus sign.
    """
    df = trials.copy()
    if "status" in df.columns:
        completed = df["status"].astype(str).str.upper().eq("COMPLETE")
    else:
        completed = pd.Series(True, index=df.index)

    if "max_drawdown" in df.columns and "abs_max_drawdown" not in df.columns:
        df["abs_max_drawdown"] = pd.to_numeric(df["max_drawdown"], errors="coerce").abs()
    if "cvar_95" in df.columns and "abs_cvar_95" not in df.columns:
        df["abs_cvar_95"] = pd.to_numeric(df["cvar_95"], errors="coerce").abs()
    if "cdar_95" in df.columns and "abs_cdar_95" not in df.columns:
        df["abs_cdar_95"] = pd.to_numeric(df["cdar_95"], errors="coerce").abs()

    r_weights = {
        "annualized_return": 1.00,
        "sharpe_ratio": 0.15,
        "sortino_ratio": 0.10,
        "dsr": 0.15,
        **(return_weights or {}),
    }
    q_weights = {
        "annualized_return": 0.35,
        "sharpe_ratio": 0.25,
        "sortino_ratio": 0.20,
        "psr": 0.10,
        "dsr": 0.25,
        "annualized_volatility": 0.25,
        "abs_max_drawdown": 0.35,
        "abs_cvar_95": 0.25,
        "abs_cdar_95": 0.0,
        "downside_deviation": 0.15,
        "turnover": 0.05,
        **(risk_weights or {}),
    }

    return_score = pd.Series(0.0, index=df.index)
    return_mass = 0.0
    for col, weight in r_weights.items():
        if col in df.columns and weight != 0:
            return_score += float(weight) * _robust_z(df.loc[completed, col]).reindex(df.index).fillna(0.0)
            return_mass += abs(float(weight))
    df["return_policy_score"] = return_score / max(return_mass, EPS)

    quality_score = pd.Series(0.0, index=df.index)
    quality_mass = 0.0
    for col, weight in q_weights.items():
        if col not in df.columns or weight == 0:
            continue
        sign = -1.0 if col in RISK_COLUMNS else 1.0
        quality_score += sign * float(weight) * _robust_z(df.loc[completed, col]).reindex(df.index).fillna(0.0)
        quality_mass += abs(float(weight))
    df["risk_control_score"] = quality_score / max(quality_mass, EPS)
    df.loc[~completed, ["return_policy_score", "risk_control_score"]] = np.nan
    return df


def choose_policy_trials(
    trials: pd.DataFrame,
    *,
    return_floor_quantile: float = 0.35,
    minimum_annual_return: float | None = None,
    minimum_dsr: float | None = None,
) -> dict[str, pd.Series]:
    """Choose one maximum-return trial and one risk-controller trial.

    The risk-controller is restricted to trials that preserve a configurable
    return floor, avoiding the trivial all-cash/near-zero-risk solution.
    """
    df = trials.copy()
    if not {"return_policy_score", "risk_control_score"}.issubset(df.columns):
        df = add_policy_scores(df)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("COMPLETE")]
    df = df.dropna(subset=["annualized_return"])
    if df.empty:
        raise ValueError("No completed trials with annualized_return are available.")

    return_order = [c for c in ["annualized_return", "dsr", "sharpe_ratio"] if c in df.columns]
    max_return = df.sort_values(return_order, ascending=[False] * len(return_order)).iloc[0]

    floor = float(df["annualized_return"].quantile(np.clip(return_floor_quantile, 0.0, 1.0)))
    if minimum_annual_return is not None:
        floor = max(floor, float(minimum_annual_return))
    eligible = df[df["annualized_return"] >= floor]
    if minimum_dsr is not None and "dsr" in eligible.columns:
        dsr_eligible = eligible[pd.to_numeric(eligible["dsr"], errors="coerce") >= float(minimum_dsr)]
        if not dsr_eligible.empty:
            eligible = dsr_eligible
    if eligible.empty:
        eligible = df
    risk_control = eligible.sort_values(
        [c for c in ["risk_control_score", "dsr", "annualized_return"] if c in eligible.columns],
        ascending=False,
    ).iloc[0]
    return {"max_return": max_return, "risk_control": risk_control}


def extract_pareto_front(
    trials: pd.DataFrame,
    *,
    return_column: str = "annualized_return",
    risk_columns: Iterable[str] = ("annualized_volatility", "abs_max_drawdown", "abs_cvar_95", "abs_cdar_95"),
) -> pd.DataFrame:
    """Return non-dominated trials: maximize return and minimize every risk."""
    df = trials.copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("COMPLETE")]
    risk_cols = [c for c in risk_columns if c in df.columns]
    cols = [return_column, *risk_cols]
    df = df.dropna(subset=cols).copy()
    if df.empty:
        return df
    values = df[cols].to_numpy(dtype=float)
    oriented = values.copy()
    oriented[:, 0] *= -1.0  # minimization convention
    efficient = np.ones(len(df), dtype=bool)
    for i, point in enumerate(oriented):
        if not efficient[i]:
            continue
        dominated_by_other = np.any(np.all(oriented <= point, axis=1) & np.any(oriented < point, axis=1))
        if dominated_by_other:
            efficient[i] = False
    return df.loc[efficient].sort_values(return_column, ascending=False).reset_index(drop=True)


def _parameter_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("param_")]


def _bin_parameter(values: pd.Series, max_bins: int = 6) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == values.notna().sum() and numeric.nunique(dropna=True) > max_bins:
        try:
            q = min(max_bins, int(numeric.nunique()))
            return pd.qcut(numeric, q=q, duplicates="drop").astype(str)
        except Exception:
            pass
    return values.fillna("<missing>").astype(str)


def build_parameter_range_summary(
    trials: pd.DataFrame,
    *,
    metrics: Iterable[str] = (
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "cvar_95",
        "cdar_95",
        "psr",
        "dsr",
        "risk_control_score",
    ),
    max_bins: int = 6,
    top_fraction: float = 0.20,
) -> pd.DataFrame:
    """Summarize how each parameter range changes each performance metric."""
    df = trials.copy()
    if not {"return_policy_score", "risk_control_score"}.issubset(df.columns):
        df = add_policy_scores(df)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("COMPLETE")]
    if df.empty:
        return pd.DataFrame()
    cutoff = float(df["risk_control_score"].quantile(1.0 - np.clip(top_fraction, 0.01, 0.99)))
    df["top_risk_control"] = df["risk_control_score"] >= cutoff
    metric_cols = [c for c in metrics if c in df.columns]
    rows: list[dict[str, Any]] = []
    for param in _parameter_columns(df):
        groups = _bin_parameter(df[param], max_bins=max_bins)
        for bucket, block in df.groupby(groups, dropna=False):
            row: dict[str, Any] = {
                "parameter": param.removeprefix("param_"),
                "range": str(bucket),
                "n_trials": int(len(block)),
                "top_risk_control_share": float(block["top_risk_control"].mean()),
            }
            for metric in metric_cols:
                x = pd.to_numeric(block[metric], errors="coerce")
                row[f"median_{metric}"] = float(x.median()) if x.notna().any() else np.nan
                row[f"mean_{metric}"] = float(x.mean()) if x.notna().any() else np.nan
                row[f"q25_{metric}"] = float(x.quantile(0.25)) if x.notna().any() else np.nan
                row[f"q75_{metric}"] = float(x.quantile(0.75)) if x.notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["parameter", "range"]).reset_index(drop=True)


def policy_comparison_table(trials: pd.DataFrame, selected: dict[str, pd.Series]) -> pd.DataFrame:
    metric_order = [
        "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "cvar_95", "downside_deviation", "psr", "dsr", "turnover",
        "return_policy_score", "risk_control_score",
    ]
    rows: list[dict[str, Any]] = []
    for policy, row in selected.items():
        record: dict[str, Any] = {"policy": policy, "trial_number": int(row.get("trial_number", row.name))}
        for col in metric_order:
            if col in row.index:
                record[col] = row[col]
        params = {c.removeprefix("param_"): row[c] for c in row.index if c.startswith("param_")}
        record["parameters"] = json.dumps(params, sort_keys=True, default=str)
        rows.append(record)
    return pd.DataFrame(rows)
