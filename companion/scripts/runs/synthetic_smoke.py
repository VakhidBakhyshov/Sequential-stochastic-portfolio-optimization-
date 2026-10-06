"""Generate deterministic synthetic artifacts to smoke-test the reporting stack.

These numbers are artificial and must never be used in the paper's empirical
claims. The script validates schemas and rendering for PSR/DSR, FWER/FDR,
causality, nested overfitting diagnostics, return heatmaps and the HTML dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.results.interactive_report import build_interactive_report
from scripts.runs.common_postprocess import (
    generate_metric_artifacts_safely,
    generate_research_validation_safely,
)
from scripts.validation.signal_research import run_signal_research


def _synthetic_trial_audit(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    recipes = [
        ("optimizer", "return_cvar_constraint", 0.25),
        ("optimizer", "return_cvar_constraint", 0.50),
        ("optimizer", "mean_cvar_sharpe", 0.75),
        ("optimizer", "mean_cvar_sharpe", 1.00),
        ("strategy", "risk_parity", 1.00),
        ("strategy", "inverse_volatility", 1.00),
    ]
    rows: list[dict[str, object]] = []
    for date in dates:
        validation = rng.normal([0.70, 0.55, 0.40, 0.28, 0.20, 0.10], 0.27)
        internal_test = 0.55 * validation + rng.normal(0.0, 0.35, len(recipes))
        selected = int(np.argmax(0.45 * validation + 0.55 * internal_test))
        for i, (source, method, alpha) in enumerate(recipes):
            candidate_id = f"{source}:{method}:recipe={i}:alpha={alpha:.2f}"
            common = {
                "rebalance_date": str(date.date()),
                "recipe_id": f"synthetic_recipe_{i}",
                "candidate_id": candidate_id,
                "source": source,
                "method": method,
                "alpha": alpha,
                "params": json.dumps({"max_weight": 0.08 + 0.01 * i}, sort_keys=True),
                "selected_candidate": i == selected,
            }
            rows.append({
                **common,
                "stage": "validation",
                "validation_sharpe": float(validation[i]),
                "validation_score": float(validation[i]),
            })
            rows.append({
                **common,
                "stage": "test",
                "validation_sharpe": float(validation[i]),
                "test_sharpe": float(internal_test[i]),
                "test_score": float(internal_test[i]),
            })
    return pd.DataFrame(rows)


def _synthetic_selection_audit(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for date in dates:
        rows.append({
            "date": str(date.date()),
            "engine_used": True,
            "history_end": str((date - pd.Timedelta(days=1)).date()),
            "train_end": str((date - pd.Timedelta(days=120)).date()),
            "validation_end": str((date - pd.Timedelta(days=40)).date()),
            "internal_test_end": str((date - pd.Timedelta(days=1)).date()),
            "fallback_reason": "",
        })
    return pd.DataFrame(rows)


def main(output: str | Path = "reports/sample_dashboard") -> Path:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260721)
    dates = pd.date_range("2020-01-31", periods=72, freq="ME")
    tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "VNQ", "DBC", "XLK", "XLF"]

    factor = rng.normal(0.006, 0.035, len(dates))
    asset_returns = np.column_stack([
        0.3 * factor + rng.normal(0.003, 0.035, len(dates)) for _ in tickers
    ])
    asset_returns[:, 0] = factor
    strategy_returns = 0.55 * factor + rng.normal(0.0035, 0.018, len(dates))
    balance = 1000.0 * np.cumprod(1.0 + strategy_returns)
    pd.DataFrame({
        "Balance": balance,
        "PnL": strategy_returns,
        "Cost": rng.uniform(0.0001, 0.001, len(dates)),
        "Returns": strategy_returns,
        "Alpha": rng.choice([0.25, 0.5, 0.75, 1.0], len(dates)),
    }, index=dates).to_csv(out / "pnl.csv")
    pd.DataFrame(asset_returns, index=dates, columns=tickers).to_csv(out / "real.csv")

    matrices: list[dict[str, object]] = []
    weights: dict[str, pd.DataFrame] = {}
    previous = np.ones(len(tickers)) / len(tickers)
    for i, date in enumerate(dates):
        eligible = rng.random(len(tickers)) > 0.15
        eligible[0] = True
        raw = rng.random(len(tickers)) * eligible
        threshold = np.quantile(raw[raw > 0], 0.35) if (raw > 0).any() else 0.0
        raw[raw < threshold] = 0.0
        target = raw / raw.sum() if raw.sum() > 0 else np.ones(len(tickers)) / len(tickers)
        exposure = float(np.clip(0.75 + 0.25 * np.sin(i / 8), 0.55, 1.0))
        target *= exposure
        executed = previous + 0.75 * (target - previous)
        previous = executed
        weights[date.strftime("%Y-%m-%d")] = pd.DataFrame({
            "Key": tickers,
            "Value": eligible.astype(int),
            "weights": executed,
            "target_weights": target,
            "executed_weights": executed,
        })
        covariance = np.cov(asset_returns[max(0, i - 35):i + 1].T) if i > 2 else np.eye(len(tickers)) * 0.001
        sd = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        correlation = covariance / np.outer(sd, sd)
        np.fill_diagonal(correlation, 1.0)
        matrices.append({
            "date": date.strftime("%Y-%m-%d"),
            "tickers": tickers,
            "covariance_horizon": covariance.tolist(),
            "correlation": correlation.tolist(),
            "horizon": 21,
            "return_type": "returns",
        })

    with pd.ExcelWriter(out / "weights.xlsx") as writer:
        for name, frame in weights.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    (out / "risk_matrices.json").write_text(json.dumps(matrices), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({
        "return_type": "returns",
        "horizon": 21,
        "synthetic_smoke_test": True,
        "warning": "Artificial QA data - not an empirical strategy result.",
    }), encoding="utf-8")

    # Synthetic risk channels share a slow latent state but retain independent noise.
    latent = pd.Series(rng.normal(0.0, 0.35, len(dates))).ewm(alpha=0.18, adjust=False).mean().to_numpy()
    backward_timer = np.clip(0.86 - 0.18 * latent + rng.normal(0, 0.07, len(dates)), 0.3, 1.0)
    forward_timer = np.clip(0.88 - 0.15 * np.roll(latent, -1) + rng.normal(0, 0.075, len(dates)), 0.3, 1.0)
    conviction_timer = np.clip(0.90 + rng.normal(0, 0.06, len(dates)), 0.3, 1.0)
    overlay_fraction = np.clip((backward_timer * forward_timer * conviction_timer) ** (1 / 3), 0.3, 1.0)
    centered_cvar = np.abs(0.035 + 0.012 * latent + rng.normal(0, 0.004, len(dates)))
    scenario_mean = 0.006 + rng.normal(0, 0.004, len(dates))
    raw_cvar = centered_cvar - scenario_mean
    pd.DataFrame({
        "date": dates,
        "cvar_model": centered_cvar,
        "cvar_model_centered": centered_cvar,
        "cvar_model_raw": raw_cvar,
        "portfolio_scenario_mean": scenario_mean,
        "forecast_cvar_centered": True,
        "vol_model": np.abs(rng.normal(0.035, 0.008, len(dates))),
        "backward_vol_timer": backward_timer,
        "forward_cvar_timer": forward_timer,
        "conviction_timer": conviction_timer,
        "overlay_fraction": overlay_fraction,
    }).to_csv(out / "forecast_risk.csv", index=False)

    signal_tables = []
    for i, _ in enumerate(dates):
        records = []
        for ticker in tickers:
            records.append({
                "ticker": ticker,
                "efficiency_ratio": float(np.clip(rng.normal(0.45 + 0.08 * latent[i], 0.12), 0, 1)),
                "atr_vol_regime": float(np.abs(rng.normal(1.0 + 0.2 * latent[i], 0.15))),
                "liquidity_score": float(np.abs(rng.lognormal(0.0, 0.25))),
                "breakout_strength": float(np.clip(rng.normal(0.55 - 0.05 * latent[i], 0.18), 0, 1)),
                "normalized_spike": float(rng.normal(0.0, 1.0)),
                "pullback_ratio": float(np.clip(rng.beta(2.0, 5.0), 0, 1)),
                "net_move": float(rng.normal(0.01, 0.04)),
                "garch_vol_forecast": float(np.abs(rng.normal(0.012 + 0.003 * latent[i], 0.002))),
                "risk_reward_ratio": float(np.abs(rng.normal(1.4, 0.3))),
                "kelly_fraction": float(np.clip(rng.normal(0.10, 0.05), 0, 0.25)),
                "signal_quality": float(np.clip(rng.normal(0.58, 0.08), 0.05, 0.95)),
            })
        signal_tables.append(json.dumps(records))
    pd.DataFrame({
        "date": dates,
        "overlay_fraction": overlay_fraction,
        "backward_vol_timer": backward_timer,
        "forward_cvar_timer": forward_timer,
        "conviction_timer": conviction_timer,
        "portfolio_kelly": np.clip(rng.normal(0.45, 0.2, len(dates)), 0.0, 1.0),
        "portfolio_risk_reward": np.abs(rng.normal(1.4, 0.35, len(dates))),
        "portfolio_garch_vol": np.abs(rng.normal(0.035, 0.008, len(dates))),
        "portfolio_realized_vol": np.abs(rng.normal(0.038, 0.010, len(dates))),
        "forward_model_cvar": centered_cvar,
        "forward_model_cvar_raw": raw_cvar,
        "portfolio_scenario_mean": scenario_mean,
        "historical_cvar_target": np.abs(rng.normal(0.035, 0.004, len(dates))),
        "historical_cvar_target_raw": np.abs(rng.normal(0.031, 0.005, len(dates))),
        "signal_centering_applied": True,
        "composite_quality": np.clip(rng.normal(0.58, 0.08, len(dates)), 0.0, 1.0),
        "smart_signal_table": signal_tables,
    }).to_csv(out / "smart_rebalance_audit.csv", index=False)

    trial_dates = dates[-36:]
    _synthetic_trial_audit(trial_dates, rng).to_csv(out / "trial_audit.csv", index=False)
    _synthetic_selection_audit(trial_dates).to_csv(out / "selection_audit.csv", index=False)

    generate_research_validation_safely(out, config={"research_validation": {"enabled": True}})
    generate_metric_artifacts_safely(out)
    run_signal_research(out, output_dir=out / "signal_research", min_train=18, distribution_bootstrap_samples=19)
    return build_interactive_report(out)


if __name__ == "__main__":
    print(main())
