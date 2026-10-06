"""End-of-run publication diagnostics for false-strategy detection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.results.stats import metric_summary
from scripts.validation.false_strategy import (
    causality_audit,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    expected_max_sharpe_reference_curve,
    false_strategy_density_surface,
    minimum_track_record_length,
    nested_selection_overfitting_proxy,
    probabilistic_sharpe_ratio,
    recipe_level_multiple_testing,
    regularize_trial_correlation,
)


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _validation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict((config or {}).get("research_validation", {}) or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "periods_per_year": int(cfg.get("periods_per_year", 12)),
        "risk_free_rate": float(cfg.get("risk_free_rate", 0.02)),
        "fdr_alpha": float(cfg.get("fdr_alpha", 0.05)),
        "psr_threshold": float(cfg.get("psr_threshold", 0.95)),
        "dsr_threshold": float(cfg.get("dsr_threshold", 0.95)),
        "num_trials_override": cfg.get("num_trials_override"),
        "trial_sharpe_mean_override": cfg.get("trial_sharpe_mean_override"),
        "trial_sharpe_std_override": cfg.get("trial_sharpe_std_override"),
        "adjust_serial_correlation": bool(cfg.get("adjust_serial_correlation", True)),
        "effective_trial_shrinkage": float(cfg.get("effective_trial_shrinkage", 0.10)),
        "dsr_trial_count_mode": str(cfg.get("dsr_trial_count_mode", "effective")).lower(),
    }


def _trial_universe_statistics(
    trials: pd.DataFrame,
    *,
    correlation_shrinkage: float = 0.10,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Summarize the searched strategy family and its dependence.

    ``raw_attempted_trials`` counts every unique recipe present in the audit unless
    ``counts_as_trial`` explicitly marks it false. Sharpe dispersion and trial
    correlation are estimated only from validation-stage recipes with numerical
    validation Sharpe values. This makes the research denominator auditable while
    keeping the effective-trial estimator tied to actually scored alternatives.
    """
    empty_stats = {
        "raw_number_of_trials": 1.0,
        "scored_number_of_trials": 1.0,
        "effective_number_of_trials": 1.0,
        "trial_sharpe_mean": 0.0,
        "trial_sharpe_std": 0.0,
        "correlation_panel_coverage": 0.0,
    }
    if trials.empty:
        return empty_stats, pd.DataFrame(), pd.DataFrame()

    all_df = trials.copy()
    if "recipe_id" not in all_df.columns:
        all_df["recipe_id"] = all_df.get(
            "candidate_id", pd.Series(np.arange(len(all_df)), index=all_df.index)
        ).astype(str)
    if "counts_as_trial" in all_df.columns:
        explicit = all_df["counts_as_trial"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        stage = all_df.get("stage", pd.Series("validation", index=all_df.index)).astype(str).str.lower()
        # Ordinary validation/test rows predate the attempt flag and count by
        # default; attempt rows count only when explicitly marked true.
        keep = (stage != "attempt") | explicit
        all_counted = all_df[keep].copy()
    else:
        all_counted = all_df
    raw_k = max(int(all_counted["recipe_id"].astype(str).nunique()), 1)

    df = all_df.copy()
    if "stage" in df.columns:
        df = df[df["stage"].astype(str).str.lower() == "validation"]
    if df.empty or "validation_sharpe" not in df.columns:
        stats = dict(empty_stats)
        stats["raw_number_of_trials"] = float(raw_k)
        return stats, pd.DataFrame(), pd.DataFrame()

    df["validation_sharpe"] = pd.to_numeric(df["validation_sharpe"], errors="coerce")
    recipe_sr = df.groupby("recipe_id")["validation_sharpe"].mean().dropna()
    scored_k = max(int(recipe_sr.size), 1)
    sr_mean = float(recipe_sr.mean()) if not recipe_sr.empty else 0.0
    sr_std = float(recipe_sr.std(ddof=1)) if len(recipe_sr) > 1 else 0.0

    corr_out = pd.DataFrame()
    coverage_out = pd.DataFrame()
    effective_k = float(scored_k)
    panel_coverage = 0.0
    if "rebalance_date" in df.columns and scored_k > 1:
        panel = df.pivot_table(
            index="rebalance_date", columns="recipe_id", values="validation_sharpe", aggfunc="mean"
        )
        corr_raw = panel.corr(min_periods=3)
        counts = panel.notna().astype(int).T @ panel.notna().astype(int)
        coverage_out = counts
        if corr_raw.shape[0] > 1:
            n = corr_raw.shape[0]
            off = ~np.eye(n, dtype=bool)
            finite = np.isfinite(corr_raw.to_numpy(dtype=float))[off]
            panel_coverage = float(finite.mean()) if finite.size else 0.0
            corr_reg = regularize_trial_correlation(
                corr_raw.to_numpy(dtype=float), shrinkage=correlation_shrinkage
            )
            corr_out = pd.DataFrame(corr_reg, index=corr_raw.index, columns=corr_raw.columns)
            effective_k = effective_number_of_trials(corr_reg)

    return {
        "raw_number_of_trials": float(max(raw_k, scored_k)),
        "scored_number_of_trials": float(scored_k),
        "effective_number_of_trials": float(np.clip(effective_k, 1.0, scored_k)),
        "trial_sharpe_mean": sr_mean,
        "trial_sharpe_std": max(sr_std, 0.0),
        "correlation_panel_coverage": panel_coverage,
    }, corr_out, coverage_out


def run_research_validation(
    result_folder: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate CSV/JSON artifacts used by the interactive report and paper."""
    folder = Path(result_folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = _validation_config(config)
    if not cfg["enabled"]:
        return {"enabled": False}

    pnl_path = folder / "pnl.csv"
    if not pnl_path.exists():
        raise FileNotFoundError(pnl_path)
    pnl = pd.read_csv(pnl_path, index_col=0, parse_dates=True)
    returns = pd.to_numeric(pnl.get("Returns"), errors="coerce").dropna()
    trials = _read_csv(folder / "trial_audit.csv")
    selection = _read_csv(folder / "selection_audit.csv")

    universe, trial_corr, trial_coverage = _trial_universe_statistics(
        trials, correlation_shrinkage=cfg["effective_trial_shrinkage"]
    )
    if not trial_corr.empty:
        trial_corr.to_csv(folder / "trial_dependency_correlation.csv")
    if not trial_coverage.empty:
        trial_coverage.to_csv(folder / "trial_dependency_pair_counts.csv")
    if cfg["num_trials_override"] is not None:
        override = max(float(cfg["num_trials_override"]), 1.0)
        universe["effective_number_of_trials"] = override
        universe["raw_number_of_trials"] = max(universe["raw_number_of_trials"], override)
    if cfg["trial_sharpe_mean_override"] is not None:
        universe["trial_sharpe_mean"] = float(cfg["trial_sharpe_mean_override"])
    if cfg["trial_sharpe_std_override"] is not None:
        universe["trial_sharpe_std"] = max(float(cfg["trial_sharpe_std_override"]), 0.0)

    if cfg["dsr_trial_count_mode"] == "raw":
        primary_trial_count = float(universe["raw_number_of_trials"])
    else:
        primary_trial_count = float(universe["effective_number_of_trials"])

    psr_details = probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=0.0,
        risk_free_rate=cfg["risk_free_rate"],
        periods_per_year=cfg["periods_per_year"],
        adjust_serial_correlation=cfg["adjust_serial_correlation"],
        return_details=True,
    )
    dsr_details = deflated_sharpe_ratio(
        returns,
        num_trials=primary_trial_count,
        trial_sharpe_mean=universe["trial_sharpe_mean"],
        trial_sharpe_std=universe["trial_sharpe_std"],
        risk_free_rate=cfg["risk_free_rate"],
        periods_per_year=cfg["periods_per_year"],
        adjust_serial_correlation=cfg["adjust_serial_correlation"],
        return_details=True,
    )
    expected_max = expected_max_sharpe(
        primary_trial_count,
        sharpe_mean=universe["trial_sharpe_mean"],
        sharpe_std=universe["trial_sharpe_std"],
    )
    min_trl = minimum_track_record_length(
        returns,
        benchmark_sharpe=expected_max,
        confidence=cfg["dsr_threshold"],
        risk_free_rate=cfg["risk_free_rate"],
        periods_per_year=cfg["periods_per_year"],
    )

    dsr_effective = deflated_sharpe_ratio(
        returns, num_trials=universe["effective_number_of_trials"],
        trial_sharpe_mean=universe["trial_sharpe_mean"], trial_sharpe_std=universe["trial_sharpe_std"],
        risk_free_rate=cfg["risk_free_rate"], periods_per_year=cfg["periods_per_year"],
        adjust_serial_correlation=cfg["adjust_serial_correlation"],
    )
    dsr_raw = deflated_sharpe_ratio(
        returns, num_trials=universe["raw_number_of_trials"],
        trial_sharpe_mean=universe["trial_sharpe_mean"], trial_sharpe_std=universe["trial_sharpe_std"],
        risk_free_rate=cfg["risk_free_rate"], periods_per_year=cfg["periods_per_year"],
        adjust_serial_correlation=cfg["adjust_serial_correlation"],
    )

    multiple = recipe_level_multiple_testing(trials, alpha=cfg["fdr_alpha"])
    if not multiple.empty:
        multiple.to_csv(folder / "multiple_testing_results.csv", index=False)
    pbo_detail, pbo_summary = nested_selection_overfitting_proxy(trials)
    if not pbo_detail.empty:
        pbo_detail.to_csv(folder / "nested_pbo_proxy.csv", index=False)
    causal = causality_audit(selection)
    if not causal.empty:
        causal.to_csv(folder / "causality_audit.csv", index=False)

    perf = metric_summary(
        pnl,
        risk_free_rate=cfg["risk_free_rate"],
        periods_per_year=cfg["periods_per_year"],
        num_trials=max(int(round(primary_trial_count)), 1),
        trial_sharpe_mean=universe["trial_sharpe_mean"],
        trial_sharpe_std=universe["trial_sharpe_std"],
    )
    fdr_discoveries = int(multiple.get("reject_fdr_by", pd.Series(dtype=bool)).sum()) if not multiple.empty else 0
    fdr_candidates = int(len(multiple))
    causality_pass_rate = float(causal["causality_pass"].mean()) if not causal.empty else np.nan
    full_process_pass_rate = float(causal["full_process_pass"].mean()) if not causal.empty else np.nan

    summary: dict[str, Any] = {
        "enabled": True,
        "observed_annualized_sharpe": float(psr_details.annualized_sharpe),
        "psr": float(psr_details.probability),
        "psr_benchmark_sharpe": 0.0,
        "dsr": float(dsr_details.probability),
        "dsr_benchmark_expected_max_sharpe": float(expected_max),
        "raw_number_of_trials": int(round(universe["raw_number_of_trials"])),
        "scored_number_of_trials": int(round(universe["scored_number_of_trials"])),
        "effective_number_of_trials": float(universe["effective_number_of_trials"]),
        "primary_dsr_trial_count": float(primary_trial_count),
        "dsr_trial_count_mode": cfg["dsr_trial_count_mode"],
        "dsr_effective_trial_count": float(dsr_effective),
        "dsr_raw_trial_count": float(dsr_raw),
        "correlation_panel_coverage": float(universe["correlation_panel_coverage"]),
        "trial_sharpe_mean": float(universe["trial_sharpe_mean"]),
        "trial_sharpe_std": float(universe["trial_sharpe_std"]),
        "observations": int(psr_details.observations),
        "effective_observations": float(psr_details.effective_observations),
        "return_skewness": float(psr_details.skewness),
        "return_kurtosis": float(psr_details.kurtosis),
        "minimum_track_record_length_for_dsr": float(min_trl),
        "track_record_length_sufficient": bool(np.isfinite(min_trl) and len(returns) >= min_trl),
        "psr_pass": bool(np.isfinite(psr_details.probability) and psr_details.probability >= cfg["psr_threshold"]),
        "dsr_pass": bool(np.isfinite(dsr_details.probability) and dsr_details.probability >= cfg["dsr_threshold"]),
        "fdr_alpha": float(cfg["fdr_alpha"]),
        "fdr_by_discoveries": fdr_discoveries,
        "fdr_tested_recipes": fdr_candidates,
        "nested_pbo_proxy": float(pbo_summary.get("nested_pbo_proxy", np.nan)),
        "mean_internal_test_rank_percentile": float(pbo_summary.get("mean_test_rank_percentile", np.nan)),
        "causality_pass_rate": causality_pass_rate,
        "full_process_pass_rate": full_process_pass_rate,
        "annualized_return": float(perf.get("Annualized Return", np.nan)),
        "annualized_volatility": float(perf.get("Annualized Volatility", np.nan)),
        "max_drawdown": float(perf.get("Max Drawdown", np.nan)),
        "cvar_95": float(perf.get("CVaR (95%)", np.nan)),
        "methodology_note": (
            "PSR is computed at the sampling frequency with Pearson kurtosis and an optional Bartlett effective sample size. "
            "DSR is reported under both dependence-adjusted and raw attempted trial counts; the primary mode is recorded explicitly. "
            "Sparse trial correlations are regularized rather than silently filled with zero. "
            "Recipe-level multiple testing uses a Bartlett/Newey-West long-run-variance mean test before FWER/FDR adjustment. "
            "The nested PBO value is a walk-forward rank diagnostic, not the exact CSCV/PBO estimator."
        ),
    }

    pd.DataFrame([summary]).to_csv(folder / "research_validation_summary.csv", index=False)
    json_summary = {
        key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
        for key, value in summary.items()
    }
    (folder / "research_validation_summary.json").write_text(
        json.dumps(json_summary, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )

    # Standardized pedagogical null: iid Gaussian no-skill Sharpe estimates with std(SR)=1.
    surface = false_strategy_density_surface(sharpe_mean=0.0, sharpe_std=1.0)
    surface.to_csv(folder / "false_strategy_surface.csv", index=False)
    # Direct numerical check of the expected-maximum ridge used in the heatmap.
    # This distinguishes the False Strategy approximation from the exact Gaussian
    # order-statistic expectation without changing the DSR definition.
    ref_counts = [1, 2, 5, 10, 20, 40, 100, 250, 500, 1_000, 10_000, 1_000_000]
    expected_max_sharpe_reference_curve(ref_counts, sharpe_mean=0.0, sharpe_std=1.0).to_csv(
        folder / "false_strategy_expected_max_reference.csv", index=False
    )

    # Research-calibrated null uses the empirical cross-recipe mean/dispersion.
    calibrated = false_strategy_density_surface(
        sharpe_mean=universe["trial_sharpe_mean"],
        sharpe_std=max(universe["trial_sharpe_std"], 1e-9),
    )
    calibrated.to_csv(folder / "false_strategy_surface_calibrated.csv", index=False)

    # DSR sensitivity across trial breadth and Sharpe dispersion.
    k_grid = np.unique(np.clip(np.geomspace(1, max(universe["raw_number_of_trials"], 2.0), 40), 1, None).round(3))
    base_std = max(float(universe["trial_sharpe_std"]), 1e-6)
    sigma_grid = np.linspace(0.25 * base_std, 2.0 * base_std, 35)
    sens_rows = []
    for k in k_grid:
        for sig in sigma_grid:
            bench = expected_max_sharpe(k, sharpe_mean=universe["trial_sharpe_mean"], sharpe_std=sig)
            prob = probabilistic_sharpe_ratio(
                returns, benchmark_sharpe=bench, risk_free_rate=cfg["risk_free_rate"],
                periods_per_year=cfg["periods_per_year"],
                adjust_serial_correlation=cfg["adjust_serial_correlation"],
            )
            sens_rows.append({
                "number_of_trials": float(k), "trial_sharpe_std": float(sig),
                "expected_max_sharpe": float(bench), "dsr": float(prob),
            })
    pd.DataFrame(sens_rows).to_csv(folder / "dsr_sensitivity_surface.csv", index=False)
    return summary
