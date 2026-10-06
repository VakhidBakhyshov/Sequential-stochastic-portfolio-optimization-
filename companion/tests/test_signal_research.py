from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.validation.signal_research import (
    distribution_fits,
    incremental_signal_research,
    stationarity_tests,
    group_cointegration_relationships,
    timer_subset_research,
    timer_incremental_comparisons,
)


def test_incremental_search_uses_causal_baseline_and_stops_without_improvement() -> None:
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.15, size=n)
    panel = pd.DataFrame({
        "strong": x,
        "noise": rng.normal(size=n),
        "next_abs_return": y,
    })
    out = incremental_signal_research(panel, ["strong", "noise"], target="next_abs_return", min_train=30)
    assert out.iloc[0]["added_signal"] == "__causal_mean_baseline__"
    assert bool(out.iloc[0]["accepted"])
    assert "strong" in out.loc[out["accepted"], "added_signal"].tolist()
    # The accepted MSE path must be strictly improving; the final rejected row, if
    # present, is the explicit stop condition rather than a mechanically added signal.
    accepted = out[out["accepted"]]
    assert np.all(np.diff(accepted["oos_mse"].to_numpy()) < 0)


def test_distribution_fit_reports_calibrated_best_family_pvalue() -> None:
    rng = np.random.default_rng(9)
    panel = pd.DataFrame({"signal": rng.normal(size=70)})
    fits = distribution_fits(panel, ["signal"], bootstrap_samples=19, random_state=7)
    assert not fits.empty
    best = fits[fits["best_by_aic"]]
    assert len(best) == 1
    p = float(best.iloc[0]["ks_pvalue_parametric_bootstrap"])
    assert 0.0 <= p <= 1.0
    assert best.iloc[0]["ks_pvalue_naive_fitted"] != best.iloc[0]["ks_pvalue_naive_fitted"]  # NaN by design


def test_bounded_timers_are_not_forced_into_group_cointegration() -> None:
    rng = np.random.default_rng(3)
    n = 90
    panel = pd.DataFrame({
        "backward_vol_timer": np.clip(0.7 + 0.08 * rng.normal(size=n), 0.3, 1.0),
        "forward_cvar_timer": np.clip(0.75 + 0.07 * rng.normal(size=n), 0.3, 1.0),
        "cvar_model": np.cumsum(rng.normal(size=n)),
        "asset_mean_trend": np.cumsum(rng.normal(size=n)),
    })
    signals = list(panel.columns)
    station = stationarity_tests(panel, signals)
    rel = group_cointegration_relationships(panel, station, signals)
    if not rel.empty:
        assert not rel["signals"].str.contains("backward_vol_timer|forward_cvar_timer", regex=True).any()


def test_timer_subset_research_carries_selection_aware_inference() -> None:
    rng = np.random.default_rng(5)
    n = 84
    base = rng.normal(0.008, 0.04, n)
    panel = pd.DataFrame({
        "reconstructed_risky_return": base,
        "overlay_fraction": np.clip(0.75 + 0.15 * rng.normal(size=n), 0.3, 1.0),
        "backward_vol_timer": np.clip(0.8 - 1.0 * np.maximum(-base, 0) + 0.05 * rng.normal(size=n), 0.3, 1.0),
        "forward_cvar_timer": np.clip(0.82 - 0.7 * np.maximum(-base, 0) + 0.05 * rng.normal(size=n), 0.3, 1.0),
        "conviction_timer": np.clip(0.85 + 0.08 * rng.normal(size=n), 0.3, 1.0),
    })
    out = timer_subset_research(panel)
    required = {"psr", "dsr_effective_trials", "dsr_raw_trials", "psr_p_fdr_bh", "psr_p_fdr_by"}
    assert required.issubset(out.columns)
    assert out["family_effective_trials"].min() >= 1.0
    increments = timer_incremental_comparisons(out)
    assert set(increments["relation"].unique()).issubset({"one_at_a_time_addition", "leave_one_out"})
