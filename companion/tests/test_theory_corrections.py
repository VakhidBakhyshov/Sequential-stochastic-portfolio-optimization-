from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from scripts.dataloader.monthly_etf_filtration_2 import percentile_score
from scripts.risk_controls.smart_signals import portfolio_signal_diagnostics
from scripts.validation.false_strategy import (
    adjusted_pvalues,
    exact_expected_max_sharpe_gaussian,
    expected_max_sharpe,
    false_strategy_density_surface,
    regularize_trial_correlation,
)
from scripts.validation.signal_research import (
    cointegration_tests,
    stationarity_tests,
    timer_subset_research,
)


def test_percentile_direction_matches_descending_selection() -> None:
    x = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    high = percentile_score(x, higher_is_better=True)
    low = percentile_score(x, higher_is_better=False)
    assert high.idxmax() == "c"
    assert low.idxmax() == "a"
    assert np.isclose(high.loc["c"], 1.0)
    assert np.isclose(low.loc["a"], 1.0)


def test_centered_forward_cvar_is_used_and_raw_value_is_audited() -> None:
    rng = np.random.default_rng(42)
    hist = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 2)), columns=["A", "B"])
    # Deliberately large positive scenario mean: the uncentered loss-CVaR is
    # materially contaminated by expected return at the horizon.
    scen = pd.DataFrame(rng.normal(0.03, 0.015, (3000, 2)), columns=["A", "B"])
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    signal_table = pd.DataFrame({"signal_quality": [0.6, 0.6]}, index=["A", "B"])
    diag = portfolio_signal_diagnostics(
        asset_weights=w,
        signal_table=signal_table,
        historical_returns=hist,
        pred_returns=scen,
        config={"center_forward_cvar": True, "horizon": 21, "cvar_alpha": 0.95},
    )
    assert diag.signal_centering_applied
    assert diag.forward_model_cvar > diag.forward_model_cvar_raw
    assert diag.portfolio_scenario_mean > 0.02


def test_false_strategy_approximation_matches_exact_gaussian_reference() -> None:
    for k in (10, 100, 1000):
        approx = expected_max_sharpe(k, sharpe_mean=0.2, sharpe_std=0.8)
        exact = exact_expected_max_sharpe_gaussian(k, sharpe_mean=0.2, sharpe_std=0.8)
        assert abs(approx - exact) < 0.08


def test_multiple_testing_adjustments_match_statsmodels() -> None:
    p = np.array([0.001, 0.01, 0.03, 0.07, 0.2, 0.8])
    methods = {
        "bonferroni": "bonferroni",
        "sidak": "sidak",
        "holm": "holm",
        "hochberg": "simes-hochberg",
        "benjamini_hochberg": "fdr_bh",
        "benjamini_yekutieli": "fdr_by",
    }
    for ours, sm in methods.items():
        expected = multipletests(p, method=sm)[1]
        actual = adjusted_pvalues(p, ours)
        assert np.allclose(actual, expected, atol=1e-12)


def test_regularized_trial_correlation_is_finite_psd() -> None:
    raw = np.array([
        [1.0, 0.8, np.nan, 0.4],
        [0.8, 1.0, 0.5, np.nan],
        [np.nan, 0.5, 1.0, -0.2],
        [0.4, np.nan, -0.2, 1.0],
    ])
    corr = regularize_trial_correlation(raw, shrinkage=0.15)
    assert np.isfinite(corr).all()
    assert np.allclose(np.diag(corr), 1.0)
    assert np.linalg.eigvalsh(corr).min() >= -1e-10


def test_false_strategy_surface_is_gaussian_null_not_uniform_and_has_cdf() -> None:
    surf = false_strategy_density_surface(trial_counts=[10, 100], max_sharpes=np.linspace(-1, 5, 40))
    assert {"cdf_max_sharpe", "exceedance_probability", "null_sharpe_mean", "null_sharpe_std"}.issubset(surf.columns)
    assert ((surf["cdf_max_sharpe"] >= 0) & (surf["cdf_max_sharpe"] <= 1)).all()
    e = surf.groupby("number_of_trials")["expected_max_sharpe"].first()
    assert e.loc[100] > e.loc[10]


def test_cointegration_only_runs_for_i1_unbounded_precursors() -> None:
    rng = np.random.default_rng(3)
    n = 320
    x = np.cumsum(rng.normal(0, 1, n))
    y = 1.8 * x + rng.normal(0, 0.6, n)
    timer = np.clip(0.7 + 0.08 * rng.normal(size=n), 0.3, 1.0)
    panel = pd.DataFrame({"x": x, "y": y, "bounded_timer": timer})
    st = stationarity_tests(panel, ["x", "y", "bounded_timer"])
    pairs, obj = cointegration_tests(panel, st, ["x", "y", "bounded_timer"])
    assert "bounded_timer" not in obj["summary"]["eligible_i1_signals"]
    assert not pairs.empty
    row = pairs[((pairs.signal_a == "x") & (pairs.signal_b == "y")) | ((pairs.signal_a == "y") & (pairs.signal_b == "x"))]
    assert not row.empty
    assert float(row.iloc[0].pvalue) < 0.05


def test_timer_subset_matched_exposure_control() -> None:
    rng = np.random.default_rng(9)
    n = 80
    panel = pd.DataFrame({
        "backward_vol_timer": np.clip(rng.normal(0.75, 0.12, n), 0.3, 1),
        "forward_cvar_timer": np.clip(rng.normal(0.82, 0.10, n), 0.3, 1),
        "conviction_timer": np.clip(rng.normal(0.90, 0.06, n), 0.3, 1),
        "overlay_fraction": np.clip(rng.normal(0.72, 0.08, n), 0.3, 1),
        "reconstructed_risky_return": rng.normal(0.009, 0.045, n),
    })
    result = timer_subset_research(panel)
    matched = result[result["matched_average_exposure"] == True]  # noqa: E712
    assert not matched.empty
    target = float(panel["overlay_fraction"].mean())
    assert np.allclose(matched["average_exposure"], target, atol=1e-5)


def test_failed_attempts_enter_research_trial_denominator() -> None:
    from scripts.runs.common_postprocess import candidate_trial_records
    from scripts.validation.research_validation import _trial_universe_statistics

    val = pd.DataFrame([{
        "candidate_id": "ok", "source": "optimizer", "method": "cvar", "params": {"x": 1},
        "alpha": 0.5, "validation_sharpe": 0.4,
    }])
    tst = pd.DataFrame()
    attempts = [{
        "source": "optimizer", "method": "cvar", "params": {"x": 2}, "alpha": 0.5,
        "status": "failed", "error": "solver failure", "counts_as_trial": True,
    }]
    rows = candidate_trial_records("2025-01-31", val, tst, attempt_records=attempts)
    audit = pd.DataFrame(rows)
    stats_, _, _ = _trial_universe_statistics(audit)
    assert stats_["scored_number_of_trials"] == 1
    assert stats_["raw_number_of_trials"] == 2
