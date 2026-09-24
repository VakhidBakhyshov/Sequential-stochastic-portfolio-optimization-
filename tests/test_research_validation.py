from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.validation.false_strategy import (
    adjusted_pvalues,
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    expected_max_sharpe_reference_curve,
    hac_mean_test,
    false_strategy_density_surface,
    probabilistic_sharpe_ratio,
)


def test_expected_max_sharpe_increases_with_trials() -> None:
    values = [expected_max_sharpe(k, sharpe_std=1.0) for k in (1, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] == 0.0


def test_psr_rewards_a_stronger_track_record() -> None:
    rng = np.random.default_rng(7)
    weak = rng.normal(0.001, 0.04, 120)
    strong = rng.normal(0.012, 0.04, 120)
    assert probabilistic_sharpe_ratio(strong, periods_per_year=12) > probabilistic_sharpe_ratio(weak, periods_per_year=12)


def test_dsr_is_no_larger_than_psr_under_multiple_testing() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(0.012, 0.035, 120)
    psr = probabilistic_sharpe_ratio(returns, periods_per_year=12)
    dsr = deflated_sharpe_ratio(
        returns,
        num_trials=100,
        trial_sharpe_mean=0.0,
        trial_sharpe_std=0.7,
        periods_per_year=12,
    )
    assert dsr <= psr


def test_bh_adjustment_is_valid_and_monotone_in_rank() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = adjusted_pvalues(p, "benjamini_hochberg")
    assert np.all((0.0 <= q) & (q <= 1.0))
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_effective_trial_count_respects_correlation() -> None:
    independent = np.eye(5)
    perfectly_correlated = np.ones((5, 5))
    assert effective_number_of_trials(independent) > effective_number_of_trials(perfectly_correlated)
    assert np.isclose(effective_number_of_trials(perfectly_correlated), 1.0)


def test_false_strategy_surface_has_expected_curve() -> None:
    surface = false_strategy_density_surface(trial_counts=[10, 100], max_sharpes=np.linspace(-1, 5, 25))
    assert set(["number_of_trials", "max_sharpe", "relative_density", "expected_max_sharpe"]).issubset(surface)
    expected = surface.groupby("number_of_trials")["expected_max_sharpe"].first()
    assert expected.loc[100] > expected.loc[10]


def test_causality_audit_treats_empty_csv_cell_as_no_fallback() -> None:
    from scripts.validation.false_strategy import causality_audit
    audit = pd.DataFrame([{
        "date": "2025-01-31",
        "history_end": "2025-01-30",
        "train_end": "2024-10-31",
        "validation_end": "2024-12-15",
        "internal_test_end": "2025-01-30",
        "engine_used": True,
        "fallback_reason": np.nan,
    }])
    result = causality_audit(audit)
    assert bool(result.loc[0, "no_fallback"])
    assert bool(result.loc[0, "full_process_pass"])


def test_causality_audit_parses_string_false_correctly() -> None:
    from scripts.validation.false_strategy import causality_audit
    audit = pd.DataFrame([{
        "date": "2025-01-31",
        "history_end": "2025-01-30",
        "train_end": "2024-10-31",
        "validation_end": "2024-12-15",
        "internal_test_end": "2025-01-30",
        "engine_used": "False",
        "fallback_reason": "legacy selector",
    }])
    result = causality_audit(audit)
    assert not bool(result.loc[0, "nested_engine_used"])
    assert not bool(result.loc[0, "full_process_pass"])

def test_false_strategy_expected_max_reference_is_close_to_exact_gaussian() -> None:
    ref = expected_max_sharpe_reference_curve([10, 40, 1000], sharpe_mean=0.0, sharpe_std=1.0)
    assert (ref["expected_max_sharpe_approx"] > 0).all()
    # The Bailey-Lopez de Prado extreme-value approximation is intentionally not
    # exact, but should remain close to the exact iid-Gaussian order statistic.
    assert (ref["relative_error"].abs() < 0.05).all()


def test_hac_mean_test_returns_valid_one_sided_probability() -> None:
    rng = np.random.default_rng(123)
    x = np.zeros(160)
    shocks = rng.normal(0.01, 0.03, len(x))
    for i in range(1, len(x)):
        x[i] = 0.45 * x[i - 1] + shocks[i]
    res = hac_mean_test(x, benchmark=0.0, max_lag=6)
    assert 0.0 <= res["pvalue_one_sided"] <= 1.0
    assert res["hac_standard_error"] > 0.0

