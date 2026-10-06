from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.dataloader.monthly_etf_filtration_2 import FilterConfig, calculate_month_metrics, select_top_etfs
from scripts.models.bayessian import BayessianModel
from scripts.optimizers.bayessian import BayessianCVaR
from scripts.risk_controls.smart_signals import smart_rebalance_diagnostics
from scripts.backtesting.walk_forward_engine import build_full_weight_series


def synthetic_returns(n=800, p=8, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    factor = rng.normal(0.00015, 0.007, (n, 1))
    r = factor + rng.normal(0.0, 0.006, (n, p))
    cols = [f"ETF{i}" for i in range(p)]
    return pd.DataFrame(r, index=idx, columns=cols)


def test_horizon_scenario_shape_and_repeatability():
    r = synthetic_returns()
    cfg = {"n_scenarios": 250, "horizon": 21, "scenario_return_type": "log-returns", "posterior": "empirical_bayes", "distribution": "normal", "random_state": 11}
    m1 = BayessianModel(cfg, list(r.columns), r, r.ewm(span=63).mean(), r.tail(21).values)
    m2 = BayessianModel(cfg, list(r.columns), r, r.ewm(span=63).mean(), r.tail(21).values)
    s1, s2 = m1.prediction(), m2.prediction()
    assert s1.shape == (250, r.shape[1])
    assert np.allclose(s1, s2)
    assert len(m1.evaluate_metrics(s1)) == 12


def test_cvar_lp_is_long_only_and_fully_invested():
    r = synthetic_returns(p=6)
    model = BayessianModel({"n_scenarios": 300, "horizon": 21, "scenario_return_type": "log-returns", "posterior": "empirical_bayes", "distribution": "normal"}, list(r.columns), r, r, r.tail(21).values)
    scenarios = model.prediction()
    opt = BayessianCVaR({"task_type": "return_cvar_constraint", "confidence_level": 0.95, "cvar_budget_mult": 1.0, "turnover_penalty": 0.001, "max_weight": 0.25, "min_weight": 0.0, "is_all_methods": False}, np.ones_like(r.values), r.values, scenarios, np.ones(6) / 6)
    w = np.asarray(opt.get_results()[1][0])
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-10)
    assert np.all(w <= 0.25 + 1e-8)


def test_cash_overlay_is_not_renormalized_away():
    r = synthetic_returns(p=5)
    selected = pd.Series(np.ones(5) / 5, index=r.columns)
    scenarios = np.tile(np.array([-0.05, -0.04, -0.03, -0.02, -0.01]), (400, 1))
    _, diag, adjusted = smart_rebalance_diagnostics(etfs_list=list(r.columns), historical_returns=r, pred_returns=scenarios, market_cap_history=np.ones_like(r.values), selected_asset_weights=selected, return_type="log-returns", config={"allow_cash_overlay": True, "min_overlay": 0.25, "target_monthly_vol": 0.01, "target_cvar": 0.005, "overlay_combination": "product"})
    assert adjusted.sum() <= 1.0 + 1e-9
    frame = pd.DataFrame({"Key": r.columns, "Value": 1})
    full = build_full_weight_series(portfolios_date=frame, mask=frame["Value"].eq(1), etfs_list=list(r.columns), selected_asset_weights=adjusted, preserve_total_exposure=True)
    assert np.isclose(full.sum(), adjusted.sum())
    assert diag.forward_cvar_timer <= 1.0


def test_filter_does_not_invent_pre_inception_observations():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    prices = pd.DataFrame({"SPY": np.linspace(100, 105, 30), "NEW": [np.nan] * 20 + list(np.linspace(10, 11, 10))}, index=idx)
    volumes = pd.DataFrame({"SPY": 1e6, "NEW": 2e5}, index=idx)
    metrics = calculate_month_metrics(prices, volumes, None, idx[-1] + pd.Timedelta(days=1), idx[0], idx[-1], "SPY")
    obs = metrics.set_index("etf").loc["NEW", "observations"]
    assert obs == 10
    scored, _ = select_top_etfs(metrics, FilterConfig(top_n=10, min_observations=15, spy_filter="none", exclude_benchmark=False))
    assert not bool(scored.set_index("etf").loc["NEW", "passed_base_filter"])
