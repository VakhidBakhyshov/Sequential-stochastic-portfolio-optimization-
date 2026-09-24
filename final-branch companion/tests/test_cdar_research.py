import numpy as np

from scripts.calculations.drawdown import drawdown_curve_from_returns, empirical_cdar, portfolio_path_returns
from scripts.optimizers.cdar import BayessianCDaR
from scripts.theory.cdar_benchmark import path_order_counterexample


def test_cdar_limiting_and_homogeneity():
    r = np.array([[0.02,-0.03,0.01,-0.02],[0.01,0.01,-0.04,0.00]])
    dd = drawdown_curve_from_returns(r, axis=1)
    assert abs(empirical_cdar(r, 0.0) - dd.mean()) < 1e-12
    c = empirical_cdar(r, .95)
    assert abs(empirical_cdar(0.4*r, .95) - 0.4*c) < 1e-12


def test_path_dependence_with_equal_terminal_return():
    d = path_order_counterexample()
    assert abs(d['terminal_a'] - d['terminal_b']) < 1e-12
    assert d['cdar95_a'] > d['cdar95_b']


def test_cdar_lp_respects_equal_weight_budget():
    rng=np.random.default_rng(7); S,H,N=30,8,4
    paths=0.0002 + rng.normal(size=(S,H,N))*np.linspace(.006,.015,N)[None,None,:]
    terminal=paths.sum(axis=1); ew=np.ones(N)/N
    cfg={"task_type":"return_cdar_constraint","cdar_confidence_level":.90,"confidence_level":.90,"cdar_budget_mult":1.0,"turnover_penalty":0.0,"penalty_type":"L1","is_all_methods":False,"min_weight":0.0,"max_weight":.6,"constraint_max_weight":True,"scenario_paths":paths}
    opt=BayessianCDaR(cfg,np.ones(N),paths.reshape(-1,N),terminal,ew)
    w=opt.get_results()[1][0]
    assert np.isclose(w.sum(),1.0)
    assert empirical_cdar(portfolio_path_returns(paths,w),.90) <= empirical_cdar(portfolio_path_returns(paths,ew),.90) + 1e-8


def test_cdar_forward_timer_is_available_in_multi_signal_layer():
    import pandas as pd
    from scripts.risk_controls.smart_signals import portfolio_signal_diagnostics

    rng = np.random.default_rng(11)
    n, h, s = 4, 5, 40
    hist = pd.DataFrame(rng.normal(0.0002, 0.01, size=(140, n)), columns=list("ABCD"))
    paths = rng.normal(0.0002, 0.01, size=(s, h, n))
    pred = paths.sum(axis=1)
    weights = pd.Series(np.ones(n) / n, index=list("ABCD"))
    signal_table = pd.DataFrame({"signal_quality": np.full(n, 0.6)}, index=list("ABCD"))
    diag = portfolio_signal_diagnostics(
        asset_weights=weights,
        signal_table=signal_table,
        historical_returns=hist,
        pred_returns=pred,
        scenario_paths=paths,
        config={
            "horizon": h,
            "forward_risk_measure": "cdar",
            "cdar_alpha": 0.90,
            "cdar_target_window": 100,
            "min_overlay": 0.30,
        },
    )
    assert np.isfinite(diag.forward_model_cdar)
    assert np.isfinite(diag.forward_cdar_timer)
    assert np.isclose(diag.forward_risk_timer, diag.forward_cdar_timer)
    assert diag.forward_risk_measure == "cdar"


def test_nested_engine_injects_training_prefix_paths_into_cdar_optimizer():
    import pandas as pd
    from scripts.backtesting.walk_forward_engine import BacktestEngineConfig, PortfolioWalkForwardBacktestEngine

    class FakePathModel:
        def __init__(self, config, etfs_list, train_returns, ewma_returns, future_returns):
            self.n = len(etfs_list)
            self.h = 4
            self.s = 24
            rng = np.random.default_rng(123)
            self.paths = rng.normal(0.0002, 0.008, size=(self.s, self.h, self.n))
        def prediction(self):
            return self.paths.sum(axis=1)
        def get_scenario_paths(self):
            return self.paths

    cols = ["A", "B", "C"]
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    train = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, size=(100, 3)), index=idx, columns=cols)
    engine = PortfolioWalkForwardBacktestEngine(
        engine_config=BacktestEngineConfig(include_optimizers=True, include_strategies=False),
        optimizer_registry={"cdar": BayessianCDaR},
        model_class=FakePathModel,
        model_config={},
        optimizer_base_config={
            "type": "cdar", "task_type": "return_cdar_constraint",
            "cdar_confidence_level": 0.90, "cdar_budget_mult": 1.0,
            "turnover_penalty": 0.0, "penalty_type": "L1",
            "is_all_methods": False, "min_weight": 0.0, "max_weight": 0.6,
        },
        optimizer_param_grid={},
    )
    pred = engine._predict_returns(
        etfs_list=cols, train_returns=train, train_ewma_returns=train,
        eval_template_returns=train.tail(10),
    )
    assert engine._last_scenario_paths is not None
    candidates = engine._candidate_weights_from_optimizer(
        etfs_list=cols,
        train_returns=train,
        pred_returns=pred,
        market_cap_train=np.ones_like(train.values),
        previous_weights=pd.Series(np.ones(3) / 3, index=cols),
    )
    assert len(candidates) >= 1
    assert np.isclose(float(candidates[0][3].sum()), 1.0)
    # Ordered scenario paths are a runtime tensor, not a searched hyperparameter.
    # Persisting them in candidate params makes trial_audit.csv enormous and makes
    # recipe ids depend on random scenario draws.
    assert "scenario_paths" not in candidates[0][2]


def test_zero_drift_brownian_terminal_cdar_exact_formula_scales_linearly():
    from scripts.theory.cdar_benchmark import exact_terminal_drawdown_cdar_zero_drift_bm
    c1 = exact_terminal_drawdown_cdar_zero_drift_bm(0.20, 1.0, 0.95)
    c2 = exact_terminal_drawdown_cdar_zero_drift_bm(0.40, 1.0, 0.95)
    c_half_horizon = exact_terminal_drawdown_cdar_zero_drift_bm(0.20, 0.25, 0.95)
    assert np.isclose(c2, 2.0 * c1)
    assert np.isclose(c_half_horizon, 0.5 * c1)


def test_terminal_cvar_and_cdar_can_rank_paths_oppositely():
    from scripts.theory.cdar_benchmark import risk_geometry_ranking_reversal
    x = risk_geometry_ranking_reversal(0.95)
    assert x["terminal_prefers_a"] == 1.0
    assert x["cdar_prefers_b"] == 1.0


def test_two_signal_mode_excludes_conviction_from_overlay():
    import pandas as pd
    from scripts.risk_controls.smart_signals import portfolio_signal_diagnostics
    rng = np.random.default_rng(1234)
    n, h, s = 4, 5, 50
    hist = pd.DataFrame(rng.normal(0.0002, 0.01, size=(160, n)), columns=list("ABCD"))
    paths = rng.normal(0.0002, 0.01, size=(s, h, n))
    pred = paths.sum(axis=1)
    weights = pd.Series(np.ones(n) / n, index=list("ABCD"))
    signal_table = pd.DataFrame({"signal_quality": np.full(n, 0.1)}, index=list("ABCD"))
    diag = portfolio_signal_diagnostics(
        asset_weights=weights,
        signal_table=signal_table,
        historical_returns=hist,
        pred_returns=pred,
        scenario_paths=paths,
        config={
            "horizon": h, "forward_risk_measure": "cdar", "cdar_alpha": 0.90,
            "include_conviction_timer": False, "overlay_combination": "product",
            "min_overlay": 0.30,
        },
    )
    expected = np.clip(diag.backward_vol_timer * diag.forward_risk_timer, 0.30, 1.0)
    assert np.isclose(diag.overlay_fraction, expected)
    assert "conviction" not in diag.comments.lower()


def test_paired_cvar_cdar_comparison_uses_block_bootstrap():
    import pandas as pd
    from scripts.runs.compare_cvar_cdar import paired_bootstrap
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    a = pd.Series(np.linspace(-0.02, 0.03, 24), index=idx)
    b = a + 0.001
    out = paired_bootstrap(a, b, draws=50, seed=7, block_length=4)
    assert set(out["bootstrap_type"]) == {"paired_circular_block"}
    assert set(out["block_length"]) == {4}
    assert len(out) == 2
