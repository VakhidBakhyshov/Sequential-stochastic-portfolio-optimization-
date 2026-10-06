import json

import numpy as np
import pandas as pd

from scripts.validation.theorem_empirical_bridge import (
    run_historical_theorem_audit,
    run_synthetic_theorem_audit,
)


def test_theorem_bridge_full_structural_audit(tmp_path):
    run_synthetic_theorem_audit(tmp_path, seed=12345)
    s = pd.read_csv(tmp_path / "theorem_to_synthetic_summary.csv").set_index("result")
    assert s.loc["Prop1/Cor1", "value"] > 0.995
    assert s.loc["Prop2", "value"] < 1e-10
    assert s.loc["Prop5", "value"] < 1e-10
    assert s.loc["Lemma3", "value"] == 0.0
    assert s.loc["Lemma4", "value"] < 1e-12
    assert s.loc["Lemma6", "value"] < 1e-12
    assert s.loc["Lemma7", "value"] < 1e-12
    assert s.loc["Prop9", "value"] < 1e-8
    assert s.loc["Property1", "value"] < 1e-12
    assert s.loc["Prop11/A2", "value"] <= 1 / 0.30 + 1e-8
    assert s.loc["Lemma8", "value"] < 1e-10
    assert s.loc["PropA1", "value"] < 1e-12
    assert s.loc["PropA3", "value"] < 1e-10
    assert s.loc["Counterexample3", "value"] > 0.9
    assert s.loc["Prop12/13", "value"] > 0

    h = pd.read_csv(tmp_path / "prop3_horizon_mean_domination.csv")
    ratio = float(h.iloc[-1].mean_to_vol_ratio / h.iloc[0].mean_to_vol_ratio)
    assert abs(ratio - np.sqrt(252)) < 1e-8

    d = pd.read_csv(tmp_path / "prop6_dimension_covariance_error.csv")
    assert d[["N_over_T", "mean_relative_cov_op_error"]].corr().iloc[0, 1] > 0.75

    coverage = pd.read_csv(tmp_path / "theorem_coverage_register.csv")
    labels = set(coverage["result"].astype(str))
    for expected in ["Prop1", "Cor1", "Prop2", "Prop9", "Property1", "PropA1", "PropA3"]:
        assert expected in labels


def test_historical_audit_is_fail_soft_and_measures_available_artifacts(tmp_path):
    pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "cvar_model_centered": [0.02, 0.03, 0.025],
        "cvar_model_raw": [0.018, 0.027, 0.024],
        "portfolio_scenario_mean": [0.002, 0.003, 0.001],
        "vol_model": [0.010, 0.015, 0.012],
        "forecast_cvar_level": [0.95, 0.95, 0.95],
        "optimizer_cvar_level": [0.94, 0.95, 0.96],
        "scenario_pit_realized_target": [0.10, 0.50, 0.90],
        "scenario_var_violation": [False, True, False],
        "optimizer_cvar_constraint_binding": [True, True, False],
        "optimizer_cvar_constraint_slack": [0.0, 0.0, 0.01],
        "optimizer_cvar_budget_shadow_price_max": [0.8, 1.2, 0.0],
    }).to_csv(tmp_path / "forecast_risk.csv", index=False)
    (tmp_path / "risk_matrices.json").write_text(json.dumps([
        {"date": "2020-01-01", "selected_dimension": 80, "estimation_observations": 252,
         "covariance_daily": [[0.01, 0.002], [0.002, 0.02]]},
        {"date": "2020-02-01", "selected_dimension": 100, "estimation_observations": 252,
         "covariance_daily": [[0.02, 0.005], [0.005, 0.03]]},
    ]))
    payload = run_historical_theorem_audit(tmp_path)
    assert payload["mode"] == "historical_real_data"
    path = tmp_path / "theorem_empirical_audit" / "historical_theorem_audit.csv"
    assert path.exists()
    assert len(pd.read_csv(path)) > 0
