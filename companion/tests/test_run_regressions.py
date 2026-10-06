from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.models.bayessian import BayessianModel
from scripts.backtesting.walk_forward_engine import as_return_frame


def test_bayessian_risk_snapshot_uses_base_model_returns_attribute():
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    cols = ["A", "B"]
    rng = np.random.default_rng(7)
    hist = pd.DataFrame(rng.normal(0.0, 0.01, size=(80, 2)), columns=cols)
    hist.insert(0, "Date", dates)
    ewma = hist.copy()
    future = rng.normal(0.0, 0.01, size=(21, 2))
    model = BayessianModel(
        {
            "horizon": 21,
            "n_scenarios": 100,
            "scenario_return_type": "log-returns",
            "distribution": "normal",
            "posterior": "empirical_bayes",
            "cov_method": "ledoit_wolf",
        },
        cols,
        hist,
        ewma,
        future,
    )
    model.prediction()
    snap = model.get_risk_matrix_snapshot()
    assert snap["estimation_observations"] == 80
    assert snap["selected_dimension"] == 2


def test_market_cap_reindex_pattern_restores_all_nan_selected_ticker():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    mc_raw = pd.DataFrame({
        "Date": dates,
        "A": np.arange(10.0),
        "BOND": np.nan,
    })
    mc = as_return_frame(mc_raw)
    # as_return_frame intentionally drops columns that are all NaN in the window.
    assert "BOND" not in mc.columns
    aligned = mc.reindex(index=dates, columns=["A", "BOND"]).fillna(0.0)
    assert list(aligned.columns) == ["A", "BOND"]
    assert float(aligned["BOND"].sum()) == 0.0


def test_calendar_split_is_used_when_date_column_is_preserved():
    from scripts.backtesting.walk_forward_engine import split_train_validation_test_by_months

    dates = pd.bdate_range("2016-01-01", "2018-12-31")
    df = pd.DataFrame({"Date": dates, "A": 0.0, "B": 0.0})
    train, val, test, info = split_train_validation_test_by_months(
        df,
        validation_months=9,
        internal_test_months=3,
        min_history_months=12,
    )
    assert info["mode"] == "calendar_months"
    assert isinstance(train.index, pd.DatetimeIndex)
    assert isinstance(val.index, pd.DatetimeIndex)
    assert isinstance(test.index, pd.DatetimeIndex)
    assert len(info["validation_months"]) == 9
    assert len(info["test_months"]) == 3


def test_candidate_trial_records_strip_runtime_scenario_paths():
    import json
    from scripts.runs.common_postprocess import candidate_trial_records

    huge_runtime_tensor = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    table = pd.DataFrame([{
        "candidate_id": "optimizer:cdar:candidate=0:alpha=1.0000",
        "source": "optimizer",
        "method": "return_cdar_constraint",
        "params": {"type": "cdar", "max_weight": 0.12, "scenario_paths": huge_runtime_tensor},
        "alpha": 1.0,
        "validation_sharpe": 1.2,
    }])
    rows = candidate_trial_records("2025-01-02", table, pd.DataFrame())
    assert len(rows) == 1
    params = json.loads(rows[0]["params"])
    assert params == {"max_weight": 0.12, "type": "cdar"}
    assert "scenario_paths" not in rows[0]["params"]
