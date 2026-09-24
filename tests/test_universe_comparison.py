import numpy as np
import pandas as pd

from scripts.runs.compare_etf_universes import equal_weight_backtest, membership_matrix


def test_equal_weight_holding_period_return_is_asset_average_without_costs():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01", "2024-03-01"])
    close = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0],
            "B": [100.0, 90.0, 99.0],
        },
        index=dates,
    )
    universe = {d: {"A", "B"} for d in dates}
    out = equal_weight_backtest(
        close,
        dates,
        universe,
        cost_bps=0.0,
        charge_initial_trade=False,
    )
    # Month 1: +10% and -10% -> equal-weight portfolio return exactly 0%.
    assert np.isclose(out.returns.iloc[0], 0.0)
    # Month 2: both assets +10% -> portfolio return +10%.
    assert np.isclose(out.returns.iloc[1], 0.10)


def test_membership_matrix_has_run_py_compatible_binary_flags():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01"])
    universe = {
        dates[0]: {"A"},
        dates[1]: {"B"},
    }
    m = membership_matrix(universe, ["A", "B"], dates)
    assert list(m.columns) == ["Date", "A", "B"]
    assert m[["A", "B"]].to_numpy().tolist() == [[1, 0], [0, 1]]
