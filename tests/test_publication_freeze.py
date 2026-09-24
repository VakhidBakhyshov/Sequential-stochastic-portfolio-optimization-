import pandas as pd

from scripts.dataloader.monthly_etf_filtration_2 import percentile_score


def test_percentile_orientation_is_publication_correct():
    x = pd.Series([1.0, 2.0, 3.0], index=["low", "mid", "high"])
    higher = percentile_score(x, higher_is_better=True)
    lower = percentile_score(x, higher_is_better=False)
    assert higher["high"] > higher["mid"] > higher["low"]
    assert lower["low"] > lower["mid"] > lower["high"]
