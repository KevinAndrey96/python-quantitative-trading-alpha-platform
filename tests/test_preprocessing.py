import numpy as np
import pandas as pd
import pytest
from src.preprocessing.winsorization import winsorize_series, cross_sectional_winsorize
from src.preprocessing.zscore import standardize_series, cross_sectional_zscore
from src.preprocessing.composite import compute_composite_alpha


def test_winsorize_series():
    # 20 points from 1 to 20
    s = pd.Series(list(range(1, 21)), dtype=float)
    winsorized = winsorize_series(s, low=0.10, high=0.90)
    # 10th percentile ~ 2.9, 90th percentile ~ 18.1
    assert winsorized.min() > 1.0
    assert winsorized.max() < 20.0
    assert len(winsorized) == len(s)


def test_cross_sectional_zscore():
    df = pd.DataFrame({
        "date": ["2023-01-01"] * 5 + ["2023-01-02"] * 5,
        "factor": [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    })
    z = cross_sectional_zscore(df, "factor")
    # Day 1 mean should be 0, std 1
    day1_z = z.iloc[:5]
    day2_z = z.iloc[5:]
    assert pytest.approx(day1_z.mean(), abs=1e-6) == 0.0
    assert pytest.approx(day1_z.std(ddof=0), abs=1e-4) == 1.0
    assert pytest.approx(day2_z.mean(), abs=1e-6) == 0.0
    assert pytest.approx(day2_z.std(ddof=0), abs=1e-4) == 1.0


def test_compute_composite_alpha():
    df = pd.DataFrame({
        "factor_a": [1.0, 2.0, 3.0],
        "factor_b": [2.0, 4.0, 6.0]
    })
    weights = {"factor_a": 1.0, "factor_b": 2.0}
    comp = compute_composite_alpha(df, weights)
    # (1*1 + 2*2)/3 = 5/3 ~ 1.666
    assert pytest.approx(comp.iloc[0], 0.01) == 5.0 / 3.0
    # (1*2 + 2*4)/3 = 10/3 ~ 3.333
    assert pytest.approx(comp.iloc[1], 0.01) == 10.0 / 3.0
