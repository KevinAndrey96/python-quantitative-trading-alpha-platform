import numpy as np
import pandas as pd
import pytest
from src.factors.value import ValueFactor
from src.factors.quality import QualityFactor
from src.factors.sentiment import SentimentFactor
from src.factors.momentum import CryptoTechnicalFactors


def test_value_factor_computation():
    df = pd.DataFrame({
        "ebit": [10.0, 20.0, 30.0],
        "enterprise_value": [100.0, 200.0, 150.0]
    })
    factor = ValueFactor()
    res = factor.compute(df)
    expected = pd.Series([0.1, 0.1, 0.2])
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_value_factor_missing_columns():
    df = pd.DataFrame({"random_col": [1, 2, 3]})
    factor = ValueFactor()
    with pytest.raises(ValueError):
        factor.compute(df)


def test_quality_factor_computation():
    df = pd.DataFrame({
        "net_income": [15.0, 25.0],
        "shareholder_equity": [100.0, 50.0]
    })
    factor = QualityFactor()
    res = factor.compute(df)
    expected = pd.Series([0.15, 0.50])
    pd.testing.assert_series_equal(res, expected, check_names=False)


def test_sentiment_factor_rolling():
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
        "sentiment_score": [0.2, 0.4, 0.6, 0.8]
    })
    factor = SentimentFactor(window_length=2)
    res = factor.compute(df)
    assert res.iloc[0] == 0.2
    assert pytest.approx(res.iloc[1], 0.01) == 0.3
    assert pytest.approx(res.iloc[2], 0.01) == 0.5
    assert pytest.approx(res.iloc[3], 0.01) == 0.7


def test_crypto_technical_factors():
    prices = pd.Series([100.0, 102.0, 101.0, 105.0, 107.0, 104.0, 108.0, 110.0, 109.0, 115.0])
    high = prices + 2.0
    low = prices - 2.0

    ema = CryptoTechnicalFactors.exponential_moving_average(prices, span=3)
    assert len(ema) == len(prices)
    assert not ema.isna().any()

    rsi = CryptoTechnicalFactors.relative_strength_index(prices, period=5)
    assert len(rsi) == len(prices)
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()

    atr = CryptoTechnicalFactors.average_true_range(high, low, prices, period=5)
    assert len(atr) == len(prices)
    assert (atr >= 0.0).all()
