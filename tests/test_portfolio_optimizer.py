import pandas as pd
import pytest
from src.portfolio.optimizer import DollarNeutralPortfolioOptimizer


def test_dollar_neutral_allocation():
    scores = pd.Series({
        "AAPL": 2.5,
        "MSFT": 1.8,
        "GOOGL": 0.5,
        "AMZN": -0.2,
        "TSLA": -1.5,
        "META": -2.8
    })
    optimizer = DollarNeutralPortfolioOptimizer(long_quantile=0.33, short_quantile=0.33, max_gross_leverage=1.0)
    weights = optimizer.allocate_weights(scores)

    # Net exposure must be 0.0 (Dollar Neutral)
    assert pytest.approx(weights.sum(), abs=1e-6) == 0.0

    # Gross exposure must be 1.0
    assert pytest.approx(weights.abs().sum(), abs=1e-6) == 1.0

    # Longs must have positive weights, shorts negative weights
    assert weights["AAPL"] > 0
    assert weights["MSFT"] > 0
    assert weights["META"] < 0
    assert weights["TSLA"] < 0
    assert weights["GOOGL"] == 0.0
    assert weights["AMZN"] == 0.0


def test_position_concentration_cap():
    scores = pd.Series({"AAPL": 2.0, "MSFT": -2.0})
    optimizer = DollarNeutralPortfolioOptimizer(
        long_quantile=0.5,
        short_quantile=0.5,
        max_gross_leverage=1.0,
        max_single_position_pct=0.25
    )
    weights = optimizer.allocate_weights(scores)
    assert weights["AAPL"] <= 0.25
    assert weights["MSFT"] >= -0.25
