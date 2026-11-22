import pandas as pd
import pytest
from src.backtesting.engine import BacktestEngine


def test_engine_single_asset_flat_prices():
    prices = pd.Series([100.0, 100.0, 100.0, 100.0], index=pd.date_range("2023-01-01", periods=4))
    weights = pd.Series([1.0, 1.0, 1.0, 1.0], index=prices.index)

    # With 0 costs
    engine = BacktestEngine(initial_capital=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    res = engine.run_single_asset(prices, weights)

    assert pytest.approx(res.equity_curve.iloc[-1], 0.01) == 10_000.0
    assert (res.daily_returns == 0.0).all()


def test_engine_single_asset_positive_returns():
    prices = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2023-01-01", periods=3))
    weights = pd.Series([1.0, 1.0, 1.0], index=prices.index)

    engine = BacktestEngine(initial_capital=10_000.0, commission_bps=0.0, slippage_bps=0.0)
    res = engine.run_single_asset(prices, weights)

    # 10,000 * 1.10 * 1.10 = 12,100
    assert pytest.approx(res.equity_curve.iloc[-1], 0.01) == 12_100.0


def test_engine_multi_asset_dollar_neutral():
    dates = pd.date_range("2023-01-01", periods=3)
    prices = pd.DataFrame({
        "AAPL": [100.0, 110.0, 120.0],
        "TSLA": [100.0, 90.0, 80.0]
    }, index=dates)

    # Long AAPL (+0.5), Short TSLA (-0.5)
    weights = pd.DataFrame({
        "AAPL": [0.5, 0.5, 0.5],
        "TSLA": [-0.5, -0.5, -0.5]
    }, index=dates)

    engine = BacktestEngine(initial_capital=100_000.0, commission_bps=0.0, slippage_bps=0.0)
    res = engine.run_multi_asset(prices, weights)

    # Day 1 to 2: AAPL +10% -> 0.5 * 0.10 = +5%
    # TSLA -10% -> -0.5 * (-0.10) = +5%
    # Total Day 2 return = +10%
    assert res.daily_returns.iloc[1] > 0.09
    assert res.equity_curve.iloc[-1] > 100_000.0
