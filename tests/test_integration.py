import pandas as pd
import pytest
from src.data.loader import load_equity_universe, load_crypto_series
from src.strategies.equity_multifactor import EquityMultiFactorStrategy
from src.analytics.tearsheet import TearSheetGenerator
from src.cli import run_equity_backtest, run_crypto_backtest


def test_data_loader_equities():
    price_matrix, panel_df = load_equity_universe()
    assert isinstance(price_matrix, pd.DataFrame)
    assert isinstance(panel_df, pd.DataFrame)
    assert len(price_matrix.columns) >= 5
    assert "date" in panel_df.columns
    assert "ticker" in panel_df.columns


def test_data_loader_crypto():
    btc_df = load_crypto_series("data/crypto/BTC_USDT_1d.csv")
    assert isinstance(btc_df, pd.DataFrame)
    assert "close" in btc_df.columns
    assert len(btc_df) >= 100


def test_equity_multifactor_full_strategy():
    price_matrix, panel_df = load_equity_universe()
    strategy = EquityMultiFactorStrategy()
    weights_df = strategy.generate_target_weights(panel_df)

    assert isinstance(weights_df, pd.DataFrame)
    assert len(weights_df) == len(price_matrix)
    # Check that on rebalance dates, weights sum close to 0 (dollar-neutral)
    daily_sums = weights_df.sum(axis=1)
    assert (daily_sums.abs() < 1e-4).all()


def test_tearsheet_and_cli_execution():
    eq_res = run_equity_backtest()
    summary = TearSheetGenerator.generate_summary_table(eq_res)
    assert "Quantopian Multi-Factor Market-Neutral" in summary
    assert "Sharpe Ratio" in summary
    assert "Alpha (vs Benchmark)" in summary

    monthly = TearSheetGenerator.generate_monthly_returns_table(eq_res.daily_returns)
    assert "Year Total" in monthly

    crypto_res = run_crypto_backtest("BTC_USDT")
    assert crypto_res.strategy_name == "Crypto Momentum (BTC_USDT)"
