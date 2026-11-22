import pandas as pd
import pytest
from src.analytics.metrics import (
    total_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    win_rate,
    compute_all_metrics,
)


def test_total_return():
    eq = pd.Series([100.0, 120.0])
    assert pytest.approx(total_return(eq), 0.001) == 0.20


def test_max_drawdown():
    # Peak at 100, drops to 70, recovers to 90 -> -30%
    eq = pd.Series([100.0, 70.0, 90.0])
    assert pytest.approx(max_drawdown(eq), 0.001) == -0.30


def test_win_rate():
    rets = pd.Series([0.01, -0.02, 0.03, 0.04, -0.01])
    # 3 positive out of 5 -> 60%
    assert pytest.approx(win_rate(rets), 0.01) == 0.60


def test_compute_all_metrics():
    eq = pd.Series([100.0, 102.0, 105.0, 103.0, 107.0])
    rets = eq.pct_change().fillna(0.0)
    metrics = compute_all_metrics(eq, rets)
    assert "Sharpe Ratio" in metrics
    assert "CAGR" in metrics
    assert "Max Drawdown" in metrics
    assert "Sortino Ratio" in metrics
    assert metrics["Total Return"] > 0.0
