import pandas as pd
import pytest
from src.config import CryptoStrategyConfig
from src.strategies.crypto_momentum import CryptoMomentumStrategy


def test_crypto_strategy_signals():
    # Synthetic uptrend series
    prices = [100.0 + i * 2.0 for i in range(50)]
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=50),
        "close": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices]
    })

    strategy = CryptoMomentumStrategy()
    weights_df = strategy.generate_target_weights(df)

    assert "weight" in weights_df.columns
    assert len(weights_df) == 50
    # In strong continuous uptrend with fast EMA > slow EMA, should hold position
    assert (weights_df["weight"] >= 0.0).all()
    assert (weights_df["weight"] <= 1.0).all()
