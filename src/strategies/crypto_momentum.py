import numpy as np
import pandas as pd
from ..config import CryptoStrategyConfig
from ..factors.momentum import CryptoTechnicalFactors
from .base import BaseStrategy


class CryptoMomentumStrategy(BaseStrategy):
    """
    Crypto Quantitative Momentum Strategy.
    Combines trend-following indicators with dynamic volatility management:
    - Dual EMA Crossover (Fast vs Slow)
    - RSI Momentum Filter (Overbought/Oversold thresholds)
    - ATR Volatility-adjusted trailing risk bounds
    """

    def __init__(self, config: CryptoStrategyConfig = None):
        super().__init__("crypto_dual_ema_momentum")
        self.config = config or CryptoStrategyConfig()

    def generate_target_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "close" not in df.columns:
            raise ValueError("Dataframe must contain 'close' column")

        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        # Compute Technical Indicators
        ema_fast = CryptoTechnicalFactors.exponential_moving_average(close, self.config.fast_ema)
        ema_slow = CryptoTechnicalFactors.exponential_moving_average(close, self.config.slow_ema)
        rsi = CryptoTechnicalFactors.relative_strength_index(close, self.config.rsi_period)
        atr = CryptoTechnicalFactors.average_true_range(high, low, close, self.config.atr_period)

        weights = pd.Series(0.0, index=df.index)
        position = 0.0
        trailing_stop = 0.0

        for i in range(1, len(df)):
            curr_close = close.iloc[i]
            prev_close = close.iloc[i - 1]
            fast_curr = ema_fast.iloc[i]
            slow_curr = ema_slow.iloc[i]
            fast_prev = ema_fast.iloc[i - 1]
            slow_prev = ema_slow.iloc[i - 1]
            curr_rsi = rsi.iloc[i]
            curr_atr = atr.iloc[i]

            # Bullish Crossover: Fast EMA crosses above Slow EMA and RSI < 70
            bullish_cross = (fast_curr > slow_curr) and (fast_prev <= slow_prev)
            # Bearish Crossover: Fast EMA crosses below Slow EMA
            bearish_cross = (fast_curr < slow_curr) and (fast_prev >= slow_prev)

            if position == 0.0:
                if bullish_cross and curr_rsi < self.config.rsi_overbought:
                    position = self.config.position_size_pct
                    trailing_stop = curr_close - (self.config.atr_stop_multiplier * curr_atr)
            elif position > 0.0:
                # Update trailing stop
                new_stop = curr_close - (self.config.atr_stop_multiplier * curr_atr)
                if new_stop > trailing_stop:
                    trailing_stop = new_stop

                # Exit on trailing stop breach or bearish cross
                if curr_close < trailing_stop or bearish_cross or curr_rsi > 85.0:
                    position = 0.0
                    trailing_stop = 0.0

            weights.iloc[i] = position

        date_col = df["date"] if "date" in df.columns else df.index
        out_df = pd.DataFrame({"weight": weights.values}, index=date_col)
        out_df.index.name = "date"
        return out_df
