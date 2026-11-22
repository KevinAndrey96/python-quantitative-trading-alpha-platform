from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FactorWeights:
    value: float = 1.0
    quality: float = 1.0
    sentiment: float = 1.0


@dataclass
class EquityBacktestConfig:
    initial_capital: float = 1_000_000.0
    max_gross_leverage: float = 1.0
    rebalance_frequency_days: int = 5  # Weekly rebalancing
    winsorize_low: float = 0.05
    winsorize_high: float = 0.95
    long_quantile: float = 0.30   # Top 30% for longs
    short_quantile: float = 0.30  # Bottom 30% for shorts
    commission_bps: float = 5.0   # 5 basis points (0.05%)
    slippage_bps: float = 5.0     # 5 basis points (0.05%)
    factor_weights: FactorWeights = field(default_factory=FactorWeights)


@dataclass
class CryptoStrategyConfig:
    initial_capital: float = 100_000.0
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0
    position_size_pct: float = 0.95
    commission_bps: float = 10.0  # 10 basis points (0.1% typical on Binance/Kraken)
    slippage_bps: float = 5.0
