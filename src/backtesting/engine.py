from dataclasses import dataclass
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    turnover: pd.Series
    holdings_value: pd.Series
    cash_value: pd.Series
    benchmark_returns: Optional[pd.Series] = None
    strategy_name: str = "Quantitative Strategy"


class BacktestEngine:
    """
    Quantitative Portfolio Backtesting Simulation Engine.
    Simulates portfolio rebalancing, execution slippage, commissions, and capital tracking.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_bps: float = 5.0,
        slippage_bps: float = 5.0
    ):
        self.initial_capital = initial_capital
        self.cost_rate = (commission_bps + slippage_bps) / 10_000.0

    def run_multi_asset(
        self,
        price_matrix: pd.DataFrame,
        target_weights: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
        strategy_name: str = "Multi-Factor Long/Short Equity"
    ) -> BacktestResult:
        """
        Executes backtest for multi-asset equity universe.
        price_matrix: DataFrame indexed by Date with tickers as columns (Closing prices).
        target_weights: DataFrame indexed by Date with tickers as columns (Portfolio weights).
        """
        # Align dates
        common_dates = sorted(price_matrix.index.intersection(target_weights.index))
        if len(common_dates) < 2:
            raise ValueError("Price matrix and target weights must have at least 2 common dates")

        prices = price_matrix.loc[common_dates].copy()
        weights = target_weights.loc[common_dates].copy().fillna(0.0)

        # Asset returns
        asset_returns = prices.pct_change().fillna(0.0)

        portfolio_equity = [self.initial_capital]
        daily_returns = [0.0]
        turnover_list = [0.0]

        prev_w = pd.Series(0.0, index=prices.columns)

        for i in range(1, len(common_dates)):
            curr_target_w = weights.iloc[i - 1]  # Trade executed on previous close / current open
            ret_t = asset_returns.iloc[i]

            # Calculate turnover and transaction costs
            delta_w = (curr_target_w - prev_w).abs().sum()
            tx_cost = delta_w * self.cost_rate

            # Gross portfolio return = sum(w_i * r_i)
            gross_ret = (curr_target_w * ret_t).sum()
            net_ret = gross_ret - tx_cost

            new_equity = portfolio_equity[-1] * (1.0 + net_ret)
            portfolio_equity.append(new_equity)
            daily_returns.append(net_ret)
            turnover_list.append(delta_w)

            # Evolve weights with asset returns before next rebalance
            prev_w = curr_target_w * (1.0 + ret_t)
            if prev_w.abs().sum() > 0:
                prev_w = prev_w / prev_w.abs().sum()

        equity_series = pd.Series(portfolio_equity, index=common_dates, name="equity")
        ret_series = pd.Series(daily_returns, index=common_dates, name="returns")
        turnover_series = pd.Series(turnover_list, index=common_dates, name="turnover")

        bench_returns = None
        if benchmark_prices is not None:
            bench_aligned = benchmark_prices.reindex(common_dates).ffill().bfill()
            bench_returns = bench_aligned.pct_change().fillna(0.0)

        return BacktestResult(
            equity_curve=equity_series,
            daily_returns=ret_series,
            turnover=turnover_series,
            holdings_value=equity_series * weights.abs().sum(axis=1),
            cash_value=equity_series * (1.0 - weights.abs().sum(axis=1)),
            benchmark_returns=bench_returns,
            strategy_name=strategy_name
        )

    def run_single_asset(
        self,
        close_prices: pd.Series,
        target_weights: pd.Series,
        strategy_name: str = "Crypto Momentum Strategy"
    ) -> BacktestResult:
        """Executes backtest for a single asset (e.g. BTC/USDT or ETH/USDT)."""
        common_idx = close_prices.index.intersection(target_weights.index)
        prices = close_prices.loc[common_idx]
        weights = target_weights.loc[common_idx].fillna(0.0)

        asset_returns = prices.pct_change().fillna(0.0)

        portfolio_equity = [self.initial_capital]
        daily_returns = [0.0]
        turnover_list = [0.0]

        prev_w = 0.0

        for i in range(1, len(common_idx)):
            curr_w = weights.iloc[i - 1]
            ret_t = asset_returns.iloc[i]

            delta_w = abs(curr_w - prev_w)
            tx_cost = delta_w * self.cost_rate

            gross_ret = curr_w * ret_t
            net_ret = gross_ret - tx_cost

            new_equity = portfolio_equity[-1] * (1.0 + net_ret)
            portfolio_equity.append(new_equity)
            daily_returns.append(net_ret)
            turnover_list.append(delta_w)

            prev_w = curr_w

        equity_series = pd.Series(portfolio_equity, index=common_idx, name="equity")
        ret_series = pd.Series(daily_returns, index=common_idx, name="returns")
        turnover_series = pd.Series(turnover_list, index=common_idx, name="turnover")

        return BacktestResult(
            equity_curve=equity_series,
            daily_returns=ret_series,
            turnover=turnover_series,
            holdings_value=equity_series * weights,
            cash_value=equity_series * (1.0 - weights),
            benchmark_returns=asset_returns,
            strategy_name=strategy_name
        )
