import argparse
import sys
from .config import EquityBacktestConfig, CryptoStrategyConfig
from .data.loader import load_equity_universe, load_crypto_series
from .strategies.equity_multifactor import EquityMultiFactorStrategy
from .strategies.crypto_momentum import CryptoMomentumStrategy
from .backtesting.engine import BacktestEngine
from .analytics.tearsheet import TearSheetGenerator


def run_equity_backtest():
    print("=" * 72)
    print("  QUANTITATIVE ALPHA RESEARCH: CROSS-SECTIONAL EQUITY STRATEGY")
    print("  Model: Multi-Factor Market-Neutral (Value + Quality + Sentiment)")
    print("=" * 72)

    price_matrix, panel_df = load_equity_universe()
    config = EquityBacktestConfig()
    strategy = EquityMultiFactorStrategy(config)

    print(f"[*] Loaded universe: {len(price_matrix.columns)} equities over {len(price_matrix)} trading days.")
    print("[*] Computing factors, winsorization, and cross-sectional Z-scores...")
    target_weights = strategy.generate_target_weights(panel_df)

    print("[*] Executing dollar-neutral backtest simulation with slippage and fees...")
    engine = BacktestEngine(
        initial_capital=config.initial_capital,
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps
    )
    # Benchmark: Equal-weighted universe buy-and-hold
    benchmark_prices = price_matrix.mean(axis=1)

    result = engine.run_multi_asset(
        price_matrix=price_matrix,
        target_weights=target_weights,
        benchmark_prices=benchmark_prices,
        strategy_name="Quantopian Multi-Factor Market-Neutral"
    )

    print("\n" + TearSheetGenerator.generate_summary_table(result))
    print("\n[+] Monthly Returns Distribution:")
    print(TearSheetGenerator.generate_monthly_returns_table(result.daily_returns))
    return result


def run_crypto_backtest(symbol: str = "BTC_USDT"):
    print("=" * 72)
    print(f"  QUANTITATIVE MOMENTUM STRATEGY: CRYPTO ({symbol})")
    print("  Model: Dual-EMA Trend Following + RSI Filter + ATR Trailing Risk")
    print("=" * 72)

    csv_path = f"data/crypto/{symbol}_1d.csv"
    crypto_df = load_crypto_series(csv_path)

    config = CryptoStrategyConfig()
    strategy = CryptoMomentumStrategy(config)

    print(f"[*] Loaded {symbol}: {len(crypto_df)} daily bars.")
    print("[*] Computing Dual-EMA (9/21), RSI(14), and ATR(14) signals...")
    target_weights = strategy.generate_target_weights(crypto_df)

    print("[*] Executing backtest simulation with exchange fees...")
    engine = BacktestEngine(
        initial_capital=config.initial_capital,
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps
    )

    result = engine.run_single_asset(
        close_prices=crypto_df["close"],
        target_weights=target_weights["weight"],
        strategy_name=f"Crypto Momentum ({symbol})"
    )

    print("\n" + TearSheetGenerator.generate_summary_table(result))
    print("\n[+] Monthly Returns Distribution:")
    print(TearSheetGenerator.generate_monthly_returns_table(result.daily_returns))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Quantitative Trading & Multi-Factor Alpha Research Platform"
    )
    subparsers = parser.add_subparsers(dest="command", help="Backtest subcommand")

    equity_parser = subparsers.add_parser("equity", help="Run Cross-Sectional Multi-Factor Equity Backtest")
    crypto_parser = subparsers.add_parser("crypto", help="Run Crypto Momentum Backtest")
    crypto_parser.add_argument(
        "--symbol",
        type=str,
        default="BTC_USDT",
        choices=["BTC_USDT", "ETH_USDT"],
        help="Crypto trading pair"
    )

    args = parser.parse_args()

    if args.command == "equity":
        run_equity_backtest()
    elif args.command == "crypto":
        run_crypto_backtest(args.symbol)
    else:
        # Default: run both
        run_equity_backtest()
        print("\n\n")
        run_crypto_backtest("BTC_USDT")


if __name__ == "__main__":
    main()
