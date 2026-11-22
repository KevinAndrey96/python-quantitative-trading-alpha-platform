import pandas as pd
from tabulate import tabulate
from ..backtesting.engine import BacktestResult
from .metrics import compute_all_metrics


class TearSheetGenerator:
    """Generates institutional quantitative tear sheets and summary performance tables."""

    @staticmethod
    def generate_summary_table(result: BacktestResult, risk_free_rate: float = 0.0) -> str:
        metrics = compute_all_metrics(
            equity_curve=result.equity_curve,
            daily_returns=result.daily_returns,
            benchmark_returns=result.benchmark_returns,
            risk_free_rate=risk_free_rate
        )

        table_data = [
            ["Strategy Name", result.strategy_name],
            ["Start Date", str(result.equity_curve.index[0])[:10]],
            ["End Date", str(result.equity_curve.index[-1])[:10]],
            ["Total Trading Days", len(result.equity_curve)],
            ["Initial Capital", f"${result.equity_curve.iloc[0]:,.2f}"],
            ["Final Equity", f"${result.equity_curve.iloc[-1]:,.2f}"],
            ["Total Return", f"{metrics['Total Return'] * 100:.2f}%"],
            ["CAGR (Annualized Return)", f"{metrics['CAGR'] * 100:.2f}%"],
            ["Annualized Volatility", f"{metrics['Annualized Volatility'] * 100:.2f}%"],
            ["Sharpe Ratio (Rf=0%)", f"{metrics['Sharpe Ratio']:.2f}"],
            ["Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}"],
            ["Max Drawdown", f"{metrics['Max Drawdown'] * 100:.2f}%"],
            ["Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}"],
            ["Win Rate (Days)", f"{metrics['Win Rate'] * 100:.2f}%"],
            ["Avg Daily Turnover", f"{result.turnover.mean() * 100:.2f}%"]
        ]

        if "Alpha" in metrics:
            table_data.append(["Alpha (vs Benchmark)", f"{metrics['Alpha'] * 100:.2f}%"])
            table_data.append(["Beta (vs Benchmark)", f"{metrics['Beta']:.2f}"])

        headers = ["Metric", "Value"]
        return tabulate(table_data, headers=headers, tablefmt="github")

    @staticmethod
    def generate_monthly_returns_table(daily_returns: pd.Series) -> str:
        """Constructs a Monthly Returns matrix table."""
        df = daily_returns.to_frame("ret")
        df.index = pd.to_datetime(df.index)
        monthly = df.groupby([df.index.year, df.index.month])["ret"].apply(
            lambda s: (1.0 + s).prod() - 1.0
        ).unstack()

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly.columns = [month_names[m - 1] for m in monthly.columns]
        monthly["Year Total"] = monthly.apply(lambda row: (1.0 + row.dropna()).prod() - 1.0, axis=1)

        # Format as percentage strings
        formatted = monthly.map(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "")
        return tabulate(formatted, headers="keys", tablefmt="github")
