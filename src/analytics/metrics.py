from typing import Dict, Tuple
import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate (CAGR)."""
    n_periods = len(equity_curve)
    if n_periods < 2 or equity_curve.iloc[0] <= 0 or equity_curve.iloc[-1] <= 0:
        return 0.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    std = daily_returns.std()
    if np.isnan(std) or std == 0:
        return 0.0
    return float(std * np.sqrt(periods_per_year))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio."""
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_ret = daily_returns - rf_daily
    std = excess_ret.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess_ret.mean() / std) * np.sqrt(periods_per_year))


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino Ratio focusing only on downside volatility."""
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_ret = daily_returns - rf_daily
    downside = excess_ret[excess_ret < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std) or len(downside) == 0:
        return 0.0
    return float((excess_ret.mean() / downside_std) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough equity decline."""
    if len(equity_curve) == 0:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(cagr(equity_curve, periods_per_year) / mdd)


def alpha_and_beta(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Tuple[float, float]:
    """Computes CAPM Alpha and Beta relative to benchmark."""
    aligned = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 1.0

    ret_s = aligned.iloc[:, 0]
    ret_b = aligned.iloc[:, 1]

    cov_matrix = np.cov(ret_s, ret_b)
    var_b = cov_matrix[1, 1]
    if var_b == 0 or np.isnan(var_b):
        return 0.0, 1.0

    beta = float(cov_matrix[0, 1] / var_b)
    annual_s = ret_s.mean() * periods_per_year
    annual_b = ret_b.mean() * periods_per_year
    alpha = float(annual_s - (risk_free_rate + beta * (annual_b - risk_free_rate)))
    return alpha, beta


def win_rate(daily_returns: pd.Series) -> float:
    nonzero = daily_returns[daily_returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).sum() / len(nonzero))


def compute_all_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """Generates complete dictionary of financial performance metrics."""
    metrics = {
        "Total Return": total_return(equity_curve),
        "CAGR": cagr(equity_curve, periods_per_year),
        "Annualized Volatility": annualized_volatility(daily_returns, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(daily_returns, risk_free_rate, periods_per_year),
        "Sortino Ratio": sortino_ratio(daily_returns, risk_free_rate, periods_per_year),
        "Max Drawdown": max_drawdown(equity_curve),
        "Calmar Ratio": calmar_ratio(equity_curve, periods_per_year),
        "Win Rate": win_rate(daily_returns)
    }

    if benchmark_returns is not None:
        alpha, beta = alpha_and_beta(daily_returns, benchmark_returns, risk_free_rate, periods_per_year)
        metrics["Alpha"] = alpha
        metrics["Beta"] = beta

    return metrics
