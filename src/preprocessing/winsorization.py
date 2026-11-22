import numpy as np
import pandas as pd


def winsorize_series(series: pd.Series, low: float = 0.05, high: float = 0.95) -> pd.Series:
    """
    Winsorize a pandas Series to limit the influence of extreme statistical outliers.
    Values below the low percentile are capped at the low percentile value;
    values above the high percentile are capped at the high percentile value.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return series.copy()

    q_low = clean.quantile(low)
    q_high = clean.quantile(high)
    return series.clip(lower=q_low, upper=q_high)


def cross_sectional_winsorize(df: pd.DataFrame, factor_col: str, date_col: str = "date",
                              low: float = 0.05, high: float = 0.95) -> pd.Series:
    """Winsorize factor values on a cross-sectional per-date basis."""
    return df.groupby(date_col)[factor_col].transform(lambda s: winsorize_series(s, low, high))
