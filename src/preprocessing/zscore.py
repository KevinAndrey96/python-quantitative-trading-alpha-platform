import numpy as np
import pandas as pd


def standardize_series(series: pd.Series) -> pd.Series:
    """
    Standardize a pandas Series to zero mean and unit variance:
    Z = (X - mu) / sigma
    """
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def cross_sectional_zscore(df: pd.DataFrame, factor_col: str, date_col: str = "date") -> pd.Series:
    """Apply cross-sectional Z-score normalization grouped by date."""
    return df.groupby(date_col)[factor_col].transform(standardize_series)
