import os
from typing import Tuple
import pandas as pd


def load_equity_universe(csv_path: str = "data/equities/sample_universe.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads cross-sectional equity dataset.
    Returns:
    - price_matrix: DataFrame of close prices indexed by Date with tickers as columns.
    - full_df: Full panel dataframe including fundamentals and sentiment scores.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Equity dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    price_matrix = df.pivot(index="date", columns="ticker", values="close")
    price_matrix.index.name = "date"
    return price_matrix, df


def load_crypto_series(csv_path: str = "data/crypto/BTC_USDT_1d.csv") -> pd.DataFrame:
    """Loads OHLCV daily data for a cryptocurrency pair."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Crypto dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return df
