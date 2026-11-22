import pandas as pd
from .base import BaseFactor


class QualityFactor(BaseFactor):
    """
    Fundamental Quality Factor: Return on Equity (ROE).
    Measures profitability and capital efficiency.
    """

    def __init__(self, name: str = "quality"):
        super().__init__(name)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "roe" in df.columns:
            return df["roe"].copy()
        if "net_income" in df.columns and "shareholder_equity" in df.columns:
            equity = df["shareholder_equity"].replace(0, float("nan"))
            return df["net_income"] / equity
        raise ValueError("Dataframe must contain 'roe' or ('net_income' and 'shareholder_equity')")
