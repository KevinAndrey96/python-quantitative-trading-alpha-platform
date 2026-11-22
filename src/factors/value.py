import pandas as pd
from .base import BaseFactor


class ValueFactor(BaseFactor):
    """
    Fundamental Value Factor: EBIT / Enterprise Value.
    Captures earnings yield relative to firm total enterprise value.
    """

    def __init__(self, name: str = "value"):
        super().__init__(name)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "ebit_ev_ratio" in df.columns:
            return df["ebit_ev_ratio"].copy()
        if "ebit" in df.columns and "enterprise_value" in df.columns:
            ev = df["enterprise_value"].replace(0, float("nan"))
            return df["ebit"] / ev
        raise ValueError("Dataframe must contain 'ebit_ev_ratio' or ('ebit' and 'enterprise_value')")
