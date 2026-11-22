from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for quantitative trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_target_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate target portfolio weights dataframe.
        For multi-asset: DataFrame indexed by Date with tickers as columns.
        For single-asset: Series or single-column DataFrame with target exposure (-1.0 to 1.0).
        """
        pass
