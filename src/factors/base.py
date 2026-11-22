from abc import ABC, abstractmethod
import pandas as pd


class BaseFactor(ABC):
    """Abstract base class for quantitative investment factors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute factor values from input market/fundamental dataframe."""
        pass
