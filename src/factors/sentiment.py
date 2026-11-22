import pandas as pd
from .base import BaseFactor


class SentimentFactor(BaseFactor):
    """
    Alternative Data Sentiment Factor: StockTwits & Twitter NLP Sentiment.
    Applies rolling smoothing to extract persistent investor sentiment signals.
    """

    def __init__(self, name: str = "sentiment", window_length: int = 3):
        super().__init__(name)
        self.window_length = window_length

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "sentiment_score" in df.columns:
            # If multi-asset with ticker column, calculate rolling mean per ticker
            if "ticker" in df.columns and "date" in df.columns:
                return (
                    df.groupby("ticker")["sentiment_score"]
                    .transform(lambda s: s.rolling(self.window_length, min_periods=1).mean())
                )
            return df["sentiment_score"].rolling(self.window_length, min_periods=1).mean()

        if "bull_messages" in df.columns and "total_messages" in df.columns:
            tot = df["total_messages"].replace(0, float("nan"))
            ratio = df["bull_messages"] / tot
            return ratio.rolling(self.window_length, min_periods=1).mean()

        raise ValueError("Dataframe must contain 'sentiment_score' or ('bull_messages' and 'total_messages')")
