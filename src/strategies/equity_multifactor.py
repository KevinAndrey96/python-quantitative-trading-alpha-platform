import pandas as pd
from ..config import EquityBacktestConfig
from ..factors.value import ValueFactor
from ..factors.quality import QualityFactor
from ..factors.sentiment import SentimentFactor
from ..preprocessing.winsorization import cross_sectional_winsorize
from ..preprocessing.zscore import cross_sectional_zscore
from ..preprocessing.composite import compute_composite_alpha
from ..portfolio.optimizer import DollarNeutralPortfolioOptimizer
from .base import BaseStrategy


class EquityMultiFactorStrategy(BaseStrategy):
    """
    Long/Short Market-Neutral Equity Multi-Factor Strategy.
    Modernized implementation of the Quantopian cross-sectional alpha model:
    - Factors: Value (EBIT/EV), Quality (ROE), Sentiment (NLP SMA)
    - Normalization: Cross-sectional winsorization & Z-score standardization
    - Optimization: Dollar-neutral long/short weighting with periodic rebalancing
    """

    def __init__(self, config: EquityBacktestConfig = None):
        super().__init__("equity_cross_sectional_multifactor")
        self.config = config or EquityBacktestConfig()
        self.value_factor = ValueFactor()
        self.quality_factor = QualityFactor()
        self.sentiment_factor = SentimentFactor()
        self.optimizer = DollarNeutralPortfolioOptimizer(
            long_quantile=self.config.long_quantile,
            short_quantile=self.config.short_quantile,
            max_gross_leverage=self.config.max_gross_leverage
        )

    def generate_target_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if "date" not in df.columns or "ticker" not in df.columns:
            raise ValueError("Dataframe must contain 'date' and 'ticker' columns")

        # 1. Compute raw factors
        df["raw_value"] = self.value_factor.compute(df)
        df["raw_quality"] = self.quality_factor.compute(df)
        df["raw_sentiment"] = self.sentiment_factor.compute(df)

        # 2. Winsorize cross-sectionally
        df["win_value"] = cross_sectional_winsorize(
            df, "raw_value", low=self.config.winsorize_low, high=self.config.winsorize_high
        )
        df["win_quality"] = cross_sectional_winsorize(
            df, "raw_quality", low=self.config.winsorize_low, high=self.config.winsorize_high
        )
        df["win_sentiment"] = cross_sectional_winsorize(
            df, "raw_sentiment", low=self.config.winsorize_low, high=self.config.winsorize_high
        )

        # 3. Z-Score standardize cross-sectionally
        df["z_value"] = cross_sectional_zscore(df, "win_value")
        df["z_quality"] = cross_sectional_zscore(df, "win_quality")
        df["z_sentiment"] = cross_sectional_zscore(df, "win_sentiment")

        # 4. Composite Alpha Score
        factor_weights = {
            "z_value": self.config.factor_weights.value,
            "z_quality": self.config.factor_weights.quality,
            "z_sentiment": self.config.factor_weights.sentiment
        }
        df["composite_alpha"] = compute_composite_alpha(df, factor_weights)

        # 5. Determine rebalance dates
        dates = sorted(df["date"].unique())
        rebal_freq = self.config.rebalance_frequency_days
        rebalance_dates = set(dates[::rebal_freq])

        # 6. Allocate dollar-neutral weights per rebalance date
        all_tickers = sorted(df["ticker"].unique())
        weights_by_date = {}

        last_weights = pd.Series(0.0, index=all_tickers)

        for d in dates:
            if d in rebalance_dates or len(weights_by_date) == 0:
                day_df = df[df["date"] == d].set_index("ticker")
                day_scores = day_df["composite_alpha"]
                allocated = self.optimizer.allocate_weights(day_scores)
                # Ensure all universe tickers are present
                last_weights = allocated.reindex(all_tickers).fillna(0.0)

            weights_by_date[d] = last_weights.copy()

        weights_df = pd.DataFrame.from_dict(weights_by_date, orient="index")
        weights_df.index.name = "date"
        return weights_df
