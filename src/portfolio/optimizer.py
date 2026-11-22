from typing import Dict, Optional
import numpy as np
import pandas as pd


class DollarNeutralPortfolioOptimizer:
    """
    Constructs an institutional dollar-neutral, market-neutral long/short equity portfolio:
    - Long bucket: Top alpha quantile
    - Short bucket: Bottom alpha quantile
    - Net exposure = 0.0
    - Gross exposure = 1.0 (or configured max leverage)
    - Position concentration limits applied equally
    """

    def __init__(
        self,
        long_quantile: float = 0.30,
        short_quantile: float = 0.30,
        max_gross_leverage: float = 1.0,
        max_single_position_pct: Optional[float] = None
    ):
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.max_gross_leverage = max_gross_leverage
        self.max_single_position_pct = max_single_position_pct

    def allocate_weights(self, alpha_scores: pd.Series) -> pd.Series:
        """
        Given a Series of composite alpha scores indexed by ticker on a rebalance date,
        allocates dollar-neutral weights.
        """
        clean_scores = alpha_scores.dropna()
        n = len(clean_scores)
        if n < 2:
            return pd.Series(0.0, index=alpha_scores.index)

        sorted_scores = clean_scores.sort_values(ascending=False)
        k_long = max(1, int(np.ceil(n * self.long_quantile)))
        k_short = max(1, int(np.ceil(n * self.short_quantile)))

        # Ensure no overlap between longs and shorts
        if k_long + k_short > n:
            k_long = n // 2
            k_short = n - k_long

        long_tickers = sorted_scores.index[:k_long]
        short_tickers = sorted_scores.index[-k_short:]

        weights = pd.Series(0.0, index=alpha_scores.index)

        # Allocate +0.5 * leverage to longs, -0.5 * leverage to shorts
        target_side_weight = 0.5 * self.max_gross_leverage
        weights.loc[long_tickers] = target_side_weight / k_long
        weights.loc[short_tickers] = -target_side_weight / k_short

        # Apply position concentration bounds if configured
        if self.max_single_position_pct is not None and self.max_single_position_pct > 0:
            weights = weights.clip(
                lower=-self.max_single_position_pct,
                upper=self.max_single_position_pct
            )

        return weights
