# Legacy Quantopian Multi-Factor Model (2020)

This directory preserves the historical implementation developed for the Quantopian hosted quantitative research and backtesting platform prior to its shutdown in late 2020.

## Overview of the Original Strategy

The strategy implemented in `Cross-Sectional Equity.py` is a classic institutional **Long/Short Equity Market-Neutral** model built upon three primary predictive factor signals:

1. **Fundamental Value Factor:**
   $$\text{Value} = \frac{\text{EBIT}}{\text{Enterprise Value}}$$
   Captures earnings yield relative to firm total enterprise value.

2. **Fundamental Quality Factor:**
   $$\text{Quality} = \text{ROE} = \frac{\text{Net Income}}{\text{Shareholder Equity}}$$
   Identifies capital-efficient firms generating superior return on equity.

3. **Alternative Sentiment Factor (NLP):**
   Aggregates retail and institutional market sentiment from StockTwits and Twitter streams:
   $$\text{Sentiment} = \text{SMA}_{3}(\text{StockTwits}_{\text{bull}} - \text{StockTwits}_{\text{bear}})$$
   Standardized alongside Twitter message ratio:
   $$\text{Twitter Ratio} = \frac{\text{Bull Scored Messages}}{\text{Total Scanned Messages}}$$

## Statistical Transformations

- **Winsorization:** Limits extreme statistical outliers to the 5th and 95th percentiles to avoid idiosyncratic single-stock distortions.
- **Cross-Sectional Z-Score:** Standardizes factors across the universe to zero mean and unit variance:
  $$Z = \frac{X - \mu}{\sigma}$$
- **Composite Alpha Signal:**
  $$\text{Alpha} = Z(\text{Value}) + Z(\text{Quality}) + Z(\text{Sentiment})$$

## Portfolio Construction & Optimization

- **Trading Universe:** Top liquid US equities (`QTradableStocksUS`, 600 assets).
- **Long Basket:** Top 300 stocks with the highest composite alpha scores.
- **Short Basket:** Bottom 300 stocks with the lowest composite alpha scores.
- **Dollar Neutrality:** Net dollar exposure constrained to 0.0.
- **Gross Leverage:** Constrained to a maximum of 1.0.
- **Risk Model Neutralization:** Factor exposure neutralization across market sectors and style risks.
- **Position Limits:** Maximum individual position size capped at $\pm \frac{2.0}{600} \approx \pm 0.33\%$ to prevent single-stock concentration risk.
