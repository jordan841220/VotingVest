# VotingVest

## Goal

This project leverages multiple technical indicators to perform a **MAJORITY VOTE** and brute‑force all possible indicator combinations, backtests each strategy’s performance, and then selects the best combination

- Backtesting: Calculates cumulative return, Sharpe ratio, win rate, total trades, maximum drawdown, etc.
- Decision Making: Generates a "Strong Buy", "Hold / Buy", or "Close / No Buy" recommendation for the current day based on the top-performing indicator combination. (Short-selling is not implemented; "Close" strictly means **CLOSING** any existing long position.)

## Prerequisites
- Python: Version 3.8 or higher
- Python Packages:
```
pip install pandas numpy yfinance TA-Lib matplotlib tqdm
```

## Indicators and Criteria
The indicators we consider in the script:

### RSI (RSI_14, RSI_6)
- <30 → Buy
- \>70 → Close

### Stochastic Oscillator (STOCH_K_14, STOCH_D_3)
- <20 → Buy
- \>80 → Close

### Williams %R (WILLIAMS_R_14)
- <-80 → Buy
- \>-20 → Close

### CCI (CCI_20)
- < -100 → Buy
- /> 100 → Close

### MFI (MFI_14)
- < 20 → Buy
- \> 80 → Close

### MACD Crossover
- MACD line crosses above signal line → Buy
- MACD line crosses below signal line → Close

### Bollinger Bands
- Price crosses above upper band → Close
- Price crosses above middle band → Buy
- Price crosses below middle band → Close
- Price crosses below lower band → Buy

### Moving Average Crossovers
- EMA_12/EMA_26: Golden cross → Buy; Death cross → Close
- SMA_50/SMA_200: Golden cross → Buy; Death cross → Close

### DMI / ADX
- +DI crosses above -DI → Buy; crosses below → Close
- ADX > 25: if +DI > -DI → Buy; if +DI < -DI → Close

### OBV (On‑Balance Volume)
- OBV increases → Buy
- OBV decreases → Close

### Parabolic SAR (SAR)
- Price above SAR → Buy
- Price below SAR → Close

### Ichimoku
- Conversion/Baseline crossover → Buy/Close
- Cloud breakout: Price breaks above or below the Senkou Span A/B → Buy/Close
- Lagging Span crossover → Buy/Close

### Other Breakouts and Momentum
- Donchian Channel (20‑period)**: Break above upper channel → Buy; break below → Close
- Keltner Channel (20‑period ATR)**: Break above upper channel → Buy; break below → Close
- ATR Breakout**: Close > previous close + ATR → Buy; close < previous close - ATR → Close
- Aroon Up/Down crossover** → Buy/Close
- ROC (ROC_10)**: ROC > 0 → Buy; ROC < 0 → Close

## Usage
```
python strategy.py <ticker> <threads> <member>
```
- [ticker]: (required) stock symbol (e.g., AAPL, MSFT, GOOG)
- [threads]: (optional) number of worker processes for vote computation; default = 20.
- [member]: (optional) size of each indicator group (must be odd); default = 7.

### The script will:

1. Download historical data (all period) from Yahoo Finance
2. Compute all technical indicators
3. Generate signals via **MAJORITY VOTE** over all "odd-length" (specified by [member]) indicator combinations
4. Backtest each combination’s performance
5. Rank combinations using Min‑Max normalization and select the top 3 (exclude the combinations with <5 trading counts)
6. Plot performance charts and output today’s recommendation based on the best combo

Happy backtesting and good luck with your trading insights! 🎯

