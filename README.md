# VotingVest
This project leverages multiple technical indicators to perform a **MAJORITY VOTE** and brute‑force all possible indicator combinations, backtests each strategy’s performance, and then selects the best combination

## Goal
這是一個自動化的股票分析工具，它會從 Yahoo Finance 下載股價，計算常見的技術指標（如 RSI、MACD、布林帶等），並根據這些指標給出買賣信號。接著，工具會測試不同指標組合的歷史表現，幫你找到最有效的策略，並生成簡易的績效圖表和最終建議。
<br>
<br>
This is an automated stock tool that downloads price data from Yahoo Finance, calculates common technical indicators (e.g. RSI, MACD, Bollinger Bands), and generates buy/sell signals. It then backtests different indicator combinations to find the best strategy and creates basic performance charts along with a final recommendation.

<br>
<br>
具體上來說，它會自動從 Yahoo Finance 抓取股票歷史價格數據，計算多達數十種常用技術指標（如 RSI、MACD、布林帶、一目均衡表等），並為每個指標生成買入／賣出信號。接著，工具透過多進程並行運行「多數投票」機制，針對不同指標組合進行回測，評估包括總報酬、夏普比率、勝率與最大回撤等績效指標，並自動產生多種圖表（例如報酬 vs. 夏普散點圖、交易收益分佈直方圖、權益曲線與回撤曲線）。最終，它會挑出表現最優的指標組合，給出當前的交易建議，並將所有結果與圖檔儲存到本地。

<br>
<br>

It automatically fetches historical stock price data from Yahoo Finance, computes dozens of popular technical indicators (such as RSI, MACD, Bollinger Bands, Ichimoku Cloud, etc.), and generates buy/sell signals for each indicator. It then employs a parallel “majority voting” process to backtest various indicator combinations, evaluating performance metrics like total return, Sharpe ratio, win rate, and maximum drawdown. The tool produces visual summaries—return vs. Sharpe scatter plots, trade return histograms, equity & drawdown curves—and identifies the top-performing combination, offering a current trading recommendation. All results and plots are saved locally for further review.


- Backtesting: Calculates cumulative return, Sharpe ratio, win rate, total trades, maximum drawdown, etc.
- Decision Making: Generates a "Strong Buy", "Hold / Buy", or "Close / No Buy" recommendation for the current day based on the top-performing indicator combination. (Short-selling is not implemented; "Close" strictly means **CLOSING** any existing long position.)

<br>

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


<br>



## Prerequisites
- Python: Version 3.8 or higher
- Python Packages:
```
pip install pandas numpy yfinance TA-Lib matplotlib tqdm
```

<br>

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


