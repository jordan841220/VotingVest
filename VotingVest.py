import pandas as pd
import numpy as np
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import datetime
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
import argparse


################################################################################################
# fetch from yahoo
def fetch_data(ticker: str,
               start: str | None = None,
               end:   str | None = None) -> pd.DataFrame:

    # end date, default = today
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    # start date, default = 'period="max"'
    if start is None:
        df = yf.download(ticker, period="max", progress=False)
        df = df.loc[:end]
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    return df



################################################################################################
# calculate for indicators (see README)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    df_ind = pd.DataFrame(index=df.index)
    close_flat = df['Close'].to_numpy(dtype='float64').ravel()
    high_flat = df['High'].to_numpy(dtype='float64').ravel()
    low_flat = df['Low'].to_numpy(dtype='float64').ravel()
    vol_flat = df['Volume'].to_numpy(dtype='float64').ravel()

    ################################################
    # RSI
    df_ind['RSI_14'] = talib.RSI(close_flat, timeperiod=14)
    df_ind['RSI_6']  = talib.RSI(close_flat, timeperiod=6)
    
    ################################################
    # STOCH K%D (14,3)
    slowk, slowd = talib.STOCH(high_flat, low_flat, close_flat,
                               fastk_period=14,
                               slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)
    df_ind['STOCH_K_14'] = slowk
    df_ind['STOCH_D_3']  = slowd
    
    ################################################
    # Williams %R
    df_ind['WILLIAMS_R_14'] = talib.WILLR(high_flat, low_flat, close_flat, timeperiod=14)

    ################################################
    # CCI
    df_ind['CCI_20'] = talib.CCI(high_flat, low_flat, close_flat, timeperiod=20)

    ################################################
    # MFI
    df_ind['MFI_14'] = talib.MFI(high_flat, low_flat, close_flat, vol_flat, timeperiod=14)

    ################################################
    # MACD，之後用 黃金交叉 來判斷買進與否
    macd, macd_signal, _ = talib.MACD(close_flat,
                                      fastperiod=12,
                                      slowperiod=26,
                                      signalperiod=9)
    df_ind['MACD']        = macd
    df_ind['MACD_SIGNAL'] = macd_signal
    
    ################################################
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close_flat,
                                        timeperiod=20,
                                        nbdevup=2, nbdevdn=2)
    df_ind['BB_UPPER']   = upper
    df_ind['BB_MIDDLE']  = middle
    df_ind['BB_LOWER']   = lower

    ################################################
    # EMA 12 & 26
    df_ind['EMA_12'] = talib.EMA(close_flat, timeperiod=12)
    df_ind['EMA_26'] = talib.EMA(close_flat, timeperiod=26)

    ################################################
    # SMA 50 & 200
    df_ind['SMA_50']  = talib.SMA(close_flat, timeperiod=50)
    df_ind['SMA_200'] = talib.SMA(close_flat, timeperiod=200)

    ################################################
    # DMI (+DI, -DI)
    df_ind['PLUS_DI_14']  = talib.PLUS_DI(high_flat, low_flat, close_flat, timeperiod=14)
    df_ind['MINUS_DI_14'] = talib.MINUS_DI(high_flat, low_flat, close_flat, timeperiod=14)

    ################################################
    # ADX
    df_ind['ADX_14'] = talib.ADX(high_flat, low_flat, close_flat, timeperiod=14)

    ################################################
    # OBV
    df_ind['OBV'] = talib.OBV(close_flat, vol_flat)

    ################################################
    # SAR
    df_ind['SAR'] = talib.SAR(high_flat, low_flat, acceleration=0.02, maximum=0.2)

    ################################################
    # Ichimoku 轉換線 & 基準線
    
    # Tenkan-sen（转换线）= (过去9期最高价 + 过去9期最低价) / 2
    df_ind['ICHIMOKU_CONVERSION'] = (
        df['High'].rolling(window=9).max() +
        df['Low'].rolling(window=9).min()
    ) / 2

    # Kijun-sen（基准线）= (过去26期最高价 + 过去26期最低价) / 2
    df_ind['ICHIMOKU_BASELINE']   = (
        df['High'].rolling(window=26).max() +
        df['Low'].rolling(window=26).min()
    ) / 2
    
    # Span A
    df_ind['ICHIMOKU_SPAN_A'] = (
        df_ind['ICHIMOKU_CONVERSION'] +
        df_ind['ICHIMOKU_BASELINE']
    ) / 2
    df_ind['ICHIMOKU_SPAN_A'] = df_ind['ICHIMOKU_SPAN_A'].shift(26)
    
    # Span B
    df_ind['ICHIMOKU_SPAN_B'] = (
        df['High'].rolling(window=52).max() +
        df['Low'].rolling(window=52).min()
    ) / 2
    df_ind['ICHIMOKU_SPAN_B'] = df_ind['ICHIMOKU_SPAN_B'].shift(26)

    # Chikou Span
    df_ind['ICHIMOKU_LAGGING'] = df['Close'].shift(-26)
    
    
    ################################################
    # Donchian Channel (20-period high/low)
    df_ind['DONCH_UPPER_20'] = df['High'].rolling(window=20).max()
    df_ind['DONCH_LOWER_20'] = df['Low'].rolling(window=20).min()

    ################################################
    # ATR & Keltner Channel
    df_ind['ATR_14'] = talib.ATR(high_flat, low_flat, close_flat, timeperiod=14)
    ema20 = talib.EMA(close_flat, timeperiod=20)
    df_ind['KELT_UPPER_20'] = ema20 + df_ind['ATR_14']
    df_ind['KELT_LOWER_20'] = ema20 - df_ind['ATR_14']

    ################################################
    # Aroon Up/Down
    aroon_up, aroon_down = talib.AROON(high_flat, low_flat, timeperiod=25)
    df_ind['AROON_UP']   = aroon_up
    df_ind['AROON_DOWN'] = aroon_down

    ################################################
    # ROC (Rate of Change)
    df_ind['ROC_10'] = talib.ROC(close_flat, timeperiod=10)

    # 加入更多指標

    df_new = pd.concat([df, df_ind], axis=1)
    df_new.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df_new.columns
    ]
    
    return df_new


################################################################################################
def generate_signal(df: pd.DataFrame, indicator: str) -> pd.Series:

    s = pd.Series(0, index=df.index) 
    
    # RSI_14, RSI_6
    # <30=buy, >70=sell
    if indicator.startswith('RSI'):
        # RSI_14, RSI_6
        s[df[indicator] < 30] =  1
        s[df[indicator] > 70] = -1
    
    # K/D
    # <20=buy, >80=sell
    elif indicator == 'STOCH_K_14':
        s[df['STOCH_K_14'] < 20] = 1
        s[df['STOCH_K_14'] > 80] = -1
    
    elif indicator == 'STOCH_D_3':
        s[df['STOCH_D_3'] < 20] = 1
        s[df['STOCH_D_3'] > 80] = -1
    
    # WILLIAMS
    # <-80=buy, >-20=sell
    elif indicator == 'WILLIAMS_R_14':
        s[df['WILLIAMS_R_14'] < -80] = 1
        s[df['WILLIAMS_R_14'] > -20] = -1
    
    # CCI
    # <-100=buy, >100=sell
    elif indicator == 'CCI_20':
        s[df['CCI_20'] < -100] = 1
        s[df['CCI_20'] > 100] = -1
    
    # MFI
    # <20=buy, >80=sell
    elif indicator == 'MFI_14':
        s[df['MFI_14'] < 20] = 1
        s[df['MFI_14'] > 80] = -1
    
    # MACD 黃金交叉
    # 黃金交叉
    elif indicator == 'MACD':
        buy  = (df['MACD'] > df['MACD_SIGNAL']) & (df['MACD'].shift(1) <= df['MACD_SIGNAL'].shift(1))
        sell = (df['MACD'] < df['MACD_SIGNAL']) & (df['MACD'].shift(1) >= df['MACD_SIGNAL'].shift(1))
        s[buy]  =  1
        s[sell] = -1
    
    # 布林帶
    elif indicator == 'BB_UPPER':
        cross_up = (df['Close'] > df['BB_UPPER']) & (df['Close'].shift(1) <= df['BB_UPPER'].shift(1))
        s[cross_up] = -1
    elif indicator == 'BB_MIDDLE':
        cross_mid_up   = (df['Close'] > df['BB_MIDDLE']) & (df['Close'].shift(1) <= df['BB_MIDDLE'].shift(1))
        cross_mid_down = (df['Close'] < df['BB_MIDDLE']) & (df['Close'].shift(1) >= df['BB_MIDDLE'].shift(1))
        s[cross_mid_up]   = 1
        s[cross_mid_down] = -1
    elif indicator == 'BB_LOWER':
        cross_low = (df['Close'] < df['BB_LOWER']) & (df['Close'].shift(1) >= df['BB_LOWER'].shift(1))
        s[cross_low] = 1
    
    # EMA 交叉 (12,26)
    elif indicator == 'EMA_CROSS_12_26':
        buy  = (df['EMA_12'] > df['EMA_26']) & (df['EMA_12'].shift(1) <= df['EMA_26'].shift(1))
        sell = (df['EMA_12'] < df['EMA_26']) & (df['EMA_12'].shift(1) >= df['EMA_26'].shift(1))
        s[buy]  = 1
        s[sell] = -1
    
    # SMA 交叉 (50,200)
    elif indicator == 'SMA_CROSS_50_200':
        buy  = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        sell = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
        s[buy]  = 1
        s[sell] = -1
    
    # DMI 指標 (+DI vs -DI)
    elif indicator == 'PLUS_DI_14':
        buy  = (df['PLUS_DI_14'] > df['MINUS_DI_14']) & (df['PLUS_DI_14'].shift(1) <= df['MINUS_DI_14'].shift(1))
        sell = (df['PLUS_DI_14'] < df['MINUS_DI_14']) & (df['PLUS_DI_14'].shift(1) >= df['MINUS_DI_14'].shift(1))
        s[buy]  = 1
        s[sell] = -1
    
    # OBV 變化趨勢
    elif indicator == 'OBV':
        buy  = df['OBV'].diff() > 0
        sell = df['OBV'].diff() < 0
        s[buy]  = 1
        s[sell] = -1
    
    # SAR 轉向
    elif indicator == 'SAR':
        buy  = df['Close'] > df['SAR']
        sell = df['Close'] < df['SAR']
        s[buy]  = 1
        s[sell] = -1
    
    # 一目均衡表 轉換線/基準線交叉
    # Ichimoku Conversion/Baseline 交叉
    elif indicator == 'ICHIMOKU_CROSS':
        buy  = (df['ICHIMOKU_CONVERSION'] > df['ICHIMOKU_BASELINE']) & \
               (df['ICHIMOKU_CONVERSION'].shift(1) <= df['ICHIMOKU_BASELINE'].shift(1))
        sell = (df['ICHIMOKU_CONVERSION'] < df['ICHIMOKU_BASELINE']) & \
               (df['ICHIMOKU_CONVERSION'].shift(1) >= df['ICHIMOKU_BASELINE'].shift(1))
        s[buy]  = 1
        s[sell] = -1

    # Price vs Cloud Breakout (Span A/B 雲層突破)
    elif indicator == 'ICHIMOKU_CLOUD':
        upper_cloud = pd.concat([df['ICHIMOKU_SPAN_A'], df['ICHIMOKU_SPAN_B']], axis=1).max(axis=1)
        lower_cloud = pd.concat([df['ICHIMOKU_SPAN_A'], df['ICHIMOKU_SPAN_B']], axis=1).min(axis=1)
        # 價格由下穿越雲層上緣 -> 買進
        buy  = (df['Close'] > upper_cloud) & (df['Close'].shift(1) <= upper_cloud.shift(1))
        # 價格由上跌破雲層下緣 -> 賣出
        sell = (df['Close'] < lower_cloud) & (df['Close'].shift(1) >= lower_cloud.shift(1))
        s[buy]  = 1
        s[sell] = -1

    # Chikou Span vs Price (Lagging Span 交叉)
    elif indicator == 'ICHIMOKU_LAG_CROSS':
        # Lagging span 由下穿過當期收盤 -> 買進
        buy  = (df['ICHIMOKU_LAGGING'] > df['Close']) & (df['ICHIMOKU_LAGGING'].shift(1) <= df['Close'].shift(1))
        # Lagging span 由上跌破當期收盤 -> 賣出
        sell = (df['ICHIMOKU_LAGGING'] < df['Close']) & (df['ICHIMOKU_LAGGING'].shift(1) >= df['Close'].shift(1))
        s[buy]  = 1
        s[sell] = -1
    
    # DONCH breakout
    elif indicator == 'DONCH_UPPER_20':
        buy  = df['Close'] > df['DONCH_UPPER_20']
        sell = df['Close'] < df['DONCH_UPPER_20']
        s[buy]  = 1
        s[sell] = -1
    
    # Keltner breakout
    elif indicator == 'KELT_UPPER_20':
        buy  = df['Close'] > df['KELT_UPPER_20']
        sell = df['Close'] < df['KELT_UPPER_20']
        s[buy]  = 1
        s[sell] = -1
    
    # Aroon 指標交叉
    elif indicator == 'AROON_CROSS':
        buy  = (df['AROON_UP'] > df['AROON_DOWN']) & (df['AROON_UP'].shift(1) <= df['AROON_DOWN'].shift(1))
        sell = (df['AROON_UP'] < df['AROON_DOWN']) & (df['AROON_UP'].shift(1) >= df['AROON_DOWN'].shift(1))
        s[buy]  = 1
        s[sell] = -1
    
    # ADX 趨勢方向
    elif indicator == 'ADX_14':
        buy  = (df['ADX_14'] > 25) & (df['PLUS_DI_14'] > df['MINUS_DI_14'])
        sell = (df['ADX_14'] > 25) & (df['PLUS_DI_14'] < df['MINUS_DI_14'])
        s[buy]  = 1
        s[sell] = -1
    
    # ATR breakout
    elif indicator == 'ATR_BREAKOUT':
        buy  = df['Close'] > df['Close'].shift(1) + df['ATR_14']
        sell = df['Close'] < df['Close'].shift(1) - df['ATR_14']
        s[buy]  = 1
        s[sell] = -1
    
    # Rate of Change
    elif indicator == 'ROC_10':
        buy  = df['ROC_10'] > 0
        sell = df['ROC_10'] < 0
        s[buy]  = 1
        s[sell] = -1

    # 其餘指標?

    return s.ffill().fillna(0)




################################################################################################
# backtest
def backtest_indicator(df: pd.DataFrame, signals: pd.Series) -> dict:

    df_bt = df[['Close']].copy()
    df_bt['signal'] = signals

    # 1=buy, 0=nothing, -1=sell)
    df_bt['position'] = (
    df_bt['signal']
      .map({1: 1, -1: 0})  # 注意：map 只處理 key 有定義的情況，其他（也就是 0）會變成 NaN
      .ffill()            
      .fillna(0)          
    )

    # daily return
    df_bt['ret'] = df_bt['Close'].pct_change() * df_bt['position'].shift(1)
    
    
    # define when to entry and when to exit
    pos = df_bt['position']
    entries = df_bt.index[(pos.shift(1) == 0) & (pos == 1)].tolist()
    exits   = df_bt.index[(pos.shift(1) == 1) & (pos == 0)].tolist()
    # forced to sell in last trade (for calculating win_rate)
    if pos.iloc[-1] == 1:
        exits.append(df_bt.index[-1])

    # entry/exit
    m = min(len(entries), len(exits))
    entries, exits = entries[:m], exits[:m]

    # win rate
    trade_returns = [
        df_bt.loc[exit, 'Close'] / df_bt.loc[entry, 'Close'] - 1
        for entry, exit in zip(entries, exits)
    ]
    total_trades = len(trade_returns)
    win_trades   = sum(r > 0 for r in trade_returns)
    win_rate     = (win_trades / total_trades * 100) if total_trades > 0 else 0.0

    # cumulative return (%)
    cumulative = (1 + df_bt['ret'].fillna(0)).cumprod()
    df_bt['cumulative'] = cumulative
    total_return = (cumulative.iloc[-1] - 1) * 100

    # sharpe
    sharpe = (
        df_bt['ret'].mean() / df_bt['ret'].std() * np.sqrt(252)
        if df_bt['ret'].std() != 0 else np.nan
    )

    # max_drawdown
    running_max  = cumulative.cummax()
    drawdown     = (running_max - cumulative) / running_max * 100
    max_drawdown = drawdown.max()

    return {
        'total_return_%': round(total_return, 2),
        'sharpe_ratio':   round(sharpe, 2),
        'win_rate_%':     round(win_rate, 2),
        'total_trades':   total_trades,
        'max_drawdown_%': round(max_drawdown, 2)
    }


################################################################################################
def backtest_indicator_plot(df: pd.DataFrame, signals: pd.Series, title: str) -> dict:

    df_bt = df[['Close']].copy()
    df_bt['signal'] = signals

    # 1=buy, 0=nothing, -1=sell)
    df_bt['position'] = (
    df_bt['signal']
      .map({1: 1, -1: 0})  # 注意：map 只處理 key 有定義的情況，其他（也就是 0）會變成 NaN
      .ffill()            
      .fillna(0)          
    )

    # cumulative return (%)
    cumulative = (1 + df_bt['ret'].fillna(0)).cumprod()
    df_bt['cumulative'] = cumulative

    # plotting
    axes = df_bt[['Close','position','cumulative']].plot(
        subplots=True,
        figsize=(10, 8),
        layout=(4, 1),
        sharex=True
    )

    fig = axes.flatten()[0].get_figure()
    fig.suptitle(f"Backtest – {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"backtest_{title.replace('/', '_')}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved backtest plot to {filename}")

################################################################################################
def calculate_score(backtest_res):

    df = pd.DataFrame(backtest_res).T.reset_index().rename(columns={'index': 'combination'})
    
    # Min-max normalization
    def minmax(series):
        return (series - series.min()) / (series.max() - series.min())

    # these metrics are better when higher
    df['s_return'] = 1 - minmax(df['total_return_%'])
    df['s_sharpe'] = 1 - minmax(df['sharpe_ratio'])
    df['s_winrate'] = 1 - minmax(df['win_rate_%'])
    
    # these metrics are better when lower
    df['s_trades'] = minmax(df['total_trades'])
    df['s_drawdown'] = minmax(df['max_drawdown_%'])

    # sum
    # the lower the score is, the better the combo is
    score_cols = [col for col in df.columns 
                if col.startswith('s_') and not df[col].isna().all()]
    df['score'] = df[score_cols].sum(axis=1)
    
    # exclude trading counts < 5
    df = df[df['total_trades'] >= 5]
    # sort by score
    df_sorted = df.sort_values(by='score').reset_index(drop=True).head(3)
    
    return df_sorted

################################################################################################
# parallel computing the indicator VOTING process
def _compute_one_consensus(args):
    
    df, comb = args
    new_col = "_".join(map(str, comb))
    subset = df.iloc[:, list(comb)]
    def majority_vote(row):
        counts = row.value_counts()
        max_count = counts.max()
        winners = [label for label, cnt in counts.items() if cnt == max_count]
        return winners[0] if len(winners) == 1 else 0
    series = subset.apply(majority_vote, axis=1)
    return new_col, series

def generate_consensus_columns_parallel(df: pd.DataFrame, 
                                        max_workers: int, 
                                        member: int) -> pd.DataFrame:
    n = df.shape[1]
    # combos = [
    #     comb
    #     for r in range(3, n+1, 2)
    #     for comb in itertools.combinations(range(n), r)
    # ]
    combos = itertools.combinations(range(n), member)

    consensus = {}
    # fut_to_comb record the ERROR comb
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fut_to_comb = {
            executor.submit(_compute_one_consensus, (df, comb)): comb
            for comb in combos
        }
        for fut in tqdm(as_completed(fut_to_comb),
                        total=len(fut_to_comb),
                        desc="Consensus combinations"):
            comb = fut_to_comb[fut]
            try:
                col_name, series = fut.result()
                consensus[col_name] = series
            except Exception as e:
                # warning but keep running
                print(f"Warning: The combo {comb} failed - {e}")

    return pd.DataFrame(consensus, index=df.index)


################################################################################################
def generate_current_decision(df_new: pd.DataFrame,
                              df_subset_voted: pd.DataFrame,
                              result: pd.DataFrame,
                              pick: int = 1) -> None:

    combo = result["combination"].iat[pick]
    subset_cols = df_new.columns[5:]
    
    # get the real indicator names from combo name
    idxs = [int(i) for i in combo.split("_")]
    indicators = [subset_cols[i] for i in idxs]
    
    # fetch the last day of signal from this combo
    signal = df_subset_voted[combo].iat[-1]
    action = {1: "Strong Buy / Add to Position", 0: "Hold / Weak Buy", -1: "Close Position / No Buy"}[signal]
    
    print(f"Best Combo of Indicators: {indicators}")
    print(f"Suggested Movement: {action}")


################################################################################################
def main(ticket, 
         threads: int,
         member: int):

    if member % 2 == 0:
        print("Error: member must be an odd number", file=sys.stderr)
        sys.exit(1)
    
    df = fetch_data(ticket)
    df_new = compute_indicators(df)

    for col_i in range(5, len(df_new.columns)):
        indicator = df_new.columns[col_i]
        df_new[indicator] = generate_signal(df_new, indicator)

    df_subset = df_new.iloc[:, 5:] # omit CHLVO
    df_subset_voted = generate_consensus_columns_parallel(df = df_subset, max_workers = threads, member = member)

    backtest_res = {}
    for indicators in df_subset_voted.columns:
        backtest_res[indicators] = backtest_indicator(
            df_new[['Close']],         
            df_subset_voted[indicators]    
        )

    # plotting
    # 1) Scatter: Sharpe vs. Total Return
    scores_df = pd.DataFrame(backtest_res).T
    plt.figure(figsize=(6,4))
    plt.scatter(scores_df['sharpe_ratio'], scores_df['total_return_%'])
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Total Return (%)')
    plt.title('Return vs. Sharpe for All Combos')
    plt.tight_layout()
    plt.savefig('return_vs_sharpe.png', dpi=300)
    plt.close()

    # 2) Histogram of all trade returns (from best combo)
    best_key = max(backtest_res, key=lambda k: backtest_res[k]['total_return_%'])
    # re-run backtest to get individual trade returns
    df_bt = df_new[['Close']].copy()
    df_bt['signal'] = df_subset_voted[best_key]
    df_bt['position'] = df_bt['signal'].map({1:1, -1:0}).ffill().fillna(0)
    df_bt['ret'] = df_bt['Close'].pct_change() * df_bt['position'].shift(1)
    trade_returns = []
    pos = df_bt['position']
    entries = df_bt.index[(pos.shift(1)==0)&(pos==1)]
    exits   = df_bt.index[(pos.shift(1)==1)&(pos==0)]
    if pos.iloc[-1]==1: exits = exits.append(df.Index[-1:])
    for e, x in zip(entries, exits):
        trade_returns.append(df_bt.at[x,'Close']/df_bt.at[e,'Close'] - 1)
    plt.figure(figsize=(6,4))
    plt.hist(trade_returns, bins=20)
    plt.xlabel('Trade Return')
    plt.ylabel('Frequency')
    plt.title('Distribution of Trade Returns\n(best combo)')
    plt.tight_layout()
    plt.savefig('trade_return_hist.png', dpi=300)
    plt.close()

    # 3) Equity & Drawdown Curve for best combo
    cum = (1 + df_bt['ret'].fillna(0)).cumprod()
    running_max = cum.cummax()
    dd = (running_max - cum) / running_max
    plt.figure(figsize=(8,4))
    plt.plot(cum, label='Equity Curve')
    plt.plot(-dd, label='Drawdown', color='red')
    plt.legend()
    plt.title(f'Equity & Drawdown – {best_key}')
    plt.tight_layout()
    plt.savefig('equity_drawdown.png', dpi=300)
    plt.close()

    result = calculate_score(backtest_res)
    result.to_csv("backtest_results.csv", index=False)

    # plot
    for indicators in result["combination"]:
        s = df_subset_voted[indicators]
        backtest_indicator_plot(df_new[['Close']], s, indicators)

    # generate current decision using our best indicators combination
    generate_current_decision(df_new, df_subset_voted, result, pick=1)



################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest voting strategy")
    parser.add_argument("ticket", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("-t", "--threads", type=int, default=20,
                        help="Number of worker processes (default: 20)")
    parser.add_argument("-m", "--member", type=int, default=7,
                        help="Size of each indicator group (odd, default: 7)")
    parser.add_argument("-s", "--start", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: earliest)")
    parser.add_argument("-e", "--end",   type=str, default=None,
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    main(args.ticket, args.threads, args.member)
