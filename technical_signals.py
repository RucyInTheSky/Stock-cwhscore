import talib
import numpy as np
import pandas as pd

def compute_technical_score(df, selected_indicators):
    df = df.copy()
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    score = 0
    max_score = 25

    for ind in selected_indicators:
        try:
            if ind == "SMA":
                sma = talib.SMA(close, 20)
                if close.iloc[-1] > sma.iloc[-1]: score += 2
            elif ind == "EMA":
                ema = talib.EMA(close, 20)
                if close.iloc[-1] > ema.iloc[-1]: score += 2
            elif ind == "MACD":
                macd, signal, hist = talib.MACD(close)
                if macd.iloc[-1] > signal.iloc[-1]: score += 3
            elif ind == "RSI":
                rsi = talib.RSI(close)
                if rsi.iloc[-1] < 30: score += 2
            elif ind == "BBANDS":
                upper, mid, lower = talib.BBANDS(close)
                if close.iloc[-1] < lower.iloc[-1]: score += 2
            elif ind == "ADX":
                adx = talib.ADX(high, low, close)
                if adx.iloc[-1] > 25: score += 2
            elif ind == "OBV":
                obv = talib.OBV(close, vol)
                if obv.diff().iloc[-1] > 0: score += 1
            elif ind == "ATR":
                atr = talib.ATR(high, low, close)
                if atr.iloc[-1] < atr.mean(): score += 1
            elif ind == "SAR":
                sar = talib.SAR(high, low)
                if close.iloc[-1] > sar.iloc[-1]: score += 2
        except Exception:
            pass

    df['technical_score'] = np.clip(score, 0, max_score)
    return df[['Ticker', 'technical_score']]
