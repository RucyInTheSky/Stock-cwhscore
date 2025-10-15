import talib
import numpy as np
import pandas as pd

TECH_LABELS = {
    "SMA": "単純移動平均（SMA20 上抜け）",
    "EMA": "指数移動平均（EMA20 上抜け）",
    "MACD": "MACDシグナル上抜け",
    "RSI": "RSIが中立域（40-60）",
    "BBANDS": "ボリンジャーバンド上向き（中心線より上）",
    "ADX": "ADXが25以上（トレンド強）",
    "OBV": "OBVが上昇傾向",
    "ATR": "ATRが平均未満（ボラ低下）",
    "SAR": "SARより上で推移",
    "GOLDENCROSS": "ゴールデンクロス（MA20 > MA50）"
}

def compute_technical_score(df: pd.DataFrame, selected_indicators, max_score: int = 30):
    df = df.copy()
    close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
    score = 0
    names = []

    for ind in selected_indicators:
        try:
            if ind == "SMA":
                sma = talib.SMA(close, 20)
                if close.iloc[-1] > sma.iloc[-1]:
                    score += 2; names.append(TECH_LABELS[ind])
            elif ind == "EMA":
                ema = talib.EMA(close, 20)
                if close.iloc[-1] > ema.iloc[-1]:
                    score += 2; names.append(TECH_LABELS[ind])
            elif ind == "MACD":
                macd, signal, hist = talib.MACD(close)
                if macd.iloc[-1] > signal.iloc[-1]:
                    score += 3; names.append(TECH_LABELS[ind])
            elif ind == "RSI":
                rsi = talib.RSI(close, 14)
                if 40 <= rsi.iloc[-1] <= 60:
                    score += 2; names.append(TECH_LABELS[ind])
            elif ind == "BBANDS":
                upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                if close.iloc[-1] > mid.iloc[-1] and mid.diff().iloc[-1] > 0:
                    score += 3; names.append(TECH_LABELS[ind])
            elif ind == "ADX":
                adx = talib.ADX(high, low, close, timeperiod=14)
                if adx.iloc[-1] >= 25:
                    score += 2; names.append(TECH_LABELS[ind])
            elif ind == "OBV":
                obv = talib.OBV(close, vol)
                if obv.diff().iloc[-1] > 0:
                    score += 2; names.append(TECH_LABELS[ind])
            elif ind == "ATR":
                atr = talib.ATR(high, low, close, timeperiod=14)
                if atr.iloc[-1] < atr.rolling(30).mean().iloc[-1]:
                    score += 1; names.append(TECH_LABELS[ind])
            elif ind == "SAR":
                sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
                if close.iloc[-1] > sar.iloc[-1]:
                    score += 2; names.append(TECH_LABELS[ind])
            elif ind == "GOLDENCROSS":
                ma20 = talib.SMA(close, 20)
                ma50 = talib.SMA(close, 50)
                if ma20.iloc[-1] > ma50.iloc[-1] and ma20.diff().iloc[-1] > 0:
                    score += 4; names.append(TECH_LABELS[ind])
        except Exception:
            pass

    score = int(min(score, max_score))
    return score, names
