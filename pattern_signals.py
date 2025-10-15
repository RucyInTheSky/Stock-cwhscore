import talib
import numpy as np
import pandas as pd

PATTERN_LABELS = {
    "CDLMORNINGSTAR": "明けの明星",
    "CDL3WHITESOLDIERS": "白三兵",
    "CDLENGULFING": "包み足（陽線）",
    "CDLPIERCING": "切り込み線",
    "CDLHAMMER": "カラカサ（下ヒゲ陽線）",
    "CDLDRAGONFLYDOJI": "トンボ十字",
    "CDL3LINESTRIKE": "スリーストライク（陽線）",
    "CDLMATCHINGLOW": "マッチングロー",
    "CDLEVENINGSTAR": "宵の明星",
    "CDLDARKCLOUDCOVER": "暗雲",
    "CDLSHOOTINGSTAR": "射撃星"
}

POSITIVE = {
    "CDLMORNINGSTAR": 5,
    "CDL3WHITESOLDIERS": 6,
    "CDLENGULFING": 4,
    "CDLPIERCING": 4,
    "CDLHAMMER": 4,
    "CDLDRAGONFLYDOJI": 3,
    "CDL3LINESTRIKE": 3,
    "CDLMATCHINGLOW": 3,
}

NEGATIVE = {
    "CDLEVENINGSTAR": -5,
    "CDLDARKCLOUDCOVER": -4,
    "CDLSHOOTINGSTAR": -3,
}

def compute_pattern_score(df: pd.DataFrame, selected_patterns, max_score: int = 20):
    if df is None or df.empty:
        return 0, []

    score = 0
    names = []

    for pat in selected_patterns:
        try:
            res = getattr(talib, pat)(df["Open"], df["High"], df["Low"], df["Close"])
            pos_count = (res.tail(10) > 0).sum()
            neg_count = (res.tail(10) < 0).sum()

            if pat in POSITIVE and pos_count > 0:
                score += POSITIVE[pat]
                names.append(PATTERN_LABELS.get(pat, pat))
            if pat in NEGATIVE and neg_count > 0:
                score += NEGATIVE[pat]
                names.append(PATTERN_LABELS.get(pat, pat))
        except Exception:
            pass

    score = int(np.clip(score, 0, max_score))
    return score, names
