import talib
import numpy as np
import pandas as pd

def compute_pattern_score(df, selected_patterns):
    df = df.copy()
    score = 0
    max_score = 25

    for pat in selected_patterns:
        try:
            values = getattr(talib, pat)(df['Open'], df['High'], df['Low'], df['Close'])
            if values.iloc[-1] > 0:
                score += 1.5
            elif values.iloc[-1] < 0:
                score -= 1.5
        except Exception:
            pass

    df['pattern_score'] = np.clip(score, 0, max_score)
    return df[['Ticker', 'pattern_score']]
