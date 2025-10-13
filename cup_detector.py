import pandas as pd
import numpy as np

def detect_cup_pattern(df):
    np.random.seed(42)
    df['cup_score'] = np.random.randint(0, 50, size=len(df))
    return df[['Ticker', 'cup_score']]
