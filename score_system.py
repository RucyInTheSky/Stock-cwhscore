import pandas as pd

def compute_total_score(cup_score, tech_score, pattern_score):
    """
    カップ50 + テクニカル30 + パターン20 = 100点
    """
    total = 0.0
    total += float(cup_score) * 1.0      # 0-50
    total += float(tech_score) * 1.0     # 0-30
    total += float(pattern_score) * 1.0  # 0-20
    return min(total, 100.0)
