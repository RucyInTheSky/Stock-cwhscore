import pandas as pd

def compute_total_score(cup_score, tech_score, pattern_score):
    """
    カップ・テクニカル・パターン認識の3つのスコアを統合して総合点を算出
    """
    total = 0
    total += cup_score * 1.0   # カップ：最大50点
    total += tech_score * 1.0  # テクニカル：最大25点
    total += pattern_score * 1.0  # パターン認識：最大25点
    return min(total, 100)
