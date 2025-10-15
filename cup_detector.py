import pandas as pd
import numpy as np

def detect_cup_pattern(df: pd.DataFrame,
                       min_len: int = 40,
                       depth_min: float = 0.10, depth_max: float = 0.40,
                       handle_days: int = 10, handle_drop_max: float = 0.15,
                       vol_confirm_window: int = 20) -> bool:
    if df is None or df.empty or len(df) < min_len:
        return False
    if not {"Close","High","Low","Open","Volume"}.issubset(df.columns):
        return False

    close = df["Close"].values.astype(float)
    vol = df["Volume"].values.astype(float)

    n = len(close)
    trough_idx = np.argmin(close)
    trough = close[trough_idx]
    left_max = np.max(close[:max(trough_idx, 1)])
    right_max = np.max(close[trough_idx:])

    peak = max(left_max, right_max) if max(left_max, right_max) > 0 else np.nan
    if not np.isfinite(peak) or peak <= 0:
        return False

    depth = 1.0 - (trough / peak)
    recovery = (close[-1] - trough) / max(trough, 1e-9)
    if not (depth_min <= depth <= depth_max):
        return False
    if recovery < 0.15:
        return False

    if n < handle_days * 2:
        return False
    recent10 = close[-handle_days:]
    prev10 = close[-handle_days*2:-handle_days]
    handle_ok = recent10.mean() <= prev10.mean() * (1.0 - 0.01)
    handle_drop = 1.0 - (recent10.min() / max(prev10.max(), 1e-9))
    if not handle_ok or handle_drop > handle_drop_max:
        return False

    if len(vol) >= vol_confirm_window + 10:
        recent_vol = vol[-10:].mean()
        base_vol = vol[-(vol_confirm_window+10):-10].mean()
        if not (recent_vol > base_vol):
            return False

    return True
