import sys, os, sqlite3, time
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import yfinance as yf
import talib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from score_system import compute_total_score

st.set_page_config(page_title="CWH株スキャナ", layout="wide")
DB_PATH = "stocks.db"

# -------------------- DB --------------------
def ensure_db_exists():
    if not os.path.exists(DB_PATH):
        st.error("❌ データベース 'stocks.db' が見つかりません。stock_data.csv をDBに変換して配置してください。")
        st.stop()

def load_filters():
    con = sqlite3.connect(DB_PATH)
    ind = pd.read_sql_query("SELECT DISTINCT industry FROM stocks WHERE industry!='' ORDER BY industry", con)["industry"].tolist()
    topix = pd.read_sql_query("SELECT DISTINCT topix FROM stocks WHERE topix!='' ORDER BY topix", con)["topix"].tolist()
    con.close()
    return ind, topix

def load_tickers(industry_multi, topix_multi):
    con = sqlite3.connect(DB_PATH)
    query = "SELECT code, name, industry, topix FROM stocks WHERE 1=1"
    params = []
    if industry_multi:
        query += " AND industry IN ({})".format(",".join(["?"] * len(industry_multi)))
        params += industry_multi
    if topix_multi:
        query += " AND topix IN ({})".format(",".join(["?"] * len(topix_multi)))
        params += topix_multi
    df = pd.read_sql_query(query, con, params=params)
    con.close()
    return [{"コード": f"{r['code']}.T", "銘柄名": r["name"], "業種": r["industry"], "TOPIX区分": r["topix"]} for _, r in df.iterrows()]

# -------------------- yfinance --------------------
def fetch_history(ticker, period="1y"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty or "Close" not in hist:
            return pd.DataFrame()
        hist.dropna(inplace=True)
        return hist
    except Exception:
        return pd.DataFrame()

# -------------------- カップウィズハンドル検出（強化版） --------------------
def detect_cup_pattern(df):
    if len(df) < 90:
        return False
    close = df["Close"].values
    vol = df["Volume"].values
    ma20 = pd.Series(close).rolling(20).mean().values
    segment = ma20[-90:]
    bottom_idx = np.argmin(segment)
    bottom = segment[bottom_idx]
    if np.isnan(bottom) or bottom <= 0:
        return False
    rise = (segment[-1] - bottom) / bottom
    # ハンドル部の再下降と出来高チェック
    handle = np.mean(segment[-15:]) < np.mean(segment[-30:-15])  # 再下降
    vol_trend = np.mean(vol[-10:]) > np.mean(vol[-30:-10])       # 出来高増加
    return rise > 0.15 and handle and vol_trend  # 25%以上上昇かつ再下降＋出来高増

# -------------------- テクニカル --------------------
def compute_technical_score(df):
    score = 0
    details = []
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi = talib.RSI(close, timeperiod=14)
    if rsi.iloc[-1] < 30: score += 4; details.append("RSI30以下")
    elif rsi.iloc[-1] < 40: score += 2; details.append("RSI40以下")

    macd, signal, _ = talib.MACD(close)
    if macd.iloc[-1] > signal.iloc[-1]: score += 5; details.append("MACD上抜け")

    upper, mid, lower = talib.BBANDS(close)
    if close.iloc[-1] < lower.iloc[-1]: score += 3; details.append("BB下限接触")

    adx = talib.ADX(high, low, close)
    if adx.iloc[-1] > 25: score += 3; details.append("ADX強気")

    sma25, sma75 = talib.SMA(close, 25), talib.SMA(close, 75)
    if sma25.iloc[-1] > sma75.iloc[-1]: score += 4; details.append("GC(25>75)")

    obv = talib.OBV(close, vol)
    if obv.diff().iloc[-1] > 0: score += 2; details.append("OBV上昇")

    return min(score, 25), details

# -------------------- パターン認識（直近10本の出現数） --------------------
def compute_pattern_score(df):
    score = 0
    details = []
    candles = [
        ("CDLENGULFING", "包み足", 4),
        ("CDLHAMMER", "カラカサ", 3),
        ("CDLMORNINGSTAR", "明けの明星", 5),
        ("CDL3WHITESOLDIERS", "白三兵", 5),
        ("CDLPIERCING", "切り込み線", 4),
        ("CDLDRAGONFLYDOJI", "トンボ十字", 3),
        ("CDLTAKURI", "たくり足", 3),
    ]
    for pat, label, val in candles:
        try:
            arr = getattr(talib, pat)(df["Open"], df["High"], df["Low"], df["Close"])
            if (arr.tail(10) > 0).sum() > 0:  # 直近10本で出現
                score += val
                details.append(label)
        except Exception:
            pass
    return min(score, 25), details

# -------------------- 総合スキャン --------------------
def scan(tickers, pause=0.2, progress_cb=None):
    results = []
    def worker(entry):
        tk, name = entry["コード"], entry["銘柄名"]
        hist = fetch_history(tk, "1y")
        if hist.empty: return None

        cup = detect_cup_pattern(hist)
        tech_score, tech_details = compute_technical_score(hist)
        pattern_score, pattern_details = compute_pattern_score(hist)
        total = compute_total_score(
            cup_score=(50 if cup else 0),
            tech_score=tech_score,
            pattern_score=pattern_score
        )
        return {
            "コード": tk.replace(".T",""),
            "銘柄名": name,
            "総合スコア": total,
            "カップ検出": "✔️" if cup else "",
            "カップ(50)": 50 if cup else 0,
            "テクニカル(25)": tech_score,
            "テクニカル指標": "・".join(tech_details) if tech_details else "-",
            "パターン(25)": pattern_score,
            "パターン認識": "・".join(pattern_details) if pattern_details else "-"
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(worker, t): t for t in tickers}
        total = len(futs)
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r: results.append(r)
            if progress_cb: progress_cb(i, total)
            time.sleep(pause)

    df = pd.DataFrame(results)
    return df.sort_values("総合スコア", ascending=False) if not df.empty else df

# -------------------- Streamlit UI --------------------
st.header("📈 カップ・テクニカル・パターン統合スキャナ（v5.2）")
st.caption("カップ検出は25％上昇＋再下降＋出来高増を条件に調整、パターンは直近10本で検出。")

ensure_db_exists()
industries, topix_cats = load_filters()
col1, col2 = st.columns(2)
with col1:
    sel_ind = st.multiselect("業種", industries)
with col2:
    sel_topix = st.multiselect("TOPIX区分", topix_cats)

if st.button("スキャン開始"):
    tickers = load_tickers(sel_ind, sel_topix)
    if not tickers:
        st.warning("条件に一致する銘柄がありません。")
    else:
        prog = st.progress(0); status = st.empty()
        def pcb(d,t): prog.progress(int(100*d/t)); status.write(f"{d}/{t} 件処理中...")
        df = scan(tickers, pause=0.25, progress_cb=pcb)
        if df.empty:
            st.info("結果が空です。")
        else:
            st.subheader("結果一覧（スコア降順）")
            cols = ["コード","銘柄名","総合スコア","カップ検出",
                    "カップ(50)","テクニカル(25)","テクニカル指標",
                    "パターン(25)","パターン認識"]
            st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)
            st.download_button("📥 CSVダウンロード", df.to_csv(index=False).encode("utf-8-sig"), "scan_results.csv")
else:
    st.info("業種・TOPIX区分を選択して『スキャン開始』を押してください。")
