import sys, os, sqlite3, time
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import yfinance as yf
import talib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from score_system import compute_total_score
from technical_signals import compute_technical_score
from pattern_signals import compute_pattern_score
from cup_detector import detect_cup_pattern

st.set_page_config(page_title="Stock-cwhscore v4.4 (Swing 2w)", layout="wide")
DB_PATH = "stocks.db"

def ensure_db_exists():
    if not os.path.exists(DB_PATH):
        st.error("❌ データベース 'stocks.db' が見つかりません。")
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

def fetch_history(ticker, period="3mo"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty or "Close" not in hist:
            return pd.DataFrame()
        hist.dropna(inplace=True)
        return hist
    except Exception:
        return pd.DataFrame()

SELECTED_INDICATORS = [
    "SMA", "EMA", "MACD", "RSI", "BBANDS", "ADX", "OBV", "ATR", "SAR",
    "GOLDENCROSS"
]

SELECTED_PATTERNS = [
    "CDLMORNINGSTAR", "CDL3WHITESOLDIERS", "CDLENGULFING",
    "CDLPIERCING", "CDLHAMMER", "CDLDRAGONFLYDOJI",
    "CDL3LINESTRIKE", "CDLMATCHINGLOW",
    "CDLEVENINGSTAR", "CDLDARKCLOUDCOVER", "CDLSHOOTINGSTAR"
]

def scan(tickers, pause=0.15, progress_cb=None):
    results = []
    def worker(entry):
        tk, name = entry["コード"], entry["銘柄名"]
        hist = fetch_history(tk, "3mo")
        if hist.empty: return None

        cup = detect_cup_pattern(hist)
        tech_score, tech_names = compute_technical_score(hist, SELECTED_INDICATORS, max_score=30)
        pattern_score, pattern_names = compute_pattern_score(hist, SELECTED_PATTERNS, max_score=20)
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
            "テクニカル(30)": tech_score,
            "テクニカル指標": "・".join(tech_names) if tech_names else "-",
            "パターン(20)": pattern_score,
            "パターン認識": "・".join(pattern_names) if pattern_names else "-"
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

st.subheader("📈 CWHスコア（v4.4）")
st.caption("カップ50 + テクニカル30 + パターン20 / 期間=直近3ヶ月")

with st.expander("📘 アプリ概要（クリックで展開）"):
    st.caption("""
📈 **Cup-with-Handle Stock Scanner v4.4（Swing Trade 2 Weeks）**

🔍 **概要**
本アプリは、株価チャートパターン「カップ・ウィズ・ハンドル」をベースに、
有名テクニカル指標およびローソク足パターン認識を組み合わせて、
短期スイングトレード（2週間前後）で上昇が期待できる銘柄を
スコア化・ランキングするStreamlitアプリです。

🧠 **主要ロジック構成**
- **app.py**：Streamlit UI・DBから銘柄読み込み・スキャン統括  
- **cup_detector.py**：カップ・ウィズ・ハンドル形状解析（深さ・回復率・出来高）  
- **technical_signals.py**：代表的なテクニカル9種＋ゴールデンクロスを解析  
- **pattern_signals.py**：TA-Libの12種ローソク足パターン（陽線・陰線）を検出  
- **score_system.py**：カップ50＋テクニカル30＋パターン20＝総合100点制  

⚙️ **スコア構成**
- 🏆 カップ検出（50点）：深さ10〜40%、回復率15%以上、出来高上昇、ハンドル形成  
- 📊 テクニカル（30点）：SMA・EMA・MACD・RSI(40–60)・BB・ADX・OBV・ATR・SAR・GC  
- 💡 パターン認識（20点）：明けの明星・白三兵・包み足・カラカサ・宵の明星・暗雲など12種  

🧾 **動作仕様**
- 期間：直近3ヶ月固定（短期トレンド重視）  
- 対象：SQLite stocks.db 内の銘柄リスト  
- 出力：総合スコア降順のDataFrame＋CSVダウンロード  
- フィルタ：業種・TOPIX区分を複数選択可能  

📈 **期待できる動作結果**
本モデルは「急騰直前の静かな上昇局面」を重視しており、
直近の過熱銘柄よりも、なだらかにトレンド形成中の銘柄を上位に抽出します。
また、直近2週間での上昇余地を狙うスイングトレードに最適化されています。
""")

ensure_db_exists()
industries, topix_cats = load_filters()
col1, col2 = st.columns(2)
with col1:
    sel_ind = st.multiselect("業種", industries)
with col2:
    sel_topix = st.multiselect("マーケット区分", topix_cats)



if st.button("スキャン開始"):
    tickers = load_tickers(sel_ind, sel_topix)
    if not tickers:
        st.warning("条件に一致する銘柄がありません。")
    else:
        prog = st.progress(0); status = st.empty()
        def pcb(d,t): prog.progress(int(100*d/t)); status.write(f"{d}/{t} 件処理中...")
        df = scan(tickers, pause=0.15, progress_cb=pcb)
        if df.empty:
            st.info("結果が空です。")
        else:
            st.subheader("結果一覧（スコア降順）")
            cols = ["コード","銘柄名","総合スコア","カップ検出",
                    "カップ(50)","テクニカル(30)","テクニカル指標",
                    "パターン(20)","パターン認識"]
            st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)
            st.download_button("📥 CSVダウンロード", df.to_csv(index=False).encode("utf-8-sig"), "scan_results.csv")
else:
    st.info("業種・マーケット区分を選択して『スキャン開始』を押してください。")
