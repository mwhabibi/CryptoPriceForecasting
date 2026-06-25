import os
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
from utils import COINS, format_price, prepare_model_input, get_data_with_indicators

# Root project = dua level di atas pages/Prediction.py  →  pages/../  = root
# Streamlit selalu menjalankan dari root, tapi Path(__file__) lebih aman.
ROOT_DIR = Path(__file__).resolve().parent.parent  # …/pages/../  = project root

# ─── 1. CONFIG & STATE ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hasil Prediksi AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "selected_coin" not in st.session_state:
    st.warning("⚠️ Tidak ada aset yang dipilih. Mengarahkan ke Beranda...")
    time.sleep(1.5)
    st.switch_page("Home.py")
    st.stop()

selected_coin = st.session_state["selected_coin"]
coin_name     = COINS.get(selected_coin, selected_coin.replace("-USD", ""))
ticker        = selected_coin.replace("-USD", "")

# ─── 2. CSS — DESIGN SYSTEM KONSISTEN ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #080B10;
    --bg2:       #0D1117;
    --surface:   #111820;
    --surface2:  #141D28;
    --border:    #1E2733;
    --border2:   #2A3544;
    --text:      #E8EDF5;
    --muted:     #5A6A7E;
    --muted2:    #7A8A9E;
    --accent:    #00D4FF;
    --accent2:   #0099CC;
    --green:     #00E5A0;
    --red:       #FF4D6A;
    --gold:      #FFB800;
    --purple:    #A78BFA;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }

/* ── Tombol Global ── */
div[data-testid="stButton"] button {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    color: var(--muted2) !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,212,255,0.06) !important;
}

/* ── Page Header ── */
.page-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 60%, #061020 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.page-header::after {
    content: ''; position: absolute; bottom: -60px; left: 25%;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,229,160,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.ph-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,229,160,0.1);
    border: 1px solid rgba(0,229,160,0.25);
    border-radius: 100px;
    padding: 4px 12px;
    font-family: var(--font-body);
    font-size: 11px; font-weight: 600;
    color: var(--green);
    letter-spacing: 0.05em; text-transform: uppercase;
    margin-bottom: 14px;
}
.ph-badge .dot {
    width: 5px; height: 5px;
    background: var(--green); border-radius: 50%;
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.ph-title {
    font-family: var(--font-head);
    font-size: 26px; font-weight: 800;
    color: var(--text); margin: 0 0 6px 0;
}
.ph-title span { color: var(--accent); }
.ph-sub {
    font-family: var(--font-body);
    font-size: 13.5px; color: var(--muted); line-height: 1.6;
}

/* ── Section Header ── */
.section-header {
    font-family: var(--font-head); font-size: 11px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
    margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Metrik Model ── */
.model-metrics-row {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px;
}
.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; position: relative; overflow: hidden;
}
.metric-card::before { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; }
.metric-card.mc-blue::before   { background: linear-gradient(90deg, var(--accent), transparent); }
.metric-card.mc-gold::before   { background: linear-gradient(90deg, var(--gold), transparent); }
.metric-card.mc-purple::before { background: linear-gradient(90deg, var(--purple), transparent); }
.mc-label { font-family: var(--font-body); font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
.mc-value { font-family: var(--font-head); font-size: 22px; font-weight: 800; color: var(--text); margin-bottom: 4px; }
.mc-sub   { font-family: var(--font-body); font-size: 11.5px; color: var(--muted); }
.quality-badge {
    display: inline-block; padding: 3px 10px; border-radius: 6px;
    font-family: var(--font-mono); font-size: 10px; font-weight: 500; margin-top: 6px;
}
.quality-excellent { background: rgba(0,229,160,0.12); color: var(--green); border: 1px solid rgba(0,229,160,0.2); }
.quality-good      { background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.2); }
.quality-fair      { background: rgba(255,184,0,0.12); color: var(--gold);   border: 1px solid rgba(255,184,0,0.2); }
.quality-poor      { background: rgba(255,77,106,0.12); color: var(--red);   border: 1px solid rgba(255,77,106,0.2); }

/* ── Summary Card (Ringkasan Prediksi) ── */
.summary-card {
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.summary-card.bullish {
    background: linear-gradient(135deg, rgba(0,229,160,0.08) 0%, rgba(0,229,160,0.03) 100%);
    border: 1px solid rgba(0,229,160,0.25);
}
.summary-card.bearish {
    background: linear-gradient(135deg, rgba(255,77,106,0.08) 0%, rgba(255,77,106,0.03) 100%);
    border: 1px solid rgba(255,77,106,0.25);
}
.summary-grid { display: grid; grid-template-columns: auto 1fr auto; gap: 32px; align-items: center; }
.summary-verdict {
    font-family: var(--font-head); font-size: 40px; font-weight: 800;
    line-height: 1;
}
.summary-verdict.bullish { color: var(--green); }
.summary-verdict.bearish { color: var(--red); }
.summary-verdict-label {
    font-family: var(--font-body); font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-top: 4px;
}
.summary-verdict-label.bullish { color: rgba(0,229,160,0.7); }
.summary-verdict-label.bearish { color: rgba(255,77,106,0.7); }
.summary-narrative {
    font-family: var(--font-body); font-size: 14px; color: var(--muted2); line-height: 1.75;
}
.summary-narrative b { color: var(--text); }
.summary-stats { display: flex; flex-direction: column; gap: 10px; }
.sum-stat { text-align: right; }
.sum-stat-label { font-family: var(--font-body); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.sum-stat-value { font-family: var(--font-mono); font-size: 14px; color: var(--text); font-weight: 500; }

/* ── Prediksi Tabel ── */
.pred-table-wrap {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 24px;
}
.pred-table { width: 100%; border-collapse: collapse; }
.pred-table thead tr { background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--border); }
.pred-table th {
    padding: 13px 20px; font-family: var(--font-body); font-size: 11px; font-weight: 600;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; text-align: right;
}
.pred-table th:first-child { text-align: left; }
.pred-table td {
    padding: 14px 20px; font-family: var(--font-mono); font-size: 13px; color: var(--text);
    text-align: right; border-bottom: 1px solid rgba(30,39,51,0.6);
}
.pred-table td:first-child { text-align: left; font-family: var(--font-body); font-weight: 500; }
.pred-table tbody tr:last-child td { border-bottom: none; }
.pred-table tbody tr:hover { background: rgba(255,255,255,0.02); }
.pred-table tbody tr.row-peak { background: rgba(0,229,160,0.04); }
.pred-table tbody tr.row-nadir { background: rgba(255,77,106,0.04); }

.up-badge { display: inline-flex; align-items: center; gap: 4px; background: rgba(0,229,160,0.12); color: var(--green); padding: 3px 8px; border-radius: 5px; font-size: 12px; }
.dn-badge { display: inline-flex; align-items: center; gap: 4px; background: rgba(255,77,106,0.12); color: var(--red);   padding: 3px 8px; border-radius: 5px; font-size: 12px; }
.tag-peak  { display: inline-block; margin-left: 8px; background: rgba(0,229,160,0.15); color: var(--green); font-size: 9px; padding: 2px 6px; border-radius: 4px; font-family: var(--font-body); font-weight: 600; text-transform: uppercase; }
.tag-nadir { display: inline-block; margin-left: 8px; background: rgba(255,77,106,0.15); color: var(--red);   font-size: 9px; padding: 2px 6px; border-radius: 4px; font-family: var(--font-body); font-weight: 600; text-transform: uppercase; }
.tag-start { display: inline-block; margin-left: 8px; background: rgba(0,212,255,0.12); color: var(--accent); font-size: 9px; padding: 2px 6px; border-radius: 4px; font-family: var(--font-body); font-weight: 600; text-transform: uppercase; }

/* ── Disclaimer ── */
.disclaimer-banner {
    background: rgba(255,184,0,0.06);
    border: 1px solid rgba(255,184,0,0.2);
    border-radius: 12px;
    padding: 18px 22px;
    display: flex; gap: 14px; align-items: flex-start;
    margin-bottom: 24px;
}
.disc-icon { font-size: 20px; flex-shrink: 0; margin-top: 1px; }
.disc-title { font-family: var(--font-head); font-size: 12px; font-weight: 700; color: var(--gold); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 5px; }
.disc-text  { font-family: var(--font-body); font-size: 12.5px; color: var(--muted2); line-height: 1.7; }
.disc-text b { color: var(--text); }

/* ── Loading Steps ── */
.loading-step {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; border-radius: 8px;
    margin-bottom: 8px;
    background: var(--surface); border: 1px solid var(--border);
    font-family: var(--font-body); font-size: 13px; color: var(--muted);
}
.ls-icon { font-size: 16px; flex-shrink: 0; }
.ls-done { color: var(--green); }

/* ── Footer ── */
.footer {
    border-top: 1px solid var(--border); padding-top: 20px; margin-top: 12px;
    display: flex; gap: 24px;
}
.footer-col { flex: 1; min-width: 220px; }
.footer-label { font-family: var(--font-head); font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.footer-text  { font-family: var(--font-body); font-size: 12.5px; color: #374151; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ─── 3. HELPER ────────────────────────────────────────────────────────────────
# Root project: pages/Prediction.py -> parent -> parent = root folder
ROOT_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_metrics():
    """Baca metrics.json dari root project, aman dari manapun dijalankan."""
    metrics_path = ROOT_DIR / "metrics.json"
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # Ditangani di UI
    except json.JSONDecodeError:
        return {}

@st.cache_resource
def load_ml_assets(t):
    try:
        model  = load_model(str(ROOT_DIR / "models" / f"{t}_best_model.keras"))
        scaler = joblib.load(str(ROOT_DIR / "scalers" / f"{t}_scaler.pkl"))
        return model, scaler
    except Exception:
        return None, None

def mape_quality(mape):
    """Kembalikan (label, css_class) berdasarkan MAPE."""
    if mape < 5:    return "Sangat Akurat",   "quality-excellent"
    if mape < 10:   return "Akurat",          "quality-good"
    if mape < 20:   return "Cukup Akurat",    "quality-fair"
    return          "Perlu Evaluasi",          "quality-poor"

def rmse_context(rmse, current_price):
    """Persentase RMSE terhadap harga sekarang — kontekstualisasi."""
    if current_price and current_price > 0:
        pct = (rmse / current_price) * 100
        return f"≈{pct:.2f}% dari harga saat ini"
    return "Tidak dapat dihitung"

# ─── 4. NAVIGASI ATAS ─────────────────────────────────────────────────────────
col_back, _, col_doc, col_refresh = st.columns([2, 5, 2.5, 2])
with col_back:
    if st.button("← Kembali ke Detail", use_container_width=True):
        st.switch_page("pages/Detail.py")
with col_doc:
    if st.button("📋 Performa Model", use_container_width=True):
        st.switch_page("pages/Documentation.py")
with col_refresh:
    if st.button("🔄 Jalankan Ulang", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── 5. PAGE HEADER ───────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="ph-badge"><span class="dot"></span>Hasil Prediksi AI · LSTM · 7 Hari ke Depan</div>
    <div class="ph-title">Prediksi Harga <span>{coin_name}</span></div>
    <div class="ph-sub">
        Model <b style="color:var(--text)">Long Short-Term Memory (LSTM)</b> telah memproses data historis, 
        indikator teknikal (RSI, MACD, ATR), dan pola volatilitas untuk menghasilkan estimasi 
        harga penutupan 7 hari ke depan. Semua angka bersifat estimasi, bukan kepastian.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 6. METRIK MODEL ──────────────────────────────────────────────────────────
metrics_data = load_metrics()
coin_metrics = metrics_data.get(selected_coin, {})
rmse_val     = float(coin_metrics.get("LSTM_RMSE") or 0)
mape_val     = float(coin_metrics.get("LSTM_MAPE") or 0)

from utils import get_market_summary
raw_market, _ = get_market_summary()
coin_info     = next((d for d in raw_market if d.get("Ticker") == selected_coin), None)
current_price = coin_info.get("Price", 0) if coin_info else 0
icon_src      = coin_info.get("Icon", "") if coin_info else ""

quality_label, quality_cls = mape_quality(mape_val)
rmse_ctx = rmse_context(rmse_val, current_price)

# Peringatan jika metrics.json kosong / tidak ditemukan
if not metrics_data:
    st.warning(
        "⚠️ File `metrics.json` tidak ditemukan di root project. "
        "Jalankan script evaluasi model terlebih dahulu agar skor RMSE & MAPE tampil."
    )

st.markdown('<div class="section-header">🎯 Evaluasi Performa Model LSTM</div>', unsafe_allow_html=True)

# Render 3 card pakai st.columns agar tidak ada bug HTML rendering
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown(f"""
    <div class="metric-card mc-blue">
        <div class="mc-label">Root Mean Square Error (RMSE)</div>
        <div class="mc-value">{format_price(rmse_val)}</div>
        <div class="mc-sub">{rmse_ctx}</div>
        <div style="margin-top:10px;font-family:var(--font-body);font-size:12px;color:var(--muted);line-height:1.6;">
            RMSE mengukur rata-rata <b style="color:var(--text)">jarak kesalahan prediksi</b>
            dalam satuan dolar — semakin kecil nilainya, semakin presisi model.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card mc-gold">
        <div class="mc-label">Mean Absolute Percentage Error (MAPE)</div>
        <div class="mc-value">{mape_val:.2f}%</div>
        <div class="mc-sub">Persentase rata-rata kesalahan prediksi</div>
        <div class="quality-badge {quality_cls}">{quality_label}</div>
        <div style="margin-top:10px;font-family:var(--font-body);font-size:12px;color:var(--muted);line-height:1.6;">
            Interpretasi: &lt;5% sangat akurat, &lt;10% akurat, &lt;20% cukup baik, &ge;20% perlu evaluasi ulang.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── 7. PROSES PREDIKSI DENGAN LOADING STEPS ──────────────────────────────────
model, scaler = load_ml_assets(selected_coin)

if model is None or scaler is None:
    st.error(f"❌ Model atau Scaler untuk **{ticker}** tidak ditemukan. Pastikan folder `models/` dan `scalers/` sudah berisi file yang benar.")
    st.stop()

# Placeholder loading bertahap
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
    <div style="padding:24px;background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:16px;">
        <div style="font-family:var(--font-head);font-size:14px;font-weight:700;color:var(--text);margin-bottom:16px;">
            ⚙️ Menjalankan Pipeline Prediksi LSTM...
        </div>
        <div class="loading-step"><span class="ls-icon">📥</span> Langkah 1/3 — Mengambil data historis 365 hari terakhir...</div>
        <div class="loading-step"><span class="ls-icon">🔧</span> Langkah 2/3 — Menghitung indikator teknikal (RSI, MACD, ATR)...</div>
        <div class="loading-step"><span class="ls-icon">🧠</span> Langkah 3/3 — Menjalankan model LSTM untuk 7 prediksi ke depan...</div>
    </div>
    """, unsafe_allow_html=True)

# Prediksi
end_date   = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
df = get_data_with_indicators(selected_coin, start_date, end_date, interval="1d")
recent_data, last_price, last_date, FEATURES = prepare_model_input(selected_coin, lookback=60)

if recent_data is None or df.empty:
    loading_placeholder.empty()
    st.error("❌ Data historis tidak mencukupi untuk menjalankan prediksi. Coba lagi nanti.")
    st.stop()

scaled_data = scaler.transform(recent_data)
X_input     = scaled_data.reshape(1, 60, len(FEATURES))
pred_scaled = model.predict(X_input, verbose=0)[0]

dummy_array        = np.zeros((7, len(FEATURES)))
dummy_array[:, 0]  = pred_scaled
pred_log_ret       = scaler.inverse_transform(dummy_array)[:, 0]

future_dates  = []
future_prices = []
changes_pct   = []
changes_abs   = []
current_p     = last_price

for i, log_r in enumerate(pred_log_ret):
    next_p  = current_p * np.exp(log_r)
    change  = ((next_p - current_p) / current_p) * 100
    abs_chg = next_p - last_price

    future_prices.append(next_p)
    changes_pct.append(change)
    changes_abs.append(abs_chg)
    future_dates.append(last_date + timedelta(days=i + 1))
    current_p = next_p

loading_placeholder.empty()

# ─── 8. RINGKASAN PREDIKSI (SUMMARY CARD) ────────────────────────────────────
total_change   = ((future_prices[-1] - last_price) / last_price) * 100
is_bullish     = total_change >= 0
trend_word     = "BULLISH ↑" if is_bullish else "BEARISH ↓"
trend_cls      = "bullish" if is_bullish else "bearish"
peak_idx       = int(np.argmax(future_prices))
nadir_idx      = int(np.argmin(future_prices))
peak_price     = future_prices[peak_idx]
nadir_price    = future_prices[nadir_idx]
peak_date_str  = future_dates[peak_idx].strftime("%d %b")
nadir_date_str = future_dates[nadir_idx].strftime("%d %b")

sign_total = "+" if total_change >= 0 else ""
narrative = (
    f"Berdasarkan pola data historis <b>{coin_name}</b> selama 60 hari terakhir, "
    f"model LSTM memperkirakan pergerakan harga ke arah "
    f"<b>{'naik (bullish)' if is_bullish else 'turun (bearish)'}</b> "
    f"dengan estimasi perubahan total <b>{sign_total}{total_change:.2f}%</b> dalam 7 hari ke depan. "
    f"Harga tertinggi diprediksi mencapai <b>{format_price(peak_price)}</b> pada <b>{peak_date_str}</b>, "
    f"sedangkan harga terendah diperkirakan <b>{format_price(nadir_price)}</b> pada <b>{nadir_date_str}</b>."
)

st.markdown('<div class="section-header">📊 Ringkasan Prediksi Otomatis</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="summary-card {trend_cls}">
    <div class="summary-grid">
        <div>
            <div class="summary-verdict {trend_cls}">{sign_total}{total_change:.1f}%</div>
            <div class="summary-verdict-label {trend_cls}">{trend_word}</div>
        </div>
        <div class="summary-narrative">{narrative}</div>
        <div class="summary-stats">
            <div class="sum-stat">
                <div class="sum-stat-label">Harga Awal (Hari Ini)</div>
                <div class="sum-stat-value">{format_price(last_price)}</div>
            </div>
            <div class="sum-stat">
                <div class="sum-stat-label">Estimasi Harga Akhir (Hari 7)</div>
                <div class="sum-stat-value">{format_price(future_prices[-1])}</div>
            </div>
            <div class="sum-stat">
                <div class="sum-stat-label">Puncak / Nadir Prediksi</div>
                <div class="sum-stat-value" style="color:var(--green);">{format_price(peak_price)}</div>
                <div class="sum-stat-value" style="color:var(--red);">{format_price(nadir_price)}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 9. GRAFIK PREDIKSI ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Visualisasi Prediksi vs Historis</div>', unsafe_allow_html=True)

# Tentukan format ticker harga
last_p_chart = df["Close"].iloc[-1]
if   last_p_chart < 0.0000001: tick_fmt = ".12f"
elif last_p_chart < 1:         tick_fmt = ".8f"
else:                          tick_fmt = ",.2f"

fig = go.Figure()

# ── Historis ──
hist_x = list(df.index[-60:])
hist_y = list(df["Close"].iloc[-60:])
fig.add_trace(go.Scatter(
    x=hist_x, y=hist_y,
    mode="lines", name="Harga Historis",
    line=dict(color="#00D4FF", width=2),
    fill="tozeroy", fillcolor="rgba(0,212,255,0.04)",
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Harga: <b>$%{y:,.6g}</b><extra>Historis</extra>",
))

# ── Confidence zone (±MAPE%) ──
conf_upper = [p * (1 + mape_val / 100) for p in [last_price] + future_prices]
conf_lower = [p * (1 - mape_val / 100) for p in [last_price] + future_prices]
conf_x     = [last_date] + future_dates

fig.add_trace(go.Scatter(
    x=conf_x + conf_x[::-1],
    y=conf_upper + conf_lower[::-1],
    fill="toself",
    fillcolor="rgba(255,184,0,0.07)",
    line=dict(width=0),
    showlegend=True, name=f"Zona Ketidakpastian (±{mape_val:.1f}% MAPE)",
    hoverinfo="skip",
))

# ── Garis prediksi ──
pred_x = [last_date] + future_dates
pred_y = [last_price] + future_prices
fig.add_trace(go.Scatter(
    x=pred_x, y=pred_y,
    mode="lines+markers", name="Prediksi LSTM",
    line=dict(color="#FFB800", width=2.5, dash="dash"),
    marker=dict(size=8, color="#FFB800", line=dict(color="#080B10", width=2)),
    hovertemplate="<b>%{x|%d %b %Y}</b><br>Prediksi: <b>$%{y:,.6g}</b><extra>Prediksi LSTM</extra>",
))

# ── Marker PEAK ──
fig.add_trace(go.Scatter(
    x=[future_dates[peak_idx]], y=[peak_price],
    mode="markers+text", name="Harga Tertinggi",
    marker=dict(size=14, color="#00E5A0", symbol="triangle-up", line=dict(color="#080B10", width=2)),
    text=[f"  Maks: {format_price(peak_price)}"],
    textposition="top center",
    textfont=dict(color="#00E5A0", size=11, family="DM Mono"),
    hovertemplate=f"<b>Prediksi Tertinggi</b><br>{peak_date_str}: <b>{format_price(peak_price)}</b><extra></extra>",
))

# ── Marker NADIR ──
fig.add_trace(go.Scatter(
    x=[future_dates[nadir_idx]], y=[nadir_price],
    mode="markers+text", name="Harga Terendah",
    marker=dict(size=14, color="#FF4D6A", symbol="triangle-down", line=dict(color="#080B10", width=2)),
    text=[f"  Min: {format_price(nadir_price)}"],
    textposition="bottom center",
    textfont=dict(color="#FF4D6A", size=11, family="DM Mono"),
    hovertemplate=f"<b>Prediksi Terendah</b><br>{nadir_date_str}: <b>{format_price(nadir_price)}</b><extra></extra>",
))

# ── Garis batas prediksi ──
fig.add_vline(
    x=last_date, line_width=1.5,
    line_dash="dot", line_color="rgba(90,106,126,0.6)",
)
fig.add_annotation(
    x=last_date, y=1, yref="paper",
    text="Hari Ini ➔",
    showarrow=False,
    font=dict(color="#5A6A7E", size=11, family="DM Sans"),
    xanchor="right", yanchor="top",
    bgcolor="rgba(8,11,16,0.7)",
    borderpad=4,
)

fig.update_layout(
    height=480,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#5A6A7E", size=11),
    margin=dict(l=8, r=90, t=20, b=8),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#111820", bordercolor="#2A3544",
        font=dict(color="#E8EDF5", size=12, family="DM Sans"), align="left",
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="left",
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7A8A9E", size=11, family="DM Sans"),
    ),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(30,39,51,0.9)", gridwidth=1,
        zeroline=False, color="#5A6A7E", showline=False,
        rangeslider=dict(visible=False), type="date",
        tickformat="%d %b", tickfont=dict(family="DM Mono", size=10),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(30,39,51,0.9)", gridwidth=1,
        zeroline=False, color="#5A6A7E", showline=False,
        side="right", tickprefix="$", tickformat=tick_fmt, nticks=7,
        tickfont=dict(family="DM Mono", size=10),
    ),
)

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": f"{ticker}_prediction", "scale": 2},
})

# ─── 10. TABEL PREDIKSI DETAIL ────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Tabel Estimasi Harga Harian</div>', unsafe_allow_html=True)

rows_html = ""
for i in range(7):
    date_str   = future_dates[i].strftime("%A, %d %b %Y")
    price_str  = format_price(future_prices[i])
    chg_day    = changes_pct[i]
    chg_total  = ((future_prices[i] - last_price) / last_price) * 100
    sign_d     = "+" if chg_day >= 0 else ""
    sign_t     = "+" if chg_total >= 0 else ""
    badge_cls  = "up-badge" if chg_day >= 0 else "dn-badge"
    arrow      = "▲" if chg_day >= 0 else "▼"

    # Tag khusus
    extra_tag = ""
    row_cls   = ""
    if i == peak_idx:
        extra_tag = '<span class="tag-peak">▲ Puncak</span>'
        row_cls   = "row-peak"
    elif i == nadir_idx:
        extra_tag = '<span class="tag-nadir">▼ Nadir</span>'
        row_cls   = "row-nadir"
    if i == 0:
        extra_tag += '<span class="tag-start">Hari 1</span>'

    rows_html += f"""
    <tr class="{row_cls}">
        <td>Hari {i+1} &nbsp;·&nbsp; {date_str}{extra_tag}</td>
        <td><b>{price_str}</b></td>
        <td><span class="{badge_cls}">{arrow} {sign_d}{chg_day:.2f}%</span></td>
        <td style="color:{'var(--green)' if chg_total>=0 else 'var(--red)'};">{sign_t}{chg_total:.2f}%</td>
        <td style="color:var(--muted);font-size:11px;">{format_price(abs(future_prices[i] - last_price))}</td>
    </tr>"""

st.markdown(f"""
<div class="pred-table-wrap">
    <table class="pred-table">
        <thead>
            <tr>
                <th>Tanggal</th>
                <th>Estimasi Harga</th>
                <th>Perubahan Harian</th>
                <th>Perubahan vs Hari Ini</th>
                <th>Selisih Absolut</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
""", unsafe_allow_html=True)

# ─── 11. DISCLAIMER PROMINENT ─────────────────────────────────────────────────
st.markdown(f"""
<div class="disclaimer-banner">
    <div class="disc-icon">⚠️</div>
    <div>
        <div class="disc-title">Peringatan Penting — Bukan Saran Finansial</div>
        <div class="disc-text">
            Seluruh prediksi yang ditampilkan dihasilkan oleh model <b>LSTM berbasis data historis</b> dan 
            bersifat <b>estimasi statistik</b>, bukan jaminan pergerakan harga di masa depan. 
            Pasar kripto sangat dipengaruhi oleh sentimen publik, regulasi, dan faktor makroekonomi 
            yang <b>tidak dapat dimodelkan oleh data historis semata</b>. 
            Lakukan riset mandiri (<i>DYOR — Do Your Own Research</i>) sebelum mengambil keputusan investasi apapun.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 12. FOOTER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-col">
        <div class="footer-label">⚙️ Informasi Sistem</div>
        <div class="footer-text">
            Halaman ini memvisualisasikan output model LSTM yang telah dilatih dengan data historis 
            aset kripto. Evaluasi performa menggunakan metrik RMSE dan MAPE sesuai standar penelitian.
        </div>
    </div>
    <div class="footer-col">
        <div class="footer-label">📚 Konteks Tugas Akhir</div>
        <div class="footer-text">
            Dashboard ini merupakan implementasi sistem prediksi berbasis Deep Learning (LSTM) 
            yang dikembangkan sebagai bagian dari penelitian Tugas Akhir tentang visualisasi 
            prediksi harga kripto yang interaktif dan mudah dipahami.
        </div>
    </div>
    <div class="footer-col">
        <div class="footer-label">⚠️ Disclaimer</div>
        <div class="footer-text">
            Konten bersifat informatif untuk keperluan akademis. 
            Bukan saran investasi atau rekomendasi perdagangan kripto.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)