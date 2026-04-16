import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
from utils import COINS, format_price, prepare_model_input, get_data_with_indicators

# --- 1. CONFIG & STATE ---
st.set_page_config(page_title="Prediction Result", layout="wide", initial_sidebar_state="collapsed")

if 'selected_coin' not in st.session_state:
    st.warning("⚠️ No cryptocurrency selected. Returning to Home...")
    st.switch_page("Home.py")

selected_coin = st.session_state['selected_coin']

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .header-title { font-size: 28px; font-weight: bold; }
    .metric-container { margin-bottom: 20px; }
    .metric-title { font-size: 14px; color: #8B949E; font-weight: bold; }
    .metric-value { font-size: 16px; color: #E3B341; font-weight: bold; }
    .metric-sub { font-size: 12px; color: #8B949E; font-style: italic; }
    .pred-table { width: 100%; border-collapse: collapse; margin-top: 15px; color: white; font-size: 14px; }
    .pred-table th { text-align: right; padding: 12px; border-bottom: 1px solid #30363D; color: #8B949E; }
    .pred-table th:first-child { text-align: left; }
    .pred-table td { text-align: right; padding: 12px; border-bottom: 1px solid #21262D; }
    .pred-table td:first-child { text-align: left; font-weight: bold; }
    .change-up { color: #00FF00; }
    .change-down { color: #FF4B4B; }
    .footer { display: flex; justify-content: space-between; font-size: 12px; color: #8B949E; margin-top: 40px; border-top: 1px solid #30363D; padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_data
def load_metrics():
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {}

metrics_data = load_metrics()
coin_metrics = metrics_data.get(selected_coin, {"RMSE": 0, "MAPE": 0})

@st.cache_resource
def load_ml_assets(ticker):
    try:
        model = load_model(f"models/{ticker}_best_model.keras")
        scaler = joblib.load(f"scalers/{ticker}_scaler.pkl")
        return model, scaler
    except:
        return None, None

# --- 4. HEADER ---
col_title, col_space, col_btn1, col_btn2, col_btn3 = st.columns([5, 1, 2, 2, 2])
with col_title:
    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 10px;'> 
            <div class='header-title'> Prediction Result</div>
            <div style='font-size: 28px; font-weight: bold; color: #00FCE7;'>{selected_coin}</div>
        </div>
    """, unsafe_allow_html=True)
with col_btn1:
    if st.button("❮ Back To Previous", width='stretch'):
        st.switch_page("pages/Detail.py")
with col_btn2:
    if st.button("Model Performance", width='stretch'):
        st.switch_page("pages/Documentation.py")
with col_btn3:
    if st.button("Re-Analysis (Re-Run)", type="primary", width='stretch'):
        st.rerun()

# Metrics Display
col_m1, col_m2, col_m3 = st.columns([2, 2, 6])
with col_m1:
    st.markdown(f'<div class="metric-container"><span class="metric-title">RMSE Score:</span> <span class="metric-value">{format_price(coin_metrics["RMSE"])}</span><br><span class="metric-sub">Lower is better</span></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown(f'<div class="metric-container"><span class="metric-title">MAPE Score:</span> <span class="metric-value">{coin_metrics["MAPE"]}%</span><br><span class="metric-sub">Average error percentage</span></div>', unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #161B22; margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# --- 5. LOGIKA PREDIKSI ---
model, scaler = load_ml_assets(selected_coin)

if model is None or scaler is None:
    st.error("Model atau Scaler tidak ditemukan. Periksa folder models/ dan scalers/.")
    st.stop()

with st.spinner("Memproses algoritma LSTM..."):
    # Kita ambil data full (df) untuk chart, dan recent_data untuk model
    # Menggunakan fungsi yang sudah kita perbaiki di utils
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
    df = get_data_with_indicators(selected_coin, start_date, end_date, interval="1d")
    
    recent_data, last_price, last_date, FEATURES = prepare_model_input(selected_coin, lookback=60)
    
    if recent_data is None or df.empty:
        st.error("Data tidak mencukupi.")
        st.stop()

    # Prediksi
    scaled_data = scaler.transform(recent_data)
    X_input = scaled_data.reshape(1, 60, len(FEATURES))
    pred_scaled = model.predict(X_input, verbose=0)[0]
    
    dummy_array = np.zeros((7, len(FEATURES)))
    dummy_array[:, 0] = pred_scaled
    pred_log_ret = scaler.inverse_transform(dummy_array)[:, 0]
    
    future_dates = []
    future_prices = []
    changes_pct = []
    current_p = last_price
    
    for i, log_r in enumerate(pred_log_ret):
        next_p = current_p * np.exp(log_r)
        # Hitung persentase perubahan dari hari sebelumnya
        change = ((next_p - current_p) / current_p) * 100 
        
        future_prices.append(next_p)
        changes_pct.append(change)
        future_dates.append(last_date + timedelta(days=i+1))
        
        current_p = next_p

# --- 6. VISUALISASI CHART ---
fig = go.Figure()
# Grafik Historis (Menggunakan df yang baru kita ambil di atas)
fig.add_trace(go.Scatter(
    x=df.index[-60:], y=df['Close'].iloc[-60:],
    mode='lines', name='Harga Asli', 
    line=dict(color='#00FCE7', width=2),
    fill='tozeroy', fillcolor='rgba(0, 252, 231, 0.05)'
))
# Grafik Prediksi
fig.add_trace(go.Scatter(
    x=[df.index[-1]] + future_dates, y=[df['Close'].iloc[-1]] + future_prices,
    mode='lines+markers', name='Prediksi AI', 
    line=dict(color='#E3B341', width=2, dash='dash'),
    marker=dict(size=6, color='#E3B341')
))

last_price = df['Close'].iloc[-1]
if last_price < 0.0000001:
    dynamic_tick_format = ".12f"
elif last_price < 1:
    dynamic_tick_format = ".8f"
else:
    dynamic_tick_format = ",.2f"

# 3. Garis Vertikal Penanda "Hari Ini" (Bypass bug internal Plotly)
# Gambar garis vertikalnya saja (tanpa teks)
fig.add_vline(
    x=df.index[-1], 
    line_width=1.5, line_dash="dash", line_color="#8B949E"
)

# Tempelkan teksnya secara independen agar tidak memicu error perhitungan
fig.add_annotation(
    x=df.index[-1],
    y=1,
    yref="paper",
    text="Hari Ini (Batas Prediksi) ➔ ",
    showarrow=False,
    font=dict(color="white", size=11),
    xanchor="right",
    yanchor="top"
)

# 4. Update Layout (Dark Mode, Bersih, Profesional)
fig.update_layout(
    height=450,
    plot_bgcolor='rgba(0,0,0,0)', # Transparan menyatu dengan background Streamlit
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8B949E'),
    margin=dict(l=0, r=0, t=10, b=0),
    
    # Grid halus agar tidak berisik
    xaxis=dict(showgrid=True, gridcolor='#1F2937', zeroline=False),
    yaxis=dict(
        showgrid=True, gridcolor='#1F2937', 
        zeroline=False, tickprefix="$", tickformat=dynamic_tick_format, side='right' # Harga di kanan (standar crypto)
    ),
    
    hovermode="x unified",
    
    # Legenda dipindah ke ATAS GRAFIK, dan dibuat transparan agar tidak menutupi apapun
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left",
        bgcolor='rgba(0,0,0,0)'
    )
)

st.plotly_chart(fig, width='stretch', config={'displayModeBar': True})

# --- 7. TABEL PREDIKSI ---
st.markdown("### Predicted Prices (Next 7 Days)")

# Membuat HTML Table agar sama persis dengan mockup
table_html = '<table class="pred-table">'
table_html += '<thead><tr><th>Date</th><th>Price</th><th>Change (%)</th></tr></thead><tbody>'

for i in range(7):
    date_str = future_dates[i].strftime('%d %b %Y')
    price_str = format_price(future_prices[i])
    change_val = changes_pct[i]
    
    color_class = "change-up" if change_val >= 0 else "change-down"
    sign = "+" if change_val >= 0 else ""
    
    table_html += f"<tr><td>{date_str}</td><td>{price_str}</td><td class='{color_class}'>{sign}{change_val:.2f}%</td></tr>"

table_html += '</tbody></table>'
st.markdown(table_html, unsafe_allow_html=True)

# --- 8. FOOTER ---
st.markdown("""
<div class="footer">
    <div class="footer-left">
        <b>System Information & Disclaimer</b><br>
        This System Uses A Long Short-Term Memory (LSTM) Algorithm To Predict Crypto Asset Prices Based On Historical Data. Accuracy Is Evaluated Using RMSE And MAPE Metrics.
    </div>
    <div class="footer-right">
        <b style="color: white;">Important Notice</b><br>
        Content Provided In This Dashboard Is For Informational Purposes Only And Does Not Constitute Financial Advice, Investment Recommendation, Or Trading Endorsement. Cryptocurrency Trading Involves High Risk And Volatility. Please Conduct Your Own Research (DYOR) Before Trading.
    </div>
</div>
""", unsafe_allow_html=True)