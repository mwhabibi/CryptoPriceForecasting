import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import json
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf
from utils import COINS, format_price

# --- 1. CONFIG & STATE ---
st.set_page_config(page_title="Prediction Result", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

if 'selected_coin' not in st.session_state:
    st.session_state['selected_coin'] = 'BTC-USD'

selected_coin = st.session_state['selected_coin']

# --- 2. CUSTOM CSS (Sesuai Mockup) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .header-title { font-size: 28px; font-weight: bold; }
    
    /* Styling Metrics */
    .metric-container { margin-bottom: 20px; }
    .metric-title { font-size: 14px; color: #8B949E; font-weight: bold; }
    .metric-value { font-size: 16px; color: #E3B341; font-weight: bold; } /* Warna oranye/emas */
    .metric-sub { font-size: 12px; color: #8B949E; font-style: italic; }
    
    /* Tabel Kustom */
    .pred-table { width: 100%; border-collapse: collapse; margin-top: 15px; color: white; font-size: 14px; }
    .pred-table th { text-align: right; padding: 12px; border-bottom: 1px solid #30363D; color: #8B949E; }
    .pred-table th:first-child { text-align: left; }
    .pred-table td { text-align: right; padding: 12px; border-bottom: 1px solid #21262D; }
    .pred-table td:first-child { text-align: left; font-weight: bold; }
    .change-up { color: #00FF00; }
    .change-down { color: #FF4B4B; }
    
    /* Footer */
    .footer { display: flex; justify-content: space-between; font-size: 12px; color: #8B949E; margin-top: 40px; border-top: 1px solid #30363D; padding-top: 20px; }
    .footer-left { max-width: 45%; }
    .footer-right { max-width: 45%; text-align: right; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD METRICS & MODEL ---
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
    except Exception as e:
        return None, None

# --- 4. TOP HEADER & METRICS LAYOUT ---
# Membagi header menjadi 4 kolom untuk mendorong tombol ke ujung kanan
col_title, col_space, col_btn1, col_btn2 = st.columns([5, 1, 2, 2])

with col_title:
    st.markdown("<div class='header-title'>Prediction Result</div>", unsafe_allow_html=True)

with col_btn1:
    if st.button("❮ Back To Previous", use_container_width=True):
        st.switch_page("pages/Detail.py")

with col_btn2:
    if st.button("Re-Analysis (Re-Run)", type="primary", use_container_width=True):
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Metrics Display
col_m1, col_m2, col_m3 = st.columns([2, 2, 6])
with col_m1:
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-title">RMSE Score:</span> <span class="metric-value">{format_price(coin_metrics['RMSE'])}</span><br>
        <span class="metric-sub">Lower is better</span>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-container">
        <span class="metric-title">MAPE Score:</span> <span class="metric-value">{coin_metrics['MAPE']}%</span><br>
        <span class="metric-sub">Average error percentage</span>
    </div>
    """, unsafe_allow_html=True)
    
st.markdown("<hr style='border: 1px solid #161B22; margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)


# --- 5. LOGIKA PREDIKSI (Bypass yfinance bug dengan history period="max") ---
model, scaler = load_ml_assets(selected_coin)

if model is None or scaler is None:
    st.error("Model atau Scaler tidak ditemukan. Pastikan file ada di folder 'models' dan 'scalers'.")
    st.stop()

with st.spinner("Memproses algoritma LSTM..."):
    # Trik Bypass yfinance: Tarik data maksimum, lalu potong
    t = yf.Ticker(selected_coin)
    df = t.history(period="1y", interval="1d") # Tarik 1 tahun terakhir agar pasti cukup
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Feature Engineering Cepat
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(window=14).mean()
    
    df.dropna(inplace=True)

    # Persiapan Input Model
    LOOKBACK = 60
    FEATURES = ['Log_Ret', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Volume']
    
    recent_data = df[FEATURES].values[-LOOKBACK:]
    scaled_data = scaler.transform(recent_data)
    X_input = scaled_data.reshape(1, LOOKBACK, len(FEATURES))
    
    # Inferensi
    pred_scaled = model.predict(X_input, verbose=0)[0]
    dummy_array = np.zeros((7, len(FEATURES)))
    dummy_array[:, 0] = pred_scaled
    pred_log_ret = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Konversi Harga
    last_price = df['Close'].iloc[-1]
    last_date = df.index[-1]
    
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

# --- 6. VISUALISASI CHART (Future Projection) ---
fig = go.Figure()

# 1. Garis Harga Asli (Cyan Terang - Konsisten dengan tema)
fig.add_trace(go.Scatter(
    x=df.index[-60:], y=df['Close'].iloc[-60:],
    mode='lines', name='Harga Asli (Historis)', 
    line=dict(color='#00FCE7', width=2),
    fill='tozeroy', fillcolor='rgba(0, 252, 231, 0.05)' # Sedikit efek glow di bawah garis
))

# 2. Garis Prediksi (Oranye/Emas Putus-putus - Melambangkan Masa Depan)
fig.add_trace(go.Scatter(
    x=[df.index[-1]] + future_dates, y=[df['Close'].iloc[-1]] + future_prices,
    mode='lines+markers', name='Prediksi AI (7 Hari)', 
    line=dict(color='#E3B341', width=2, dash='dash'),
    marker=dict(size=6, color='#E3B341')
))

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
        zeroline=False, tickprefix="$", side='right' # Harga di kanan (standar crypto)
    ),
    
    hovermode="x unified",
    
    # Legenda dipindah ke ATAS GRAFIK, dan dibuat transparan agar tidak menutupi apapun
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor='rgba(0,0,0,0)'
    )
)

st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

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