import time
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import (
    get_data_with_indicators, get_market_summary,
    format_price, format_big_number, COINS, COIN_DESCRIPTIONS
)
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ─── 1. CONFIG & STATE ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Kripto - Analisis Coin",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "selected_coin" not in st.session_state:
    # Menampilkan warning kecil (opsional, bisa dihapus jika hanya ingin spinner)
    st.warning("Sesi telah diperbarui. Silakan pilih coin kripto kembali.")
    
    # Menampilkan animasi loading dengan pesan
    with st.spinner("🔄 Memuat ulang data... Mengarahkan kembali ke Beranda (Home)."):
        time.sleep(2)  # Beri jeda 2 detik agar animasi terlihat
        st.switch_page("Home.py")
        st.stop()

selected_coin = st.session_state["selected_coin"]
coin_name     = COINS.get(selected_coin, selected_coin.replace("-USD", ""))
ticker        = selected_coin.replace("-USD", "")
coin_desc     = COIN_DESCRIPTIONS.get(selected_coin, "Informasi koin tidak tersedia.")

# ─── 2. CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #080B10;
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

/* ── Page Header ── */
.page-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 60%, #061020 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}
.page-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.header-left { display: flex; align-items: center; gap: 20px; }
.coin-avatar {
    width: 56px; height: 56px; border-radius: 50%;
    border: 2px solid var(--border2); object-fit: cover; background: var(--surface);
}
.header-coin-name { font-family: var(--font-head); font-size: 26px; font-weight: 800; color: var(--text); margin: 0; }
.header-coin-ticker { font-family: var(--font-mono); font-size: 13px; color: var(--muted); margin-top: 2px; }
.header-right { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }
.header-price { font-family: var(--font-head); font-size: 32px; font-weight: 800; color: var(--accent); }
.header-change-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 12px; border-radius: 8px;
    font-family: var(--font-mono); font-size: 13px; font-weight: 500;
}
.pill-up   { background: rgba(0,229,160,0.12); color: var(--green); }
.pill-down { background: rgba(255,77,106,0.12); color: var(--red); }

/* ── KPI Row ── */
.kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
.kpi-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; position: relative; overflow: hidden;
}
.kpi-card::before { content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; }
.kpi-card.accent-blue::before  { background: linear-gradient(90deg, var(--accent), transparent); }
.kpi-card.accent-green::before { background: linear-gradient(90deg, var(--green), transparent); }
.kpi-card.accent-red::before   { background: linear-gradient(90deg, var(--red), transparent); }
.kpi-card.accent-gold::before  { background: linear-gradient(90deg, var(--gold), transparent); }
.kpi-icon { font-size: 18px; margin-bottom: 10px; }
.kpi-label { font-family: var(--font-body); font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
.kpi-value { font-family: var(--font-head); font-size: 18px; font-weight: 700; color: var(--text); }
.kpi-sub { font-family: var(--font-body); font-size: 11px; color: var(--muted); margin-top: 3px; }

/* ── Section Header ── */
.section-header {
    font-family: var(--font-head); font-size: 11px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted);
    margin-bottom: 14px; display: flex; align-items: center; gap: 10px;
}
.section-header::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Indicators ── */
.indicator-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 28px; }
.indicator-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.ind-label { font-family: var(--font-body); font-size: 11px; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
.ind-value { font-family: var(--font-mono); font-size: 16px; font-weight: 500; color: var(--text); margin-bottom: 4px; }
.ind-signal { display: inline-flex; align-items: center; gap: 5px; font-family: var(--font-body); font-size: 11px; padding: 2px 8px; border-radius: 4px; }
.signal-buy     { background: rgba(0,229,160,0.12); color: var(--green); }
.signal-sell    { background: rgba(255,77,106,0.12); color: var(--red); }
.signal-neutral { background: rgba(255,184,0,0.12); color: var(--gold); }

/* ── Info Box ── */
.info-box {
    background: linear-gradient(135deg, rgba(167,139,250,0.06), rgba(0,212,255,0.03));
    border: 1px solid rgba(167,139,250,0.18); border-radius: 12px; padding: 20px 24px;
    margin-bottom: 28px; display: flex; gap: 14px; align-items: flex-start;
}
.info-icon { font-size: 24px; flex-shrink: 0; margin-top: 1px; }
.info-title { font-family: var(--font-head); font-size: 13px; font-weight: 700; color: var(--purple); margin-bottom: 6px; }
.info-text { font-family: var(--font-body); font-size: 13px; color: var(--muted2); line-height: 1.75; }
.info-text b { color: var(--text); }

/* ── CTA Card — hanya teks, tombol ada di bawahnya via Streamlit column ── */
.cta-card {
    background: linear-gradient(135deg, #0A1A2E 0%, #061020 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 16px;
    padding: 32px 40px;
    position: relative;
    overflow: hidden;
    text-align: center;
    margin-bottom: 20px;
}
.cta-card::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.cta-text-title {
    font-family: var(--font-head); font-size: 20px; font-weight: 800;
    color: var(--text); margin-bottom: 10px;
}
.cta-text-sub {
    font-family: var(--font-body); font-size: 14px;
    color: var(--muted); line-height: 1.7;
    max-width: 560px; margin: 0 auto;
}

/* ── Tombol Navigasi (Kembali & Refresh) — style default semua tombol ── */
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

/* ── Tombol Prediksi — .predict-col adalah div marker tepat sebelum kolom ──
   Karena Streamlit merender kolom sebagai sibling setelah marker div,
   kita pakai ~ (adjacent sibling combinator) untuk menarget elemen berikutnya. ── */
.predict-col ~ div div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%) !important;
    color: #000 !important;
    font-family: var(--font-head) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    border-radius: 12px !important;
    border: none !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.predict-col ~ div div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #00E5FF 0%, #00AADD 100%) !important;
    box-shadow: 0 6px 24px rgba(0,212,255,0.3) !important;
    color: #000 !important;
    transform: translateY(-1px) !important;
}
.predict-col ~ div div[data-testid="stButton"] button:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}

/* ── Footer ── */
.footer { border-top: 1px solid var(--border); padding-top: 20px; margin-top: 28px; display: flex; gap: 24px; }
.footer-col { flex: 1; min-width: 240px; }
.footer-label { font-family: var(--font-head); font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
.footer-text { font-family: var(--font-body); font-size: 12.5px; color: #374151; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ─── 3. HELPER FUNGSI ─────────────────────────────────────────────────────────
def get_start_date(timeframe):
    end = datetime.now()
    if   timeframe == "1M":  start = end - relativedelta(months=1)
    elif timeframe == "3M":  start = end - relativedelta(months=3)
    elif timeframe == "6M":  start = end - relativedelta(months=6)
    elif timeframe == "1Y":  start = end - relativedelta(years=1)
    elif timeframe == "ALL": start = datetime(2022, 1, 1)
    else:                    start = end - relativedelta(months=6)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def get_rsi_signal(rsi):
    if rsi is None: return "Tidak Tersedia", "signal-neutral"
    if rsi < 30:    return "Oversold (Area Beli)", "signal-buy"
    if rsi > 70:    return "Overbought (Area Jual)", "signal-sell"
    return "Netral", "signal-neutral"

def get_macd_signal(macd, signal):
    if macd is None or signal is None: return "Tidak Tersedia", "signal-neutral"
    if macd > signal: return "Bullish ↑", "signal-buy"
    return "Bearish ↓", "signal-sell"

# ─── 4. AMBIL DATA PASAR ──────────────────────────────────────────────────────
raw_market_data, fetch_time = get_market_summary()
market_data = [d for d in raw_market_data if isinstance(d, dict)]
coin_info   = next((d for d in market_data if d.get("Ticker") == selected_coin), None)

current_price = coin_info.get("Price",     0) if coin_info else 0
change_24h    = coin_info.get("Change",    0) if coin_info else 0
market_cap    = coin_info.get("MarketCap", 0) if coin_info else 0
volume_24h    = coin_info.get("Volume",    0) if coin_info else 0
atl           = coin_info.get("ATL",       0) if coin_info else 0
icon_src      = coin_info.get("Icon",     "") if coin_info else ""

# ─── 5. NAVIGASI ATAS ─────────────────────────────────────────────────────────
col_back, _, col_refresh = st.columns([2, 8, 2])
with col_back:
    if st.button("← Kembali ke Beranda", use_container_width=True):
        st.switch_page("Home.py")
with col_refresh:
    if st.button("🔄 Refresh Data", use_container_width=True, help="Perbarui data pasar"):
        st.cache_data.clear()
        st.rerun()

# ─── 6. HEADER ────────────────────────────────────────────────────────────────
is_up    = change_24h >= 0
arrow    = "▲" if is_up else "▼"
cls_pill = "pill-up" if is_up else "pill-down"

st.markdown(f"""
<div class="page-header">
    <div class="header-left">
        <img src="{icon_src}" class="coin-avatar" onerror="this.style.display='none'">
        <div>
            <div class="header-coin-name">{coin_name}</div>
            <div class="header-coin-ticker">{ticker} · USD · Data Langsung (Live)</div>
        </div>
    </div>
    <div class="header-right">
        <div class="header-price">{format_price(current_price)}</div>
        <span class="header-change-pill {cls_pill}">
            {arrow} {abs(change_24h):.2f}% (24 Jam)
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 7. PROFIL Coin ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid var(--border);
            border-radius:12px;padding:20px 24px;margin-bottom:28px;">
    <div style="font-family:var(--font-head);font-size:14px;font-weight:700;
                color:var(--accent);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.1em;">
        Apa itu {coin_name}?
    </div>
    <div style="font-family:var(--font-body);font-size:14px;color:var(--muted2);line-height:1.6;">
        {coin_desc}
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 8. KPI CARDS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card accent-blue">
        <div class="kpi-icon">💰</div>
        <div class="kpi-label">Kapitalisasi Pasar</div>
        <div class="kpi-value">{format_big_number(market_cap)}</div>
        <div class="kpi-sub">Total nilai koin beredar</div>
    </div>
    <div class="kpi-card accent-green">
        <div class="kpi-icon">📊</div>
        <div class="kpi-label">Volume Perdagangan (24j)</div>
        <div class="kpi-value">{format_big_number(volume_24h)}</div>
        <div class="kpi-sub">Aktivitas transaksi hari ini</div>
    </div>
    <div class="kpi-card accent-red">
        <div class="kpi-icon">📉</div>
        <div class="kpi-label">Harga Terendah (ATL)</div>
        <div class="kpi-value">{format_price(atl)}</div>
        <div class="kpi-sub">Titik terendah sepanjang masa</div>
    </div>
    <div class="kpi-card accent-gold">
        <div class="kpi-icon">🔄</div>
        <div class="kpi-label">Perubahan 24 Jam</div>
        <div class="kpi-value" style="color:{'var(--green)' if is_up else 'var(--red)'}">
            {arrow} {abs(change_24h):.2f}%
        </div>
        <div class="kpi-sub">Dibandingkan kemarin</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 9. GRAFIK HARGA ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Grafik Harga & Data Historis</div>', unsafe_allow_html=True)

timeframe_options  = ["1M", "3M", "6M", "1Y", "ALL"]
timeframe_selected = st.radio(
    "Rentang Waktu:", options=timeframe_options, index=2,
    horizontal=True, label_visibility="collapsed",
)

start_date, end_date = get_start_date(timeframe_selected)

with st.spinner(f"Memuat data historis ({timeframe_selected})..."):
    df = get_data_with_indicators(selected_coin, start_date, end_date, interval="1d")

if not df.empty:
    last_p = df["Close"].iloc[-1]
    if   last_p < 0.0000001: tick_fmt = ".12f"
    elif last_p < 1:         tick_fmt = ".8f"
    else:                    tick_fmt = ",.2f"

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        subplot_titles=("", "Volume", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC", hoverinfo="skip",
        increasing=dict(line=dict(color="#00E5A0", width=1), fillcolor="rgba(0,229,160,0.85)"),
        decreasing=dict(line=dict(color="#FF4D6A", width=1), fillcolor="rgba(255,77,106,0.85)"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", line=dict(width=0),
        showlegend=False, name="",
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "Open : $%{customdata[0]:,.6g}<br>"
            "High : $%{customdata[1]:,.6g}<br>"
            "Low  : $%{customdata[2]:,.6g}<br>"
            "Close: <b>$%{customdata[3]:,.6g}</b><br>"
            "Vol  : %{customdata[4]:,.0f}<extra></extra>"
        ),
        customdata=list(zip(df["Open"], df["High"], df["Low"], df["Close"], df["Volume"])),
    ), row=1, col=1)

    latest_close = df["Close"].iloc[-1]
    fig.add_annotation(
        x=df.index[-1], y=latest_close, xref="x", yref="y",
        text=f"  ${latest_close:,.6g}", showarrow=False,
        font=dict(color="#00D4FF", size=11, family="DM Mono"),
        xanchor="left",
        bgcolor="rgba(0,212,255,0.12)", bordercolor="rgba(0,212,255,0.35)",
        borderwidth=1, borderpad=4,
    )
    fig.add_hline(y=latest_close, line_dash="dot",
                  line_color="rgba(0,212,255,0.25)", line_width=1, row=1, col=1)

    colors_vol = ["rgba(0,229,160,0.65)" if c >= o else "rgba(255,77,106,0.65)"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], marker_color=colors_vol,
        showlegend=False, hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], fill="tozeroy",
            fillcolor="rgba(167,139,250,0.06)",
            line=dict(color="#A78BFA", width=1.8), showlegend=False,
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,77,106,0.05)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,229,160,0.05)",  line_width=0, row=3, col=1)
        for lvl, clr in [(70, "rgba(255,77,106,0.45)"), (30, "rgba(0,229,160,0.45)"), (50, "rgba(90,106,126,0.3)")]:
            fig.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=1, row=3, col=1)

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd_hist   = df["MACD"] - df["MACD_Signal"]
        hist_colors = ["rgba(0,229,160,0.55)" if v >= 0 else "rgba(255,77,106,0.55)" for v in macd_hist]
        fig.add_trace(go.Bar(x=df.index, y=macd_hist, marker_color=hist_colors,
                             showlegend=False, hovertemplate="Histogram: %{y:.6g}<extra></extra>"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines",
                                 line=dict(color="#00D4FF", width=1.6), showlegend=False,
                                 hovertemplate="MACD: %{y:.6g}<extra></extra>"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines",
                                 line=dict(color="#FFB800", width=1.6, dash="dot"), showlegend=False,
                                 hovertemplate="Signal: %{y:.6g}<extra></extra>"), row=4, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="rgba(90,106,126,0.35)", line_width=1, row=4, col=1)

    ax = dict(showgrid=True, gridcolor="rgba(30,39,51,0.9)", gridwidth=1,
              zeroline=False, color="#5A6A7E", showline=False,
              tickfont=dict(family="DM Mono", size=10, color="#5A6A7E"))
    fig.update_layout(
        height=640,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#5A6A7E", size=11),
        margin=dict(l=8, r=80, t=28, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#111820", bordercolor="#2A3544",
                        font=dict(color="#E8EDF5", size=12, family="DM Sans"), align="left"),
        showlegend=False,
        xaxis =dict(**ax, rangeslider=dict(visible=False), type="date", showticklabels=False),
        xaxis2=dict(**ax, rangeslider=dict(visible=False), showticklabels=False),
        xaxis3=dict(**ax, rangeslider=dict(visible=False), showticklabels=False),
        xaxis4=dict(**ax, rangeslider=dict(visible=False), tickformat="%b %Y", dtick="M1"),
        yaxis =dict(**ax, side="right", tickprefix="$", tickformat=tick_fmt, nticks=6),
        yaxis2=dict(**ax, side="right", title=dict(text="Vol",  font=dict(size=9, color="#5A6A7E")), nticks=3),
        yaxis3=dict(**ax, side="right", title=dict(text="RSI",  font=dict(size=9, color="#5A6A7E")), range=[0,100], tickvals=[0,30,50,70,100]),
        yaxis4=dict(**ax, side="right", title=dict(text="MACD", font=dict(size=9, color="#5A6A7E")), nticks=4),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(family="DM Sans", size=10, color="#5A6A7E")

    st.plotly_chart(fig, use_container_width=True, config={
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": f"{ticker}_chart", "scale": 2},
    })
else:
    st.warning(f"Data tidak tersedia untuk rentang waktu {timeframe_selected}.")

# ─── 10. INDIKATOR TEKNIKAL ───────────────────────────────────────────────────
st.markdown('<br><div class="section-header">🧮 Ringkasan Indikator Teknikal</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <div class="info-icon">💡</div>
    <div>
        <div class="info-title">Apa arti indikator teknikal ini?</div>
        <div class="info-text">
            <b>RSI (Relative Strength Index)</b> mengukur kekuatan tren — di bawah 30 berarti
            <b>oversold</b> (potensi beli), di atas 70 berarti <b>overbought</b> (potensi jual).<br>
            <b>MACD</b> mendeteksi perubahan momentum — jika garis MACD di atas garis sinyal,
            tren cenderung <b>bullish</b> (naik) dan sebaliknya.<br>
            <b>ATR (Average True Range)</b> mengukur tingkat volatilitas — seberapa besar
            rata-rata fluktuasi harga di pasar.<br>
            Indikator ini bersifat <b>informatif</b> sebagai variabel input model AI,
            bukan sinyal beli/jual yang pasti.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not df.empty:
    rsi_val  = round(df["RSI"].iloc[-1],        2) if "RSI"         in df.columns else None
    macd_val = round(df["MACD"].iloc[-1],        6) if "MACD"        in df.columns else None
    sig_val  = round(df["MACD_Signal"].iloc[-1], 6) if "MACD_Signal" in df.columns else None
    atr_val  = round(df["ATR"].iloc[-1],         6) if "ATR"         in df.columns else None

    rsi_text,  rsi_cls  = get_rsi_signal(rsi_val)
    macd_text, macd_cls = get_macd_signal(macd_val, sig_val)

    rsi_display  = f"{rsi_val}"      if rsi_val  else "N/A"
    macd_display = f"{macd_val:.6f}" if macd_val else "N/A"
    atr_display  = f"{atr_val:.6f}"  if atr_val  else "N/A"

    st.markdown(f"""
    <div class="indicator-grid">
        <div class="indicator-card">
            <div class="ind-label">Relative Strength Index (RSI)</div>
            <div class="ind-value">{rsi_display}</div>
            <span class="ind-signal {rsi_cls}">{rsi_text}</span>
        </div>
        <div class="indicator-card">
            <div class="ind-label">MACD vs Signal</div>
            <div class="ind-value" style="font-size:13px">{macd_display}</div>
            <span class="ind-signal {macd_cls}">{macd_text}</span>
        </div>
        <div class="indicator-card">
            <div class="ind-label">Average True Range (ATR)</div>
            <div class="ind-value" style="font-size:13px">{atr_display}</div>
            <span class="ind-signal signal-neutral">Indeks Volatilitas Pasar</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <div class="info-icon">💡</div>
    <div>
        <div class="info-title">Variabel Input Model LSTM</div>
        <div class="info-text">
            Indikator teknikal di atas (<b>RSI, MACD, ATR</b>) merupakan fitur asli (input feature) yang
            diekstraksi dari data historis dan dimasukkan ke dalam arsitektur <i>Deep Learning</i> LSTM
            untuk menemukan pola prediksi harga. Urutan fitur: Log_Ret → RSI → MACD → MACD_Signal → ATR → Volume.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── 11. CTA CARD + TOMBOL PREDIKSI ──────────────────────────────────────────
st.markdown(f"""
<div class="cta-card">
    <div class="cta-text-title">🔮 Siap memprediksi arah harga {coin_name}?</div>
    <div class="cta-text-sub">
        Model cerdas LSTM telah mempelajari fitur-fitur teknikal di atas dan siap
        mensimulasikan harga penutupan (Close Price) selama 7 hari ke depan.
    </div>
</div>
""", unsafe_allow_html=True)

# Marker div kosong — tidak membungkus tombol, hanya sebagai CSS anchor
st.markdown('<div class="predict-col"></div>', unsafe_allow_html=True)

# Kolom 3:4:3 → tombol mengisi kolom tengah yang lebarnya ±36% halaman
_, col_btn, _ = st.columns([3, 4, 3])
with col_btn:
    if st.button(
        f"🚀 Jalankan Prediksi AI · {ticker}",
        key="btn_predict",
        use_container_width=True,
    ):
        # session_state["selected_coin"] sudah ter-set sejak Home.py
        # st.switch_page membawa seluruh session_state ke halaman berikutnya
        st.switch_page("pages/Prediction.py")

# ─── 12. FOOTER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-col">
        <div class="footer-label">⚙️ Informasi Sistem</div>
        <div class="footer-text">
            Halaman ini memvisualisasikan data mentah (OHLCV) beserta indikator hasil ekstraksi
            yang sepenuhnya sejalan dengan skenario pengujian parameter model pada Tugas Akhir.
        </div>
    </div>
    <div class="footer-col">
        <div class="footer-label">⚠️ Peringatan Risiko</div>
        <div class="footer-text">
            Data teknikal yang disajikan di sini murni berfungsi sebagai alat bantu riset analisis
            teknikal dan tidak bisa diartikan sebagai jaminan investasi (financial advice).
        </div>
    </div>
</div>
""", unsafe_allow_html=True)