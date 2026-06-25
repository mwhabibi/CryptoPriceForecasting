import streamlit as st
import pandas as pd
from utils import get_market_summary, format_big_number, format_price

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Kripto",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #080B10;
    --bg2:       #0D1117;
    --surface:   #111820;
    --border:    #1E2733;
    --border2:   #2A3544;
    --text:      #E8EDF5;
    --muted:     #5A6A7E;
    --accent:    #00D4FF;
    --accent2:   #0099CC;
    --green:     #00E5A0;
    --red:       #FF4D6A;
    --gold:      #FFB800;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

.stApp { background-color: var(--bg) !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 40%, #061020 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -60px; left: 30%;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(0,229,160,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-family: var(--font-body);
    font-size: 12px; font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-badge .dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.8); }
}
.hero h1 {
    font-family: var(--font-head);
    font-size: 38px; font-weight: 800;
    color: var(--text);
    margin: 0 0 8px 0; line-height: 1.1;
}
.hero h1 span { color: var(--accent); }
.hero-sub {
    font-family: var(--font-body);
    font-size: 15px; font-weight: 400; color: var(--muted);
    margin: 0 0 28px 0; line-height: 1.7;
    max-width: 560px;
}
.how-steps {
    display: flex; gap: 12px; flex-wrap: wrap;
}
.step-pill {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 10px 16px;
    font-family: var(--font-body); font-size: 13px;
    color: var(--muted);
}
.step-num {
    width: 22px; height: 22px;
    background: var(--accent); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700;
    color: #000; flex-shrink: 0;
}
.step-pill strong { color: var(--text); font-size: 13px; display: block; margin-bottom: 1px; font-weight: 600; }

/* ── Section Title ── */
.section-header {
    font-family: var(--font-head);
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1; height: 1px;
    background: var(--border);
}

/* ── Coin Cards ── */
.coin-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    transition: border-color 0.2s, transform 0.2s;
    cursor: default;
    height: 100%;
}
.coin-card:hover { border-color: var(--border2); }

.card-top {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px;
}
.card-identity { display: flex; align-items: center; gap: 10px; }
.coin-logo {
    width: 34px; height: 34px; border-radius: 50%;
    border: 1px solid var(--border2);
    object-fit: cover;
}
.coin-ticker {
    font-family: var(--font-head);
    font-size: 15px; font-weight: 700; color: var(--text);
}
.coin-fullname {
    font-family: var(--font-body);
    font-size: 11px; color: var(--muted);
}
.change-badge {
    font-family: var(--font-mono);
    font-size: 11px; font-weight: 500;
    padding: 4px 9px; border-radius: 6px;
}
.change-up   { background: rgba(0,229,160,0.12); color: var(--green); }
.change-down { background: rgba(255,77,106,0.12); color: var(--red); }

.coin-price-main {
    font-family: var(--font-body);
    font-size: 16px; font-weight: 800;
    color: var(--text);
    margin-bottom: 12px;
}
.card-meta {
    display: flex; gap: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
}
.meta-item { flex: 1; }
.meta-label {
    font-family: var(--font-body);
    font-size: 10px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 2px;
}
.meta-value {
    font-family: var(--font-mono);
    font-size: 12px; color: var(--text); font-weight: 500;
}

/* ── Table ── */
.table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 32px;
}
.styled-table {
    width: 100%; border-collapse: collapse;
    font-family: var(--font-body);
}
.styled-table thead tr {
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid var(--border);
}
.styled-table th {
    padding: 13px 18px;
    font-size: 11px; font-weight: 600;
    color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    text-align: right;
}
.styled-table th:first-child { text-align: left; }
.styled-table td {
    padding: 13px 18px;
    font-size: 13px;
    color: var(--text);
    text-align: right;
    border-bottom: 1px solid rgba(30,39,51,0.6);
    font-family: var(--font-mono); /* numbers only */
}
.styled-table td:first-child {
    text-align: left;
    font-family: var(--font-body); /* asset name = body font */
}
.styled-table tbody tr:last-child td { border-bottom: none; }
.styled-table tbody tr:hover { background: rgba(255,255,255,0.02); }

.asset-cell { display: flex; align-items: center; gap: 10px; }
.asset-name { font-weight: 700; color: var(--text); }
.asset-sym  { color: var(--muted); font-size: 10px; margin-left: 4px; }
.tbl-logo   { width: 22px; height: 22px; border-radius: 50%; }

/* ── Explanation Box ── */
.explain-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.04), rgba(0,153,204,0.02));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 28px;
    display: flex; gap: 16px; align-items: flex-start;
}
.explain-icon { font-size: 28px; flex-shrink: 0; margin-top: 2px; }
.explain-title {
    font-family: var(--font-head);
    font-size: 14px; font-weight: 700; color: var(--accent);
    margin-bottom: 6px;
}
.explain-text {
    font-family: var(--font-body);
    font-size: 13.5px; color: var(--muted); line-height: 1.75;
}
.explain-text b { color: var(--text); }

/* ── Stats Bar ── */
.stats-bar {
    display: flex; gap: 2px;
    margin-bottom: 28px;
}
.stat-item {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 18px;
    text-align: center;
}
.stat-item:first-child { border-radius: 10px 0 0 10px; }
.stat-item:last-child  { border-radius: 0 10px 10px 0; }
.stat-num {
    font-family: var(--font-body);
    font-size: 20px; font-weight: 800; color: var(--accent);
}
.stat-desc {
    font-family: var(--font-body);
    font-size: 12px; color: var(--muted); margin-top: 3px;
}

/* ── Footer ── */
.footer {
    border-top: 1px solid var(--border);
    padding-top: 20px; margin-top: 20px;
    display: flex; justify-content: space-between; gap: 24px;
    flex-wrap: wrap;
}
.footer-col { flex: 1; min-width: 240px; }
.footer-label {
    font-family: var(--font-head);
    font-size: 11px; font-weight: 700;
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 6px;
}
.footer-text {
    font-family: var(--font-body);
    font-size: 13px; color: #374151; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Memuat data pasar..."):
    data, last_updated = get_market_summary()

# ─── HERO BANNER ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">
        <span class="dot"></span>
        Live Market Data · Diperbarui: {last_updated}
    </div>
    <h1>Prediksi Harga <span>Kripto</span><br>berbasis LSTM</h1>
    <p class="hero-sub">
        Dashboard ini menggunakan model kecerdasan buatan <b style="color:#E8EDF5">Long Short-Term Memory (LSTM)</b> 
        untuk memprediksi harga aset kripto berdasarkan pola data historis.
        Pilih aset di bawah untuk melihat prediksi dan analisis selengkapnya.
    </p>
    <div class="how-steps">
        <div class="step-pill">
            <span class="step-num">1</span>
            <div><strong>Lihat Pasar</strong>Pantau harga & pergerakan 24 jam</div>
        </div>
        <div class="step-pill">
            <span class="step-num">2</span>
            <div><strong>Pilih Aset</strong>Klik tombol "Analisis" pada koin</div>
        </div>
        <div class="step-pill">
            <span class="step-num">3</span>
            <div><strong>Baca Prediksi</strong>Lihat grafik & hasil model AI</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── WHAT IS LSTM EXPLAINER ───────────────────────────────────────────────────
st.markdown("""
<div class="explain-box">
    <div class="explain-icon">🧠</div>
    <div>
        <div class="explain-title">Apa itu LSTM dan bagaimana cara kerjanya?</div>
        <div class="explain-text">
            <b>LSTM (Long Short-Term Memory)</b> adalah jenis jaringan saraf tiruan yang dirancang khusus untuk 
            mengenali pola dalam data berurutan seperti harga historis. Model ini "mengingat" tren 
            masa lalu untuk membuat estimasi ke depan — mirip seperti seorang analis yang mempelajari 
            grafik harga selama berbulan-bulan sebelum membuat keputusan.
            <br><br>
            Performa model diukur menggunakan <b>RMSE</b> (kesalahan akar rata-rata kuadrat), 
            dan <b>MAPE</b> (persentase kesalahan rata-rata). Semakin kecil nilainya, semakin akurat prediksinya.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── STATS BAR ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-num">{len(data)}</div>
        <div class="stat-desc">Aset Dipantau</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">LSTM</div>
        <div class="stat-desc">Algoritma Prediksi</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">Live</div>
        <div class="stat-desc">Sumber Data Real-Time</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── COIN CARDS ───────────────────────────────────────────────────────────────
if not data:
    st.error("⚠️ Gagal mengambil data pasar. Periksa koneksi internet Anda.")
else:
    st.markdown('<div class="section-header">🪙 Pilih Aset untuk Dianalisis</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(data))
    for i, item in enumerate(data):
        is_up = item["Change"] >= 0
        arrow  = "▲" if is_up else "▼"
        cls    = "change-up" if is_up else "change-down"
        ticker = item['Ticker'].replace('-USD', '')

        with cols[i]:
            st.markdown(f"""
            <div class="coin-card">
                <div class="card-top">
                    <div class="card-identity">
                        <img src="{item['Icon']}" class="coin-logo" onerror="this.style.display='none'">
                        <div>
                            <div class="coin-ticker">{ticker}</div>
                            <div class="coin-fullname">{item['Name']}</div>
                        </div>
                    </div>
                    <span class="change-badge {cls}">{arrow} {item['Change']:.2f}%</span>
                </div>
                <div class="coin-price-main">{format_price(item['Price'])}</div>
                <div class="card-meta">
                    <div class="meta-item">
                        <div class="meta-label">Mkt Cap</div>
                        <div class="meta-value">{format_big_number(item['MarketCap'])}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Vol 24h</div>
                        <div class="meta-value">{format_big_number(item['Volume'])}</div>
                    </div>
                </div>
            </div>
            <br>""", unsafe_allow_html=True)

            if st.button(f"🔮 Analisis {ticker}", key=f"btn_{i}", use_container_width=True):
                st.session_state['selected_coin'] = item['Ticker']
                st.switch_page("pages/Detail.py")

    # ─── MARKET TABLE ─────────────────────────────────────────────────────────
    st.markdown('<br><div class="section-header">📋 Ringkasan Pasar Lengkap</div>', unsafe_allow_html=True)

    rows = ""
    for item in data:
        is_up  = item['Change'] >= 0
        arrow  = "▲" if is_up else "▼"
        color  = "var(--green)" if is_up else "var(--red)"
        ticker = item['Ticker'].replace('-USD', '')
        rows += f"""
        <tr>
            <td>
                <div class="asset-cell">
                    <img src="{item['Icon']}" class="tbl-logo" onerror="this.style.display='none'">
                    <span class="asset-name">{item['Name']}</span>
                    <span class="asset-sym">{ticker}</span>
                </div>
            </td>
            <td>{format_price(item['Price'])}</td>
            <td style="color:{color};">{arrow} {item['Change']:.2f}%</td>
            <td>{format_price(item['ATL'])}</td>
            <td>{format_big_number(item['MarketCap'])}</td>
            <td>{format_big_number(item['Volume'])}</td>
        </tr>"""

    st.markdown(f"""
    <div class="table-wrap">
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Aset</th>
                    <th>Harga Saat Ini</th>
                    <th>Perubahan 24 Jam</th>
                    <th>Harga Terendah Sepanjang Masa</th>
                    <th>Kapitalisasi Pasar</th>
                    <th>Volume Perdagangan (24j)</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-col">
        <div class="footer-label">⚙️ Tentang Sistem</div>
        <div class="footer-text">
            Dashboard ini menggunakan algoritma LSTM (Long Short-Term Memory) 
            untuk memprediksi harga aset kripto berdasarkan data historis. 
            Akurasi model dievaluasi menggunakan metrik RMSE dan MAPE.
        </div>
    </div>
    <div class="footer-col">
        <div class="footer-label">⚠️ Disclaimer</div>
        <div class="footer-text">
            Seluruh konten bersifat informatif dan bukan merupakan saran finansial. 
            Perdagangan kripto memiliki risiko tinggi. Lakukan riset mandiri (DYOR) 
            sebelum membuat keputusan investasi apapun.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)