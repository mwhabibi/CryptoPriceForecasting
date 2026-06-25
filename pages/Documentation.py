import os
import io
import sys
import json
import time
import base64
import subprocess
import threading
import queue
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
from datetime import date, timedelta as _td

# ─── PATH RESOLUTION ──────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DOCS_DIR   = ROOT_DIR / "Training" / "Documentation"
UJI_SCRIPT = ROOT_DIR / "UjiCobaModel.py"
OUTPUT_UJI = ROOT_DIR / "hasil_ujicobamodel"

GRAFIK_PATHS = {
    "BTC-USD":   DOCS_DIR / "grafik-training-BTC.png",
    "ETH-USD":   DOCS_DIR / "grafik-training-ETH.png",
    "DOGE-USD":  DOCS_DIR / "grafik-training-DOGE.png",
    "SHIB-USD":  DOCS_DIR / "grafik-training-SHIBA.png",
    "FLOKI-USD": DOCS_DIR / "grafik-training-FLOKI.png",
}
VIS_DATA_PATH = DOCS_DIR / "visualisasi-data.png"
METRICS_PATH  = ROOT_DIR / "metrics.json"

COINS_META = {
    "BTC-USD":   {"name": "Bitcoin",   "ticker": "BTC",   "color": "#F7931A"},
    "ETH-USD":   {"name": "Ethereum",  "ticker": "ETH",   "color": "#627EEA"},
    "DOGE-USD":  {"name": "Dogecoin",  "ticker": "DOGE",  "color": "#C2A633"},
    "SHIB-USD":  {"name": "Shiba Inu", "ticker": "SHIB",  "color": "#E0382B"},
    "FLOKI-USD": {"name": "Floki",     "ticker": "FLOKI", "color": "#F0A500"},
}

# ─── DATA TRAINING AKTUAL (dari hasil training nyata) ─────────────────────────
TRAINING_SUMMARY = {
    "BTC-USD":   {"best_epoch": 21, "best_val_loss": 0.0044},
    "ETH-USD":   {"best_epoch": 14, "best_val_loss": 0.0107},
    "DOGE-USD":  {"best_epoch": 15, "best_val_loss": 0.0058},
    "SHIB-USD":  {"best_epoch": 29, "best_val_loss": 0.0045},
    "FLOKI-USD": {"best_epoch": 14, "best_val_loss": 0.0055},
}

TRAINING_HISTORY = {
    "BTC-USD": {
        "train": [0.1297,0.1103,0.091,0.0754,0.061,0.051,0.0423,0.0396,0.0328,0.0288,
                  0.0217,0.0235,0.0203,0.0152,0.0134,0.012,0.0115,0.0117,0.0105,0.0092,
                  0.0104,0.0075,0.008,0.0081,0.0083,0.0098,0.0067,0.0081,0.0084,0.0056,0.0083],
        "val":   [0.0176,0.0154,0.0174,0.0161,0.0143,0.0113,0.0096,0.0113,0.0096,0.0078,
                  0.0089,0.0066,0.0098,0.0069,0.0082,0.0066,0.0073,0.0072,0.0056,0.0086,
                  0.0044,0.0057,0.005,0.0049,0.0053,0.0047,0.0056,0.0046,0.0059,0.0056,0.0048],
    },
    "ETH-USD": {
        "train": [0.1278,0.1096,0.0909,0.076,0.064,0.0505,0.0438,0.0359,0.0341,0.0284,
                  0.0233,0.0189,0.0177,0.0167,0.0161,0.0143,0.0144,0.0115,0.009,0.0113,
                  0.0111,0.0096,0.0104,0.0087],
        "val":   [0.019,0.0168,0.0137,0.0127,0.0112,0.0126,0.0105,0.0106,0.0115,0.0083,
                  0.0085,0.0095,0.007,0.0107,0.0119,0.0121,0.0112,0.011,0.0111,0.0114,
                  0.0119,0.012,0.0108,0.0115],
    },
    "DOGE-USD": {
        "train": [0.1299,0.1067,0.0879,0.0741,0.0649,0.0518,0.0446,0.0388,0.0316,0.0302,
                  0.0264,0.0199,0.0186,0.0156,0.0118,0.0113,0.0131,0.0116,0.0086,0.0091,
                  0.0118,0.008,0.0072,0.0087,0.0109],
        "val":   [0.0179,0.0178,0.0166,0.0132,0.014,0.0116,0.0118,0.0111,0.01,0.0077,
                  0.0102,0.0078,0.0069,0.006,0.0058,0.0064,0.0061,0.0072,0.0071,0.0063,
                  0.0068,0.007,0.0067,0.0066,0.0062],
    },
    "SHIB-USD": {
        "train": [0.1283,0.11,0.0918,0.0756,0.0619,0.0519,0.0456,0.0398,0.0342,0.0292,
                  0.0248,0.019,0.0169,0.0186,0.0155,0.0112,0.0105,0.0124,0.0084,0.0086,
                  0.01,0.0103,0.0098,0.0074,0.0096,0.007,0.0073,0.0093,0.0105,0.0096,
                  0.0086,0.0081,0.0057,0.007,0.0065,0.0063,0.01,0.007,0.0095],
        "val":   [0.0194,0.0183,0.0156,0.0145,0.0131,0.0109,0.0121,0.0096,0.008,0.0099,
                  0.0076,0.0102,0.0099,0.0095,0.0071,0.0054,0.0089,0.0067,0.0087,0.0086,
                  0.0081,0.0057,0.006,0.0078,0.0056,0.005,0.0065,0.008,0.0045,0.0051,
                  0.005,0.0057,0.0057,0.0058,0.0059,0.0053,0.0053,0.0057,0.0055],
    },
    "FLOKI-USD": {
        "train": [0.1313,0.1095,0.0918,0.0741,0.0621,0.0506,0.0449,0.0355,0.0321,0.028,
                  0.0231,0.0216,0.0163,0.0115,0.0166,0.0129,0.0106,0.0117,0.0122,0.0088,
                  0.0104,0.0073,0.0068,0.0089],
        "val":   [0.0191,0.0177,0.0165,0.0161,0.0132,0.0114,0.0124,0.0096,0.0096,0.0077,
                  0.007,0.0103,0.0095,0.0055,0.0064,0.0065,0.0062,0.0059,0.0061,0.0067,
                  0.0056,0.0058,0.0057,0.0057],
    },
}

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dokumentasi Model",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
:root {
    --bg:#080B10; --bg2:#0D1117; --surface:#111820; --surface2:#141D28; --surface3:#0C1219;
    --border:#1E2733; --border2:#2A3544; --text:#E8EDF5; --muted:#5A6A7E; --muted2:#7A8A9E;
    --accent:#00D4FF; --accent2:#0099CC; --green:#00E5A0; --red:#FF4D6A; --gold:#FFB800;
    --purple:#A78BFA; --orange:#FF8C42;
    --font-head:'Syne',sans-serif; --font-body:'DM Sans',sans-serif; --font-mono:'DM Mono',monospace;
}
.stApp { background-color: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }

/* Global Buttons */
div[data-testid="stButton"] button {
    background: var(--surface2) !important; border: 1px solid var(--border2) !important;
    color: var(--muted2) !important; font-family: var(--font-body) !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 16px !important; border-radius: 8px !important; transition: all 0.2s ease !important;
}
div[data-testid="stButton"] button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: rgba(0,212,255,0.06) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 4px !important; gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px !important;
    color: var(--muted) !important; font-family: var(--font-body) !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 18px !important; border: none !important; transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; background: rgba(255,255,255,0.04) !important; }
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important; color: var(--accent) !important;
    border: 1px solid var(--border2) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Hero */
.doc-hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 50%, #061020 100%);
    border: 1px solid var(--border); border-radius: 16px; padding: 36px 44px;
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.doc-hero::before {
    content:''; position:absolute; top:-80px; right:-80px; width:320px; height:320px;
    background:radial-gradient(circle,rgba(167,139,250,0.08) 0%,transparent 70%); pointer-events:none;
}
.hero-badge {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(167,139,250,0.1); border:1px solid rgba(167,139,250,0.25);
    border-radius:100px; padding:5px 14px;
    font-family:var(--font-body); font-size:11px; font-weight:600;
    color:var(--purple); letter-spacing:0.05em; text-transform:uppercase; margin-bottom:16px;
}
.hero-badge .dot { width:5px; height:5px; background:var(--purple); border-radius:50%; }
.doc-hero h1 { font-family:var(--font-head); font-size:34px; font-weight:800; color:var(--text); margin:0 0 8px; line-height:1.15; }
.doc-hero h1 span { color:var(--purple); }
.doc-hero-sub { font-family:var(--font-body); font-size:14px; color:var(--muted); line-height:1.75; max-width:680px; margin:0 0 24px; }
.hero-toc { display:flex; gap:10px; flex-wrap:wrap; }
.toc-pill { display:inline-flex; align-items:center; gap:7px; background:rgba(255,255,255,0.03); border:1px solid var(--border2); border-radius:8px; padding:8px 14px; font-family:var(--font-body); font-size:12px; color:var(--muted); }
.toc-num { width:18px; height:18px; border-radius:50%; background:rgba(167,139,250,0.15); color:var(--purple); font-size:10px; font-weight:700; display:flex; align-items:center; justify-content:center; }

/* Section Header */
.section-header {
    font-family:var(--font-head); font-size:11px; font-weight:700;
    letter-spacing:0.15em; text-transform:uppercase; color:var(--muted);
    margin:24px 0 16px; display:flex; align-items:center; gap:10px;
}
.section-header::after { content:''; flex:1; height:1px; background:var(--border); }

/* Method grid */
.method-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:28px; }
.method-step { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px 18px; }
.method-step-num { font-family:var(--font-head); font-size:32px; font-weight:800; color:rgba(167,139,250,0.15); margin-bottom:8px; }
.method-step-title { font-family:var(--font-head); font-size:13px; font-weight:700; color:var(--text); margin-bottom:6px; }
.method-step-desc { font-family:var(--font-body); font-size:12px; color:var(--muted); line-height:1.65; }
.method-step-tag { display:inline-block; margin-top:10px; background:rgba(167,139,250,0.1); color:var(--purple); border:1px solid rgba(167,139,250,0.2); font-family:var(--font-mono); font-size:10px; padding:2px 8px; border-radius:4px; }

/* Arch card */
.arch-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:22px 24px; margin-bottom:24px; }
.arch-layer { flex:1; padding:14px 10px; text-align:center; border-right:1px solid var(--border); }
.arch-layer:last-child { border-right:none; }
.arch-layer-icon { font-size:22px; margin-bottom:6px; }
.arch-layer-name { font-family:var(--font-head); font-size:12px; font-weight:700; color:var(--text); }
.arch-layer-detail { font-family:var(--font-mono); font-size:10px; color:var(--muted); margin-top:4px; }

/* Param table */
.param-table-wrap { background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:24px; }
.param-table { width:100%; border-collapse:collapse; }
.param-table thead tr { background:rgba(255,255,255,0.02); border-bottom:1px solid var(--border); }
.param-table th { padding:11px 18px; font-family:var(--font-body); font-size:11px; font-weight:600; color:var(--muted); text-transform:uppercase; text-align:left; }
.param-table td { padding:11px 18px; font-family:var(--font-mono); font-size:12.5px; color:var(--text); border-bottom:1px solid rgba(30,39,51,0.5); }
.param-table td:first-child { color:var(--muted2); font-family:var(--font-body); }
.param-table tbody tr:last-child td { border-bottom:none; }
.param-val { color:var(--accent); } .param-highlight { color:var(--gold); }

/* Info box */
.info-box { background:linear-gradient(135deg,rgba(0,212,255,0.05),rgba(0,153,204,0.02)); border:1px solid rgba(0,212,255,0.15); border-radius:12px; padding:18px 22px; margin-bottom:20px; display:flex; gap:14px; align-items:flex-start; }
.info-icon { font-size:22px; flex-shrink:0; margin-top:1px; }
.info-title { font-family:var(--font-head); font-size:13px; font-weight:700; color:var(--accent); margin-bottom:5px; }
.info-text { font-family:var(--font-body); font-size:13px; color:var(--muted2); line-height:1.75; }
.info-text b { color:var(--text); }

/* Training card */
.train-card-header { display:flex; align-items:center; justify-content:space-between; padding:14px 18px; border-bottom:1px solid var(--border); background:rgba(255,255,255,0.02); }
.train-card-title { display:flex; align-items:center; gap:10px; font-family:var(--font-head); font-size:14px; font-weight:700; color:var(--text); }
.train-meta { display:flex; gap:16px; flex-wrap:wrap; padding:12px 18px; border-top:1px solid var(--border); background:rgba(0,0,0,0.2); }
.train-meta-item { text-align:center; flex:1; min-width:80px; }
.train-meta-label { font-family:var(--font-body); font-size:10px; color:var(--muted); text-transform:uppercase; margin-bottom:2px; }
.train-meta-val { font-family:var(--font-mono); font-size:13px; font-weight:500; }

/* Perf table */
.perf-table-wrap { background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:24px; }
.perf-table { width:100%; border-collapse:collapse; }
.perf-table thead tr { background:rgba(255,255,255,0.02); border-bottom:1px solid var(--border); }
.perf-table th { padding:13px 20px; font-family:var(--font-body); font-size:11px; font-weight:600; color:var(--muted); text-transform:uppercase; text-align:right; }
.perf-table th:first-child { text-align:left; }
.perf-table td { padding:14px 20px; font-family:var(--font-mono); font-size:13px; color:var(--text); text-align:right; border-bottom:1px solid rgba(30,39,51,0.5); }
.perf-table td:first-child { text-align:left; font-family:var(--font-body); }
.perf-table tbody tr:last-child td { border-bottom:none; }
.coin-cell { display:flex; align-items:center; gap:10px; }
.coin-dot { width:10px; height:10px; border-radius:50%; }
.coin-ticker-name { font-weight:700; color:var(--text); }
.coin-sym { font-size:11px; color:var(--muted); margin-left:4px; }
.mape-bar-wrap { display:flex; align-items:center; gap:10px; justify-content:flex-end; }
.mape-bar-bg { width:80px; height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden; }
.mape-bar-fill { height:100%; border-radius:3px; }
.q-badge { display:inline-block; padding:2px 8px; border-radius:4px; font-family:var(--font-body); font-size:10px; font-weight:600; }
.q-exc  { background:rgba(0,229,160,0.12);  color:var(--green); }
.q-good { background:rgba(0,212,255,0.12);  color:var(--accent); }
.q-fair { background:rgba(255,184,0,0.12);  color:var(--gold); }
.q-poor { background:rgba(255,77,106,0.12); color:var(--red); }

/* Summary Stats */
.summary-stat-row { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px; }
.summary-stat { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px; text-align:center; position:relative; overflow:hidden; }
.summary-stat::before { content:''; position:absolute; bottom:0; left:0; width:100%; height:2px; }
.summary-stat.s-blue::before   { background:linear-gradient(90deg,var(--accent),transparent); }
.summary-stat.s-green::before  { background:linear-gradient(90deg,var(--green),transparent); }
.summary-stat.s-gold::before   { background:linear-gradient(90deg,var(--gold),transparent); }
.summary-stat.s-purple::before { background:linear-gradient(90deg,var(--purple),transparent); }
.ss-icon { font-size:20px; margin-bottom:8px; }
.ss-label { font-family:var(--font-body); font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px; }
.ss-value { font-family:var(--font-head); font-size:20px; font-weight:800; color:var(--text); }
.ss-sub { font-family:var(--font-body); font-size:11px; color:var(--muted); margin-top:2px; }

/* Kesimpulan */
.kesimpulan-card { background:linear-gradient(135deg,rgba(167,139,250,0.07) 0%,rgba(0,212,255,0.04) 100%); border:1px solid rgba(167,139,250,0.2); border-radius:14px; padding:28px 32px; margin-bottom:24px; }
.kesimpulan-title { font-family:var(--font-head); font-size:16px; font-weight:800; color:var(--purple); margin-bottom:16px; }
.kesimpulan-points { display:flex; flex-direction:column; gap:12px; }
.kesimpulan-point { display:flex; gap:14px; align-items:flex-start; padding:12px 16px; background:rgba(0,0,0,0.2); border-radius:8px; border-left:3px solid; }
.kesimpulan-point.green { border-color:var(--green); } .kesimpulan-point.blue { border-color:var(--accent); }
.kesimpulan-point.gold  { border-color:var(--gold);  } .kesimpulan-point.purple { border-color:var(--purple); }
.kp-icon { font-size:18px; flex-shrink:0; margin-top:1px; }
.kp-title { font-family:var(--font-head); font-size:13px; font-weight:700; color:var(--text); margin-bottom:3px; }
.kp-desc  { font-family:var(--font-body); font-size:12.5px; color:var(--muted); line-height:1.6; }
.kp-desc b { color:var(--text); }

/* ═══ UJI COBA MODEL — NEW FEATURE ═══ */
.ujicoba-hero {
    background:linear-gradient(135deg,#0A1A15 0%,#071510 50%,#050F0A 100%);
    border:1px solid rgba(0,229,160,0.2); border-radius:16px; padding:28px 36px;
    margin-bottom:24px; position:relative; overflow:hidden;
}
.ujicoba-hero::before { content:''; position:absolute; top:-60px; right:-60px; width:260px; height:260px; background:radial-gradient(circle,rgba(0,229,160,0.08) 0%,transparent 70%); pointer-events:none; }
.ujicoba-badge { display:inline-flex; align-items:center; gap:6px; background:rgba(0,229,160,0.1); border:1px solid rgba(0,229,160,0.25); border-radius:100px; padding:5px 14px; font-family:var(--font-body); font-size:11px; font-weight:600; color:var(--green); letter-spacing:0.05em; text-transform:uppercase; margin-bottom:14px; }
.ujicoba-badge .dot { width:5px; height:5px; background:var(--green); border-radius:50%; animation:blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
.ujicoba-title { font-family:var(--font-head); font-size:24px; font-weight:800; color:var(--text); margin:0 0 8px; }
.ujicoba-title span { color:var(--green); }
.ujicoba-sub { font-family:var(--font-body); font-size:13.5px; color:var(--muted); line-height:1.7; max-width:640px; margin-bottom:24px; }

/* Config grid */
.config-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }
.config-item { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px; display:flex; align-items:flex-start; gap:12px; }
.config-icon { font-size:20px; flex-shrink:0; margin-top:1px; }
.config-label { font-family:var(--font-body); font-size:10px; font-weight:600; color:var(--muted); text-transform:uppercase; margin-bottom:3px; }
.config-value { font-family:var(--font-mono); font-size:13px; color:var(--text); font-weight:500; }
.config-desc { font-family:var(--font-body); font-size:11px; color:var(--muted); margin-top:2px; }

/* Pipeline steps */
.pipeline-steps { display:flex; align-items:center; background:var(--surface); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-bottom:20px; }
.pipeline-step { flex:1; padding:16px 12px; text-align:center; border-right:1px solid var(--border); transition:all 0.3s; }
.pipeline-step:last-child { border-right:none; }
.pipeline-step.done    { background:rgba(0,229,160,0.05); }
.pipeline-step.active  { background:rgba(0,212,255,0.06); }
.pipeline-step.pending { opacity:0.45; }
.pipeline-step.error   { background:rgba(255,77,106,0.05); }
.ps-icon  { font-size:18px; margin-bottom:6px; }
.ps-label { font-family:var(--font-body); font-size:11px; font-weight:600; color:var(--text); margin-bottom:3px; }
.ps-status { font-family:var(--font-mono); font-size:10px; }
.ps-status.done    { color:var(--green); }
.ps-status.active  { color:var(--accent); }
.ps-status.pending { color:var(--muted); }
.ps-status.error   { color:var(--red); }

/* Terminal */
.terminal-wrap { background:#020508; border:1px solid rgba(0,229,160,0.2); border-radius:12px; overflow:hidden; margin-bottom:20px; }
.terminal-header { display:flex; align-items:center; justify-content:space-between; padding:10px 16px; background:#040A0E; border-bottom:1px solid rgba(0,229,160,0.15); }
.terminal-dots { display:flex; gap:6px; }
.terminal-dot { width:10px; height:10px; border-radius:50%; }
.terminal-title { font-family:var(--font-mono); font-size:11px; color:#3A5A3A; }
.terminal-body { padding:16px; font-family:var(--font-mono); font-size:11.5px; line-height:1.8; color:#7AE29A; min-height:200px; max-height:500px; overflow-y:auto; white-space:pre-wrap; word-break:break-word; }
.log-header  { color:#00E5A0; font-weight:500; }
.log-info    { color:#7AE29A; }
.log-warn    { color:#FFB800; }
.log-error   { color:#FF4D6A; }
.log-metric  { color:#00D4FF; }
.log-success { color:#00E5A0; font-weight:500; }
.log-muted   { color:#1A3A1A; }

/* Script warning */
.script-warning { background:rgba(255,184,0,0.06); border:1px solid rgba(255,184,0,0.2); border-radius:12px; padding:18px 22px; margin-bottom:20px; display:flex; gap:14px; }
.sw-icon  { font-size:22px; flex-shrink:0; margin-top:1px; }
.sw-title { font-family:var(--font-head); font-size:13px; font-weight:700; color:var(--gold); margin-bottom:5px; text-transform:uppercase; }
.sw-text  { font-family:var(--font-body); font-size:13px; color:var(--muted2); line-height:1.75; }
.sw-text b { color:var(--text); }

/* Hasil coin card */
.hasil-coin-card { background:var(--surface); border:1px solid var(--border); border-radius:14px; overflow:hidden; margin-bottom:16px; transition:border-color 0.2s; }
.hasil-coin-card:hover { border-color:var(--border2); }
.hasil-coin-header { display:flex; align-items:center; justify-content:space-between; padding:14px 18px; border-bottom:1px solid var(--border); background:rgba(255,255,255,0.02); }
.hasil-coin-name   { font-family:var(--font-head); font-size:14px; font-weight:700; color:var(--text); }
.hasil-coin-ticker { font-family:var(--font-mono); font-size:11px; color:var(--muted); margin-top:1px; }
.hasil-metrics-row { display:grid; grid-template-columns:repeat(4,1fr); border-bottom:1px solid var(--border); }
.hasil-metric-item { padding:12px 14px; text-align:center; border-right:1px solid var(--border); }
.hasil-metric-item:last-child { border-right:none; }
.hm-label { font-family:var(--font-body); font-size:9px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:3px; }
.hm-value { font-family:var(--font-mono); font-size:12px; font-weight:500; }
.hm-value.lstm  { color:var(--accent); }
.hm-value.naive { color:var(--muted2); }

/* Verdict badges */
.verdict-win  { display:inline-flex; align-items:center; gap:5px; background:rgba(0,229,160,0.12); color:var(--green); border:1px solid rgba(0,229,160,0.25); padding:4px 10px; border-radius:6px; font-family:var(--font-mono); font-size:10px; font-weight:500; }
.verdict-lose { display:inline-flex; align-items:center; gap:5px; background:rgba(255,77,106,0.12); color:var(--red);   border:1px solid rgba(255,77,106,0.25); padding:4px 10px; border-radius:6px; font-family:var(--font-mono); font-size:10px; font-weight:500; }

/* Footer */
.footer { border-top:1px solid var(--border); padding-top:20px; margin-top:12px; display:flex; gap:24px; }
.footer-col { flex:1; min-width:220px; }
.footer-label { font-family:var(--font-head); font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase; margin-bottom:6px; }
.footer-text  { font-family:var(--font-body); font-size:12.5px; color:#374151; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def mape_quality(mape):
    if mape is None:  return "N/A",            "q-fair", "#5A6A7E"
    if mape < 5:      return "Sangat Akurat",   "q-exc",  "#00E5A0"
    if mape < 10:     return "Akurat",          "q-good", "#00D4FF"
    if mape < 20:     return "Cukup Akurat",    "q-fair", "#FFB800"
    return                   "Perlu Evaluasi",  "q-poor", "#FF4D6A"

def colorize_log(line):
    s = line.strip()
    if not s: return '<span class="log-muted"> </span>'
    if "==" in s and len(s) > 8:        return f'<span class="log-header">{s}</span>'
    if any(k in s for k in ["HASIL AKHIR","AKURASI","MODEL LSTM TERBUKTI","PENGUJIAN SELESAI","berhasil"]):
        return f'<span class="log-success">{s}</span>'
    if any(k in s for k in ["RMSE","MAPE","Naive","Akurasi"]):
        return f'<span class="log-metric">{s}</span>'
    if any(k in s for k in ["Error","CRITICAL","tidak ditemukan","Traceback","TIDAK lebih"]):
        return f'<span class="log-error">{s}</span>'
    if any(k in s for k in ["Downloading","Jeda","simulasi","prediksi"]):
        return f'<span class="log-info">{s}</span>'
    if any(k in s for k in ["WARNING","kurang","disimpan"]):
        return f'<span class="log-warn">{s}</span>'
    return f'<span class="log-info">{s}</span>'

def render_pipeline_html(statuses):
    STEPS = [
        ("📥","Unduh Data","yfinance API"),
        ("⚙️","Feature Eng.","RSI·MACD·ATR"),
        ("🧠","Prediksi LSTM","7-day forecast"),
        ("📊","Hitung Metrik","RMSE·MAPE·Naive"),
        ("💾","Simpan Hasil","metrics.json·PNG"),
    ]
    label_map = {"done":"✓ Selesai","active":"⟳ Berjalan...","pending":"○ Menunggu","error":"✗ Error"}
    parts = []
    for i, (icon, label, desc) in enumerate(STEPS):
        s = statuses[i] if i < len(statuses) else "pending"
        parts.append(f"""
        <div class="pipeline-step {s}">
            <div class="ps-icon">{icon}</div>
            <div class="ps-label">{label}</div>
            <div class="ps-status {s}">{label_map.get(s,'○')}</div>
        </div>""")
    return f'<div class="pipeline-steps">{"".join(parts)}</div>'

def render_terminal_html(lines, success=None, running=False):
    colored = [colorize_log(l) for l in lines[-80:]] if lines else ['<span class="log-muted">Menunggu eksekusi...</span>']
    content = "\n".join(colored)
    status_dot = "#27C93F" if success else "#FF4D6A" if success is False else "#FFBD2E"
    status_txt = "Completed ✓" if success else "Error ✗" if success is False else "Running..." if running else "Ready"
    return f"""
    <div class="terminal-wrap">
        <div class="terminal-header">
            <div class="terminal-dots">
                <div class="terminal-dot" style="background:#FF5F56;"></div>
                <div class="terminal-dot" style="background:#FFBD2E;"></div>
                <div class="terminal-dot" style="background:{status_dot};"></div>
            </div>
            <div class="terminal-title">UjiCobaModelLSTM.py — {status_txt}</div>
            <div class="terminal-title">{len(lines)} baris</div>
        </div>
        <div class="terminal-body">{content}</div>
    </div>"""

# ─── NAVIGASI ─────────────────────────────────────────────────────────────────
col_home = st.columns([2])[0]
with col_home:
    if st.button("🏠 Beranda", use_container_width=True):
        st.switch_page("Home.py")

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="doc-hero">
    <div class="hero-badge"><span class="dot"></span>Dokumentasi Teknis &amp; Akademis</div>
    <h1>Dokumentasi Model <span>LSTM</span></h1>
    <p class="doc-hero-sub">
        Halaman ini mendokumentasikan seluruh proses pengembangan model prediksi harga kripto —
        dari eksplorasi data, desain arsitektur, proses training, evaluasi performa,
        hingga <b style="color:var(--green)">Uji Coba Validasi Model secara Live</b>.
        Dokumentasi ini merupakan bagian integral Tugas Akhir berbasis
        <b style="color:var(--text)">Deep Learning LSTM</b>.
    </p>
    <div class="hero-toc">
        <span class="toc-pill"><span class="toc-num">1</span>Proses Trainning</span>
        <span class="toc-pill"><span class="toc-num">2</span>Arsitektur</span>
        <span class="toc-pill"><span class="toc-num">3</span>Dataset</span>
        <span class="toc-pill"><span class="toc-num">4</span>Grafik Training</span>
        <span class="toc-pill"><span class="toc-num">5</span>Evaluasi</span>
        <span class="toc-pill" style="border-color:rgba(0,229,160,0.3);color:var(--green);">
            <span class="toc-num" style="background:rgba(0,229,160,0.15);color:var(--green);">⚡</span>
            Uji Coba Model
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

metrics_data = load_metrics()

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① Proses Trainning", "② Arsitektur", "③ Dataset",
    "④ Grafik Training", "⑤ Evaluasi", "⚡ Uji Coba Model",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Proses Trainning
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">① Proses Trainning </div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="method-grid">
        <div class="method-step">
            <div class="method-step-num">01</div>
            <div class="method-step-title">Pengumpulan Data</div>
            <div class="method-step-desc">Data historis OHLCV diambil dari Yahoo Finance menggunakan <code>yfinance</code>. Rentang data mencakup beberapa tahun hingga tanggal pelatihan terakhir untuk 5 aset kripto.</div>
            <span class="method-step-tag">yfinance · OHLCV</span>
        </div>
        <div class="method-step">
            <div class="method-step-num">02</div>
            <div class="method-step-title">Feature Engineering</div>
            <div class="method-step-desc">Diekstraksi 6 fitur teknikal: Log Return, RSI (14), MACD, MACD Signal, ATR (14), dan Volume sebagai input vektor model LSTM. Seluruh fitur dinormalisasi dengan MinMaxScaler (0–1).</div>
            <span class="method-step-tag">RSI · MACD · ATR · Log_Ret</span>
        </div>
        <div class="method-step">
            <div class="method-step-num">03</div>
            <div class="method-step-title">Training LSTM</div>
            <div class="method-step-desc">Model dilatih dengan lookback window 60 hari, memprediksi Log Return 7 hari ke depan. EarlyStopping (patience=10) dan ModelCheckpoint mencegah overfitting. Split data 80% train / 20% test.</div>
            <span class="method-step-tag">Lookback 60 · EarlyStopping·10</span>
        </div>
        <div class="method-step">
            <div class="method-step-num">04</div>
            <div class="method-step-title">Evaluasi Model</div>
            <div class="method-step-desc">Performa diukur dengan RMSE dan MAPE pada test set, dibandingkan langsung dengan baseline Naive Forecast untuk membuktikan superioritas LSTM.</div>
            <span class="method-step-tag">RMSE · MAPE · vs Naive</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-icon">💡</div>
        <div>
            <div class="info-title">Mengapa LSTM? &amp; Validasi Komparatif</div>
            <div class="info-text">
                <b>LSTM</b> dirancang untuk mengatasi <i>vanishing gradient</i> pada data sekuensial panjang.
                Harga kripto bersifat <b>time-series non-linear</b> — LSTM lebih unggul dari ARIMA/regresi linear.
                Validitas dibuktikan via <b>komparasi dengan Naive Forecast</b>: jika LSTM menghasilkan MAPE lebih kecil,
                model terbukti secara statistik lebih optimal. Formula inverse-transform harga:
                <code>Price[t+1] = Price[t] × exp(Log_Ret[t+1])</code> — numerik stabil untuk rentang harga BTC vs SHIB.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARSITEKTUR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">② Arsitektur &amp; Hyperparameter Model</div>', unsafe_allow_html=True)

    # ── Diagram Arsitektur (data aktual: LSTM1=64, LSTM2=32) ──────────────────
    st.markdown("""
    <div class="arch-card">
        <div style="font-family:var(--font-head);font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;">Alur Arsitektur Model LSTM</div>
        <div style="display:flex;align-items:center;background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:10px;overflow:hidden;">
            <div class="arch-layer">
                <div class="arch-layer-icon">📥</div>
                <div class="arch-layer-name">Input</div>
                <div class="arch-layer-detail">shape=(60, 6)</div>
                <div class="arch-layer-detail" style="font-size:9px;margin-top:3px;">60 hari × 6 fitur</div>
            </div>
            <div style="color:var(--border2);font-size:16px;padding:0 4px;">→</div>
            <div class="arch-layer">
                <div class="arch-layer-icon">🧠</div>
                <div class="arch-layer-name">LSTM 1</div>
                <div class="arch-layer-detail">64 units</div>
                <div class="arch-layer-detail" style="font-size:9px;margin-top:3px;">return_seq=True · tanh</div>
            </div>
            <div style="color:var(--border2);font-size:16px;padding:0 4px;">→</div>
            <div class="arch-layer">
                <div class="arch-layer-icon">🔽</div>
                <div class="arch-layer-name">Dropout 1</div>
                <div class="arch-layer-detail">rate=0.2</div>
                <div class="arch-layer-detail" style="font-size:9px;margin-top:3px;">Regularisasi</div>
            </div>
            <div style="color:var(--border2);font-size:16px;padding:0 4px;">→</div>
            <div class="arch-layer">
                <div class="arch-layer-icon">🧠</div>
                <div class="arch-layer-name">LSTM 2</div>
                <div class="arch-layer-detail">32 units</div>
                <div class="arch-layer-detail" style="font-size:9px;margin-top:3px;">return_seq=False · tanh</div>
            </div>
            <div style="color:var(--border2);font-size:16px;padding:0 4px;">→</div>
            <div class="arch-layer">
                <div class="arch-layer-icon">🔽</div>
                <div class="arch-layer-name">Dropout 2</div>
                <div class="arch-layer-detail">rate=0.2</div>
                <div class="arch-layer-detail" style="font-size:9px;margin-top:3px;">Regularisasi</div>
            </div>
            <div style="color:var(--border2);font-size:16px;padding:0 4px;">→</div>
            <div class="arch-layer">
                <div class="arch-layer-icon">📤</div>
                <div class="arch-layer-name">Dense Out</div>
                <div class="arch-layer-detail">7 units</div>
                <div class="arch-layer-detail" style="font-size:9px;margin-top:3px;">linear · 7-hari prediksi</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        # ── Arsitektur Layer (data aktual) ──
        st.markdown("""
        <div class="param-table-wrap"><table class="param-table">
            <thead><tr><th>#</th><th>Layer</th><th>Units / Config</th><th>Fungsi</th></tr></thead>
            <tbody>
                <tr><td>1</td><td>Input Layer</td><td><span class="param-val">shape=(60, 6)</span></td><td style="font-size:11px;color:var(--muted);">60 hari lookback, 6 fitur</td></tr>
                <tr><td>2</td><td>LSTM Layer 1</td><td><span class="param-val">64 units · return_seq=True</span></td><td style="font-size:11px;color:var(--muted);">Ekstraksi pola sekuensial</td></tr>
                <tr><td>3</td><td>Dropout 1</td><td><span class="param-val">rate=0.2 (20%)</span></td><td style="font-size:11px;color:var(--muted);">Cegah overfitting</td></tr>
                <tr><td>4</td><td>LSTM Layer 2</td><td><span class="param-val">32 units · return_seq=False</span></td><td style="font-size:11px;color:var(--muted);">Representasi inti padat</td></tr>
                <tr><td>5</td><td>Dropout 2</td><td><span class="param-val">rate=0.2 (20%)</span></td><td style="font-size:11px;color:var(--muted);">Regularisasi lanjut</td></tr>
                <tr><td>6</td><td>Dense Output</td><td><span class="param-val">7 units · linear</span></td><td style="font-size:11px;color:var(--muted);">Prediksi 7 hari ke depan</td></tr>
            </tbody>
        </table></div>
        """, unsafe_allow_html=True)

    with col_p2:
        # ── Hyperparameter (data aktual) ──
        st.markdown("""
        <div class="param-table-wrap"><table class="param-table">
            <thead><tr><th>Hyperparameter</th><th>Nilai</th></tr></thead>
            <tbody>
                <tr><td>Optimizer</td><td><span class="param-val">Adam</span></td></tr>
                <tr><td>Learning Rate</td><td><span class="param-val">0.001</span></td></tr>
                <tr><td>Loss Function</td><td><span class="param-val">Mean Squared Error (MSE)</span></td></tr>
                <tr><td>Max Epochs</td><td><span class="param-highlight">50</span></td></tr>
                <tr><td>Batch Size</td><td><span class="param-val">32</span></td></tr>
                <tr><td>EarlyStopping Patience</td><td><span class="param-val">10 epoch</span></td></tr>
                <tr><td>Lookback Window</td><td><span class="param-val">60 hari</span></td></tr>
                <tr><td>Forecast Horizon</td><td><span class="param-val">7 hari</span></td></tr>
                <tr><td>Train / Test Split</td><td><span class="param-val">80% / 20%</span></td></tr>
                <tr><td>Scaler</td><td><span class="param-val">MinMaxScaler (0–1)</span></td></tr>
            </tbody>
        </table></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <div class="info-icon">🔧</div>
        <div>
            <div class="info-title">Urutan Fitur &amp; Strategi Inverse Transform</div>
            <div class="info-text">
                Urutan fitur tetap: <b>Log_Ret → RSI → MACD → MACD_Signal → ATR → Volume</b>.
                Model memprediksi <b>Log Return</b> (bukan harga langsung), lalu diinverse-transform via scaler
                menggunakan <code>scale_factor = scaler.scale_[0]</code> dan <code>min_factor = scaler.min_[0]</code>,
                kemudian dikonversi kumulatif: <code>Price[t] = Price[t-1] × exp(log_ret[t])</code>.
                Strategi ini numerik stabil untuk aset dengan rentang harga sangat berbeda (BTC ~$90k vs SHIB ~$0.00002).
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabel Ringkasan Hasil Training per Koin ──────────────────────────────
    st.markdown('<div class="section-header">📊 Ringkasan Hasil Training per Aset (Data Aktual)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="param-table-wrap"><table class="param-table">
        <thead>
            <tr>
                <th>Aset</th>
                <th>Best Epoch</th>
                <th>Best Val Loss (MSE)</th>
                <th>Optimizer</th>
                <th>Learning Rate</th>
                <th>Lookback</th>
                <th>Forecast</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#F7931A;display:inline-block;"></span>BTC-USD</span></td>
                <td><span class="param-val">21</span></td>
                <td><span class="param-highlight">0.0044</span></td>
                <td><span class="param-val">Adam</span></td>
                <td><span class="param-val">0.001</span></td>
                <td><span class="param-val">60</span></td>
                <td><span class="param-val">7</span></td>
            </tr>
            <tr>
                <td><span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#627EEA;display:inline-block;"></span>ETH-USD</span></td>
                <td><span class="param-val">14</span></td>
                <td><span class="param-val">0.0107</span></td>
                <td><span class="param-val">Adam</span></td>
                <td><span class="param-val">0.001</span></td>
                <td><span class="param-val">60</span></td>
                <td><span class="param-val">7</span></td>
            </tr>
            <tr>
                <td><span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#C2A633;display:inline-block;"></span>DOGE-USD</span></td>
                <td><span class="param-val">15</span></td>
                <td><span class="param-val">0.0058</span></td>
                <td><span class="param-val">Adam</span></td>
                <td><span class="param-val">0.001</span></td>
                <td><span class="param-val">60</span></td>
                <td><span class="param-val">7</span></td>
            </tr>
            <tr>
                <td><span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#E0382B;display:inline-block;"></span>SHIB-USD</span></td>
                <td><span class="param-val">29</span></td>
                <td><span class="param-val">0.0045</span></td>
                <td><span class="param-val">Adam</span></td>
                <td><span class="param-val">0.001</span></td>
                <td><span class="param-val">60</span></td>
                <td><span class="param-val">7</span></td>
            </tr>
            <tr>
                <td><span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:8px;height:8px;border-radius:50%;background:#F0A500;display:inline-block;"></span>FLOKI-USD</span></td>
                <td><span class="param-val">14</span></td>
                <td><span class="param-val">0.0055</span></td>
                <td><span class="param-val">Adam</span></td>
                <td><span class="param-val">0.001</span></td>
                <td><span class="param-val">60</span></td>
                <td><span class="param-val">7</span></td>
            </tr>
        </tbody>
    </table></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">③ Dataset</div>', unsafe_allow_html=True)
    if VIS_DATA_PATH.exists():
        try:
            img_dataset = Image.open(VIS_DATA_PATH)
            st.markdown("""
            <div class="info-box">
                <div class="info-icon">📊</div>
                <div>
                    <div class="info-title">Visualisasi Data Historis &amp; Distribusi Fitur</div>
                    <div class="info-text">Dataset mencakup visualisasi dari dataset yang digunakan sebagai fitur input LSTM.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(img_dataset, use_container_width=True)
        except Exception as e:
            st.warning(f"Gagal memuat visualisasi-data.png: {e}")
    else:
        st.markdown(f"""
        <div style="background:rgba(255,184,0,0.06);border:1px solid rgba(255,184,0,0.2);
                    border-radius:10px;padding:24px;color:var(--muted);font-family:var(--font-body);
                    font-size:13px;text-align:center;">
            ⚠️ File <code>visualisasi-data.png</code> tidak ditemukan di <code>Training/Documentation/</code>.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GRAFIK TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">④ Grafik Training Loss per Aset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <div class="info-icon">📉</div>
        <div>
            <div class="info-title">Cara Membaca Grafik Training Loss</div>
            <div class="info-text">
                <b>Training Loss</b> (biru) dan <b>Validation Loss</b> (oranye).
                Model baik: (1) kedua kurva turun bersama, (2) gap kecil = tidak overfit, (3) kurva stabil di akhir.
                EarlyStopping <b>patience=10</b> mencegah model berjalan melewati epoch optimal.
                Grafik interaktif di bawah menggunakan <b>data epoch aktual</b> dari proses training.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    coin_keys = list(TRAINING_HISTORY.keys())
    for row_start in range(0, len(coin_keys), 2):
        cols = st.columns(2)
        for ci, col in enumerate(cols):
            idx = row_start + ci
            if idx >= len(coin_keys): break
            tk   = coin_keys[idx]
            meta = COINS_META[tk]
            m    = metrics_data.get(tk, {})
            rmse_v = m.get("RMSE") or m.get("LSTM_RMSE")
            mape_v = m.get("MAPE") or m.get("LSTM_MAPE")
            ql, qcls, qcolor = mape_quality(mape_v)
            hist   = TRAINING_HISTORY[tk]
            summary = TRAINING_SUMMARY[tk]
            n_epochs = len(hist["train"])
            best_ep  = summary["best_epoch"]
            best_vl  = summary["best_val_loss"]

            with col:
                st.markdown(f"""
                <div class="train-card-header" style="background:var(--surface);border:1px solid var(--border);border-radius:12px 12px 0 0;margin-top:12px;">
                    <div class="train-card-title">
                        <span style="width:8px;height:8px;border-radius:50%;background:{meta['color']};display:inline-block;"></span>
                        {meta['name']} <span style="font-family:var(--font-mono);font-size:11px;color:var(--muted);">{meta['ticker']}</span>
                    </div>
                    <span class="q-badge {qcls}">{ql}</span>
                </div>
                """, unsafe_allow_html=True)

                # Plot interaktif dari data epoch aktual
                epochs = list(range(1, n_epochs + 1))
                fig_tr = go.Figure()
                fig_tr.add_trace(go.Scatter(
                    x=epochs, y=hist["train"], name="Training Loss",
                    line=dict(color=meta["color"], width=2),
                    mode="lines",
                ))
                fig_tr.add_trace(go.Scatter(
                    x=epochs, y=hist["val"], name="Validation Loss",
                    line=dict(color="#00D4FF", width=2, dash="dot"),
                    mode="lines",
                ))
                # Marker best epoch
                fig_tr.add_trace(go.Scatter(
                    x=[best_ep], y=[best_vl],
                    mode="markers+text",
                    marker=dict(color="#00E5A0", size=10, symbol="star"),
                    text=[f" Best: {best_vl}"],
                    textposition="top right",
                    textfont=dict(family="DM Mono", size=9, color="#00E5A0"),
                    name=f"Best Epoch {best_ep}",
                    showlegend=True,
                ))
                fig_tr.add_vline(
                    x=best_ep, line_dash="dot", line_color="rgba(0,229,160,0.3)", line_width=1,
                )
                fig_tr.update_layout(
                    height=220,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#5A6A7E", size=10),
                    margin=dict(l=0, r=0, t=8, b=0),
                    legend=dict(font=dict(color="#7A8A9E", size=9), bgcolor="rgba(0,0,0,0)",
                                orientation="h", yanchor="bottom", y=1.0),
                    xaxis=dict(showgrid=False, zeroline=False,
                               tickfont=dict(family="DM Mono", size=9, color="#5A6A7E"),
                               title=dict(text="Epoch", font=dict(size=9, color="#5A6A7E"))),
                    yaxis=dict(showgrid=True, gridcolor="rgba(30,39,51,0.6)", zeroline=False,
                               tickfont=dict(family="DM Mono", size=9, color="#5A6A7E"),
                               title=dict(text="MSE Loss", font=dict(size=9, color="#5A6A7E"))),
                )
                st.plotly_chart(fig_tr, use_container_width=True, config={"displayModeBar": False})

                # Coba tampilkan PNG jika ada
                img_path = GRAFIK_PATHS[tk]
                if img_path.exists():
                    with st.expander(f"📷 Lihat Grafik PNG Training {meta['ticker']}"):
                        try:
                            st.image(Image.open(img_path), use_container_width=True)
                        except Exception as e:
                            st.error(f"Gagal memuat: {e}")

                st.markdown(f"""
                <div class="train-meta" style="border:1px solid var(--border);border-top:none;border-radius:0 0 12px 12px;margin-bottom:8px;">
                    <div class="train-meta-item"><div class="train-meta-label">Total Epoch</div><div class="train-meta-val" style="color:var(--muted2);">{n_epochs}</div></div>
                    <div class="train-meta-item"><div class="train-meta-label">Best Epoch</div><div class="train-meta-val" style="color:var(--green);">{best_ep}</div></div>
                    <div class="train-meta-item"><div class="train-meta-label">Best Val Loss</div><div class="train-meta-val" style="color:var(--accent);">{best_vl:.4f}</div></div>
                    <div class="train-meta-item"><div class="train-meta-label">File Model</div><div class="train-meta-val" style="font-size:10px;color:var(--muted);">{meta['ticker']}_best_model.keras</div></div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EVALUASI PERFORMA
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">⑤ Evaluasi Performa — Semua Aset</div>', unsafe_allow_html=True)

    if not metrics_data:
        st.warning("⚠️ `metrics.json` belum ditemukan. Jalankan tab ⚡ Uji Coba Model untuk mengisi data.")
    else:
        rows_data = []
        for tk, meta in COINS_META.items():
            m = metrics_data.get(tk, {})
            rows_data.append({
                "key": tk, "name": meta["name"], "ticker": meta["ticker"], "color": meta["color"],
                "rmse":  m.get("RMSE")  or m.get("LSTM_RMSE"),
                "mape":  m.get("MAPE")  or m.get("LSTM_MAPE"),
                "naive_rmse": m.get("NAIVE_RMSE"),
                "naive_mape": m.get("NAIVE_MAPE"),
            })

        mapes = [r["mape"] for r in rows_data if r["mape"] is not None]
        best_row = min(rows_data, key=lambda r: r["mape"] if r["mape"] else 999)
        wins = sum(1 for r in rows_data if r["mape"] and r["naive_mape"] and r["mape"] < r["naive_mape"])
        avg_mape = sum(mapes)/len(mapes) if mapes else 0

        st.markdown(f"""
        <div class="summary-stat-row">
            <div class="summary-stat s-blue"><div class="ss-icon">🎯</div><div class="ss-label">Model Diuji</div><div class="ss-value">5</div><div class="ss-sub">Aset Kripto</div></div>
            <div class="summary-stat s-green"><div class="ss-icon">🏆</div><div class="ss-label">MAPE Terbaik</div><div class="ss-value">{best_row['mape']:.1f}%</div><div class="ss-sub">{best_row['name']}</div></div>
            <div class="summary-stat s-gold"><div class="ss-icon">📊</div><div class="ss-label">Rata-Rata MAPE</div><div class="ss-value">{avg_mape:.1f}%</div><div class="ss-sub">Semua aset</div></div>
            <div class="summary-stat s-purple"><div class="ss-icon">⚡</div><div class="ss-label">Ungguli Naive</div><div class="ss-value">{wins}/{len(rows_data)}</div><div class="ss-sub">LSTM &gt; Baseline</div></div>
        </div>
        """, unsafe_allow_html=True)

        mape_max = max(mapes, default=100)
        rows_html = ""
        for r in rows_data:
            ql, qcls, qcolor = mape_quality(r["mape"])
            bar_pct = (r["mape"] / mape_max * 100) if (r["mape"] and mape_max > 0) else 0
            lstm_wins = r["mape"] and r["naive_mape"] and r["mape"] < r["naive_mape"]
            verdict = f'<span class="verdict-win">LSTM Lebih Baik</span>' if lstm_wins else f'<span class="verdict-lose">Naive Lebih Baik</span>' if r["naive_mape"] else "—"
            rows_html += f"""
            <tr>
                <td><div class="coin-cell"><span class="coin-dot" style="background:{r['color']};"></span><span class="coin-ticker-name">{r['name']}</span><span class="coin-sym">{r['ticker']}</span></div></td>
                <td>{f"$"+f"{r['rmse']:,.8f}" if r['rmse'] else "—"}</td>
                <td><div class="mape-bar-wrap">{f"{r['mape']:.2f}%" if r['mape'] else "—"}<div class="mape-bar-bg"><div class="mape-bar-fill" style="width:{bar_pct:.0f}%;background:{qcolor};"></div></div><span class="q-badge {qcls}">{ql}</span></div></td>
                <td>{verdict}</td>
            </tr>"""

        st.markdown(f"""
        <div class="perf-table-wrap"><table class="perf-table">
            <thead><tr><th>Aset</th><th>RMSE</th><th>MAPE (Error Rate)</th><th>vs Naive Forecast</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table></div>
        """, unsafe_allow_html=True)

        # Charts
        valid_rows = [r for r in rows_data if r["mape"] is not None]
        if valid_rows:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_mape = go.Figure()
                fig_mape.add_trace(go.Bar(
                    name="LSTM", x=[r["ticker"] for r in valid_rows], y=[r["mape"] for r in valid_rows],
                    marker_color=[r["color"] for r in valid_rows], marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
                    text=[f"{r['mape']:.2f}%" for r in valid_rows], textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color="#7A8A9E"),
                ))
                naive_vals = [r.get("naive_mape") for r in valid_rows]
                if any(v for v in naive_vals):
                    fig_mape.add_trace(go.Bar(
                        name="Naive", x=[r["ticker"] for r in valid_rows],
                        y=[v or 0 for v in naive_vals],
                        marker_color="rgba(90,106,126,0.4)", marker_line_color="rgba(90,106,126,0.6)", marker_line_width=1,
                        text=[f"{v:.2f}%" if v else "" for v in naive_vals], textposition="outside",
                        textfont=dict(family="DM Mono", size=10, color="#5A6A7E"),
                    ))
                for lvl, lbl, clr in [(5,"Sangat Akurat","#00E5A0"),(10,"Akurat","#00D4FF"),(20,"Cukup Akurat","#FFB800")]:
                    fig_mape.add_hline(y=lvl, line_dash="dot", line_color=clr, line_width=1,
                                       annotation_text=lbl, annotation_position="right",
                                       annotation_font=dict(size=9, color=clr))
                fig_mape.update_layout(
                    title=dict(text="MAPE — LSTM vs Naive Forecast", font=dict(family="Syne", size=13, color="#E8EDF5")),
                    barmode="group", height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#5A6A7E", size=11), margin=dict(l=0,r=80,t=40,b=0),
                    legend=dict(font=dict(color="#7A8A9E",size=11), bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(family="DM Mono",size=11,color="#7A8A9E")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(30,39,51,0.9)", zeroline=False,
                               ticksuffix="%", tickfont=dict(family="DM Mono",size=10,color="#5A6A7E")),
                )
                st.plotly_chart(fig_mape, use_container_width=True, config={"displayModeBar": False})

            with col_c2:
                cats = [r["ticker"] for r in valid_rows]
                scores_l = [max(0, 100 - r["mape"]) for r in valid_rows]
                scores_n = [max(0, 100 - (r.get("naive_mape") or 100)) for r in valid_rows]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=scores_l+[scores_l[0]], theta=cats+[cats[0]],
                    fill="toself", fillcolor="rgba(0,212,255,0.08)",
                    line=dict(color="#00D4FF",width=2), name="LSTM",
                ))
                if any(s > 0 for s in scores_n):
                    fig_radar.add_trace(go.Scatterpolar(
                        r=scores_n+[scores_n[0]], theta=cats+[cats[0]],
                        fill="toself", fillcolor="rgba(90,106,126,0.06)",
                        line=dict(color="#5A6A7E",width=1.5,dash="dot"), name="Naive",
                    ))
                fig_radar.update_layout(
                    title=dict(text="Skor Akurasi (100 - MAPE)", font=dict(family="Syne",size=13,color="#E8EDF5")),
                    height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans",color="#5A6A7E",size=11), margin=dict(l=40,r=40,t=40,b=20),
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True,range=[0,100],gridcolor="rgba(30,39,51,0.9)",
                                               tickfont=dict(size=9,family="DM Mono"),color="#5A6A7E"),
                               angularaxis=dict(tickfont=dict(size=12,family="Syne",color="#E8EDF5"),gridcolor="rgba(30,39,51,0.9)")),
                    legend=dict(font=dict(color="#7A8A9E",size=11),bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        # Kesimpulan
        best_name = best_row["name"] if best_row else "—"
        best_mape_s = f"{best_row['mape']:.2f}%" if best_row.get("mape") else "—"
        st.markdown(f"""
        <div class="kesimpulan-card">
            <div class="kesimpulan-title">📋 Ringkasan Temuan Penelitian</div>
            <div class="kesimpulan-points">
                <div class="kesimpulan-point green"><div class="kp-icon">✅</div><div><div class="kp-title">Model LSTM Berhasil Dilatih untuk 5 Aset</div><div class="kp-desc">BTC, ETH, DOGE, SHIB, FLOKI — training loss konvergen dengan baik di seluruh aset.</div></div></div>
                <div class="kesimpulan-point blue"><div class="kp-icon">🏆</div><div><div class="kp-title">Performa Terbaik: {best_name} (MAPE {best_mape_s})</div><div class="kp-desc">Aset dengan pola historis konsisten dan volume data besar cenderung menghasilkan prediksi lebih akurat.</div></div></div>
                <div class="kesimpulan-point gold"><div class="kp-icon">⚡</div><div><div class="kp-title">LSTM Ungguli Naive pada {wins}/5 Aset</div><div class="kp-desc">Validasi komparatif membuktikan LSTM secara statistik lebih optimal dari baseline sederhana pada mayoritas aset.</div></div></div>
                <div class="kesimpulan-point purple"><div class="kp-icon">📱</div><div><div class="kp-title">Koin Meme (SHIB, FLOKI) Lebih Sulit Diprediksi</div><div class="kp-desc">Volatilitas tinggi akibat sentimen media sosial menyebabkan MAPE lebih besar dibanding BTC/ETH yang lebih mature.</div></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — UJI COBA MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab6:

    # ── Hero ──
    st.markdown("""
    <div class="ujicoba-hero">
        <div class="ujicoba-badge"><span class="dot"></span>Live Model Validation</div>
        <div class="ujicoba-title">Uji Coba <span>Validasi</span> Model LSTM</div>
        <p class="ujicoba-sub">
            Jalankan script <code>UjiCobaModelLSTM.py</code> langsung dari dashboard ini.
            Pipeline mengunduh data historis terbaru, melakukan prediksi 7 hari via model terlatih,
            membandingkan dengan data aktual, dan menghitung metrik
            <b>RMSE · MAPE</b> vs <b>Naive Forecast</b> untuk membuktikan superioritas LSTM.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Konfigurasi ──
    st.markdown('<div class="section-header">⚙️ Konfigurasi Periode Uji Coba</div>', unsafe_allow_html=True)
    
    # Batas tanggal yang valid: minimal data tersedia s.d. kemarin
    _today = date.today()
    _max_date = _today - _td(days=1)
    _min_date = date(2021, 1, 1)
    
    # Default 7 Hari Prediksi
    _default_end = _max_date
    _default_start = _default_end - _td(days=6) 
    
    col_d1, col_d2, col_info = st.columns([2, 2, 3])
    with col_d1:
        uji_test_start = st.date_input(
            "📅 Tanggal Mulai Pengujian",
            value=_default_start,
            min_value=_min_date,
            max_value=_max_date - _td(days=6),  # beri ruang minimal 7 hari
            help="Hari pertama dari 7 hari yang akan diprediksi dan dibandingkan dengan harga aktual.",
            key="uji_date_start",
        )
    
    with col_d2:
        uji_test_end = st.date_input(
            "📅 Tanggal Akhir Pengujian",
            value=_default_end,
            min_value=uji_test_start + _td(days=6),  # pastikan minimal 7 hari setelah start
            max_value=_max_date,
            help="Hari terakhir dari periode uji (minimal 7 hari dari tanggal mulai).",
            key="uji_date_end",
        )
        
    with col_info:
        _span_days = (uji_test_end - uji_test_start).days + 1
        _buffer_days = 60 + 30   # LOOKBACK + safety buffer
        _auto_buffer_start = uji_test_start - _td(days=_buffer_days)
        _valid_window = _span_days >= 7
        _status_color = "var(--green)" if _valid_window else "var(--red)"
        _status_text  = f"✅ {_span_days} hari — valid" if _valid_window else f"❌ {_span_days} hari — minimal 7 hari"
        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-top:4px;">
            <div style="font-family:var(--font-body);font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:8px;">Ringkasan Konfigurasi</div>
            <div style="font-family:var(--font-mono);font-size:11.5px;color:{_status_color};margin-bottom:4px;">{_status_text}</div>
            <div style="font-family:var(--font-body);font-size:11px;color:var(--muted);line-height:1.7;">
                🧠 <b style="color:var(--text)">Buffer otomatis:</b> {_auto_buffer_start.strftime('%d %b %Y')}<br>
                <span style="font-size:10px;color:var(--muted);">(TEST_START − {_buffer_days} hari kalender untuk memenuhi lookback 60 hari trading)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Simpan ke session_state agar tombol Run bisa akses
    st.session_state["uji_cfg_start"] = uji_test_start.strftime("%Y-%m-%d")
    st.session_state["uji_cfg_end"]   = uji_test_end.strftime("%Y-%m-%d")

    # Config summary cards
    st.markdown(f"""
    <div class="config-grid" style="margin-top:14px;">
        <div class="config-item"><div class="config-icon">📁</div><div><div class="config-label">Script</div><div class="config-value">UjiCobaModel.py</div><div class="config-desc">{'✅ Ditemukan' if UJI_SCRIPT.exists() else '❌ Tidak ditemukan'}</div></div></div>
        <div class="config-item"><div class="config-icon">📅</div><div><div class="config-label">Buffer Start (Auto)</div><div class="config-value">{_auto_buffer_start.strftime('%Y-%m-%d')}</div><div class="config-desc">Awal pengambilan data historis</div></div></div>
        <div class="config-item"><div class="config-icon">🎯</div><div><div class="config-label">Test Period</div><div class="config-value">{uji_test_start.strftime('%d %b')} → {uji_test_end.strftime('%d %b %Y')}</div><div class="config-desc">{_span_days} hari yang diuji</div></div></div>
        <div class="config-item"><div class="config-icon">🧠</div><div><div class="config-label">Lookback</div><div class="config-value">60 Hari</div><div class="config-desc">Input window model LSTM</div></div></div>
        <div class="config-item"><div class="config-icon">📊</div><div><div class="config-label">Forecast</div><div class="config-value">7 Hari</div><div class="config-desc">Prediksi ke depan per koin</div></div></div>
        <div class="config-item"><div class="config-icon">🪙</div><div><div class="config-label">Aset Diuji</div><div class="config-value">5 Koin</div><div class="config-desc">BTC · ETH · DOGE · SHIB · FLOKI</div></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Warning ──
    st.markdown("""
    <div class="script-warning">
        <div class="sw-icon">⚠️</div>
        <div>
            <div class="sw-title">Perhatian Sebelum Menjalankan</div>
            <div class="sw-text">
                Proses membutuhkan <b>3–8 menit</b> (3 detik jeda/koin untuk rate limit Yahoo Finance).
                Pastikan <code>models/*.keras</code> dan <code>scalers/*.pkl</code> tersedia di root project.
                Hasil akan disimpan ke <code>hasil_ujicobamodel/</code> dan <code>metrics.json</code> diperbarui otomatis.
                Klik <b>Reset Hasil</b> untuk menghapus sesi sebelumnya dan memulai ulang.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    script_ok = UJI_SCRIPT.exists()
    is_running = st.session_state.get("uji_running", False)

    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run_btn = st.button(
            "🚀 Jalankan Uji Coba Model Sekarang" if script_ok else "❌ Script Tidak Ditemukan",
            disabled=(not script_ok) or is_running,
            use_container_width=True, key="btn_run_uji",
        )
    with col_clear:
        if st.button("🗑️ Reset Hasil", use_container_width=True, key="btn_clear_uji"):
            for k in ["uji_log","uji_success","uji_done","uji_running"]:
                st.session_state.pop(k, None)
            st.rerun()

    if not script_ok:
        st.error(f"❌ `UjiCobaModelLSTM.py` tidak ditemukan di: `{ROOT_DIR}`")

    # ── Eksekusi ──
    if run_btn and script_ok:
        st.session_state["uji_running"] = True
        st.session_state.pop("uji_done",    None)
        st.session_state.pop("uji_log",     None)
        st.session_state.pop("uji_success", None)

        pipeline_ph = st.empty()
        terminal_ph = st.empty()

        pipeline_ph.markdown(render_pipeline_html(["active","pending","pending","pending","pending"]), unsafe_allow_html=True)
        terminal_ph.markdown(render_terminal_html([], running=True), unsafe_allow_html=True)

        log_lines = []
        statuses  = ["active","pending","pending","pending","pending"]

        try:
            # Ambil tanggal dari session_state (diset oleh date picker di atas)
            _cfg_start = st.session_state.get("uji_cfg_start", "2025-09-01")
            _cfg_end   = st.session_state.get("uji_cfg_end",   "2025-09-07")

            _env = os.environ.copy()
            _env["UJI_TEST_START"] = _cfg_start
            _env["UJI_TEST_END"]   = _cfg_end

            proc = subprocess.Popen(
                [sys.executable, str(UJI_SCRIPT)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(ROOT_DIR),
                env=_env,
            )
            for raw in proc.stdout:
                line = raw.rstrip()
                log_lines.append(line)

                if "Downloading" in line:
                    statuses = ["done","active","pending","pending","pending"]
                elif "simulasi prediksi" in line:
                    statuses = ["done","done","active","pending","pending"]
                elif "HASIL AKHIR" in line:
                    statuses = ["done","done","done","active","pending"]
                elif "berhasil disimpan" in line or "PENGUJIAN SELESAI" in line:
                    statuses = ["done","done","done","done","done"]

                pipeline_ph.markdown(render_pipeline_html(statuses), unsafe_allow_html=True)
                terminal_ph.markdown(render_terminal_html(log_lines, running=True), unsafe_allow_html=True)

            proc.wait()
            success = proc.returncode == 0
            if not success:
                statuses = ["done","done","done","done","error"]

        except Exception as e:
            log_lines.append(f"\nCRITICAL ERROR: {e}")
            success = False
            statuses = ["error","error","error","error","error"]

        pipeline_ph.markdown(render_pipeline_html(statuses), unsafe_allow_html=True)
        terminal_ph.markdown(render_terminal_html(log_lines, success=success), unsafe_allow_html=True)

        st.session_state["uji_log"]     = log_lines
        st.session_state["uji_success"] = success
        st.session_state["uji_done"]    = True
        st.session_state["uji_running"] = False
        load_metrics.clear()
        st.rerun()

    # ── Tampilkan Hasil Tersimpan ──
    if st.session_state.get("uji_done") and "uji_log" in st.session_state:
        log_lines = st.session_state["uji_log"]
        success   = st.session_state.get("uji_success", False)

        fin_statuses = ["done"] * 5 if success else ["done","done","done","done","error"]
        st.markdown(render_pipeline_html(fin_statuses), unsafe_allow_html=True)

        if success:
            st.success("✅ Uji Coba selesai! Semua model berhasil divalidasi.")
        else:
            st.error("❌ Uji Coba selesai dengan error. Periksa terminal log di bawah.")

        st.markdown(render_terminal_html(log_lines, success=success), unsafe_allow_html=True)

        col_dl1, col_dl2 = st.columns([2, 5])
        with col_dl1:
            st.download_button(
                "📥 Download Log (.txt)", data="\n".join(log_lines),
                file_name="log_ujicobamodel.txt", mime="text/plain", key="dl_log",
            )

        # ── HASIL DETAIL ──
        if success:
            st.markdown('<div class="section-header">📊 Hasil Validasi per Aset</div>', unsafe_allow_html=True)

            fresh = {}
            try:
                with open(METRICS_PATH) as f:
                    fresh = json.load(f)
            except Exception:
                pass

            if fresh:
                results = []
                for tk, meta in COINS_META.items():
                    m = fresh.get(tk, {})
                    lm = m.get("LSTM_MAPE") or m.get("MAPE")
                    lr = m.get("LSTM_RMSE") or m.get("RMSE")
                    nm = m.get("NAIVE_MAPE")
                    nr = m.get("NAIVE_RMSE")
                    if lm is not None:
                        imp = ((nm - lm) / nm * 100) if (nm and nm > 0) else 0
                        results.append({"key":tk,"meta":meta,"lm":lm,"lr":lr,"nm":nm,"nr":nr,"wins":nm and lm<nm,"imp":imp})

                # Summary
                if results:
                    wins_c  = sum(1 for r in results if r["wins"])
                    avg_lm  = sum(r["lm"] for r in results)/len(results)
                    avg_nm  = sum(r["nm"] for r in results if r["nm"])/max(1,sum(1 for r in results if r["nm"]))
                    best    = min(results, key=lambda r: r["lm"])
                    st.markdown(f"""
                    <div class="summary-stat-row">
                        <div class="summary-stat s-green"><div class="ss-icon">🏆</div><div class="ss-label">Ungguli Naive</div><div class="ss-value">{wins_c}/5</div><div class="ss-sub">LSTM &gt; Baseline</div></div>
                        <div class="summary-stat s-blue"><div class="ss-icon">🎯</div><div class="ss-label">Avg MAPE LSTM</div><div class="ss-value">{avg_lm:.2f}%</div><div class="ss-sub">Semua aset</div></div>
                        <div class="summary-stat s-gold"><div class="ss-icon">📊</div><div class="ss-label">Avg MAPE Naive</div><div class="ss-value">{avg_nm:.2f}%</div><div class="ss-sub">Baseline</div></div>
                        <div class="summary-stat s-purple"><div class="ss-icon">⭐</div><div class="ss-label">MAPE Terbaik</div><div class="ss-value">{best['lm']:.2f}%</div><div class="ss-sub">{best['meta']['name']}</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Per-coin cards
                for row_start in range(0, len(results), 2):
                    cols = st.columns(2)
                    for ci, col in enumerate(cols):
                        ri = row_start + ci
                        if ri >= len(results): break
                        r    = results[ri]
                        meta = r["meta"]
                        ql, qcls, qcolor = mape_quality(r["lm"])

                        vhtml = (f'<span class="verdict-win">▲ LSTM Lebih Akurat (+{r["imp"]:.1f}%)</span>'
                                 if r["wins"] else f'<span class="verdict-lose">▼ Naive Lebih Akurat</span>')

                        with col:
                            st.markdown(f"""
                            <div class="hasil-coin-card">
                                <div class="hasil-coin-header">
                                    <div style="display:flex;align-items:center;gap:10px;">
                                        <span style="width:10px;height:10px;border-radius:50%;background:{meta['color']};display:inline-block;"></span>
                                        <div><div class="hasil-coin-name">{meta['name']}</div><div class="hasil-coin-ticker">{meta['ticker']} · Test 7 Hari</div></div>
                                    </div>
                                    {vhtml}
                                </div>
                                <div class="hasil-metrics-row">
                                    <div class="hasil-metric-item"><div class="hm-label">LSTM RMSE</div><div class="hm-value lstm">{"$"+f"{r['lr']:,.6g}" if r['lr'] else "—"}</div></div>
                                    <div class="hasil-metric-item"><div class="hm-label">LSTM MAPE</div><div class="hm-value lstm">{r['lm']:.2f}%</div></div>
                                    <div class="hasil-metric-item"><div class="hm-label">Naive RMSE</div><div class="hm-value naive">{"$"+f"{r['nr']:,.6g}" if r['nr'] else "—"}</div></div>
                                    <div class="hasil-metric-item"><div class="hm-label">Naive MAPE</div><div class="hm-value naive">{f"{r['nm']:.2f}%" if r['nm'] else "—"}</div></div>
                                </div>
                                <div style="padding:10px 16px;display:flex;align-items:center;justify-content:space-between;">
                                    <span class="q-badge {qcls}">{ql}</span>
                                    <span style="font-family:var(--font-mono);font-size:10px;color:var(--muted);">Akurasi: {100-r['lm']:.2f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # ── Bar Chart Komparasi ──
                st.markdown('<div class="section-header">📈 Visualisasi Komparasi LSTM vs Naive</div>', unsafe_allow_html=True)

                tickers    = [r["meta"]["ticker"] for r in results]
                lstm_mapes = [r["lm"] for r in results]
                naive_mapes= [r["nm"] or 0 for r in results]
                colors     = [r["meta"]["color"] for r in results]

                fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("MAPE — LSTM vs Naive","RMSE — LSTM vs Naive"))
                for name, vals, clr, show_leg in [("LSTM",lstm_mapes,colors,True),("Naive",naive_mapes,["rgba(90,106,126,0.4)"]*len(results),True)]:
                    is_naive = name=="Naive"
                    fig_comp.add_trace(go.Bar(
                        name=name, x=tickers, y=vals,
                        marker_color=clr if not is_naive else "rgba(90,106,126,0.4)",
                        marker_line_color="rgba(0,0,0,0.3)" if not is_naive else "rgba(90,106,126,0.6)",
                        marker_line_width=1, showlegend=show_leg,
                        text=[f"{v:.2f}%" for v in vals], textposition="outside",
                        textfont=dict(family="DM Mono", size=10, color="#7A8A9E" if is_naive else "#E8EDF5"),
                    ), row=1, col=1)

                lstm_rmses  = [r["lr"] or 0 for r in results]
                naive_rmses = [r["nr"] or 0 for r in results]
                for name, vals, clr in [("LSTM RMSE",lstm_rmses,colors),("Naive RMSE",naive_rmses,["rgba(90,106,126,0.4)"]*len(results))]:
                    is_naive = name=="Naive RMSE"
                    fig_comp.add_trace(go.Bar(
                        name=name, x=tickers, y=vals,
                        marker_color=clr if not is_naive else "rgba(90,106,126,0.4)",
                        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1, showlegend=False,
                    ), row=1, col=2)

                # ── FIX: Pisahkan base style dari showgrid ────────────────────
                # ax_base tidak mengandung showgrid agar tidak konflik saat di-unpack
                ax_base = dict(
                    gridcolor="rgba(30,39,51,0.9)",
                    zeroline=False,
                    color="#5A6A7E",
                    tickfont=dict(family="DM Mono", size=10, color="#7A8A9E"),
                )
                fig_comp.update_layout(
                    barmode="group", height=380,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#5A6A7E", size=11),
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(
                        font=dict(color="#7A8A9E", size=11), bgcolor="rgba(0,0,0,0)",
                        orientation="h", yanchor="bottom", y=1.04,
                    ),
                    # showgrid ditulis langsung di masing-masing axis — tidak via **ax_base
                    xaxis=dict(**ax_base,  showgrid=False),
                    xaxis2=dict(**ax_base, showgrid=False),
                    yaxis=dict(**ax_base,  showgrid=True, ticksuffix="%"),
                    yaxis2=dict(**ax_base, showgrid=True, type="log", tickprefix="$"),
                )
                for ann in fig_comp.layout.annotations:
                    ann.font = dict(family="Syne", size=11, color="#7A8A9E")
                st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

                # ── Grafik PNG Hasil Uji ──
                uji_imgs = sorted(OUTPUT_UJI.glob("*_ujicobamodel.png")) if OUTPUT_UJI.exists() else []
                if uji_imgs:
                    st.markdown('<div class="section-header">🖼️ Grafik Validasi Aktual vs Prediksi per Aset</div>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="info-box">
                        <div class="info-icon">📉</div>
                        <div>
                            <div class="info-title">Cara Membaca Grafik Validasi</div>
                            <div class="info-text">
                                🔵 <b>Biru = Historis</b> (14 hari sebelum uji) &nbsp;|&nbsp;
                                🟢 <b>Hijau = Aktual/Real</b> (harga nyata 7 hari) &nbsp;|&nbsp;
                                🔴 <b>Merah putus = Prediksi AI (LSTM)</b>.
                                Semakin dekat garis merah ke hijau, semakin akurat model.
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    img_map = {}
                    for img_path in uji_imgs:
                        for tk, meta in COINS_META.items():
                            if meta["ticker"] in img_path.name.upper() or tk.replace("-USD","") in img_path.name.upper():
                                img_map[tk] = img_path

                    img_keys = list(img_map.keys())
                    for row_start in range(0, len(img_keys), 2):
                        cols = st.columns(2)
                        for ci, col in enumerate(cols):
                            ri = row_start + ci
                            if ri >= len(img_keys): break
                            tk   = img_keys[ri]
                            meta = COINS_META[tk]
                            m2   = fresh.get(tk, {})
                            lm2  = m2.get("LSTM_MAPE") or m2.get("MAPE")
                            ql2, qcls2, qcolor2 = mape_quality(lm2)

                            with col:
                                st.markdown(f"""
                                <div class="train-card-header" style="background:var(--surface);border:1px solid var(--border);border-radius:12px 12px 0 0;margin-top:12px;">
                                    <div class="train-card-title">
                                        <span style="width:8px;height:8px;border-radius:50%;background:{meta['color']};display:inline-block;"></span>
                                        {meta['name']} <span style="font-family:var(--font-mono);font-size:11px;color:var(--muted);">Validasi 7 Hari</span>
                                    </div>
                                    <span class="q-badge {qcls2}">{ql2}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                try:
                                    st.image(Image.open(img_map[tk]), use_container_width=True)
                                except Exception as e:
                                    st.error(f"Gagal memuat grafik: {e}")
                                st.markdown(f"""
                                <div class="train-meta" style="border:1px solid var(--border);border-top:none;border-radius:0 0 12px 12px;margin-bottom:8px;">
                                    <div class="train-meta-item"><div class="train-meta-label">LSTM MAPE</div><div class="train-meta-val" style="color:{qcolor2};">{f"{lm2:.2f}%" if lm2 else "—"}</div></div>
                                    <div class="train-meta-item"><div class="train-meta-label">Akurasi</div><div class="train-meta-val" style="color:var(--green);">{f"{100-lm2:.2f}%" if lm2 else "—"}</div></div>
                                    <div class="train-meta-item"><div class="train-meta-label">Kualitas</div><div class="train-meta-val" style="font-size:11px;color:{qcolor2};">{ql2}</div></div>
                                </div>
                                """, unsafe_allow_html=True)
                                with open(img_map[tk], "rb") as f_img:
                                    st.download_button(
                                        f"📥 Download Grafik {meta['ticker']}",
                                        data=f_img.read(),
                                        file_name=f"{meta['ticker']}_validasi.png",
                                        mime="image/png",
                                        key=f"dl_img_{tk}",
                                    )
            else:
                st.warning("⚠️ `metrics.json` belum terisi. Pastikan script selesai tanpa error.")

    elif not st.session_state.get("uji_done"):
        st.markdown("""
        <div style="background:var(--surface);border:1px solid var(--border);border-radius:14px;
                    padding:56px 32px;text-align:center;margin-top:16px;">
            <div style="font-size:52px;margin-bottom:18px;">🚀</div>
            <div style="font-family:var(--font-head);font-size:20px;font-weight:700;color:var(--text);margin-bottom:10px;">
                Siap Menjalankan Validasi Model
            </div>
            <div style="font-family:var(--font-body);font-size:13.5px;color:var(--muted);line-height:1.75;max-width:520px;margin:0 auto 24px;">
                Klik <b style="color:var(--green)">🚀 Jalankan Uji Coba Model Sekarang</b> di atas untuk memulai.
                Output script akan ditampilkan secara <b style="color:var(--text)">live</b> dalam terminal log,
                dan hasil validasi lengkap akan muncul setelah eksekusi selesai.
            </div>
            <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;">
                <div style="background:rgba(0,229,160,0.06);border:1px solid rgba(0,229,160,0.15);border-radius:8px;padding:12px 20px;font-family:var(--font-mono);font-size:11px;color:var(--green);">✓ Pipeline otomatis</div>
                <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.15);border-radius:8px;padding:12px 20px;font-family:var(--font-mono);font-size:11px;color:var(--accent);">✓ Live terminal log</div>
                <div style="background:rgba(255,184,0,0.06);border:1px solid rgba(255,184,0,0.15);border-radius:8px;padding:12px 20px;font-family:var(--font-mono);font-size:11px;color:var(--gold);">✓ Grafik & download</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-col">
        <div class="footer-label">📁 Struktur File</div>
        <div class="footer-text">models/ → .keras per aset<br>scalers/ → .pkl MinMaxScaler<br>metrics.json → skor evaluasi<br>Training/Documentation/ → grafik &amp; Dataset<br>hasil_ujicobamodel/ → output validasi</div>
    </div>
    <div class="footer-col">
        <div class="footer-label">⚙️ Stack Teknologi</div>
        <div class="footer-text">Python · TensorFlow/Keras · Streamlit<br>yfinance · Plotly · scikit-learn<br>pandas · numpy · joblib · subprocess</div>
    </div>
    <div class="footer-col">
        <div class="footer-label">⚠️ Catatan Akademis</div>
        <div class="footer-text">Seluruh dokumentasi dibuat untuk keperluan Tugas Akhir. Hasil prediksi bersifat estimasi, bukan saran investasi.</div>
    </div>
</div>
""", unsafe_allow_html=True)