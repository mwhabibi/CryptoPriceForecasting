import os
import time
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import json
from datetime import datetime, timedelta

metrics_dict = {}

# Matikan peringatan agar terminal bersih
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# SETUP PATH (LOKASI FILE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SCALERS_DIR = os.path.join(BASE_DIR, 'scalers')
OUTPUT_DIR = os.path.join(BASE_DIR, 'hasil_ujicobamodel')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Buka file dengan mode 'w' (write)
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message) # Tulis ke Layar
        self.log.write(message)      # Tulis ke File

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Aktifkan Pencatatan Otomatis
log_filename = os.path.join(OUTPUT_DIR, "log_ujicobamodel.txt")
sys.stdout = DualLogger(log_filename)

print(f"Working Directory Script: {BASE_DIR}")
print(f"Folder Models terdeteksi di: {MODELS_DIR}")
print(f"Folder Scalers terdeteksi di: {SCALERS_DIR}")
print(f"Gambar akan disimpan di: {OUTPUT_DIR}")

# KONFIGURASI
COINS = ["BTC-USD", "ETH-USD", "DOGE-USD", "SHIB-USD", "FLOKI-USD"]
LOOKBACK     = 60
FORECAST     = 7 # Ditambahkan: Menegaskan bahwa model memprediksi 7 hari ke depan

_test_start_raw = os.environ.get("TEST_START", "2025-09-01")
_test_end_raw = os.environ.get("TEST_END", "2025-09-07")

from datetime import datetime, timedelta
_test_start_dt = datetime.strptime(_test_start_raw, "%Y-%m-%d")
_test_end_dt   = datetime.strptime(_test_end_raw,   "%Y-%m-%d")

# Validasi: test window minimal FORECAST hari
if (_test_end_dt - _test_start_dt).days + 1 < FORECAST:
    print(f"WARNING: Rentang uji ({_test_start_raw} → {_test_end_raw}) kurang dari {FORECAST} hari. "
          f"DOWNLOAD_END disesuaikan otomatis.")
    _test_end_dt = _test_start_dt + timedelta(days=FORECAST - 1)

# START_BUFFER harus cukup jauh sebelum TEST_START agar ada ≥ LOOKBACK hari trading.
# Pasar kripto ~365 hari/tahun, tapi untuk keamanan tambah 30 hari buffer kalender
# (mengantisipasi hari kosong karena data yfinance bisa tidak kontinu).
_buffer_days   = LOOKBACK + 30
START_BUFFER   = (_test_start_dt - timedelta(days=_buffer_days)).strftime("%Y-%m-%d")
TEST_START     = _test_start_raw
# DOWNLOAD_END = hari setelah test_end agar yfinance memasukkan data hari terakhir (slice eksklusif)
DOWNLOAD_END   = (_test_end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"[CONFIG] TEST_START  = {TEST_START}")
print(f"[CONFIG] TEST_END    = {_test_end_dt.strftime('%Y-%m-%d')} (inklusif)")
print(f"[CONFIG] DOWNLOAD_END= {DOWNLOAD_END} (eksklusif, untuk yfinance)")
print(f"[CONFIG] START_BUFFER= {START_BUFFER} (otomatis: TEST_START - {_buffer_days} hari)")

# 1. FUNGSI AMBIL DATA & HITUNG INDIKATOR
def get_data_with_indicators(ticker, start, end):
    print(f"Downloading data {ticker}...")
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        raise ValueError(f"Data kosong untuk {ticker}")

    # FEATURE ENGINEERING
    # Log Return
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.clip(lower=0)).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    df.dropna(inplace=True)
    return df

# EKSEKUSI PENGUJIAN UTAMA
print("\n" + "="*70)
print(f"MEMULAI PENGUJIAN VALIDASI MODEL MULTI-STEP (Prediksi {FORECAST} Hari)")
print("="*70)

for ticker in COINS:
    print(f"\nAnalisis Koin: {ticker}")
    print("Jeda 3 detik agar aman dari blokir Yahoo Finance API...")
    time.sleep(3) 

    try:
        # A. LOAD FILE PENTING
        model_path = os.path.join(MODELS_DIR, f"{ticker}_best_model.keras")
        scaler_path = os.path.join(SCALERS_DIR, f"{ticker}_scaler.pkl")
        
        if not os.path.exists(model_path):
            print(f"Error Path: File tidak ditemukan di {model_path}")
            continue
            
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # B. AMBIL DATA LENGKAP
        df_full = get_data_with_indicators(ticker, START_BUFFER, DOWNLOAD_END)
        
        # C. TENTUKAN DATA AKTUAL 7 HARI YANG AKAN DIPREDIKSI
        mask_future = df_full.index >= TEST_START
        future_data = df_full.loc[mask_future]
        
        if len(future_data) < FORECAST:
            print(f"Data masa depan setelah {TEST_START} kurang dari {FORECAST} hari.")
            continue
            
        # Mengambil 7 hari (FORECAST) pertama sebagai data Aktual/Real
        actual_7_days_df = future_data.iloc[:FORECAST]
        actual_dates = actual_7_days_df.index
        actual_prices = actual_7_days_df['Close'].values
        
        # D. TENTUKAN DATA INPUT 60 HARI (LOOKBACK) SEBELUM TANGGAL PREDIKSI
        start_idx = df_full.index.get_loc(actual_dates[0])
        
        if start_idx < LOOKBACK:
            print("Data histori sebelum tanggal uji coba kurang dari LOOKBACK (60 hari).")
            continue
            
        input_window = df_full.iloc[start_idx-LOOKBACK : start_idx]
        features = ['Log_Ret', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'Volume']
        input_values = input_window[features].values
        
        # E. LAKUKAN PREDIKSI (Satu kali tembak langsung 7 hari)
        print(f"Melakukan simulasi prediksi 7 hari ke depan...")
        input_scaled = scaler.transform(input_values)
        input_reshaped = np.expand_dims(input_scaled, axis=0) # Shape (1, 60, 6)
        
        # Model memprediksi 7 nilai log_return yang diskalakan
        pred_log_ret_scaled = model.predict(input_reshaped, verbose=0)[0] 
        
        # F. DENORMALISASI & KONVERSI LOG RETURN KE HARGA ASLI
        scale_factor = scaler.scale_[0] 
        min_factor = scaler.min_[0]    
        pred_log_ret = (pred_log_ret_scaled - min_factor) / scale_factor
        
        predicted_prices = []
        last_close_price = input_window['Close'].iloc[-1]
        current_price = last_close_price
        
        for lr in pred_log_ret:
            current_price = current_price * np.exp(lr) # Ubah log return menjadi harga secara kumulatif
            predicted_prices.append(current_price)
            
        predicted_prices = np.array(predicted_prices)

        # G. HITUNG ERROR UNTUK RENTANG 7 HARI INI
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
        accuracy = 100 * (1 - mape)
        
        print(f"HASIL AKHIR ({actual_dates[0].strftime('%Y-%m-%d')} s.d {actual_dates[-1].strftime('%Y-%m-%d')}):")
        print(f"RMSE    : ${rmse:.8f}")
        print(f"MAE     : ${mae:.8f}")
        print(f"MAPE    : {mape:.2%}")
        print(f"AKURASI : {accuracy:.2f}%")
        
        # Naive Forecast untuk melakukan komparasi hasil prediksi model LSTM
        naive_predictions = np.full(FORECAST, last_close_price)
        
        naive_rmse = np.sqrt(mean_squared_error(actual_prices, naive_predictions))
        naive_mape = mean_absolute_percentage_error(actual_prices, naive_predictions)
        
        print(f"\n--- KOMPARASI BASELINE (NAIVE FORECAST) ---")
        print(f"Naive RMSE: ${naive_rmse:.10f}")
        print(f"Naive MAPE: {naive_mape:.4%}")
        
        if mape < naive_mape:
            print("MODEL LSTM TERBUKTI LEBIH OPTIMAL dibandingkan baseline (Naive).")
        else:
            print("Model LSTM TIDAK lebih baik dari Naive Forecast.")

        metrics_dict[ticker] = {
            "LSTM_RMSE": round(float(rmse), 8),
            "LSTM_MAPE": round(float(mape * 100), 2),
            "NAIVE_RMSE": round(float(naive_rmse), 8),
            "NAIVE_MAPE": round(float(naive_mape * 100), 2),
        }

        # H. VISUALISASI GRAFIK
        plt.figure(figsize=(12, 6))
        
        # Tambahkan 14 hari data historis agar grafik tidak cuma isi 7 titik saja
        history_plot_days = 14
        historical_dates = input_window.index[-history_plot_days:]
        historical_prices = input_window['Close'].iloc[-history_plot_days:]
        
        plt.plot(historical_dates, historical_prices, label='History', color='blue', marker='.')
        plt.plot(actual_dates, actual_prices, label='Actual (Real)', color='green', marker='o')
        plt.plot(actual_dates, predicted_prices, label='Predicted 7-Days (AI)', color='red', linestyle='--', marker='x')
        
        # Garis penghubung titik historis terakhir ke titik pertama (biar estetik tidak terputus)
        plt.plot([historical_dates[-1], actual_dates[0]], [historical_prices.iloc[-1], actual_prices[0]], color='green')
        plt.plot([historical_dates[-1], actual_dates[0]], [historical_prices.iloc[-1], predicted_prices[0]], color='red', linestyle='--')

        plt.title(f"{ticker} - Validasi Model Multi-Step 7 Hari\n({actual_dates[0].strftime('%Y-%m-%d')} s.d {actual_dates[-1].strftime('%Y-%m-%d')})")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"{ticker}_ujicobamodel.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath)
        print(f"Gambar grafik disimpan di: {filepath}")
        plt.close()
        
        print("-" * 70)

    except Exception as e:
        print(f"CRITICAL ERROR pada {ticker}: {e}")
        import traceback
        traceback.print_exc()

# I. SIMPAN METRIK
metrics_path = os.path.join(BASE_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=4)
print(f"\nFile metrik berhasil disimpan di: {metrics_path}")

print("\nPENGUJIAN SELESAI.")