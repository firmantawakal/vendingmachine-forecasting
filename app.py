# ======================================
# === app.py: Complete Manual Refresh Version ===
# ======================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

# ======================================
# === STEP 1: Konfigurasi & Load Model ===
# ======================================

# Load base models
rf = joblib.load("RandomForest_model.pkl")
svm = joblib.load("SVM_model.pkl")
xgb = joblib.load("XGBoost_model.pkl")

# Load meta LSTM (tanpa compile)
lstm_meta = tf.keras.models.load_model("stacking_lstm_meta.h5", compile=False)

# Load scaler, encoder, dan feature names
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Google Sheets URLs
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSL4OW8HCT1zxfojPSx2Pm2gAS1KVTzqWtmv4VuYl6XgDDQ7aHN2hYuOzDfS7LMRA/pub?output=csv"
SHEET_VIEW_URL = "https://docs.google.com/spreadsheets/d/1nK7Z_KLPML52WPnOztTOrRTNLBQKsYaa/edit?usp=sharing&ouid=114455522393064977140&rtpof=true&sd=true"

# ======================================
# === STEP 2: MANUAL DATA LOADING ===
# ======================================

def load_data_and_skus():
    """Load data from Google Sheets and extract all unique PRODUCTs - MANUAL REFRESH ONLY"""
    try:
        df = pd.read_csv(SHEET_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H.%M.%S", errors="coerce")
        
        # Extract all unique PRODUCTs from real data
        unique_skus = df["product"].dropna().unique().tolist()
        unique_skus.sort()  # Sort alphabetically
        
        return df, unique_skus, None  # Return success
    except Exception as e:
        return None, le.classes_.tolist(), str(e)  # Return error

def get_sku_encoded(sku, unique_skus):
    """Get encoded value for PRODUCT, with fallback for unknown PRODUCTs"""
    try:
        return le.transform([sku])[0]
    except:
        if sku in unique_skus:
            return len(le.classes_) + unique_skus.index(sku)
        return 0  # Default fallback

# ======================================
# === STEP 3: Fungsi Prediksi ===
# ======================================

def predict_sales(input_df):
    """Enhanced prediction function"""
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_names]
    X_scaled = scaler.transform(input_df)
    
    # Base predictions
    base_preds = [
        rf.predict(X_scaled).reshape(-1, 1),
        svm.predict(X_scaled).reshape(-1, 1),
        xgb.predict(X_scaled).reshape(-1, 1)
    ]
    
    # Meta LSTM prediction
    meta_X = np.hstack(base_preds)
    meta_X_lstm = meta_X.reshape((meta_X.shape[0], 1, meta_X.shape[1]))
    y_pred = lstm_meta.predict(meta_X_lstm, verbose=0)
    
    return y_pred.flatten()

# ======================================
# === STEP 4: Streamlit GUI ===
# ======================================

st.set_page_config(page_title="Vending Machine Prediction", layout="wide")
st.title("🤖 Vending Machine Monitoring & Prediction (Manual Refresh)")

# ======================================
# === MANUAL REFRESH CONTROLS ===
# ======================================

# Initialize session state for data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_main = None
    st.session_state.dynamic_skus = []
    st.session_state.last_refresh = None
    st.session_state.data_error = None

# Header with refresh controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

with col1:
    st.markdown("### 🔄 Data Synchronization")

with col2:
    # 🔥 MANUAL REFRESH BUTTON
    if st.button("🔄 Refresh Data", type="primary"):
        with st.spinner("Loading fresh data from Google Sheets..."):
            df_main, dynamic_skus, error = load_data_and_skus()
            
            if error is None:
                st.session_state.df_main = df_main
                st.session_state.dynamic_skus = dynamic_skus
                st.session_state.last_refresh = datetime.now()
                st.session_state.data_error = None
                st.session_state.data_loaded = True
                st.success(f"✅ Data refreshed! Found {len(dynamic_skus)} unique PRODUCTs")
                st.rerun()
            else:
                st.session_state.data_error = error
                st.error(f"❌ Error loading data: {error}")

with col3:
    # 🔧 FORCE RELOAD BUTTON
    if st.button("⚡ Force Reload"):
        # Clear all session state and reload
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

with col4:
    # Display last refresh time
    if st.session_state.last_refresh:
        st.write(f"**Last Refresh:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    else:
        st.write("**Status:** No data loaded yet")

# Load initial data if not loaded
if not st.session_state.data_loaded:
    with st.spinner("Loading initial data..."):
        df_main, dynamic_skus, error = load_data_and_skus()
        
        if error is None:
            st.session_state.df_main = df_main
            st.session_state.dynamic_skus = dynamic_skus
            st.session_state.last_refresh = datetime.now()
            st.session_state.data_loaded = True
            st.success(f"✅ Initial data loaded! Found {len(dynamic_skus)} unique PRODUCTs")
        else:
            st.session_state.data_error = error
            st.error(f"❌ Error loading initial data: {error}")

# Get data from session state
df_main = st.session_state.df_main
dynamic_skus = st.session_state.dynamic_skus

# Display current status in sidebar
with st.sidebar:
    st.markdown("### 📊 Data Status")
    
    if st.session_state.data_loaded and df_main is not None:
        st.success("✅ Data Loaded")
        st.metric("📦 Total PRODUCTs", len(dynamic_skus))
        st.metric("📊 Total Records", len(df_main))
        
        if st.session_state.last_refresh:
            st.write(f"**Last Refresh:** {st.session_state.last_refresh.strftime('%d/%m/%Y %H:%M:%S')}")
        
        # Show recent PRODUCTs
        st.write("**Recent PRODUCTs:**")
        for i, sku in enumerate(dynamic_skus[:8]):
            st.write(f"{i+1}. {sku}")
        if len(dynamic_skus) > 8:
            st.write(f"... and {len(dynamic_skus) - 8} more")
            
    else:
        st.warning("⚠️ No data loaded")
        if st.session_state.data_error:
            st.error(f"Error: {st.session_state.data_error}")
    
    st.markdown("---")
    st.write("🔄 **Manual Refresh Mode**")
    st.write("Data only updates when you click refresh button")

# ======================================
# === TABS WITH DATA DEPENDENCY ===
# ======================================

if not st.session_state.data_loaded or df_main is None:
    st.warning("⚠️ Please refresh data to continue")
    st.info("Click '🔄 Refresh Data' button above to load data from Google Sheets")
    st.stop()

# Continue with tabs only if data is loaded
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Monitoring",
    "🔮 Prediksi per Produk", 
    "📊 Prediksi Besok",
    "📅 Prediksi Minggu Depan",
    "📆 Prediksi Bulan Depan",
    "📂 Prediksi dari File XLSX"
])

# ================================
# Tab 1: ENHANCED MONITORING
# ================================
with tab1:
    st.subheader("📡 Monitoring Harian (All PRODUCTs)")
    
    if df_main is not None:
        # Filter today's data
        today = pd.Timestamp.today().date()
        df_today = df_main[df_main["timestamp"].dt.date == today]
        
        if df_today.empty:
            st.warning("⚠️ Tidak ada data penjualan untuk hari ini.")
            # Show recent data as fallback
            recent_data = df_main.head(20)
            st.write("**Data Terbaru (20 records):**")
            st.dataframe(recent_data)
        else:
            # Environmental info
            latest = df_today.sort_values("timestamp").iloc[-1]
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric("🌡️ Suhu (°C)", f"{latest['temperature_c']}")
            with c2:
                st.metric("💧 Kelembaban (%)", f"{latest['humidity']}")
            with c3:
                st.metric("📊 PRODUCTs Terjual Hari Ini", len(df_today["product"].unique()))
            
            # 🔥 AGGREGATE ALL PRODUCTs FROM TODAY'S DATA
            sales_today = df_today.groupby("product").agg({
                "sold": "sum",
                "price": "mean"
            }).reset_index()
            
            # Generate predictions for all PRODUCTs that had sales today
            preds = []
            tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
            
            for _, row in sales_today.iterrows():
                sku_encoded = get_sku_encoded(row["product"], dynamic_skus)
                
                features = pd.DataFrame([{
                    "avg_price": row["price"],
                    "day_of_week": tomorrow.dayofweek,
                    "is_weekend": 1 if tomorrow.dayofweek >= 5 else 0,
                    "sku_encoded": sku_encoded,
                    "lag_1": row["sold"]
                }])
                
                pred = predict_sales(features)
                
                preds.append({
                    "product": row["product"],
                    "Terjual Hari Ini": int(row["sold"]),
                    "Harga Rata-rata": f"Rp {row['price']:,.0f}",
                    "Prediksi Besok": round(float(pred[0]), 2)
                })
            
            # Sort by sales descending
            preds = sorted(preds, key=lambda x: x["Terjual Hari Ini"], reverse=True)
            
            st.markdown("### 📊 Ringkasan Penjualan Harian & Prediksi Besok (Semua PRODUCT)")
            st.dataframe(pd.DataFrame(preds), use_container_width=True)
            
            # 🔥 SHOW ALL UNIQUE PRODUCTs FROM DATA
            st.markdown("### 📦 Semua PRODUCT yang Tersedia di Database")
            sku_df = pd.DataFrame({
                "No": range(1, len(dynamic_skus) + 1),
                "product": dynamic_skus,
                "Status": ["✅ Active" if sku in sales_today["product"].values else "⏸️ No Sales Today" for sku in dynamic_skus]
            })
            st.dataframe(sku_df, use_container_width=True)
            
            # Database link
            st.markdown(f'<a href="{SHEET_VIEW_URL}" target="_blank" class="btn">📂 Lihat Database Lengkap</a>', unsafe_allow_html=True)
    else:
        st.error("❌ Gagal memuat data dari Google Sheets")

# ================================
# Tab 2: DYNAMIC PREDIKSI PER PRODUK
# ================================
with tab2:
    st.subheader("🔮 Prediksi per Produk (Dynamic PRODUCT)")
    
    # 🔥 USE DYNAMIC PRODUCTs INSTEAD OF STATIC le.classes_
    sku = st.selectbox("Pilih Produk (PRODUCT)", dynamic_skus, key="tab2_sku")
    avg_price = st.number_input("Harga Rata-rata", min_value=1000, max_value=20000, value=10000, step=500)
    pred_scope = st.radio("Pilih Periode Prediksi", ["Harian", "Mingguan", "Bulanan"])
    
    day_of_week = pd.Timestamp.today().dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    
    if st.button("Prediksi Penjualan (per Produk)"):
        sku_encoded = get_sku_encoded(sku, dynamic_skus)
        
        features = pd.DataFrame([{
            "avg_price": avg_price,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "sku_encoded": sku_encoded
        }])
        
        pred = predict_sales(features)
        
        factor = 1
        if pred_scope == "Mingguan":
            factor = 7
        elif pred_scope == "Bulanan":
            factor = 30
            
        st.success(f"🔮 Prediksi {pred_scope} untuk {sku}: {pred[0] * factor:.2f} unit")

# ================================
# Tab 3: DYNAMIC PREDIKSI BESOK
# ================================
with tab3:
    st.subheader("📊 Prediksi Penjualan Besok (All Dynamic PRODUCTs)")
    
    st.info(f"📦 Total PRODUCTs available: {len(dynamic_skus)}")
    
    # Auto-populate with recent sales data if available
    if df_main is not None and not df_main.empty:
        recent_sales = df_main.groupby("product").agg({
            "price": "mean",
            "sold": "mean"
        }).round(0).to_dict()
    else:
        recent_sales = {"price": {}, "sold": {}}
    
    input_today = {}
    
    # Create inputs for ALL dynamic PRODUCTs
    for i, sku in enumerate(dynamic_skus):
        with st.expander(f"📦 {sku} (PRODUCT #{i+1})"):
            c1, c2 = st.columns(2)
            
            with c1:
                default_price = int(recent_sales["price"].get(sku, 10000))
                price = st.number_input(
                    f"Harga {sku}", 
                    min_value=1000, 
                    max_value=50000, 
                    value=default_price, 
                    step=500, 
                    key=f"price_{sku}_besok"
                )
            
            with c2:
                default_sold = int(recent_sales["sold"].get(sku, 1))
                sold = st.number_input(
                    f"Penjualan Hari Ini {sku}", 
                    min_value=0, 
                    max_value=100, 
                    value=default_sold, 
                    step=1, 
                    key=f"sold_{sku}_besok"
                )
            
            input_today[sku] = {"price": price, "sold": sold}
    
    if st.button("🔮 Prediksi Besok (Semua PRODUCT)"):
        preds = []
        tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
        
        for sku, vals in input_today.items():
            sku_encoded = get_sku_encoded(sku, dynamic_skus)
            
            features = pd.DataFrame([{
                "avg_price": vals["price"],
                "day_of_week": tomorrow.dayofweek,
                "is_weekend": 1 if tomorrow.dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded,
                "lag_1": vals["sold"]
            }])
            
            pred = predict_sales(features)
            preds.append({
                "product": sku, 
                "Input Harga": f"Rp {vals['price']:,}",
                "Input Sold Today": vals["sold"],
                "Prediksi Besok": round(float(pred[0]), 2)
            })
        
        # Sort by prediction descending
        preds = sorted(preds, key=lambda x: x["Prediksi Besok"], reverse=True)
        st.table(pd.DataFrame(preds))

# ================================
# Tab 4: DYNAMIC PREDIKSI MINGGU
# ================================
with tab4:
    st.subheader("📅 Prediksi Penjualan Minggu Depan (All Dynamic PRODUCTs)")
    
    st.info(f"📦 Predicting for {len(dynamic_skus)} PRODUCTs")
    
    input_today = {}
    
    # Batch input section
    st.markdown("### ⚡ Quick Batch Input")
    c1, c2 = st.columns(2)
    with c1:
        batch_price = st.number_input("Default Price for All PRODUCTs", value=10000, step=500, key="batch_price_minggu")
    with c2:
        batch_sold = st.number_input("Default Sold Today for All PRODUCTs", value=1, step=1, key="batch_sold_minggu")
    
    if st.button("Apply Batch Values"):
        st.session_state.apply_batch_minggu = True
    
    # Individual inputs for each PRODUCT
    for i, sku in enumerate(dynamic_skus):
        c1, c2 = st.columns(2)
        
        with c1:
            default_price = batch_price if st.session_state.get("apply_batch_minggu") else 10000
            price = st.number_input(
                f"Harga {sku}", 
                min_value=1000, 
                max_value=50000, 
                value=default_price, 
                step=500, 
                key=f"price_{sku}_minggu"
            )
        
        with c2:
            default_sold = batch_sold if st.session_state.get("apply_batch_minggu") else 1
            sold = st.number_input(
                f"Penjualan Hari Ini {sku}", 
                min_value=0, 
                max_value=100, 
                value=default_sold, 
                step=1, 
                key=f"sold_{sku}_minggu"
            )
        
        input_today[sku] = {"price": price, "sold": sold}
    
    if st.button("📅 Prediksi Minggu Depan"):
        preds = []
        
        for sku, vals in input_today.items():
            sku_encoded = get_sku_encoded(sku, dynamic_skus)
            
            features = pd.DataFrame([{
                "avg_price": vals["price"],
                "day_of_week": pd.Timestamp.today().dayofweek,
                "is_weekend": 1 if pd.Timestamp.today().dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded,
                "lag_1": vals["sold"]
            }])
            
            pred = predict_sales(features)
            preds.append({
                "product": sku, 
                "Prediksi Mingguan": round(float(pred[0] * 7), 2)
            })
        
        preds = sorted(preds, key=lambda x: x["Prediksi Mingguan"], reverse=True)
        st.table(pd.DataFrame(preds))

# ================================
# Tab 5: DYNAMIC PREDIKSI BULAN
# ================================
with tab5:
    st.subheader("📆 Prediksi Penjualan Bulan Depan (All Dynamic PRODUCTs)")
    
    st.info(f"📦 Predicting for {len(dynamic_skus)} PRODUCTs")
    
    input_today = {}
    
    # Batch input
    st.markdown("### ⚡ Quick Batch Input")
    c1, c2 = st.columns(2)
    with c1:
        batch_price = st.number_input("Default Price for All PRODUCTs", value=10000, step=500, key="batch_price_bulan")
    with c2:
        batch_sold = st.number_input("Default Sold Today for All PRODUCTs", value=1, step=1, key="batch_sold_bulan")
    
    if st.button("Apply Batch Values", key="batch_bulan"):
        st.session_state.apply_batch_bulan = True
    
    for i, sku in enumerate(dynamic_skus):
        c1, c2 = st.columns(2)
        
        with c1:
            default_price = batch_price if st.session_state.get("apply_batch_bulan") else 10000
            price = st.number_input(
                f"Harga {sku}", 
                min_value=1000, 
                max_value=50000, 
                value=default_price, 
                step=500, 
                key=f"price_{sku}_bulan"
            )
        
        with c2:
            default_sold = batch_sold if st.session_state.get("apply_batch_bulan") else 1
            sold = st.number_input(
                f"Penjualan Hari Ini {sku}", 
                min_value=0, 
                max_value=100, 
                value=default_sold, 
                step=1, 
                key=f"sold_{sku}_bulan"
            )
        
        input_today[sku] = {"price": price, "sold": sold}
    
    if st.button("📆 Prediksi Bulan Depan"):
        preds = []
        
        for sku, vals in input_today.items():
            sku_encoded = get_sku_encoded(sku, dynamic_skus)
            
            features = pd.DataFrame([{
                "avg_price": vals["price"],
                "day_of_week": pd.Timestamp.today().dayofweek,
                "is_weekend": 1 if pd.Timestamp.today().dayofweek >= 5 else 0,
                "sku_encoded": sku_encoded,
                "lag_1": vals["sold"]
            }])
            
            pred = predict_sales(features)
            preds.append({
                "product": sku, 
                "Prediksi Bulanan": round(float(pred[0] * 30), 2)
            })
        
        preds = sorted(preds, key=lambda x: x["Prediksi Bulanan"], reverse=True)
        st.table(pd.DataFrame(preds))

# ================================
# Tab 6: ENHANCED FILE UPLOAD
# ================================
with tab6:
    st.subheader("📂 Upload File XLSX untuk Prediksi (Dynamic PRODUCT Support)")
    
    st.info(f"📦 System recognizes {len(dynamic_skus)} unique PRODUCTs from database")
    
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            st.success("✅ File berhasil dibaca")
            st.dataframe(df.head(10))
            
            # Check PRODUCTs in uploaded file vs database
            file_skus = df["product"].unique() if "product" in df.columns else []
            st.markdown("### 📊 PRODUCT Analysis")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**PRODUCTs in File:**")
                for sku in file_skus:
                    status = "✅" if sku in dynamic_skus else "❓"
                    st.write(f"{status} {sku}")
            
            with c2:
                st.write("**PRODUCT Status:**")
                known = sum(1 for sku in file_skus if sku in dynamic_skus)
                unknown = len(file_skus) - known
                st.metric("Known PRODUCTs", known)
                st.metric("Unknown PRODUCTs", unknown)
            
            mode = st.radio("Pilih Mode Prediksi", ["Harian", "Mingguan", "Bulanan"])
            
            if st.button("🔮 Proses Prediksi"):
                preds = []
                tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
                
                for _, row in df.iterrows():
                    sku_encoded = get_sku_encoded(row["product"], dynamic_skus)
                    
                    features = pd.DataFrame([{
                        "avg_price": row["price"],
                        "day_of_week": tomorrow.dayofweek,
                        "is_weekend": 1 if tomorrow.dayofweek >= 5 else 0,
                        "sku_encoded": sku_encoded,
                        "lag_1": row["sold"]
                    }])
                    
                    pred = predict_sales(features)
                    
                    factor = 1
                    if mode == "Mingguan":
                        factor = 7
                    elif mode == "Bulanan":
                        factor = 30
                    
                    preds.append({
                        "product": row["product"], 
                        "Input Price": f"Rp {row['price']:,}",
                        "Input Sold": row["sold"],
                        f"Prediksi {mode}": round(float(pred[0] * factor), 2),
                        "PRODUCT Status": "✅ Known" if row["product"] in dynamic_skus else "❓ New"
                    })
                
                st.table(pd.DataFrame(preds))
                
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")


# ================================
# FOOTER INFO
# ================================
st.markdown("---")
status_text = f"**📊 System Status:** {len(dynamic_skus)} Dynamic PRODUCTs Loaded | 🔄 Manual Refresh Mode"
if st.session_state.last_refresh:
    status_text += f" | Last Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}"

st.markdown(status_text)
st.markdown("**💡 Tips:** Click '🔄 Refresh Data' to get latest data from Google Sheets")
