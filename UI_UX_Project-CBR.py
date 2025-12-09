import streamlit as st
import pandas as pd
import numpy as np
import gower
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. KONFIGURASI HALAMAN & FUNGSI UTAMA
# ==========================================
st.set_page_config(
    page_title="Sistem Rekomendasi HP (Gower + k-NN)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nama file dataset dan model
DATA_FILE = 'dataset_handphone_cleaned.csv'
MODEL_FILE = 'gower_recommender_model.pkl'

# Fitur yang digunakan untuk perhitungan (Harus sama persis dengan saat training)
FEATURES = ['Brand', 'Harga', 'RAM', 'Storage', 'Layar', 'Kamera', 'Baterai', 'OS']

@st.cache_resource
def load_or_train_model(force_retrain=False):
    """
    Mencoba memuat model yang tersimpan. 
    Jika tidak ada atau force_retrain=True, akan melatih ulang model.
    """
    model_data = {}
    
    # Coba load dataset dulu untuk referensi UI
    try:
        # Pastikan delimiter sesuai dengan file CSV Anda (biasanya , atau ;)
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"File '{DATA_FILE}' tidak ditemukan! Pastikan file csv ada di folder yang sama.")
        return None

    # Cek apakah perlu load file model atau train ulang
    if not force_retrain and os.path.exists(MODEL_FILE):
        try:
            model_data = joblib.load(MODEL_FILE)
        except Exception as e:
            st.warning(f"File model rusak, melatih ulang... (Error: {e})")
            force_retrain = True
    else:
        force_retrain = True

    # Proses Training Ulang
    if force_retrain:
        with st.spinner("Sedang melatih model baru..."):
            # 1. Buat Label Kelas Harga
            df['Kelas_Harga'] = pd.qcut(df['Harga'], q=3, labels=['Budget/Entry', 'Mid-Range', 'Flagship'])
            
            # 2. Siapkan Data
            X = df[FEATURES].copy()
            y = df['Kelas_Harga']
            
            # Pastikan tipe data benar
            X['Brand'] = X['Brand'].astype('object')
            X['OS'] = X['OS'].astype('object')
            
            # 3. Hitung Matriks Gower
            try:
                gower_dist = gower.gower_matrix(X)
            except Exception as e:
                st.error(f"Gagal menghitung Gower Distance: {e}")
                st.stop()
            
            # 4. Train k-NN
            knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
            knn.fit(gower_dist, y)
            
            # Simpan ke dictionary
            model_data = {
                'knn_model': knn,
                'reference_data': X,     # Data X (Features)
                'full_data': df          # Data Lengkap
            }
            
            # Simpan jadi file
            joblib.dump(model_data, MODEL_FILE)
            
    return model_data

# ==========================================
# 2. LOAD DATA & MODEL
# ==========================================
# Tombol reset di sidebar untuk mengatasi error model lama
st.sidebar.title("⚙️ Pengaturan")
if st.sidebar.button("🔄 Reset / Latih Ulang Model"):
    st.cache_resource.clear()
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    st.rerun()

data_model = load_or_train_model()

if data_model is None:
    st.stop()

knn = data_model['knn_model']
X_ref = data_model['reference_data']
df_full = data_model['full_data']

# ==========================================
# 3. INTERFACE PENGGUNA (SIDEBAR INPUT)
# ==========================================
st.sidebar.header("🛠️ Input Spesifikasi Idaman")

user_input = {}

# --- A. Input Brand & OS (Kategorikal) ---
available_brands = sorted(X_ref['Brand'].astype(str).unique())
available_os = sorted(X_ref['OS'].astype(str).unique())

user_input['Brand'] = st.sidebar.selectbox("Brand", options=available_brands)
user_input['OS'] = st.sidebar.selectbox("Sistem Operasi", options=available_os, index=0)

# --- B. Input Harga (Numerik Kontinu) ---
min_price = int(X_ref['Harga'].min())
max_price = int(X_ref['Harga'].max())
default_price = int(X_ref['Harga'].median())

user_input['Harga'] = st.sidebar.number_input(
    "Budget Harga (Rp)", 
    min_value=min_price, 
    max_value=max_price * 3, 
    value=default_price, 
    step=100000
)

# --- C. Input RAM & Storage (Numerik Diskrit) ---
available_ram = sorted(X_ref['RAM'].unique())
available_storage = sorted(X_ref['Storage'].unique())

def get_closest_index(lst, val):
    try:
        return lst.index(min(lst, key=lambda x: abs(x-val)))
    except:
        return 0

idx_ram = get_closest_index(available_ram, 8) 
idx_store = get_closest_index(available_storage, 128)

user_input['RAM'] = st.sidebar.selectbox("RAM (GB)", options=available_ram, index=idx_ram)
user_input['Storage'] = st.sidebar.selectbox("Memori Internal (GB)", options=available_storage, index=idx_store)

# --- D. Input Lainnya (Slider) ---
user_input['Layar'] = st.sidebar.slider("Ukuran Layar (inch)", 4.0, 8.0, 6.5, 0.1)
user_input['Kamera'] = st.sidebar.slider("Kamera Utama (MP)", 8, 200, 50, 2)
user_input['Baterai'] = st.sidebar.slider("Kapasitas Baterai (mAh)", 2000, 7000, 5000, 100)

# ==========================================
# 4. LOGIKA UTAMA (PREDIKSI & REKOMENDASI)
# ==========================================
st.title("📱 AI Rekomendasi Handphone")
st.markdown("Cari HP yang paling mirip dengan spesifikasi impianmu menggunakan **Gower Distance**.")

if st.sidebar.button("🔍 Cari Rekomendasi", type="primary"):
    
    # 1. Konversi Input User ke DataFrame
    df_input = pd.DataFrame([user_input])
    
    # [PENTING] Re-order kolom agar sesuai urutan FEATURES saat training
    # Ini mencegah error mismatch tipe data (misal: String dibandingkan dengan Float)
    df_input = df_input[FEATURES]
    
    # Pastikan tipe data konsisten
    df_input['Brand'] = df_input['Brand'].astype('object')
    df_input['OS'] = df_input['OS'].astype('object')

    with st.spinner('Sedang menghitung kemiripan...'):
        try:
            # 2. Hitung Jarak Gower
            distances = gower.gower_matrix(df_input, X_ref)[0]
            
            # 3. Prediksi Kelas Harga
            prediksi_kelas = knn.predict([distances])[0]
            
            # 4. Cari Top N
            top_n = 10
            sorted_indices = distances.argsort()[:top_n]
            
            # --- TAMPILKAN HASIL ---
            st.divider()
            col1, col2 = st.columns([1, 3])
            with col1:
                st.info("Prediksi Kategori")
            with col2:
                st.success(f"HP ini masuk kategori: **{prediksi_kelas}**")

            st.subheader(f"Top {top_n} Rekomendasi")
            
            results = []
            for idx in sorted_indices:
                row_data = df_full.iloc[idx]
                similarity_score = (1 - distances[idx]) * 100 
                
                results.append({
                    'Nama HP': row_data['Nama_HP'],
                    'Kemiripan': f"{similarity_score:.2f}%",
                    'Harga': f"Rp {row_data['Harga']:,.0f}",
                    'RAM': f"{row_data['RAM']} GB",
                    'Storage': f"{row_data['Storage']} GB",
                    'Layar': f"{row_data['Layar']}\"",
                    'Kamera': f"{row_data['Kamera']} MP",
                    'Baterai': f"{row_data['Baterai']} mAh",
                    'Kelas': row_data['Kelas_Harga']
                })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat perhitungan: {e}")
            st.write("Tips: Coba klik tombol 'Reset / Latih Ulang Model' di sidebar paling atas.")

else:
    st.info("👈 Masukkan spesifikasi di sidebar kiri, lalu klik tombol **Cari Rekomendasi**.")
    with st.expander("Lihat Database HP"):
        st.dataframe(df_full.drop(columns=['Kelas_Harga'], errors='ignore'))