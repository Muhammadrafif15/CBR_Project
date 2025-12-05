import streamlit as st
import pandas as pd
import numpy as np

# --- 1. Definisi Fitur (PENTING) ---
# Pisahkan nama kolom berdasarkan tipenya
NUMERICAL_COLS = [
    'Harga', 'Ram', 'Memori_internal', 'Ukuran_layar', 
    'Resolusi_kamera', 'Kapasitas_baterai', 'Rating_pengguna'
]
CATEGORICAL_COLS = ['Brand', 'Os', 'Stok_tersedia']
IDENTIFIER_COL = 'Nama_hp'


# --- 2. Fungsi Memuat, MEMBERSIHKAN, dan Menghitung Rentang ---
@st.cache_data
def load_clean_and_precompute(filepath):
    """
    Memuat data MENTAH dari 'dataset.handphone.csv', membersihkannya, 
    dan menghitung rentang (Range).
    """
    try:
        # 1. Muat data mentah dengan delimiter ;
        df = pd.read_csv(filepath, delimiter=';')
        
        # 2. Tentukan fitur yang akan kita gunakan
        all_cols_to_keep = [IDENTIFIER_COL] + NUMERICAL_COLS + CATEGORICAL_COLS
        
        # Ambil hanya kolom yang kita perlukan
        df = df[all_cols_to_keep]

        # --- 3. Pembersihan Data (di dalam Streamlit) ---
        # 3a. Bersihkan Harga
        df['Harga'] = df['Harga'].astype(str).str.replace(r'[.]', '', regex=True).str.replace(r'[,]', '.', regex=True)
        
        # 3b. Bersihkan Resolusi_kamera (hapus 'MP')
        df['Resolusi_kamera'] = df['Resolusi_kamera'].astype(str).str.replace('MP', '', regex=False)
        
        # 3c. Konversi 'Stok_tersedia' ke string
        df['Stok_tersedia'] = df['Stok_tersedia'].astype(str)

        # 3d. Konversi semua kolom numerik
        df[NUMERICAL_COLS] = df[NUMERICAL_COLS].apply(pd.to_numeric, errors='coerce')
        
        # 3e. Isi NaNs di kolom kategorikal dengan 'Unknown'
        for col in CATEGORICAL_COLS:
            df[col] = df[col].fillna('Unknown')
            
        # 3f. Hapus baris yang memiliki NaN di kolom penting
        df_cleaned = df.dropna(subset=[IDENTIFIER_COL] + NUMERICAL_COLS).reset_index(drop=True)
        
        # 3g. Ubah tipe data numerik yang seharusnya integer (agar tampilan dropdown bersih, tidak ada .0)
        int_cols = ['Ram', 'Memori_internal', 'Resolusi_kamera', 'Kapasitas_baterai']
        for col in int_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype(int)
        
        # --- 4. Penghitungan Rentang (Pre-computation) ---
        # Hitung rentang HANYA untuk kolom numerik
        ranges = df_cleaned[NUMERICAL_COLS].max() - df_cleaned[NUMERICAL_COLS].min()
        ranges[ranges == 0] = 1  # Hindari pembagian dengan nol
        
        # 5. Dapatkan nilai unik untuk UI (UPDATE: Mengambil unik untuk SEMUA kolom agar bisa jadi dropdown)
        # Kita gabungkan kolom kategorikal dan numerik (kecuali Harga biasanya input manual, tapi kita masukkan saja)
        ui_cols = CATEGORICAL_COLS + NUMERICAL_COLS
        ui_options = {col: sorted(df_cleaned[col].unique().tolist()) for col in ui_cols}
        
        return df_cleaned, ranges, ui_options
        
    except FileNotFoundError:
        st.error(f"File {filepath} tidak ditemukan. Pastikan 'dataset.handphone.csv' ada di folder yang sama.")
        return None, None, None
    except KeyError as e:
        st.error(f"Kolom yang diperlukan tidak ditemukan: {e}. Pastikan file CSV Anda memiliki kolom yang benar.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau membersihkan data: {e}")
        return None, None, None

# --- 3. Fungsi Gower's Similarity (MIXED DATA) ---
def calculate_gower_similarity(case_base, new_case, db_ranges):
    """
    Menghitung Gower's Similarity untuk TIPE DATA CAMPURAN.
    """
    similarity_scores = []
    
    # 1. Hitung similaritas untuk fitur NUMERIK
    for col in NUMERICAL_COLS:
        # Jika user input tidak sama persis dengan database, kita hitung jaraknya
        abs_diff = (case_base[col] - new_case[col]).abs()
        s_i = 1 - (abs_diff / db_ranges[col])
        similarity_scores.append(s_i)
        
    # 2. Hitung similaritas untuk fitur KATEGORIKAL
    for col in CATEGORICAL_COLS:
        s_i = (case_base[col] == new_case[col]).astype(int)
        similarity_scores.append(s_i)

    # 3. Gabungkan semua skor
    all_scores = pd.concat(similarity_scores, axis=1)
    
    # 4. Total similaritas adalah rata-rata dari semua skor fitur
    total_similarity = all_scores.mean(axis=1)
    
    # 5. Buat DataFrame hasil
    results_df = pd.DataFrame({
        IDENTIFIER_COL: case_base[IDENTIFIER_COL],
        'Similarity': total_similarity,
    }).sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    
    return results_df

# --- 4. Setup UI Streamlit ---
st.set_page_config(page_title="CBR Handphone", layout="wide")
st.title("📱 CBR Handphone (Gower's Similarity - Dropdown UI)")
st.write("Silahkan pilih spesifikasi yang diinginkan dari opsi yang tersedia.")

# Muat, BERSIHKAN, dan hitung rentang dari file ASLI
df_case_base, db_ranges, ui_options = load_clean_and_precompute('dataset.handphone.csv')

if df_case_base is not None:
    with st.expander("Lihat Database Kasus (Telah Dibersihkan)"):
        st.dataframe(df_case_base)

    # Dapatkan min/max untuk slider/input harga
    min_vals = df_case_base[NUMERICAL_COLS].min()
    max_vals = df_case_base[NUMERICAL_COLS].max()

    # --- Sidebar untuk Input Kasus Baru ---
    st.sidebar.header("Masukkan Spesifikasi (Kasus Baru):")
    
    new_case_input = {}
    
    # --- Input Harga (Tetap Number Input atau Slider biasanya lebih fleksibel, tapi sisa numerik jadi dropdown) ---
    st.sidebar.subheader("Fitur Numerik")
    
    # Harga tetap number input karena budget orang spesifik, tidak terpaku pada nilai diskrit database
    new_case_input['Harga'] = st.sidebar.number_input(
        "Harga (Rp) - Masukkan Budget", 
        min_value=float(min_vals['Harga']), 
        max_value=float(max_vals['Harga'] * 1.5),
        value=float(df_case_base['Harga'].median()),
        step=100000.0
    )
    
    # --- BAGIAN YANG DIUBAH: SLIDER MENJADI DROPDOWN (SELECTBOX) ---
    # Loop mulai dari index 1 karena index 0 adalah 'Harga' yang sudah dihandle diatas
    for col in NUMERICAL_COLS[1:]: 
        # Ambil opsi unik yang tersedia dari fungsi load data
        available_options = ui_options[col]
        
        # Cari nilai median untuk dijadikan default index agar tidak otomatis memilih yang terkecil
        median_val = df_case_base[col].median()
        # Cari index di list options yang nilainya paling dekat dengan median
        default_index = min(range(len(available_options)), key=lambda i: abs(available_options[i]-median_val))
        
        new_case_input[col] = st.sidebar.selectbox(
            col.replace("_", " ").title(), # Label cantik
            options=available_options,     # Pilihan dari data yang ada
            index=default_index            # Default value
        )
        
    # --- Input Kategorikal (Selectbox) - Tidak Berubah ---
    st.sidebar.subheader("Fitur Kategorikal")
    for col in CATEGORICAL_COLS:
        new_case_input[col] = st.sidebar.selectbox(
            col.replace("_", " ").title(), 
            options=sorted(ui_options[col]) 
        )

    # --- Tombol dan Tampilan Hasil ---
    if st.sidebar.button("🔍 Cari Handphone Serupa"):
        st.header("Hasil Rekomendasi (Paling Mirip)")
        
        # Hitung similaritas
        results = calculate_gower_similarity(df_case_base, new_case_input, db_ranges)
        
        # Gabungkan hasil dengan data asli untuk perbandingan
        results_detailed = pd.merge(results, df_case_base, on=IDENTIFIER_COL)
        
        # Format kolom similarity
        results_detailed['Similarity'] = results_detailed['Similarity'].map('{:.2%}'.format)
        
        # Tampilkan 10 hasil teratas
        st.dataframe(results_detailed.head(10))

else:
    st.error("Aplikasi tidak dapat dimulai karena gagal memuat data.")