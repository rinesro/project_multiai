import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="EnergiCerdas AI (Local Edge)", page_icon="⚡", layout="wide")

# ==========================================
# 2. DATA INGESTION & VALIDATOR AGENT
# ==========================================
class DataIngestionValidatorAgent:
    def __init__(self):
        self.TARIF_KWH = 1699.53
        self.PPJ = 0.03
        self.BIAYA_BEBAN = 47500
        self.FAKTOR_EMISI = 0.87
        self.BATAS_TOLERANSI = 0.15

    def proses_data(self, luas_rumah, penghuni, tagihan_asli, daftar_alat):
        total_kwh = 0.0
        alat_valid = []
        for alat in daftar_alat:
            kwh = (alat['watt'] * alat['jam'] * 30) / 1000
            total_kwh += kwh
            alat_valid.append({
                'Kategori_Alat': alat['kategori'], 'Watt': alat['watt'],
                'Jam_Per_Hari': alat['jam'], 'SKEM_Bintang': alat['skem'],
                'kWh_Bulanan': kwh, 'Nama': alat['nama']
            })
        total_kwh += 50.0  # Base load
        
        ike = total_kwh / max(1.0, float(luas_rumah))
        emisi = total_kwh * self.FAKTOR_EMISI
        
        rp_pemakaian = total_kwh * self.TARIF_KWH
        estimasi_tagihan = rp_pemakaian + (rp_pemakaian * self.PPJ) + self.BIAYA_BEBAN
        
        selisih_pct = abs(tagihan_asli - estimasi_tagihan) / max(1.0, float(tagihan_asli))
        is_anomali = selisih_pct > self.BATAS_TOLERANSI
        
        return {
            "total_kwh": total_kwh, "estimasi_rp": estimasi_tagihan, "ike": ike,
            "emisi": emisi, "is_anomali": is_anomali, "selisih_pct": selisih_pct,
            "alat_valid": alat_valid
        }

# ==========================================
# 3. LOAD SEMUA MODEL (KLASIK & LLM)
# ==========================================
@st.cache_resource(show_spinner="Memuat Model Machine Learning Klasik...")
def load_classic_models():
    dt_macro = joblib.load("dt_macro_profiler.pkl")
    dt_micro = joblib.load("dt_micro_appliance.pkl")
    knn_model = joblib.load("knn_biaya_lingkungan_recommender.pkl")
    scaler = joblib.load("knn_scaler.pkl")
    df_role_model = pd.read_csv("database_role_model.csv")
    return dt_macro, dt_micro, knn_model, scaler, df_role_model

@st.cache_resource(show_spinner="Memuat Llama-3 AI Agent ke GPU (Harap Tunggu)...")
def load_local_llm():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "llama3_energicerdas_lora", # Folder model lokalmu
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Mode inferensi agar cepat 2x lipat
    return model, tokenizer

try:
    dt_macro, dt_micro, knn_model, scaler, df_role_model = load_classic_models()
    llama_model, llama_tokenizer = load_local_llm()
except Exception as e:
    st.error(f"❌ Gagal memuat model. Error: {e}")
    st.stop()

# ==========================================
# 4. FUNGSI ORCHESTRATOR & LOKAL LLM
# ==========================================
def dapatkan_rekomendasi_knn(luas, penghuni, list_intent):
    user_baru = pd.DataFrame([{'Luas_Rumah_m2': luas, 'Jumlah_Penghuni': penghuni}])
    user_baru_scaled = scaler.transform(user_baru)
    jarak, indeks = knn_model.kneighbors(user_baru_scaled)
    kandidat = df_role_model.iloc[indeks[0]].copy()
    
    biaya_min, biaya_max = kandidat['Total_Tagihan_Rp'].min(), kandidat['Total_Tagihan_Rp'].max()
    kandidat['Skor_Biaya'] = (kandidat['Total_Tagihan_Rp'] - biaya_min) / (biaya_max - biaya_min + 1e-9)
    emisi_min, emisi_max = kandidat['Emisi_CO2_kg'].min(), kandidat['Emisi_CO2_kg'].max()
    kandidat['Skor_Lingkungan'] = (kandidat['Emisi_CO2_kg'] - emisi_min) / (emisi_max - emisi_min + 1e-9)
    
    kandidat['Skor_Akhir'] = 0.0
    if 'Biaya' in list_intent: kandidat['Skor_Akhir'] += kandidat['Skor_Biaya']
    if 'Lingkungan' in list_intent: kandidat['Skor_Akhir'] += kandidat['Skor_Lingkungan']
    if not list_intent: kandidat['Skor_Akhir'] = kandidat['IKE_kWh_per_m2']

    return kandidat.sort_values(by='Skor_Akhir', ascending=True).iloc[0]

def generate_local_explainability(profil_makro, hasil_mikro, payload_dasar, knn_target, intent_user):
    # Format Alpaca sesuai dengan saat kita melatih model
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""
    
    # 1. Konversi format ke string teks persis seperti dataset latih
    anomali_str = "ADA INDIKASI KEBOCORAN" if payload_dasar['is_anomali'] else "Normal"
    perangkat_boros = [p['Nama'] for p in hasil_mikro if p['Status'] == 'Boros']
    str_perangkat = ", ".join(perangkat_boros) if perangkat_boros else "Tidak ada"
    fokus_str = intent_user[0] if intent_user else "Biaya"
    
    target_tagihan = int(knn_target['Total_Tagihan_Rp'])
    target_emisi = round(knn_target['Emisi_CO2_kg'], 1)
    
    instruksi = "Kamu adalah konsultan energi pintar. Berikan penjelasan hasil analisis AI rumah tangga dan rekomendasi SDG dalam bahasa Indonesia yang ramah."
    input_teks = f"Anomali: {anomali_str}. Profil Makro: {profil_makro}. Perangkat Boros: {str_perangkat}. Target KNN: Tagihan Rp{target_tagihan}, Emisi {target_emisi}kg. Fokus: {fokus_str}."
    
    # 2. Tokenisasi & Prediksi
    inputs = llama_tokenizer([alpaca_prompt.format(instruksi, input_teks)], return_tensors="pt").to("cuda")
    
    # Generate Output (Maks 256 kata)
    outputs = llama_model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=llama_tokenizer.eos_token_id)
    
    # 3. Parsing Hasil (Membuang prompt instruksi dan hanya mengambil bagian Response)
    decoded_output = llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    hasil_akhir = decoded_output.split("### Response:\n")[-1].strip()
    
    return hasil_akhir

# ==========================================
# 5. ANTARMUKA STREAMLIT
# ==========================================
st.title("⚡ EnergiCerdas AI (100% Private & Offline)")
st.markdown("Sistem Multi-AI menggunakan Custom Llama-3 8B. Data Anda tidak dikirim ke server pihak ketiga.")

# Form Input seperti biasa
st.header("1. Profil Rumah Tangga")
c1, c2, c3 = st.columns(3)
with c1: luas_rumah = st.number_input("Luas Bangunan (m²)", 20, 500, 45)
with c2: penghuni = st.number_input("Jumlah Penghuni", 1, 15, 3)
with c3: tagihan_asli = st.number_input("Tagihan Bulan Lalu (Rp)", 50000, 5000000, 600000)

st.header("2. Preferensi Optimasi")
intent_user = st.multiselect("Pilih fokus rekomendasi:", ["Biaya", "Lingkungan"], default=["Biaya"])

st.header("3. Perangkat Listrik Utama")
col_a, col_b = st.columns(2)
with col_a:
    ac_jam = st.number_input("Jam AC/Hari", 0.0, 24.0, 8.0)
    ac_watt = st.number_input("Watt AC", 200, 2000, 400)
    ac_skem = st.selectbox("Label SKEM AC", ["Bintang 4/5 (Inverter)", "Bintang 1 (Non-Inverter)"])
    
with col_b:
    kulkas_watt = st.number_input("Watt Kulkas", 50, 500, 150)
    tv_jam = st.number_input("Jam TV/PC", 0.0, 24.0, 4.0)
    mcuci_jam = st.number_input("Jam Mesin Cuci", 0.0, 10.0, 1.5)

daftar_perangkat = [
    {"kategori": 1, "nama": "AC", "watt": ac_watt, "jam": ac_jam, "skem": 4 if "Inverter" in ac_skem else 1},
    {"kategori": 2, "nama": "Kulkas", "watt": kulkas_watt, "jam": 24.0, "skem": 4},
    {"kategori": 3, "nama": "TV/PC", "watt": 100, "jam": tv_jam, "skem": 0},
    {"kategori": 4, "nama": "Mesin Cuci", "watt": 350, "jam": mcuci_jam, "skem": 4}
]

# ==========================================
# 6. EKSEKUSI PIPELINE
# ==========================================
if st.button("🚀 Mulai Analisis Multi-AI", type="primary", use_container_width=True):
    with st.spinner("Mengorkestrasi Agen AI & Generasi Narasi (Llama-3)..."):
        # Pipeline 1-4
        agent = DataIngestionValidatorAgent()
        payload = agent.proses_data(luas_rumah, penghuni, tagihan_asli, daftar_perangkat)
        
        df_mikro = pd.DataFrame(payload['alat_valid'])
        prediksi_mikro = dt_micro.predict(df_mikro[['Kategori_Alat', 'Watt', 'Jam_Per_Hari', 'SKEM_Bintang', 'kWh_Bulanan']])
        hasil_mikro = [{"Nama": a['Nama'], "Status": p} for a, p in zip(payload['alat_valid'], prediksi_mikro)]
        
        profil_makro = dt_macro.predict(pd.DataFrame([{'Luas_Rumah_m2': luas_rumah, 'Jumlah_Penghuni': penghuni, 'Total_Konsumsi_kWh': payload['total_kwh']}]))[0]
        target_knn = dapatkan_rekomendasi_knn(luas_rumah, penghuni, intent_user)
        
        # Pipeline 5 (Local LLM)
        narasi_akhir = generate_local_explainability(profil_makro, hasil_mikro, payload, target_knn, intent_user)
        
    # Render Hasil
    st.divider()
    st.subheader("📊 Dasbor Analisis (Rule-Based & Decision Tree)")
    if payload['is_anomali']: st.error("⚠️ ANOMALI TERDETEKSI: Arus bocor atau data tidak wajar.")
    else: st.success("✅ Tagihan Wajar.")
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimasi Sistem", f"Rp {payload['estimasi_rp']:,.0f}")
    m2.metric("Profil IKE", profil_makro)
    m3.metric("Status Anomali", "Ya" if payload['is_anomali'] else "Tidak")
    m4.metric("Emisi Karbon", f"{payload['emisi']:.1f} kg")
    
    st.divider()
    st.subheader("🤖 Penjelasan Llama-3 AI")
    st.markdown(narasi_akhir)