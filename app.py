import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import warnings
import google.generativeai as genai

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EnergiCerdas AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); min-height: 100vh; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #00d4ff);
    background-size: 200% auto; -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite; line-height: 1.1; margin-bottom: 0.3rem;
}
@keyframes shimmer { 0%{background-position:0% center} 100%{background-position:200% center} }
.hero-sub { color: #8892a4; font-size: 1rem; font-weight: 300; letter-spacing: 0.05em; margin-bottom: 2.5rem; }

.section-label {
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase; color: #00d4ff; margin-bottom: 0.5rem;
}
.section-title { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #e8edf5; margin-bottom: 1.5rem; }

.card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.card-accent { background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.card-warn  { background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.25); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }

.metric-box { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 1.2rem; text-align: center; }
.metric-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #6b7a8d; margin-bottom: 0.4rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #e8edf5; line-height: 1.1; }
.metric-unit  { font-size: 0.75rem; color: #6b7a8d; margin-top: 0.2rem; }

.profile-badge { display: inline-block; padding: 0.4rem 1rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; }
.badge-hemat { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-sedang{ background: rgba(251,191,36,0.15); color: #f59e0b; border: 1px solid rgba(251,191,36,0.3); }
.badge-boros { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-anomali{background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }

.rekom-box {
    background: linear-gradient(135deg, rgba(123,47,247,0.1), rgba(0,212,255,0.05));
    border: 1px solid rgba(123,47,247,0.25); border-radius: 16px; padding: 1.8rem;
    color: #c8d3e0; font-size: 0.95rem; line-height: 1.8; white-space: pre-wrap;
}
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 2rem 0; }

.stSelectbox > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e8edf5 !important; border-radius: 10px !important; }
.stCheckbox label { color: #c8d3e0 !important; font-size: 0.9rem; }
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d4ff) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important; width: 100%;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(123,47,247,0.4) !important; }
div[data-testid="stNumberInput"] input { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e8edf5 !important; }
div[data-testid="stTextInput"] input  { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: #e8edf5 !important; }
label { color: #8892a4 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    dt_profiler    = joblib.load("models/decision_tree_profiler.pkl")
    anomaly_model  = joblib.load("models/ai_anomaly_model.pkl")
    anomaly_scaler = joblib.load("models/ai_anomaly_scaler.pkl")
    with open("models/ai_anomaly_features.pkl", "rb") as f:
        anomaly_features = pickle.load(f)
    return dt_profiler, anomaly_model, anomaly_scaler, anomaly_features

# ─── CONSTANTS ─────────────────────────────────────────────────
# Tarif PLN Triwulan II 2026 (April–Juni 2026)
TARIF_PLN = {
    "Rumah Tangga 900 VA (RTM)":    1352.00,
    "Rumah Tangga 1.300 VA":        1444.70,
    "Rumah Tangga 2.200 VA":        1444.70,
    "Rumah Tangga 3.500 VA":        1699.53,
    "Rumah Tangga 4.400 VA":        1699.53,
    "Rumah Tangga 5.500 VA":        1699.53,
    "Rumah Tangga 6.600 VA":        1699.53,
    "Rumah Tangga ≥ 6.600 VA":      1699.53,
    "Bisnis 6.600 VA – 200 kVA":    1444.70,
    "Bisnis & Industri ≥ 200 kVA":  1122.00,
    "Industri ≥ 30.000 kVA":         996.74,
}

EMISSION_FACTOR = 0.87  # kgCO2/kWh (ESDM 2020)

# IKE threshold dari dataset sintetis (Nilai_IKE = Konsumsi/Luas)
# Boros: IKE median 7.74 | Tidak Boros: IKE median 3.00
IKE_THRESHOLD_BOROS    = 5.0   # kWh/m²/bulan
IKE_THRESHOLD_SANGAT   = 10.0  # kWh/m²/bulan → sangat boros

# Mapping peralatan form → Nama_Perangkat dataset DT
# Dataset punya: TV, Mesin Cuci, AC, Rice Cooker, Kulkas
APPLIANCES = [
    {"name": "AC",                    "watt": 800,  "icon": "❄️",  "dt_name": "AC",          "luas_ref": 15},
    {"name": "Kulkas",                "watt": 150,  "icon": "🧊",  "dt_name": "Kulkas",       "luas_ref": 5},
    {"name": "Mesin Cuci",            "watt": 400,  "icon": "🫧",  "dt_name": "Mesin Cuci",   "luas_ref": 4},
    {"name": "TV",                    "watt": 100,  "icon": "📺",  "dt_name": "TV",           "luas_ref": 12},
    {"name": "Rice Cooker",           "watt": 400,  "icon": "🍚",  "dt_name": "Rice Cooker",  "luas_ref": 4},
    {"name": "Pompa Air",             "watt": 250,  "icon": "💧",  "dt_name": None,           "luas_ref": 4},
    {"name": "Laptop / PC",           "watt": 65,   "icon": "💻",  "dt_name": None,           "luas_ref": 4},
    {"name": "Lampu (semua ruangan)", "watt": 60,   "icon": "💡",  "dt_name": None,           "luas_ref": 10},
    {"name": "Water Heater/Dispenser","watt": 350,  "icon": "🚿",  "dt_name": None,           "luas_ref": 3},
    {"name": "Microwave / Oven",      "watt": 800,  "icon": "🍳",  "dt_name": None,           "luas_ref": 4},
]

# ─── COMPUTE ───────────────────────────────────────────────────
def compute_konsumsi(items):
    """
    Hitung kWh per peralatan dan total.
    items: list of dict {name, watt, qty, hours, icon, dt_name, luas_ref}
    Returns: kwh_total, breakdown list, jam_siang, jam_malam
    """
    kwh_total = 0.0
    jam_siang = jam_malam = 0.0
    breakdown = []

    for item in items:
        kwh = item["watt"] * item["qty"] * item["hours"] * 30 / 1000
        kwh_total += kwh

        # Estimasi proporsi siang vs malam per jenis alat
        if item["name"] in ["Lampu (semua ruangan)", "TV"]:
            jam_malam += item["hours"] * 0.65
            jam_siang += item["hours"] * 0.35
        elif item["name"] in ["AC"]:
            jam_malam += item["hours"] * 0.55
            jam_siang += item["hours"] * 0.45
        elif item["name"] in ["Laptop / PC", "Microwave / Oven", "Rice Cooker"]:
            jam_siang += item["hours"] * 0.70
            jam_malam += item["hours"] * 0.30
        else:
            jam_siang += item["hours"] * 0.50
            jam_malam += item["hours"] * 0.50

        breakdown.append({**item, "kwh": kwh})

    return kwh_total, breakdown, jam_siang, jam_malam

# ─── AI PROFILER: DECISION TREE ────────────────────────────────
def dt_classify_appliances(breakdown, luas_rumah, dt_model):
    """
    Jalankan Decision Tree per peralatan listrik.
    Output: list hasil per alat + ringkasan profil rumah tangga.

    DT features: Kategori_Energi, Nama_Perangkat, Satuan_Energi,
                 Luas_Ruangan_m2, Konsumsi_Bulanan
    DT classes:  Boros | Tidak Boros | N/A (Non-Listrik)
    """
    results = []
    boros_count = 0
    tidak_boros_count = 0
    total_listrik = 0

    for item in breakdown:
        dt_name = item.get("dt_name")
        kwh     = item["kwh"]

        # Hanya peralatan listrik yang bisa diklasifikasi DT
        # Alat tanpa dt_name: pakai IKE manual
        if dt_name:
            luas_ref = item.get("luas_ref", luas_rumah)
            df_input = pd.DataFrame([{
                "Kategori_Energi":  "Listrik",
                "Nama_Perangkat":   dt_name,
                "Satuan_Energi":    "kWh",
                "Luas_Ruangan_m2":  luas_ref,
                "Konsumsi_Bulanan": kwh,
            }])
            label = dt_model.predict(df_input)[0]
            proba = dt_model.predict_proba(df_input)[0]
            conf  = max(proba) * 100
        else:
            # IKE manual untuk alat yang tidak ada di dataset DT
            ike = kwh / max(item.get("luas_ref", 5), 1)
            if ike > IKE_THRESHOLD_BOROS:
                label = "Boros"
            else:
                label = "Tidak Boros"
            conf = 75.0  # confidence default untuk rule-based

        if label == "Boros":
            boros_count += 1
        elif label == "Tidak Boros":
            tidak_boros_count += 1

        total_listrik += 1
        results.append({
            "name":   item["name"],
            "icon":   item["icon"],
            "kwh":    kwh,
            "qty":    item["qty"],
            "hours":  item["hours"],
            "label":  label,
            "conf":   conf,
        })

    # Profil rumah tangga dari agregasi hasil DT
    if total_listrik == 0:
        profil = "Tidak Ada Data"
        badge  = "badge-sedang"
        desc   = "Tidak ada peralatan listrik yang terdeteksi."
    else:
        rasio_boros = boros_count / total_listrik
        if rasio_boros == 0:
            profil = "Hemat Energi"
            badge  = "badge-hemat"
            desc   = "Semua peralatan beroperasi dalam batas konsumsi yang efisien. Pertahankan kebiasaan ini!"
        elif rasio_boros <= 0.4:
            profil = "Cukup Efisien"
            badge  = "badge-sedang"
            desc   = f"{boros_count} dari {total_listrik} peralatan tergolong boros. Ada ruang perbaikan yang cukup besar."
        elif rasio_boros <= 0.7:
            profil = "Konsumsi Tinggi"
            badge  = "badge-boros"
            desc   = f"{boros_count} dari {total_listrik} peralatan tergolong boros. Diperlukan optimasi segera."
        else:
            profil = "Sangat Boros"
            badge  = "badge-boros"
            desc   = f"Mayoritas peralatan ({boros_count}/{total_listrik}) tergolong boros. Perlu perhatian serius."

    return results, profil, badge, desc, boros_count, tidak_boros_count

# ─── AI ANOMALY DETECTION ──────────────────────────────────────
def detect_anomaly(kwh_total, breakdown, jam_siang, jam_malam,
                   anomaly_model, anomaly_scaler, anomaly_features):
    """
    Deteksi pola konsumsi tidak wajar menggunakan Isolation Forest.
    Features: total_kwh, avg_kwh, std_kwh, max_kwh, rasio_malam,
              delta_kwh_pct, cv_kwh, avg_sub1, avg_sub2, avg_sub3
    """
    kwh_values = [item["kwh"] for item in breakdown]
    if not kwh_values:
        return False, 0.0, "Normal"

    avg_kwh = np.mean(kwh_values)
    std_kwh = np.std(kwh_values) if len(kwh_values) > 1 else 0
    max_kwh = max(kwh_values)
    rasio_malam = jam_malam / max(jam_siang + jam_malam, 1)
    cv_kwh = std_kwh / max(avg_kwh, 1)

    # delta_kwh_pct: selisih max-min sebagai % dari total
    min_kwh = min(kwh_values)
    delta_kwh_pct = ((max_kwh - min_kwh) / max(kwh_total, 1)) * 100

    # sub_metering proxy (normalized ke unit dataset)
    kitchen_names = {"Rice Cooker", "Microwave / Oven"}
    laundry_names = {"Mesin Cuci", "Kulkas", "Water Heater/Dispenser", "Laptop / PC", "Lampu (semua ruangan)"}
    heavy_names   = {"AC", "Pompa Air"}

    sub1 = sum(i["kwh"] for i in breakdown if i["name"] in kitchen_names) / 30 / 24 * 60 / 1000
    sub2 = sum(i["kwh"] for i in breakdown if i["name"] in laundry_names) / 30 / 24 * 60 / 1000
    sub3 = sum(i["kwh"] for i in breakdown if i["name"] in heavy_names)   / 30 / 24 * 60 / 1000

    feature_values = {
        "total_kwh":      kwh_total,
        "avg_kwh":        avg_kwh,
        "std_kwh":        std_kwh,
        "max_kwh":        max_kwh,
        "rasio_malam":    rasio_malam,
        "delta_kwh_pct":  delta_kwh_pct,
        "cv_kwh":         cv_kwh,
        "avg_sub1":       sub1,
        "avg_sub2":       sub2,
        "avg_sub3":       sub3,
    }

    arr    = np.array([[feature_values[f] for f in anomaly_features]])
    scaled = anomaly_scaler.transform(arr)
    score  = anomaly_model.score_samples(scaled)[0]
    pred   = anomaly_model.predict(scaled)[0]  # -1=anomali, 1=normal

    is_anomali = (pred == -1)
    # Normalize score ke 0-100 (makin tinggi = makin anomali)
    anomali_pct = max(0, min(100, (-score - 0.35) / 0.45 * 100))

    status = "Anomali Terdeteksi" if is_anomali else "Pola Normal"
    return is_anomali, anomali_pct, status

# ─── GEMINI ────────────────────────────────────────────────────
def get_gemini_recommendation(profil, kwh, biaya, emisi, breakdown,
                               golongan, dt_results, is_anomali, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    # Buat daftar peralatan dengan hasil DT
    ap_lines = []
    for r in dt_results:
        status_icon = "🔴" if r["label"] == "Boros" else "🟢"
        qty_label = f" ×{r['qty']} unit" if r.get("qty", 1) > 1 else ""
        ap_lines.append(
            f"  {status_icon} {r['icon']} {r['name']}{qty_label}: "
            f"{r['kwh']:.1f} kWh/bln — DT: **{r['label']}**"
        )

    anomali_note = ""
    if is_anomali:
        anomali_note = "\n⚠️ CATATAN: Sistem mendeteksi ANOMALI pada pola konsumsi ini. Kemungkinan ada peralatan rusak, bocor arus, atau pola penggunaan tidak wajar.\n"

    prompt = f"""Kamu adalah konsultan energi rumah tangga yang ahli dan ramah. Berikan rekomendasi berbasis data analisis AI berikut:

DATA RUMAH TANGGA:
- Golongan PLN: {golongan}
- Profil AI (Decision Tree): {profil}
- Estimasi Konsumsi: {kwh:.1f} kWh/bulan
- Estimasi Tagihan: Rp{biaya:,.0f}/bulan
- Emisi CO₂: {emisi:.1f} kg/bulan
{anomali_note}
HASIL ANALISIS DECISION TREE PER PERALATAN:
{chr(10).join(ap_lines)}

Berikan respons dalam format berikut:

🔍 ANALISIS SITUASI
[Analisis singkat 2-3 kalimat tentang profil dan hasil DT per peralatan]

⚡ PERALATAN PALING BOROS
[Fokus pada peralatan berlabel Boros dari hasil DT, jelaskan mengapa dan dampaknya]

💡 REKOMENDASI HEMAT ENERGI
1. [Rekomendasi spesifik + estimasi penghematan Rp atau %]
2. [Rekomendasi spesifik + estimasi penghematan Rp atau %]
3. [Rekomendasi spesifik + estimasi penghematan Rp atau %]

🌱 DAMPAK LINGKUNGAN
[Dampak penghematan terhadap emisi CO₂ dan SDG ke-7]

Gunakan Bahasa Indonesia yang ramah dan spesifik berdasarkan data di atas."""

    return model.generate_content(prompt).text

# ─── MAIN ──────────────────────────────────────────────────────
def main():
    st.markdown('<div class="hero-title">⚡ EnergiCerdas AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Analisis konsumsi listrik rumah tangga · '
        'Decision Tree Profiler + Anomaly Detection + Gemini AI</div>',
        unsafe_allow_html=True
    )

    try:
        dt_model, anomaly_model, anomaly_scaler, anomaly_features = load_models()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.info("Pastikan folder `models/` berisi: decision_tree_profiler.pkl, ai_anomaly_model.pkl, ai_anomaly_scaler.pkl, ai_anomaly_features.pkl")
        st.stop()

    # API Key dari Streamlit Secrets
    try:
        gemini_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_key = ""

    # ── STEP 1: INFO DASAR ─────────────────────────────────────
    st.markdown('<div class="section-label">Langkah 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Informasi Dasar</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        golongan = st.selectbox("🏠 Golongan Listrik PLN", list(TARIF_PLN.keys()), index=1)
    with c2:
        luas_rumah = st.number_input(
            "📐 Luas Rumah / Ruangan (m²)",
            min_value=5, max_value=500, value=36, step=1,
            help="Total luas ruangan yang menggunakan listrik. Dipakai untuk menghitung Indeks Konsumsi Energi (IKE)."
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── STEP 2: PERALATAN ──────────────────────────────────────
    st.markdown('<div class="section-label">Langkah 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pilih Peralatan Listrik yang Anda Miliki</div>', unsafe_allow_html=True)

    selected_items = []
    cols = st.columns(2)

    for i, ap in enumerate(APPLIANCES):
        with cols[i % 2]:
            checked = st.checkbox(
                f"{ap['icon']} **{ap['name']}**  \n`{ap['watt']} Watt`",
                key=f"chk_{i}"
            )
            if checked:
                cc1, cc2, cc3 = st.columns(3)
                with cc1: qty   = st.number_input("Jumlah unit", 1, 20, 1,          key=f"qty_{i}")
                with cc2: hours = st.number_input("Jam/hari",    0.0, 24.0, 4.0, 0.5, key=f"hrs_{i}")
                with cc3: watt  = st.number_input("Watt/unit",   1, 10000, ap["watt"], key=f"watt_{i}")
                selected_items.append({
                    **ap, "qty": qty, "hours": hours, "watt": watt,
                })
            st.write("")

    # ── STEP 3: CUSTOM ─────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Langkah 3 · Opsional</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Peralatan Lainnya</div>', unsafe_allow_html=True)

    n_custom = st.number_input("Berapa peralatan tambahan?", 0, 15, 0, 1)
    for i in range(int(n_custom)):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1: cname  = st.text_input(f"Nama #{i+1}", key=f"cn_{i}", placeholder="Contoh: Kipas Angin")
        with c2: cwatt  = st.number_input(f"Watt #{i+1}", 1, 10000, 60, key=f"cw_{i}")
        with c3: cqty   = st.number_input(f"Unit #{i+1}", 1, 20, 1, key=f"cq_{i}")
        with c4: chours = st.number_input(f"Jam/hari #{i+1}", 0.0, 24.0, 4.0, 0.5, key=f"ch_{i}")
        if cname:
            selected_items.append({
                "name": cname, "icon": "🔌", "watt": cwatt,
                "qty": cqty, "hours": chours,
                "dt_name": None, "luas_ref": luas_rumah // 4
            })

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── ANALYZE ────────────────────────────────────────────────
    if st.button("🔍  Analisis Konsumsi Energi Saya"):
        if not selected_items:
            st.warning("⚠️ Pilih minimal satu peralatan listrik.")
            st.stop()

        with st.spinner("🤖 Multi-AI sedang menganalisis..."):
            # 1. Hitung konsumsi
            kwh_total, breakdown, jam_siang, jam_malam = compute_konsumsi(selected_items)

            # 2. AI Profiler: Decision Tree per peralatan
            dt_results, profil, badge, desc, boros_cnt, hemat_cnt = dt_classify_appliances(
                breakdown, luas_rumah, dt_model
            )

            # 3. Anomaly Detection: Isolation Forest
            is_anomali, anomali_pct, anomali_status = detect_anomaly(
                kwh_total, breakdown, jam_siang, jam_malam,
                anomaly_model, anomaly_scaler, anomaly_features
            )

            # 4. Finishing System
            tarif = TARIF_PLN[golongan]
            biaya = kwh_total * tarif
            emisi = kwh_total * EMISSION_FACTOR
            ike   = kwh_total / max(luas_rumah, 1)

        # ── RESULTS ────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Hasil Analisis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Laporan Konsumsi Energi Rumah Tangga</div>', unsafe_allow_html=True)

        # Anomaly alert (jika terdeteksi)
        if is_anomali:
            st.markdown(f"""
            <div class="card-warn">
                <div class="section-label">⚠️ AI Anomaly Detection · Isolation Forest</div>
                <span class="profile-badge badge-anomali">🔮 Anomali Terdeteksi ({anomali_pct:.0f}%)</span>
                <p style="color:#c084fc; font-size:0.9rem; margin-top:0.5rem; margin-bottom:0;">
                    Pola konsumsi Anda tidak lazim dibanding pola rumah tangga pada umumnya.
                    Kemungkinan ada peralatan rusak, arus bocor, atau penggunaan yang tidak wajar.
                    Disarankan pengecekan instalasi listrik.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Profil DT
        st.markdown(f"""
        <div class="card-accent">
            <div class="section-label">AI Profiler · Decision Tree Classifier</div>
            <span class="profile-badge {badge}">📊 {profil}</span>
            <p style="color:#8892a4; font-size:0.9rem; margin-top:0.5rem; margin-bottom:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Konsumsi</div>
                <div class="metric-value">{kwh_total:.1f}</div>
                <div class="metric-unit">kWh / bulan</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Tagihan</div>
                <div class="metric-value">Rp{biaya/1000:.0f}K</div>
                <div class="metric-unit">per bulan</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Emisi CO₂</div>
                <div class="metric-value">{emisi:.1f}</div>
                <div class="metric-unit">kg / bulan</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            ike_color = "#ef4444" if ike > IKE_THRESHOLD_BOROS else "#10b981"
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">IKE</div>
                <div class="metric-value" style="color:{ike_color}">{ike:.2f}</div>
                <div class="metric-unit">kWh/m²/bulan</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Tarif PLN</div>
                <div class="metric-value">Rp{tarif:,.0f}</div>
                <div class="metric-unit">per kWh · Q2 2026</div>
            </div>""", unsafe_allow_html=True)

        # Hasil DT per peralatan
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Hasil Decision Tree per Peralatan</div>', unsafe_allow_html=True)

        sorted_dt = sorted(dt_results, key=lambda x: x["kwh"], reverse=True)
        max_kwh   = sorted_dt[0]["kwh"] if sorted_dt else 1

        for r in sorted_dt:
            pct      = r["kwh"] / kwh_total * 100 if kwh_total > 0 else 0
            bar_w    = r["kwh"] / max_kwh * 100
            is_boros = r["label"] == "Boros"
            bar_color = "linear-gradient(90deg,#ef4444,#f97316)" if is_boros else "linear-gradient(90deg,#10b981,#00d4ff)"
            label_color = "#ef4444" if is_boros else "#10b981"
            label_bg    = "rgba(239,68,68,0.12)" if is_boros else "rgba(16,185,129,0.12)"
            qty_txt  = f" ×{r['qty']} unit" if r.get("qty", 1) > 1 else ""

            st.markdown(f"""
            <div class="card" style="padding:1rem 1.5rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <span style="color:#e8edf5; font-weight:500;">{r['icon']} {r['name']}{qty_txt}</span>
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <span style="background:{label_bg};color:{label_color};padding:0.2rem 0.6rem;border-radius:999px;font-size:0.75rem;font-weight:700;">
                            {'🔴 Boros' if is_boros else '🟢 Tidak Boros'}
                        </span>
                        <span style="color:#00d4ff; font-weight:600;">{r['kwh']:.1f} kWh · {pct:.0f}%</span>
                    </div>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:999px; height:4px; width:100%;">
                    <div style="background:{bar_color}; height:4px; border-radius:999px; width:{bar_w}%;"></div>
                </div>
                <div style="color:#6b7a8d; font-size:0.75rem; margin-top:0.4rem;">
                    {r['watt']}W × {r.get('qty',1)} unit × {r['hours']} jam/hari × 30 hari
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Ringkasan boros vs hemat
        total_alat = len(dt_results)
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-bottom:1rem;">
            <div style="flex:1;background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#ef4444;">{boros_cnt}</div>
                <div style="color:#6b7a8d;font-size:0.8rem;">Peralatan Boros</div>
            </div>
            <div style="flex:1;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#10b981;">{hemat_cnt}</div>
                <div style="color:#6b7a8d;font-size:0.8rem;">Peralatan Hemat</div>
            </div>
            <div style="flex:1;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#e8edf5;">{total_alat}</div>
                <div style="color:#6b7a8d;font-size:0.8rem;">Total Peralatan</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gemini Rekomendasi
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">AI Rekomendasi · Gemini</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Rekomendasi Penghematan Energi</div>', unsafe_allow_html=True)

        if not gemini_key:
            st.markdown("""
            <div class="card" style="border-color:rgba(251,191,36,0.3);background:rgba(251,191,36,0.05);">
                <p style="color:#fbbf24;margin:0;font-weight:600;">⚠️ GEMINI_API_KEY belum diset di Streamlit Secrets.</p>
                <p style="color:#8892a4;font-size:0.85rem;margin:0.4rem 0 0;">
                    Tambahkan di <b>Settings → Secrets</b>:<br>
                    <code>GEMINI_API_KEY = "AIza..."</code><br>
                    Dapatkan gratis di <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#00d4ff;">aistudio.google.com/apikey</a>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("✨ Gemini sedang menulis rekomendasi..."):
                try:
                    rekom = get_gemini_recommendation(
                        profil, kwh_total, biaya, emisi,
                        breakdown, golongan, dt_results, is_anomali, gemini_key
                    )
                    st.markdown(f'<div class="rekom-box">{rekom}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Gemini error: {e}")

        st.markdown("""
        <div style="text-align:center;color:#1e2d42;font-size:0.72rem;margin-top:3rem;">
            Tarif PLN Q2 2026 · Emisi 0,87 kgCO₂/kWh (ESDM 2020) ·
            Decision Tree trained on synthetic dataset · Anomaly: Isolation Forest
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()