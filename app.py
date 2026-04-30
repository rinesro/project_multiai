import streamlit as st
import numpy as np
import pickle
import joblib
import warnings
import google.generativeai as genai
import os

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="EnergiCerdas AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    min-height: 100vh;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* Hero section */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-sub {
    color: #8892a4;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-bottom: 2.5rem;
}

/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 0.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8edf5;
    margin-bottom: 1.5rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.card-accent {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result metrics */
.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7a8d;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8edf5;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.75rem;
    color: #6b7a8d;
    margin-top: 0.2rem;
}

/* Profile badge */
.profile-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.badge-hemat { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.badge-mewah { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-wfh { background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }
.badge-malam { background: rgba(168,85,247,0.15); color: #a855f7; border: 1px solid rgba(168,85,247,0.3); }

/* Rekomendasi box */
.rekom-box {
    background: linear-gradient(135deg, rgba(123,47,247,0.1), rgba(0,212,255,0.05));
    border: 1px solid rgba(123,47,247,0.25);
    border-radius: 16px;
    padding: 1.8rem;
    color: #c8d3e0;
    font-size: 0.95rem;
    line-height: 1.8;
    white-space: pre-wrap;
}

/* Appliance cards */
.appliance-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #c8d3e0;
    margin-bottom: 0.3rem;
}
.appliance-watt {
    font-size: 0.75rem;
    color: #6b7a8d;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

/* Streamlit widget overrides */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8edf5 !important;
    border-radius: 10px !important;
}
.stCheckbox label { color: #c8d3e0 !important; font-size: 0.9rem; }
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(123,47,247,0.4) !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8edf5 !important;
    border-radius: 10px !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e8edf5 !important;
}
label { color: #8892a4 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models/ai_profile_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    profile_model = joblib.load("models/ai_profile_model.pkl")
    profile_scaler = joblib.load("models/ai_profile_scaler.pkl")
    estimator_features = joblib.load("models/ai_estimator_features.pkl")
    estimator_model = joblib.load("models/ai_estimator_model.pkl")
    return labels, profile_model, profile_scaler, estimator_features, estimator_model

# ─── CONSTANTS ─────────────────────────────────────────────────
TARIF_PLN = {
    "900 VA RTM": 1352.00,
    "1.300 – 2.200 VA": 1444.70,
    "≥ 3.500 VA": 1699.53,
}
EMISSION_FACTOR = 0.87  # kgCO2/kWh (ESDM 2020)

# Watt standar & mapping sub_metering
APPLIANCES = {
    "Pompa Air": {"watt": 250, "sub": "other", "icon": "💧"},
    "Laptop / PC": {"watt": 65, "sub": "other", "icon": "💻"},
    "Lampu (semua ruangan)": {"watt": 60, "sub": "other", "icon": "💡"},
    "Microwave / Oven": {"watt": 800, "sub": "sub1", "icon": "🍳"},
    "Water Heater / Dispenser": {"watt": 350, "sub": "sub2", "icon": "🚿"},
    "Rice Cooker": {"watt": 400, "sub": "sub1", "icon": "🍚"},
}

PROFILE_BADGE = {
    "Keluarga Hemat & Stabil": ("badge-hemat", "✅"),
    "Rumah Mewah / Konsumsi Ekstrem": ("badge-mewah", "🔴"),
    "Aktif Siang Hari (WFH/Bisnis)": ("badge-wfh", "🏠"),
    "Pekerja Kantoran (Boros Malam)": ("badge-malam", "🌙"),
}

PROFILE_DESC = {
    "Keluarga Hemat & Stabil": "Pola konsumsi Anda sangat efisien dan konsisten. Distribusi penggunaan listrik seimbang sepanjang hari.",
    "Rumah Mewah / Konsumsi Ekstrem": "Konsumsi listrik Anda tergolong sangat tinggi. Ada potensi penghematan signifikan dengan optimasi peralatan.",
    "Aktif Siang Hari (WFH/Bisnis)": "Pola penggunaan terkonsentrasi di siang hari, khas pengguna WFH atau home-based business.",
    "Pekerja Kantoran (Boros Malam)": "Konsumsi listrik dominan di malam hari. Peralatan standby dan penggunaan malam berkontribusi besar.",
}

# ─── HELPER FUNCTIONS ──────────────────────────────────────────
def compute_sub_metering(selected_appliances, hours_map, custom_appliances):
    """Convert appliance usage to sub_metering analog values (watt-hours per minute avg)"""
    sub1 = sub2 = sub3 = other = 0.0
    kwh_total = 0.0
    breakdown = {}

    for name, info in APPLIANCES.items():
        if name in selected_appliances:
            hours = hours_map.get(name, 0)
            watt = info["watt"]
            kwh_month = watt * hours * 30 / 1000
            kwh_total += kwh_month
            breakdown[name] = {"watt": watt, "hours": hours, "kwh": kwh_month, "icon": info["icon"]}
            # sub_metering in watt-hour per minute average daily
            whpm = (watt * hours) / (24 * 60)
            if info["sub"] == "sub1":
                sub1 += whpm
            elif info["sub"] == "sub2":
                sub2 += whpm
            elif info["sub"] == "sub3":
                sub3 += whpm
            else:
                other += whpm

    for item in custom_appliances:
        if item["name"] and item["watt"] > 0 and item["hours"] > 0:
            kwh_month = item["watt"] * item["hours"] * 30 / 1000
            kwh_total += kwh_month
            breakdown[item["name"]] = {
                "watt": item["watt"], "hours": item["hours"],
                "kwh": kwh_month, "icon": "🔌"
            }
            whpm = (item["watt"] * item["hours"]) / (24 * 60)
            other += whpm

    # Sub3 proxy: high-wattage continuous = other (AC-like devices)
    sub3 = other * 0.4  # rough proxy for dominant load

    return sub1, sub2, sub3, kwh_total, breakdown

def get_ai_profile(sub1, sub2, sub3, profile_model, profile_scaler, labels):
    arr = np.array([[sub1, sub2, sub3]])
    scaled = profile_scaler.transform(arr)
    cluster = profile_model.predict(scaled)[0]
    return labels[cluster]

def get_rf_estimate(sub1, sub2, sub3, estimator_model):
    """Aggregate RF prediction across hours for daily watt estimate"""
    total_watt = 0
    for hour in range(24):
        for dow in range(7):
            pred = estimator_model.predict(
                np.array([[sub1, sub2, sub3, hour, dow]])
            )[0]
            total_watt += max(pred, 0)
    avg_daily_watt = total_watt / (24 * 7)
    kwh_rf = avg_daily_watt * 24 * 30 / 1000
    return kwh_rf

def get_gemini_recommendation(profile, kwh, biaya, emisi, breakdown, golongan, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    appliance_list = "\n".join([
        f"  - {name}: {info['watt']}W × {info['hours']} jam/hari = {info['kwh']:.1f} kWh/bulan"
        for name, info in breakdown.items()
    ])

    prompt = f"""Kamu adalah konsultan energi rumah tangga yang ahli dan ramah. Berikan rekomendasi penghematan energi yang spesifik, praktis, dan berbasis data berikut:

DATA RUMAH TANGGA:
- Golongan PLN: {golongan}
- Profil Konsumsi AI: {profile}
- Estimasi Konsumsi: {kwh:.1f} kWh/bulan
- Estimasi Tagihan: Rp{biaya:,.0f}/bulan
- Emisi CO₂: {emisi:.1f} kg/bulan

DETAIL PERALATAN LISTRIK:
{appliance_list}

Berikan respons dalam format berikut (gunakan emoji yang sesuai):

🔍 ANALISIS SITUASI
[Analisis singkat 2-3 kalimat tentang pola konsumsi berdasarkan profil dan data]

⚡ PERALATAN PALING BOROS
[Sebutkan 2-3 peralatan teratas yang paling boros dari daftar di atas]

💡 REKOMENDASI HEMAT ENERGI
1. [Rekomendasi spesifik 1 + estimasi penghematan %]
2. [Rekomendasi spesifik 2 + estimasi penghematan %]
3. [Rekomendasi spesifik 3 + estimasi penghematan %]

🌱 DAMPAK LINGKUNGAN
[Motivasi singkat tentang dampak penghematan terhadap emisi CO₂]

Gunakan Bahasa Indonesia yang ramah, informatif, dan tidak menggurui."""

    response = model.generate_content(prompt)
    return response.text


# ─── MAIN APP ──────────────────────────────────────────────────
def main():
    # Hero
    st.markdown('<div class="hero-title">⚡ EnergiCerdas AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Analisis konsumsi listrik rumah tangga berbasis Multi-AI · Didukung K-Means + Random Forest + Gemini</div>', unsafe_allow_html=True)

    # Load models
    try:
        labels, profile_model, profile_scaler, est_features, estimator_model = load_models()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}\nPastikan folder `models/` berisi semua file .pkl")
        st.stop()

    # ── FORM ──────────────────────────────────────────
    st.markdown('<div class="section-label">Langkah 1</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Informasi Dasar</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        golongan = st.selectbox("🏠 Golongan Listrik PLN", list(TARIF_PLN.keys()))
    with col2:
        gemini_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza...")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── PERALATAN ──────────────────────────────────────
    st.markdown('<div class="section-label">Langkah 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pilih Peralatan Listrik yang Anda Miliki</div>', unsafe_allow_html=True)

    selected = []
    hours_map = {}

    cols = st.columns(2)
    for i, (name, info) in enumerate(APPLIANCES.items()):
        with cols[i % 2]:
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            checked = st.checkbox(f"{info['icon']} **{name}**  \n`{info['watt']} Watt`", key=f"chk_{name}")
            if checked:
                selected.append(name)
                hours = st.number_input(
                    f"Jam pemakaian/hari", min_value=0.0, max_value=24.0,
                    value=4.0, step=0.5, key=f"hrs_{name}"
                )
                hours_map[name] = hours
            st.markdown('</div>', unsafe_allow_html=True)

    # Custom appliances
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Tambahan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Peralatan Lainnya (Opsional)</div>', unsafe_allow_html=True)

    n_custom = st.number_input("Jumlah peralatan tambahan", min_value=0, max_value=10, value=0, step=1)
    custom_appliances = []
    for i in range(n_custom):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            cname = st.text_input(f"Nama peralatan #{i+1}", key=f"cname_{i}", placeholder="Contoh: Kipas Angin")
        with c2:
            cwatt = st.number_input(f"Watt #{i+1}", min_value=0, max_value=10000, value=100, key=f"cwatt_{i}")
        with c3:
            chours = st.number_input(f"Jam/hari #{i+1}", min_value=0.0, max_value=24.0, value=4.0, step=0.5, key=f"chrs_{i}")
        custom_appliances.append({"name": cname, "watt": cwatt, "hours": chours})

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── SUBMIT ─────────────────────────────────────────
    if st.button("🔍 Analisis Konsumsi Energi Saya"):
        if not selected and not any(c["name"] for c in custom_appliances):
            st.warning("⚠️ Pilih minimal satu peralatan listrik terlebih dahulu.")
            st.stop()

        with st.spinner("🤖 Multi-AI sedang menganalisis data Anda..."):
            # Step 1: Compute sub metering & kWh
            sub1, sub2, sub3, kwh_direct, breakdown = compute_sub_metering(
                selected, hours_map, custom_appliances
            )

            # Step 2: AI Profiler
            profile_label = get_ai_profile(sub1, sub2, sub3, profile_model, profile_scaler, labels)

            # Step 3: AI Estimator (RF)
            kwh_rf = get_rf_estimate(sub1, sub2, sub3, estimator_model)

            # Blend: 70% direct calculation + 30% RF estimate
            kwh_final = 0.7 * kwh_direct + 0.3 * kwh_rf

            # Step 4: Finishing system
            tarif = TARIF_PLN[golongan]
            biaya = kwh_final * tarif
            emisi = kwh_final * EMISSION_FACTOR

        # ── RESULTS ────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Hasil Analisis</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Laporan Konsumsi Energi Rumah Tangga Anda</div>', unsafe_allow_html=True)

        # Profile
        badge_cls, badge_icon = PROFILE_BADGE.get(profile_label, ("badge-hemat", "✅"))
        st.markdown(f"""
        <div class="card-accent">
            <div class="section-label">AI Profiler · K-Means Clustering</div>
            <span class="profile-badge {badge_cls}">{badge_icon} {profile_label}</span>
            <p style="color:#8892a4; font-size:0.9rem; margin-top:0.5rem; margin-bottom:0;">{PROFILE_DESC.get(profile_label, '')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Estimasi Konsumsi</div>
                <div class="metric-value">{kwh_final:.1f}</div>
                <div class="metric-unit">kWh / bulan</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Estimasi Tagihan</div>
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
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Tarif PLN</div>
                <div class="metric-value">Rp{tarif:,.0f}</div>
                <div class="metric-unit">per kWh · Q2 2026</div>
            </div>""", unsafe_allow_html=True)

        # Breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Detail Konsumsi per Peralatan</div>', unsafe_allow_html=True)

        sorted_bd = sorted(breakdown.items(), key=lambda x: x[1]["kwh"], reverse=True)
        max_kwh = sorted_bd[0][1]["kwh"] if sorted_bd else 1

        for name, info in sorted_bd:
            pct = info["kwh"] / kwh_final * 100 if kwh_final > 0 else 0
            bar_w = info["kwh"] / max_kwh * 100
            st.markdown(f"""
            <div class="card" style="padding:1rem 1.5rem; margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <span style="color:#e8edf5; font-weight:500;">{info['icon']} {name}</span>
                    <span style="color:#00d4ff; font-weight:600;">{info['kwh']:.1f} kWh &nbsp;·&nbsp; {pct:.0f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:999px; height:4px; width:100%;">
                    <div style="background:linear-gradient(90deg,#7b2ff7,#00d4ff); height:4px; border-radius:999px; width:{bar_w}%;"></div>
                </div>
                <div style="color:#6b7a8d; font-size:0.75rem; margin-top:0.4rem;">{info['watt']}W × {info['hours']} jam/hari × 30 hari</div>
            </div>
            """, unsafe_allow_html=True)

        # Gemini Recommendation
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">AI Rekomendasi · Gemini</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Rekomendasi Penghematan Energi</div>', unsafe_allow_html=True)

        if not gemini_key:
            st.markdown("""
            <div class="card" style="border-color:rgba(251,191,36,0.3); background:rgba(251,191,36,0.05);">
                <p style="color:#fbbf24; margin:0;">⚠️ Masukkan Gemini API Key di bagian atas untuk mendapatkan rekomendasi AI yang dipersonalisasi.</p>
                <p style="color:#8892a4; font-size:0.85rem; margin-top:0.5rem; margin-bottom:0;">Dapatkan API Key gratis di <a href="https://aistudio.google.com/apikey" target="_blank" style="color:#00d4ff;">aistudio.google.com</a></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("✨ Gemini AI sedang membuat rekomendasi..."):
                try:
                    rekom = get_gemini_recommendation(
                        profile_label, kwh_final, biaya, emisi,
                        breakdown, golongan, gemini_key
                    )
                    st.markdown(f'<div class="rekom-box">{rekom}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Gagal mendapatkan rekomendasi Gemini: {e}")

        # Footer note
        st.markdown("""
        <br>
        <div style="text-align:center; color:#3a4557; font-size:0.75rem; padding-top:1rem;">
            Tarif PLN berdasarkan Triwulan II 2026 · Faktor emisi 0,87 kgCO₂/kWh (ESDM 2020) · 
            Model K-Means + Random Forest trained on UCI & Smart Meters London dataset
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()