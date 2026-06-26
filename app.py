import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# REFERENSI REGULASI YANG DIGUNAKAN
# ============================================================
# [1] TARIF PLN
#     Penetapan Penyesuaian Tarif Tenaga Listrik (Tariff Adjustment)
#     Periode April–Juni 2026, PT PLN (Persero)
#
# [2] FAKTOR EMISI GRK
#     Faktor Emisi GRK Sistem Ketenagalistrikan Tahun 2019
#     Kementerian ESDM RI — Grid Jamali (DKI Jakarta)
#     Operating Margin (OM) = 0,80 kgCO₂/kWh
#
# [3] PBJT TENAGA LISTRIK JAKARTA
#     Perda DKI Jakarta No. 1 Tahun 2024
#     Rumah tangga: 2,4%
#     Sumber: https://dpp.jakarta.go.id/berita/sobat-pajak-ini-dia-
#             segala-hal-tentang-pbjt-tenaga-listrik
#
# [4] REKENING MINIMUM / BIAYA BEBAN
#     Rumus PLN: RM1 = 40 jam × daya (kVA) × tarif (Rp/kWh)
#     Berlaku hanya pelanggan pascabayar.
#
# [5] IKE RUMAH TANGGA
#     Pedoman Konservasi Energi Depdiknas RI
#     Satuan: kWh/m²/bulan, dibedakan ber-AC dan tidak ber-AC.
# ============================================================

# ── Import modul internal ─────────────────────────────────────────────────────
from fuzzy.ike_profiler      import profil_ike
from models.dsm_classifier   import DSMClassifier
from models.knn_recommender  import KNNRecommender
from optimizer.brute_force   import optimasi

# ============================================================
# KONSTANTA REGULASI
# ============================================================

TARIF_PER_GOLONGAN = {
    (0,      900):  1352.00,   # R-1/TR 900 VA RTM       [1]
    (901,   2200):  1444.70,   # R-1/TR 1.300 & 2.200 VA [1]
    (2201,  5500):  1699.53,   # R-2/TR 3.500–5.500 VA   [1]
    (5501, 999999): 1699.53,   # R-3/TR 6.600 VA ke atas [1]
}
FAKTOR_EMISI_JAMALI_OM  = 0.80   # kgCO₂/kWh [2]
PBJT_RUMAH_TANGGA       = 0.024  # 2,4% [3]
BATAS_TOLERANSI_ANOMALI = 0.15   # 15%

KATEGORI_OPTIONS = [
    "Pendingin",
    "Pemanas",
    "Laundry",
    "Memasak",
    "Hiburan/Elektronik",
    "Pencahayaan",
    "Pompa/Motor",
    "Lainnya",
]


def get_tarif(daya_va: int) -> float:
    for (lo, hi), tarif in TARIF_PER_GOLONGAN.items():
        if lo <= daya_va <= hi:
            return tarif
    return 1699.53


def hitung_biaya_beban(daya_va: int) -> float:
    """RM1 = 40 jam × daya (kVA) × tarif [4]"""
    return 40 * (daya_va / 1000.0) * get_tarif(daya_va)


def hitung_emisi(total_kwh: float) -> dict:
    """Hitung emisi CO₂ dari konsumsi listrik [2]"""
    emisi_bulan = round(total_kwh * FAKTOR_EMISI_JAMALI_OM, 3)
    return {
        "emisi_kg_bulan": emisi_bulan,
        "emisi_kg_tahun": round(emisi_bulan * 12, 2),
        "faktor_emisi"  : FAKTOR_EMISI_JAMALI_OM,
        "referensi"     : "Faktor Emisi GRK Grid Jamali OM, ESDM 2019",
    }


# ============================================================
# DATA INGESTION & VALIDATOR AGENT
# Deteksi anomali: if-else sederhana
# Anomali = selisih estimasi vs tagihan asli > 15%
# ============================================================

class DataIngestionValidatorAgent:
    def __init__(self, daya_va: int = 1300, is_prabayar: bool = False):
        self.daya_va     = daya_va
        self.is_prabayar = is_prabayar
        self.TARIF_KWH   = get_tarif(daya_va)
        self.PBJT        = PBJT_RUMAH_TANGGA
        self.BIAYA_BEBAN = 0.0 if is_prabayar else hitung_biaya_beban(daya_va)

    def proses_data(self, luas_rumah, penghuni, tagihan_asli, daftar_alat):
        total_kwh  = 0.0
        alat_valid = []

        for alat in daftar_alat:
            watt = alat['tegangan'] * alat['arus']
            kwh  = (watt * alat['jam'] * 30) / 1000
            total_kwh += kwh
            alat_valid.append({
                'nama'     : alat['nama'],
                'kategori' : alat['kategori'],
                'tegangan' : alat['tegangan'],
                'arus'     : alat['arus'],
                'watt'     : round(watt, 2),
                'jam'      : alat['jam'],
                'kwh_bulan': round(kwh, 3),
            })

        biaya_pemakaian  = total_kwh * self.TARIF_KWH
        biaya_pbjt       = biaya_pemakaian * self.PBJT
        estimasi_tagihan = biaya_pemakaian + biaya_pbjt + self.BIAYA_BEBAN
        ike              = total_kwh / max(1.0, float(luas_rumah))
        emisi_sebelum    = hitung_emisi(total_kwh)

        # ── Deteksi anomali (if-else) ─────────────────────────────────────────
        selisih_pct = (
            abs(tagihan_asli - estimasi_tagihan) / max(1.0, float(tagihan_asli))
        )
        is_anomali = selisih_pct > BATAS_TOLERANSI_ANOMALI

        return {
            "total_kwh"      : round(total_kwh, 3),
            "biaya_pemakaian": round(biaya_pemakaian, 0),
            "biaya_pbjt"     : round(biaya_pbjt, 0),
            "biaya_beban"    : round(self.BIAYA_BEBAN, 0),
            "estimasi_rp"    : round(estimasi_tagihan, 0),
            "ike"            : round(ike, 4),
            "emisi_sebelum"  : emisi_sebelum,
            "is_anomali"     : is_anomali,
            "selisih_pct"    : round(selisih_pct * 100, 1),
            "tarif_digunakan": self.TARIF_KWH,
            "golongan_daya"  : f"{self.daya_va} VA",
            "alat_valid"     : alat_valid,
        }


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Memuat model DSM Classifier...")
def load_dsm():
    return DSMClassifier()

@st.cache_resource(show_spinner="Memuat model KNN Recommender...")
def load_knn():
    return KNNRecommender()


# ============================================================
# GEN AI — GEMINI
# ============================================================

def generate_gemini_narasi(api_key       : str,
                            label_ike     : str,
                            payload       : dict,
                            hasil_dsm     : list,
                            knn_ctx       : dict,
                            hasil_opt     : dict,
                            intent_user   : list) -> str:
    """
    Menghasilkan narasi rekomendasi menggunakan Gemini.

    Konteks yang dikirim ke Gemini:
        - Status anomali tagihan
        - Profil IKE (zona efisiensi)
        - Ringkasan peralatan per label DSM
        - Hasil optimasi brute force (langkah + penghematan + emisi)
        - Target role model KNN
        - Fokus user (Biaya / Lingkungan)
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Susun ringkasan peralatan
    fleksibel     = [a for a in hasil_dsm if a['label_dsm'] == 'Fleksibel']
    tdk_fleksibel = [a for a in hasil_dsm if a['label_dsm'] == 'Tidak Fleksibel']

    str_fleksibel = ", ".join(
        f"{a['nama']} ({a['watt']:.0f}W, {a['jam']}j/hari)"
        for a in fleksibel
    ) or "tidak ada"

    str_tdk = ", ".join(a['nama'] for a in tdk_fleksibel) or "tidak ada"

    # Susun konteks optimasi
    if hasil_opt and hasil_opt.get('aktif'):
        opt_status = hasil_opt['status']
        if opt_status in ('efisien', 'cukup_efisien'):
            langkah_str = "\n".join(
                f"  - {l['nama']}: kurangi dari {l['jam_awal']} jam → "
                f"{l['jam_rekomendasi']} jam "
                f"(hemat Rp {l['hemat_rp']:,}/bulan, "
                f"kurangi {l['hemat_emisi_kg']} kgCO₂/bulan)"
                for l in hasil_opt['langkah']
            )
            konteks_opt = f"""
Hasil Optimasi Penggunaan (Brute Force IKE Optimizer):
  Status       : Berhasil mencapai zona {opt_status.replace('_', ' ').title()}
  IKE sebelum  : {hasil_opt['ike_awal']} → IKE setelah: {hasil_opt['ike_akhir']} kWh/m²/bulan
  Hemat biaya  : Rp {hasil_opt['hemat_rp']:,}/bulan ({hasil_opt['persen_hemat_rp']}%)
  Kurang emisi : {hasil_opt['hemat_emisi_kg']} kgCO₂/bulan ({hasil_opt['persen_hemat_emisi']}%)
  Emisi sesudah: {hasil_opt['emisi_akhir']} kgCO₂/bulan
Langkah spesifik yang direkomendasikan:
{langkah_str}"""
        else:
            konteks_opt = f"""
Hasil Optimasi: Target IKE tidak tercapai meski semua peralatan fleksibel sudah dimaksimalkan.
  IKE terbaik yang dicapai: {hasil_opt['ike_akhir']} kWh/m²/bulan
  Hemat biaya  : Rp {hasil_opt['hemat_rp']:,}/bulan ({hasil_opt['persen_hemat_rp']}%)
  Kurang emisi : {hasil_opt['hemat_emisi_kg']} kgCO₂/bulan
  Saran        : Pertimbangkan mengganti peralatan dengan label SKEM bintang tinggi."""
    else:
        konteks_opt = "Optimasi tidak diperlukan — konsumsi sudah dalam zona efisien."

    # Susun konteks KNN
    konteks_knn = (
        f"Target rumah tangga serupa yang lebih efisien (KNN): "
        f"tagihan Rp {knn_ctx['target_tagihan_rp']:,}/bulan, "
        f"emisi {knn_ctx['target_emisi_kg']} kgCO₂/bulan, "
        f"IKE {knn_ctx['target_ike']} kWh/m²/bulan. "
        f"Potensi hemat biaya: {knn_ctx['persen_hemat_biaya']}%, "
        f"potensi kurang emisi: {knn_ctx['persen_hemat_emisi']}%."
    )

    anomali_str  = (
        f"TERDETEKSI — selisih {payload['selisih_pct']}% dari tagihan asli. "
        "Kemungkinan ada perangkat yang belum diinput atau kebocoran arus."
        if payload['is_anomali']
        else "Normal"
    )
    fokus_str    = " + ".join(intent_user) if intent_user else "Efisiensi Umum"
    emisi_sblm   = payload['emisi_sebelum']

    prompt = f"""
Kamu adalah EnergiCerdas AI — konsultan energi rumah tangga Jakarta yang ramah, \
suportif, dan berbasis data regulasi resmi Indonesia.

Teks ini muncul TEPAT DI BAWAH dasbor metrik angka di aplikasi web.

DATA ANALISIS:
- Anomali tagihan     : {anomali_str}
- Profil IKE          : {label_ike} (IKE {payload['ike']:.4f} kWh/m²/bulan)
- Emisi sekarang      : {emisi_sblm['emisi_kg_bulan']} kgCO₂/bulan \
({emisi_sblm['emisi_kg_tahun']} kgCO₂/tahun)
- Fokus user          : {fokus_str}
- Peralatan fleksibel : {str_fleksibel}
- Peralatan tetap     : {str_tdk}

{konteks_opt}

{konteks_knn}

ATURAN PENULISAN:
1. JANGAN gunakan kalimat pembuka seperti "Berikut adalah..." atau \
"Berdasarkan analisis...". Langsung merujuk ke dasbor di atas, \
contoh: "Melihat dasbor di atas, kondisi kelistrikan Anda..."
2. Jika ada langkah optimasi, jelaskan dengan antusias dan spesifik \
— sebut nama peralatannya, jam pengurangannya, dan dampak rupiahnya.
3. Akhiri dengan kalimat penyemangat yang hangat dan personal — \
buat user merasa bangga dan termotivasi untuk implementasi.
4. Jika konsumsi sudah efisien, berikan tips penghematan tingkat lanjut \
dan pujian yang tulus.
5. Hubungkan dampak penghematan energi dengan SDG 7 (energi bersih) \
dan SDG 13 (aksi iklim) secara natural, bukan sebagai daftar.
6. Gunakan bahasa Indonesia yang hangat, mudah dipahami, \
dan tidak terlalu teknis.
7. Panjang respons: 3–4 paragraf, tidak perlu bullet point.
"""
    response = model.generate_content(prompt)
    return response.text


# ============================================================
# KONFIGURASI HALAMAN & SESSION STATE
# ============================================================

st.set_page_config(
    page_title="EnergiCerdas AI",
    page_icon="⚡",
    layout="wide"
)

# Load model sekali
dsm_clf = load_dsm()
knn_rec = load_knn()

# ============================================================
# SIDEBAR — API KEY
# ============================================================

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API Key terhubung.")
    except KeyError:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Dapatkan di https://aistudio.google.com/app/apikey"
        )
        if not gemini_api_key:
            st.warning("⚠️ Masukkan API Key untuk mengaktifkan narasi AI.")

    st.divider()
    st.caption(
        "Model DSM: LightGBM\n"
        "Profil IKE: Fuzzy Mamdani\n"
        "Role Model: KNN\n"
        "Optimizer: Brute Force IKE\n"
        "Gen AI: Gemini 2.5 Flash"
    )

# ============================================================
# JUDUL
# ============================================================

st.title("⚡ EnergiCerdas AI")
st.markdown(
    "Sistem rekomendasi energi rumah tangga Jakarta "
    "berbasis standar regulasi resmi Indonesia."
)

# ============================================================
# INPUT 1 — PROFIL RUMAH TANGGA
# ============================================================

st.header("1. Profil Rumah Tangga")
c1, c2, c3 = st.columns(3)
with c1: luas_rumah   = st.number_input("Luas Bangunan (m²)", 10, 500, 45)
with c2: penghuni     = st.number_input("Jumlah Penghuni", 1, 20, 3)
with c3: tagihan_asli = st.number_input(
    "Tagihan Bulan Lalu (Rp)", 50_000, 10_000_000, 600_000
)

# ============================================================
# INPUT 2 — DAYA & METERAN
# ============================================================

st.header("2. Informasi Daya & Meteran")
c4, c5 = st.columns(2)
with c4:
    daya_va = st.selectbox(
        "Daya Tersambung PLN",
        options=[900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 13200],
        index=1,
        help="Tertera di meteran atau rekening listrik Anda"
    )
with c5:
    jenis_meteran = st.radio(
        "Jenis Meteran",
        ["Prabayar (Token)", "Pascabayar (Tagihan)"],
        horizontal=True,
    )
    is_prabayar = (jenis_meteran == "Prabayar (Token)")

tarif_aktif = get_tarif(daya_va)
bb_aktif    = 0 if is_prabayar else hitung_biaya_beban(daya_va)
st.info(
    f"Golongan {daya_va} VA · "
    f"Tarif Rp {tarif_aktif:,.2f}/kWh · "
    f"PBJT 2,4% (Perda DKI No.1/2024) · "
    f"Biaya beban: "
    f"{'Rp 0 (prabayar)' if is_prabayar else f'Rp {bb_aktif:,.0f}/bulan'}"
)

# ============================================================
# INPUT 3 — PREFERENSI OPTIMASI
# ============================================================

st.header("3. Preferensi Optimasi")
intent_user = st.multiselect(
    "Fokus rekomendasi:",
    ["Biaya", "Lingkungan"],
    default=["Biaya"],
)

# ============================================================
# INPUT 4 — INVENTARISASI PERALATAN LISTRIK
# Form dinamis dengan session state (mengikuti pola app.py lama)
# ============================================================

st.header("4. Inventarisasi Peralatan Listrik")
st.caption("P = V × I — daya dihitung otomatis dari tegangan dan arus.")

# Inisialisasi session state
if 'daftar_perangkat_saved' not in st.session_state:
    st.session_state.daftar_perangkat_saved = []

# Tampilkan peralatan yang sudah disimpan
if st.session_state.daftar_perangkat_saved:
    st.write("**Peralatan tersimpan:**")
    for i, alat in enumerate(st.session_state.daftar_perangkat_saved):
        col_info, col_del = st.columns([5, 1])
        with col_info:
            st.info(
                f"🔌 **{alat['nama']}** ({alat['kategori']}) | "
                f"{alat['tegangan']}V × {alat['arus']}A = "
                f"**{alat['tegangan']*alat['arus']:,.0f} W** | "
                f"{alat['jam']} jam/hari | "
                f"{alat['tegangan']*alat['arus']*alat['jam']*30/1000:.2f} kWh/bulan"
            )
        with col_del:
            if st.button("🗑️", key=f"del_{i}", help="Hapus peralatan ini"):
                st.session_state.daftar_perangkat_saved.pop(i)
                st.rerun()

    if st.button("🗑️ Hapus Semua Peralatan"):
        st.session_state.daftar_perangkat_saved = []
        st.rerun()
else:
    st.warning("Belum ada peralatan. Tambahkan melalui form di bawah.")

st.divider()

# Form tambah peralatan baru
with st.form("form_tambah_perangkat", clear_on_submit=True):
    st.subheader("➕ Tambah Peralatan Baru")
    col1, col2 = st.columns(2)
    with col1:
        input_nama     = st.text_input("Nama Alat", placeholder="cth: AC Kamar Tidur")
        input_kategori = st.selectbox("Kategori", KATEGORI_OPTIONS)
        input_jam      = st.number_input(
            "Durasi Nyala (Jam/Hari)",
            min_value=0.1, max_value=24.0, value=4.0, step=0.5
        )
    with col2:
        input_volt   = st.number_input(
            "Tegangan (V)", value=220.0, step=1.0,
            help="Standar PLN Indonesia: 220V"
        )
        input_ampere = st.number_input(
            "Arus (A)", min_value=0.01, max_value=100.0,
            value=1.5, step=0.1, format="%.2f"
        )
        watt_preview = input_volt * input_ampere
        st.metric(
            "Estimasi Daya",
            f"{watt_preview:.1f} W",
            f"{watt_preview * input_jam * 30 / 1000:.2f} kWh/bulan"
        )

    submit = st.form_submit_button("➕ Tambahkan ke Daftar", use_container_width=True)
    if submit and input_nama:
        st.session_state.daftar_perangkat_saved.append({
            "nama"    : input_nama,
            "kategori": input_kategori,
            "tegangan": input_volt,
            "arus"    : input_ampere,
            "jam"     : input_jam,
        })
        st.rerun()

# ============================================================
# TOMBOL ANALISIS
# ============================================================

st.divider()
if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):

    if not st.session_state.daftar_perangkat_saved:
        st.error("⚠️ Tambahkan minimal 1 peralatan terlebih dahulu!")
        st.stop()

    if not gemini_api_key:
        st.error("⚠️ Masukkan Gemini API Key di sidebar untuk mengaktifkan narasi AI.")
        st.stop()

    daftar_perangkat = st.session_state.daftar_perangkat_saved

    with st.spinner("⚙️ Lapis 1 — Kalkulasi tagihan & deteksi anomali..."):
        # ── LAPIS 1: Kalkulasi & Anomali ─────────────────────────────────────
        agent   = DataIngestionValidatorAgent(int(daya_va), is_prabayar)
        payload = agent.proses_data(
            luas_rumah, penghuni, tagihan_asli, daftar_perangkat
        )

    with st.spinner("🧠 Lapis 2 — Klasifikasi IKE & DSM..."):
        # ── LAPIS 2a: Fuzzy IKE ───────────────────────────────────────────────
        ada_ac    = any(a['kategori'] == 'Pendingin' for a in daftar_perangkat)
        label_ike = profil_ike(
            luas_rumah, penghuni, payload['total_kwh'], ada_ac
        )

        # ── LAPIS 2b: DSM Classifier ──────────────────────────────────────────
        hasil_dsm  = dsm_clf.prediksi_batch(daftar_perangkat)
        ringkasan  = dsm_clf.ringkasan_dsm(hasil_dsm)

        # ── LAPIS 2c: KNN Role Model ──────────────────────────────────────────
        role_model = knn_rec.cari_role_model(luas_rumah, penghuni, intent_user)
        knn_ctx    = knn_rec.format_untuk_narasi(role_model, payload)

    with st.spinner("⚡ Lapis 3 — Optimasi jadwal penggunaan (Brute Force IKE)..."):
        # ── LAPIS 3: Brute Force Optimizer ───────────────────────────────────
        hasil_opt = optimasi(
            ringkasan_dsm = ringkasan,
            luas_m2       = float(luas_rumah),
            ada_ac        = ada_ac,
            tarif_kwh     = agent.TARIF_KWH,
            pbjt          = agent.PBJT,
            biaya_beban   = agent.BIAYA_BEBAN,
            kwh_awal      = payload['total_kwh'],
            tagihan_awal  = payload['estimasi_rp'],
            emisi_awal    = payload['emisi_sebelum']['emisi_kg_bulan'],
        )

        # Hitung emisi sesudah optimasi
        if hasil_opt['aktif']:
            emisi_sesudah = hitung_emisi(hasil_opt['total_kwh_akhir'])
            hasil_opt['emisi_sesudah'] = emisi_sesudah

    with st.spinner("✍️ Membangkitkan narasi rekomendasi (Gemini)..."):
        # ── GEN AI: Gemini ────────────────────────────────────────────────────
        narasi = generate_gemini_narasi(
            api_key     = gemini_api_key,
            label_ike   = label_ike,
            payload     = payload,
            hasil_dsm   = hasil_dsm,
            knn_ctx     = knn_ctx,
            hasil_opt   = hasil_opt,
            intent_user = intent_user,
        )

    # ============================================================
    # RENDER HASIL
    # ============================================================
    st.divider()
    st.subheader("📊 Hasil Analisis")

    # ── Anomali ───────────────────────────────────────────────────────────────
    if payload['is_anomali']:
        st.error(
            f"⚠️ Anomali terdeteksi — selisih estimasi vs tagihan asli "
            f"{payload['selisih_pct']}% (ambang batas: 15%). "
            "Kemungkinan ada perangkat yang belum diinput atau indikasi "
            "kebocoran arus."
        )
    else:
        st.success(f"✅ Tagihan wajar (selisih {payload['selisih_pct']}%).")

    # ── Metrik utama ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimasi Tagihan",  f"Rp {payload['estimasi_rp']:,.0f}")
    m2.metric("Profil IKE",        label_ike)
    m3.metric("Total Konsumsi",    f"{payload['total_kwh']} kWh/bln")
    m4.metric("Emisi CO₂",
              f"{payload['emisi_sebelum']['emisi_kg_bulan']} kg/bln")

    # ── Hasil optimasi (jika aktif) ───────────────────────────────────────────
    if hasil_opt['aktif'] and hasil_opt['langkah']:
        st.divider()
        st.subheader("⚡ Hasil Optimasi Penggunaan")

        zona_label = {
            'efisien'       : '🟢 Efisien',
            'cukup_efisien' : '🟡 Cukup Efisien',
            'tidak_tercapai': '🔴 Belum tercapai',
        }
        st.info(
            f"**Status:** {zona_label.get(hasil_opt['status'], hasil_opt['status'])} · "
            f"IKE {hasil_opt['ike_awal']} → {hasil_opt['ike_akhir']} kWh/m²/bulan"
        )

        o1, o2, o3 = st.columns(3)
        o1.metric(
            "Hemat Biaya/Bulan",
            f"Rp {hasil_opt['hemat_rp']:,}",
            f"{hasil_opt['persen_hemat_rp']}%"
        )
        o2.metric(
            "Kurang Emisi/Bulan",
            f"{hasil_opt['hemat_emisi_kg']} kgCO₂",
            f"-{hasil_opt['persen_hemat_emisi']}%"
        )
        o3.metric(
            "Emisi Setelah Optimasi",
            f"{hasil_opt.get('emisi_sesudah', {}).get('emisi_kg_bulan', '-')} kgCO₂/bln"
        )

        st.write("**Langkah rekomendasi per peralatan:**")
        for l in hasil_opt['langkah']:
            st.markdown(
                f"- 🔌 **{l['nama']}** — kurangi dari {l['jam_awal']} jam → "
                f"**{l['jam_rekomendasi']} jam/hari** "
                f"(hemat Rp {l['hemat_rp']:,}/bulan · "
                f"kurangi {l['hemat_emisi_kg']} kgCO₂/bulan)"
            )

    # ── DSM Classifier ────────────────────────────────────────────────────────
    with st.expander("🏷️ Klasifikasi DSM Peralatan"):
        for a in hasil_dsm:
            icon = "🟢" if a['label_dsm'] == 'Fleksibel' else "🔴"
            st.markdown(
                f"{icon} **{a['nama']}** — {a['label_dsm']} "
                f"({a['watt']:.0f}W · {a['jam']}j/hari · "
                f"{a['kwh_bulan']} kWh/bln)"
                + (" ⚠️ *fallback*" if a['metode'] == 'fallback' else "")
            )

    # ── Rincian tagihan ───────────────────────────────────────────────────────
    with st.expander("📄 Rincian Tagihan"):
        st.markdown(f"""
| Komponen | Nilai |
|---|---|
| Golongan daya | {payload['golongan_daya']} |
| Tarif | Rp {payload['tarif_digunakan']:,.2f}/kWh |
| Total pemakaian | {payload['total_kwh']} kWh |
| Biaya pemakaian | Rp {payload['biaya_pemakaian']:,.0f} |
| PBJT 2,4% (Perda DKI No.1/2024) | Rp {payload['biaya_pbjt']:,.0f} |
| Biaya beban/RM | Rp {payload['biaya_beban']:,.0f} |
| **Estimasi total** | **Rp {payload['estimasi_rp']:,.0f}** |
        """)

    # ── Rincian emisi ─────────────────────────────────────────────────────────
    with st.expander("🌿 Rincian Emisi CO₂"):
        e = payload['emisi_sebelum']
        st.markdown(f"""
| Komponen | Nilai |
|---|---|
| Faktor emisi | {e['faktor_emisi']} kgCO₂/kWh ({e['referensi']}) |
| Emisi per bulan | {e['emisi_kg_bulan']} kgCO₂ |
| Emisi per tahun | {e['emisi_kg_tahun']} kgCO₂ |
        """)
        if hasil_opt['aktif'] and 'emisi_sesudah' in hasil_opt:
            es = hasil_opt['emisi_sesudah']
            st.markdown("**Setelah optimasi:**")
            st.markdown(f"""
| Komponen | Nilai |
|---|---|
| Emisi sesudah | {es['emisi_kg_bulan']} kgCO₂/bulan |
| Pengurangan | {hasil_opt['hemat_emisi_kg']} kgCO₂/bulan |
| Pengurangan/tahun | {round(hasil_opt['hemat_emisi_kg']*12,2)} kgCO₂/tahun |
| Persentase reduksi | {hasil_opt['persen_hemat_emisi']}% |
            """)

    # ── Rincian peralatan ─────────────────────────────────────────────────────
    with st.expander("🔌 Rincian Peralatan"):
        st.dataframe(
            pd.DataFrame(payload['alat_valid']),
            use_container_width=True
        )

    # ── Narasi Gemini ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Rekomendasi EnergiCerdas AI")
    st.markdown(narasi)