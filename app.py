import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import warnings
import sys as _sys
from pathlib import Path as _Path

warnings.filterwarnings("ignore")

_sys.path.insert(0, str(_Path(__file__).parent))

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
# Semua konstanta regulasi & rumus kalkulasi dasar tinggal di
# core/kalkulasi.py sebagai satu sumber kebenaran — dipakai bersama oleh
# app.py (Lapis 1), models/dsm_classifier.py, dan optimizer/brute_force.py.
#
# CATATAN DESAIN: Komponen KNN Role Model Recommender yang sebelumnya ada
# di Lapis 2 SENGAJA DIHILANGKAN dari sistem. Alasannya: KNN butuh basis
# data rumah tangga riil (luas, penghuni, tagihan, emisi) sebagai bahan
# perbandingan "role model", namun setelah ditelusuri tidak ditemukan
# dataset rumah tangga Indonesia yang terbuka dan gratis dengan granularitas
# tersebut — data BPS (SUSENAS) yang paling mendekati bersifat berbayar
# (PP No.13/2024 tentang tarif diseminasi data mikro). Menggunakan data
# sintetis sebagai basis perbandingan "rumah tangga lain" dinilai kurang
# dapat dipertanggungjawabkan dibanding tetap fokus pada rekomendasi
# berbasis data milik pengguna sendiri (brute force di Lapis 3), sehingga
# komponen KNN dihapus dari arsitektur final.
from core.kalkulasi import (
    get_tarif, hitung_biaya_beban, hitung_watt, hitung_kwh_alat,
    hitung_tagihan, hitung_emisi, hitung_ike, hitung_kwh_per_org,
    deteksi_anomali, GOLONGAN_DAYA, PBJT_RUMAH_TANGGA,
)
from fuzzy.ike_profiler      import profil_ike
from models.dsm_classifier   import DSMClassifier
from optimizer.brute_force   import optimasi

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


# ============================================================
# DATA INGESTION & VALIDATOR AGENT — LAPIS 1
# Deteksi anomali: if-else sederhana
# Anomali = selisih estimasi vs tagihan asli > 15%
#
# Semua kalkulasi yang BISA dihitung dari data yang tersedia di sini
# (kWh per alat, total kWh, IKE, kWh per penghuni, tagihan, emisi)
# dihitung SEKALI di kelas ini. Layer 2 (fuzzy IKE, DSM classifier)
# dan Layer 3 (brute force) menerima hasilnya lewat 'payload' dan
# 'alat_valid', tidak menghitung ulang dari nol.
# ============================================================

class DataIngestionValidatorAgent:
    def __init__(self, daya_va: int = 1300, is_prabayar: bool = False):
        self.daya_va     = daya_va
        self.is_prabayar = is_prabayar
        self.TARIF_KWH   = get_tarif(daya_va)
        self.PBJT        = PBJT_RUMAH_TANGGA  # alias, dipakai pipeline & optimasi()
        self.BIAYA_BEBAN = hitung_biaya_beban(daya_va, is_prabayar)

    def proses_data(self, luas_rumah, penghuni, tagihan_asli, daftar_alat):
        """
        Parameters:
            daftar_alat : list of dict, tiap dict berisi
                          nama, kategori, tegangan, arus, jam, jumlah
                          ('jumlah' opsional, default 1 kalau tidak ada)
        """
        total_kwh  = 0.0
        alat_valid = []

        for alat in daftar_alat:
            jumlah = alat.get('jumlah', 1)
            watt   = hitung_watt(alat['tegangan'], alat['arus'])
            kwh    = hitung_kwh_alat(watt, alat['jam'], jumlah)
            total_kwh += kwh
            alat_valid.append({
                'nama'     : alat['nama'],
                'kategori' : alat['kategori'],
                'tegangan' : alat['tegangan'],
                'arus'     : alat['arus'],
                'watt'     : watt,
                'jam'      : alat['jam'],
                'jumlah'   : jumlah,
                'kwh_bulan': kwh,
            })

        total_kwh = round(total_kwh, 3)

        rincian_tagihan  = hitung_tagihan(
            total_kwh, self.TARIF_KWH, PBJT_RUMAH_TANGGA, self.BIAYA_BEBAN
        )
        ike           = hitung_ike(total_kwh, luas_rumah)
        kwh_per_org   = hitung_kwh_per_org(total_kwh, penghuni)
        emisi_sebelum = hitung_emisi(total_kwh)
        anomali       = deteksi_anomali(tagihan_asli, rincian_tagihan['total'])

        return {
            "total_kwh"      : total_kwh,
            "biaya_pemakaian": rincian_tagihan['biaya_pemakaian'],
            "biaya_pbjt"     : rincian_tagihan['biaya_pbjt'],
            "biaya_beban"    : rincian_tagihan['biaya_beban'],
            "estimasi_rp"    : rincian_tagihan['total'],
            "ike"            : ike,
            "kwh_per_org"    : kwh_per_org,
            "emisi_sebelum"  : emisi_sebelum,
            "is_anomali"     : anomali['is_anomali'],
            "selisih_pct"    : anomali['selisih_pct'],
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


# ============================================================
# GEN AI — GEMINI
# ============================================================

def generate_gemini_narasi(api_key       : str,
                            label_ike     : str,
                            payload       : dict,
                            hasil_dsm     : list,
                            hasil_opt     : dict,
                            intent_user   : list) -> str:
    """
    Menghasilkan narasi rekomendasi menggunakan Gemini.

    Konteks yang dikirim ke Gemini:
        - Status anomali tagihan
        - Profil IKE (zona efisiensi)
        - Ringkasan peralatan per label DSM
        - Hasil optimasi brute force (langkah + penghematan + emisi)
        - Fokus user (Biaya / Lingkungan) — memengaruhi penekanan narasi

    CATATAN: Komponen pembanding "role model KNN" yang sebelumnya ada
    di sini sudah dihapus dari arsitektur (lihat catatan desain di
    bagian import). Narasi sekarang murni berbasis data milik
    pengguna sendiri dan hasil optimasi brute force.
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
        options=GOLONGAN_DAYA,
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
bb_aktif    = hitung_biaya_beban(daya_va, is_prabayar)
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
        jumlah = alat.get('jumlah', 1)
        watt   = hitung_watt(alat['tegangan'], alat['arus'])
        kwh    = hitung_kwh_alat(watt, alat['jam'], jumlah)
        teks_jumlah = f" × {jumlah} unit" if jumlah > 1 else ""

        col_info, col_del = st.columns([5, 1])
        with col_info:
            st.info(
                f"🔌 **{alat['nama']}** ({alat['kategori']}) | "
                f"{alat['tegangan']}V × {alat['arus']}A = "
                f"**{watt:,.0f} W/unit**{teks_jumlah} | "
                f"{alat['jam']} jam/hari | "
                f"{kwh:.2f} kWh/bulan (total)"
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
        input_nama     = st.text_input(
            "Nama Alat",
            placeholder="cth: AC Kamar Tidur, atau Mesin Cuci [Mode Olahraga]",
            help="Kalau satu alat fisik punya beberapa mode dengan arus "
                 "berbeda (misal mesin cuci mode bayi vs olahraga, atau "
                 "kipas kecepatan 1 vs 3), tambahkan sebagai entri "
                 "terpisah dengan nama yang jelas, contoh: "
                 "'Kipas Angin [Kecepatan 3]'."
        )
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
        sub_a, sub_b = st.columns([2, 1])
        with sub_a:
            input_ampere = st.number_input(
                "Arus per Unit (A)", min_value=0.01, max_value=100.0,
                value=1.5, step=0.1, format="%.2f",
                help="Arus SATU unit alat, bukan dikali jumlah"
            )
        with sub_b:
            input_jumlah = st.number_input(
                "Jumlah Unit", min_value=1, max_value=200,
                value=1, step=1,
                help="Berapa banyak unit identik, mis. 12 titik lampu"
            )
        watt_preview = hitung_watt(input_volt, input_ampere)
        kwh_preview  = hitung_kwh_alat(watt_preview, input_jam, input_jumlah)
        st.metric(
            "Estimasi Daya",
            f"{watt_preview:.1f} W/unit" + (
                f" × {input_jumlah}" if input_jumlah > 1 else ""
            ),
            f"{kwh_preview:.2f} kWh/bulan (total)"
        )

    submit = st.form_submit_button("➕ Tambahkan ke Daftar", use_container_width=True)
    if submit and input_nama:
        st.session_state.daftar_perangkat_saved.append({
            "nama"    : input_nama,
            "kategori": input_kategori,
            "tegangan": input_volt,
            "arus"    : input_ampere,
            "jam"     : input_jam,
            "jumlah"  : input_jumlah,
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
        # ike & kwh_per_org sudah dihitung di Lapis 1 (payload) —
        # profil_ike() tinggal menerima, tidak menghitung ulang.
        ada_ac    = any(a['kategori'] == 'Pendingin' for a in daftar_perangkat)
        label_ike = profil_ike(
            payload['ike'], payload['kwh_per_org'], ada_ac
        )

        # ── LAPIS 2b: DSM Classifier ──────────────────────────────────────────
        # Pakai payload['alat_valid'] (hasil Lapis 1, sudah ada watt/kwh_bulan/
        # jumlah) — BUKAN daftar_perangkat mentah, supaya kWh tidak dihitung
        # ulang dan konsisten dengan angka yang ditampilkan ke user.
        hasil_dsm  = dsm_clf.prediksi_batch(payload['alat_valid'])
        ringkasan  = dsm_clf.ringkasan_dsm(hasil_dsm)

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