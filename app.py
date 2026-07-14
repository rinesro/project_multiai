import streamlit as st
import pandas as pd
import joblib
import warnings
import sys as _sys
from pathlib import Path as _Path
from datetime import date

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
    hitung_kwh_dari_token, hitung_hari_berjalan,
    hitung_estimasi_kwh_periode, hitung_saldo_token_awal,
    hitung_token_terpakai_aktual,
    GOLONGAN_DAYA, PBJT_RUMAH_TANGGA, KATEGORI_ALAT,
)
from fuzzy.ike_profiler      import profil_ike
from models.dsm_classifier   import DSMClassifier
from optimizer.brute_force   import optimasi

# KATEGORI_OPTIONS = alias lokal dari core.kalkulasi.KATEGORI_ALAT (sumber
# kebenaran tunggal) — supaya tidak ada 3 salinan daftar kategori yang
# berisiko tidak sinkron (app.py, dsm_classifier.py, backend/schemas.py).
KATEGORI_OPTIONS = KATEGORI_ALAT


# ============================================================
# DATA INGESTION & VALIDATOR AGENT — LAPIS 1
# Kelas ini sekarang ada di services/ingestion.py — dipakai bersama
# oleh app.py (Streamlit) dan backend FastAPI, supaya logika
# kalkulasi/validasi tidak ditulis dua kali di dua tempat berbeda.
# ============================================================

from services.ingestion import DataIngestionValidatorAgent


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Memuat model DSM Classifier...")
def load_dsm():
    return DSMClassifier()


# ============================================================
# GEN AI — GEMINI
# ============================================================

# ============================================================
# HELPER FORMAT HASIL OPTIMASI & NARASI GEMINI
# Sekarang ada di services/narasi.py — dipakai bersama app.py
# (Streamlit) dan backend FastAPI.
# ============================================================

from services.narasi import (
    generate_gemini_narasi,
    bucket_rekomendasi,
    format_hemat_langkah as _format_hemat_langkah,
)


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
# INPUT 1 — DAYA & METERAN
# Ditaruh SEBELUM profil rumah tangga karena field riwayat
# pemakaian di Section 2 bercabang tergantung jenis meteran
# (prabayar butuh field berbeda dari pascabayar).
# ============================================================

st.header("1. Informasi Daya & Meteran")
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
# INPUT 2 — PROFIL RUMAH TANGGA & RIWAYAT PEMAKAIAN
# Field riwayat pemakaian BERCABANG tergantung jenis meteran:
#   - Pascabayar : tagihan bulan lalu (Rp), siklus tetap ~30 hari.
#   - Prabayar   : saldo token, karena siklus top-up TIDAK tetap
#                  30 hari — dipakai untuk deteksi kebocoran arus
#                  lewat selisih saldo, bukan lewat tagihan Rp.
# ============================================================

st.header("2. Profil Rumah Tangga & Riwayat Pemakaian")
c1, c2 = st.columns(2)
with c1: luas_rumah = st.number_input("Luas Bangunan (m²)", 10, 500, 45)
with c2: penghuni   = st.number_input("Jumlah Penghuni", 1, 20, 3)

if is_prabayar:
    st.caption(
        "Data token dipakai untuk deteksi anomali — membandingkan "
        "konsumsi aktual (dari selisih saldo token) dengan estimasi "
        "dari daftar peralatan di Section 4."
    )
    t1, t2 = st.columns(2)
    with t1:
        tanggal_pembelian = st.date_input(
            "Tanggal Pembelian Token Terakhir",
            value=date.today(),
            max_value=date.today(),
            help="Tanggal top-up yang dijadikan titik awal periode analisis"
        )
        sisa_sebelum_beli = st.number_input(
            "Sisa Token Sebelum Beli (kWh)",
            min_value=0.0, value=5.0, step=0.1,
            help="Sisa kWh di meteran TEPAT SEBELUM top-up terakhir"
        )
    with t2:
        nominal_dibeli = st.number_input(
            "Nominal Token Dibeli (Rp)",
            min_value=0, value=100_000, step=5_000,
            help="Sesuai struk/nominal token — BUKAN termasuk biaya "
                 "admin bank/e-wallet"
        )
        sisa_saat_ini = st.number_input(
            "Sisa Token Saat Ini (kWh)",
            min_value=0.0, value=20.0, step=0.1,
            help="Sisa kWh di meteran HARI INI, saat analisis dilakukan"
        )

    kwh_preview_token = hitung_kwh_dari_token(
        nominal_dibeli, tarif_aktif, PBJT_RUMAH_TANGGA
    )
    hari_preview = hitung_hari_berjalan(tanggal_pembelian)
    p1, p2 = st.columns(2)
    p1.metric("Estimasi kWh dari Nominal Token", f"{kwh_preview_token:.2f} kWh")
    p2.metric(
        "Hari Sejak Pembelian",
        f"{hari_preview} hari" if hari_preview >= 0 else "⚠️ Tanggal tidak valid"
    )

    token_context = {
        "tanggal_pembelian": tanggal_pembelian,
        "sisa_sebelum_beli": sisa_sebelum_beli,
        "nominal_dibeli"   : nominal_dibeli,
        "sisa_saat_ini"    : sisa_saat_ini,
    }
    tagihan_asli = None
else:
    tagihan_asli = st.number_input(
        "Tagihan Bulan Lalu (Rp)", 50_000, 10_000_000, 600_000
    )
    token_context = None

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

    with st.spinner("⚙️ Lapis 1 — Kalkulasi & deteksi anomali..."):
        # ── LAPIS 1: Kalkulasi & Anomali ─────────────────────────────────────
        agent   = DataIngestionValidatorAgent(int(daya_va), is_prabayar)
        payload = agent.proses_data(
            luas_rumah, penghuni, daftar_perangkat,
            tagihan_asli  = tagihan_asli,
            token_context = token_context,
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

    # ── Status Anomali ───────────────────────────────────────────────────────
    # 5 kemungkinan status (2 untuk pascabayar, 5 untuk prabayar) —
    # pesan_anomali sudah lengkap dari core/anomaly_detector.py, tidak
    # perlu disusun ulang di sini.
    _RENDER_ANOMALI = {
        "anomali"             : (st.error,   "⚠️ "),
        "normal"               : (st.success, "✅ "),
        "data_belum_cukup"     : (st.warning, "ℹ️ "),
        "data_tidak_konsisten" : (st.warning, "⚠️ "),
        "tanggal_tidak_valid"  : (st.warning, "⚠️ "),
    }
    _render_fn, _prefix = _RENDER_ANOMALI.get(
        payload['status_anomali'], (st.info, "")
    )
    _render_fn(f"{_prefix}{payload['pesan_anomali']}")

    # ── KONTEN UTAMA: narasi + rekomendasi aksi ───────────────────────────────
    # Dipindah ke ATAS (sebelumnya di paling bawah) — ini yang dibaca duluan
    # oleh user awam, bukan pelengkap. Narasi Gemini murni strategi/motivasi
    # (tidak menyebut nama alat), disusul daftar aksi konkret yang dihitung
    # deterministik oleh kode lewat bucket_rekomendasi() — bukan LLM — supaya
    # nama alat & angka teknis selalu presisi.
    st.subheader("💡 Rekomendasi EnergiCerdas AI")
    st.markdown(narasi)

    rekomendasi = bucket_rekomendasi(hasil_dsm, hasil_opt)
    if rekomendasi['kurangi']:
        st.markdown("**⏱️ Bisa kurangi penggunaan perangkat berikut:**")
        for l in rekomendasi['kurangi']:
            st.markdown(
                f"- **{l['nama']}** — dari {l['jam_awal']} jam menjadi "
                f"{l['jam_rekomendasi']} jam/hari (hemat "
                f"{_format_hemat_langkah(l, payload['is_prabayar'])})"
            )
    if rekomendasi['ganti']:
        st.markdown("**🔄 Ganti penggunaan perangkat berikut dengan yang lebih hemat:**")
        for a in rekomendasi['ganti']:
            st.markdown(
                f"- **{a['nama']}** "
                f"`(tegangan = {a['tegangan']}V, arus listrik = {a['arus']}A, "
                f"daya = {a['watt']}W)`"
            )

    st.divider()

    # ── DETAIL PENDUKUNG: angka & rincian teknis ──────────────────────────────
    st.caption("Detail teknis di bawah ini bersifat opsional — pendukung angka di atas.")

    # ── Metrik utama ──────────────────────────────────────────────────────────
    label_biaya = (
        "Estimasi Nilai Konsumsi" if payload['is_prabayar'] else "Estimasi Tagihan"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label_biaya,          f"Rp {payload['estimasi_rp']:,.0f}")
    m2.metric("Tingkat Efisiensi", label_ike)
    m3.metric("Total Konsumsi",    f"{payload['total_kwh']} kWh/bln")
    m4.metric("Emisi CO₂",
              f"{payload['emisi_sebelum']['emisi_kg_bulan']} kg/bln")

    # ── Hasil optimasi (jika aktif) — angka ringkasan saja, daftar per-alat
    # sudah ditampilkan di bagian rekomendasi atas ────────────────────────────
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
            f"Efisiensi naik dari skor {hasil_opt['ike_awal']} ke {hasil_opt['ike_akhir']}"
        )

        o1, o2, o3 = st.columns(3)
        if payload['is_prabayar']:
            o1.metric(
                "Token Dihemat/Bulan",
                f"{hasil_opt['hemat_kwh']} kWh",
                f"≈ Rp {hasil_opt['hemat_rp']:,}"
            )
        else:
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
            f"{hasil_opt['emisi_akhir']} kgCO₂/bln"
        )

    # ── DSM Classifier ────────────────────────────────────────────────────────
    with st.expander("🔧 Klasifikasi Peralatan (detail teknis, opsional)"):
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

    # ── Rincian token (khusus prabayar) ─────────────────────────────────────
    if payload['is_prabayar'] and 'token_context' in payload:
        tc = payload['token_context']
        with st.expander("🔋 Rincian Token"):
            st.markdown(f"""
| Komponen | Nilai |
|---|---|
| Tanggal pembelian | {tc['tanggal_pembelian'].strftime('%d %B %Y')} |
| Hari berjalan | {tc['hari_berjalan']} hari |
| Sisa token sebelum beli | {tc['sisa_sebelum_beli']} kWh |
| Nominal token dibeli | Rp {tc['nominal_dibeli']:,.0f} |
| kWh dari pembelian (PBJT 2,4% terpotong) | {tc['kwh_dari_pembelian']} kWh |
| Saldo awal periode | {tc['saldo_awal']} kWh |
| Sisa token saat ini | {tc['sisa_saat_ini']} kWh |
| **Token terpakai aktual** | **{tc['token_terpakai_aktual']} kWh** |
| Estimasi terpakai dari daftar peralatan | {tc['estimasi_terpakai_perangkat']} kWh |
            """)
            st.caption(
                "Kolom 'Token terpakai aktual' dihitung dari selisih saldo "
                "(bukti fisik) — dibandingkan dengan estimasi dari daftar "
                "peralatan untuk deteksi anomali di atas."
            )

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