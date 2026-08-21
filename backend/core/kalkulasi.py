"""
core/kalkulasi.py
==================
Sumber kebenaran tunggal (single source of truth) untuk seluruh konstanta
regulasi dan rumus kalkulasi dasar yang dipakai di lebih dari satu layer.

Sebelum modul ini dibuat, konstanta dan rumus yang sama ditulis ulang
secara terpisah di app.py, models/dsm_classifier.py, models/knn_recommender.py,
optimizer/greedy_optimizer.py, dan train_knn.py — berisiko saling tidak sinkron
kalau salah satu diubah tanpa mengubah yang lain.

Prinsip pemakaian:
    - Semua kalkulasi yang BISA dihitung dari data yang sudah tersedia
      di Lapis 1 (app.py) dihitung SEKALI di sana.
    - Layer 2 dan Layer 3 TIDAK menghitung ulang dari nol — mereka
      menerima hasil Lapis 1 sebagai input, atau memanggil fungsi yang
      SAMA dari modul ini kalau memang butuh menghitung skenario baru
      yang belum pernah dihitung Lapis 1 (mis. simulasi greedy).

CATATAN PEMISAHAN — deteksi anomali:
    Fungsi deteksi_anomali() dan konstanta BATAS_TOLERANSI_ANOMALI
    SUDAH DIPINDAH ke core/anomaly_detector.py. Modul ini (kalkulasi.py)
    HANYA berisi rumus aritmatika/regulasi murni, tanpa keputusan bisnis
    "apa yang dianggap anomali" — supaya kalkulasi.py tetap gampang diuji
    secara terisolasi, dan supaya semua aturan anomali (termasuk kasus
    khusus prabayar: data belum cukup, data tidak konsisten, tanggal
    tidak valid) terkumpul di satu tempat yang jelas tanggung jawabnya.

Referensi regulasi:
    [1] Tarif PLN — PT PLN (Persero), stabil sejak 1 Januari 2022 hingga
        Triwulan III 2026 (Juli–September 2026), dikonfirmasi ulang
        untuk periode 1–7 Agustus 2026: "Cek Tarif Listrik PLN Terbaru
        Agustus 2026", RRI.co.id (dikutip dari laman resmi PLN),
        https://rri.co.id/ekonomi/bisnis/2620057/
        cek-tarif-listrik-pln-terbaru-agustus-2026 (diakses 2 Agustus 2026)
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO₂/kWh — ESDM 2019
    [3] PBJT 2,4% — Perda DKI Jakarta No.1/2024 (khusus rumah tangga
        DKI Jakarta — tarif PBJT-TL nasional berkisar 3%-10% tergantung
        daerah, contoh: ULP Pekanbaru berbeda dari DKI Jakarta)
    [4] Rumus dasar tagihan pascabayar & Bea Materai — infografis resmi
        PLN "Memahami Perhitungan Tagihan Listrik PLN Pascabayar"
        (berlaku sejak 1 Januari 2022); UU No.10/2020 tentang Bea
        Meterai (Rp10.000, ambang >Rp5.000.000)
    [5] Dasar pengenaan PBJT tenaga listrik prabayar dihitung dari
        nominal pembelian token (bukan hasil hitung-mundur dari kWh) —
        Badan Pendapatan Daerah Provinsi DKI Jakarta (2025),
        https://dpp.jakarta.go.id/berita/sobat-pajak-ini-dia-segala-
        hal-tentang-pbjt-tenaga-listrik

CATATAN PERBAIKAN TARIF 450/900 VA (per sumber [1], Agustus 2026):
    Status subsidi TIDAK BISA disimpulkan dari daya (VA) saja untuk
    golongan 450 & 900 VA — beda dari golongan ≥1.300 VA yang cuma
    punya SATU tarif tanpa distingsi subsidi. Rinciannya:
        450 VA             : Rp415/kWh   (cuma ada 1 tarif, selalu subsidi)
        900 VA bersubsidi   : Rp605/kWh
        900 VA RTM (non-subsidi) : Rp1.352/kWh
    Sebelumnya modul ini SELALU mengasumsikan 900 VA = RTM (non-subsidi,
    Rp1.352) tanpa opsi subsidi sama sekali, dan 450 VA tidak ada
    opsinya di dropdown — celah ini sekarang diperbaiki lewat parameter
    is_subsidi di get_tarif().

CATATAN PERUBAHAN RUMUS TAGIHAN (perombakan besar):
    Komponen "Biaya Beban"/Rekening Minimum (RM1 = 40 jam × daya ×
    tarif) yang SEBELUMNYA ada di modul ini SUDAH DIHAPUS TOTAL.
    Setelah verifikasi ulang terhadap infografis resmi PLN [4], rumus
    dasar tagihan pascabayar PLN Persero (yang berlaku untuk DKI
    Jakarta) TIDAK menyebutkan komponen Rekening Minimum sama sekali —
    rumus lama ternyata bersumber dari skema PLN Batam (badan usaha
    terpisah dari PLN Persero, tarif/skema sendiri), bukan skema
    nasional. Sebagai gantinya, ditambahkan Biaya Materai sesuai rumus
    resmi PLN [4] dan UU Bea Meterai.
"""

from datetime import date
from typing import Optional

# ── Konstanta regulasi ────────────────────────────────────────────────────────

# 450 & 900 VA: status subsidi mengubah tarif drastis, TIDAK bisa
# disimpulkan dari VA saja. 450 VA cuma ada 1 tarif (selalu subsidi). [1]
TARIF_DAYA_RENDAH = {
    450: 415.00,
    900: {"subsidi": 605.00, "non_subsidi": 1352.00},
}

# ≥1.300 VA: cuma SATU tarif per golongan, tidak ada distingsi subsidi. [1]
TARIF_PER_GOLONGAN = {
    (901,   2200):  1444.70,   # R-1/TR 1.300 & 2.200 VA [1]
    (2201,  5500):  1699.53,   # R-2/TR 3.500–5.500 VA   [1]
    (5501, 999999): 1699.53,   # R-3/TR 6.600 VA ke atas [1]
}

FAKTOR_EMISI_JAMALI_OM  = 0.80    # kgCO₂/kWh — Grid Jamali OM [2]
PBJT_RUMAH_TANGGA       = 0.024   # 2,4% [3]

# Golongan daya standar PLN (dipakai UI & generator data).
# Dipangkas sampai 6.600 VA -- batas tertinggi yang eksplisit disebut
# sumber [1] untuk konteks rumah tangga ("6.600 VA ke atas"). VA lebih
# tinggi (7.700/10.600/13.200) SENGAJA tidak dimasukkan: jarang dipakai
# rumah tangga murni (lebih umum bisnis/gedung campuran), di luar
# cakupan "rumah tangga DKI Jakarta" yang jadi fokus penelitian ini.
GOLONGAN_DAYA = [450, 900, 1300, 2200, 3500, 4400, 5500, 6600]

# 8 kategori peralatan standar — sebelumnya diduplikasi terpisah di
# app.py (KATEGORI_OPTIONS) dan models/dsm_classifier.py (KATEGORI_VALID).
# Disatukan di sini supaya kalau kategori baru ditambah/dihapus, cukup
# ubah satu tempat.
KATEGORI_ALAT = [
    "Pendingin",
    "Pemanas",
    "Laundry",
    "Memasak",
    "Hiburan/Elektronik",
    "Pencahayaan",
    "Pompa/Motor",
    "Lainnya",
]


# ── Fungsi dasar tarif ─────────────────────────────────────────────────────────

def get_tarif(daya_va: int, is_subsidi: bool = False) -> float:
    """
    Tarif Rp/kWh berdasarkan daya tersambung (VA) dan status subsidi. [1]

    Status subsidi HANYA benar-benar relevan untuk 900 VA (dua tarif
    berbeda tergantung status). Untuk 450 VA maupun golongan ≥1.300 VA,
    parameter is_subsidi diabaikan sepenuhnya — 450 VA cuma punya satu
    tarif (selalu subsidi, tidak ada opsi lain), dan ≥1.300 VA juga
    cuma punya satu tarif per golongan tanpa distingsi subsidi.
    """
    if daya_va <= 450:
        return TARIF_DAYA_RENDAH[450]
    if daya_va <= 900:
        return (
            TARIF_DAYA_RENDAH[900]["subsidi"] if is_subsidi
            else TARIF_DAYA_RENDAH[900]["non_subsidi"]
        )
    for (lo, hi), tarif in TARIF_PER_GOLONGAN.items():
        if lo <= daya_va <= hi:
            return tarif
    return 1699.53


# ── Fungsi kalkulasi daya & energi per peralatan ──────────────────────────────

def hitung_watt(tegangan: float, arus: float) -> float:
    """Daya per-unit peralatan (W) = V × I."""
    return round(tegangan * arus, 2)


def hitung_kwh_alat(watt: float, jam: float, jumlah: int = 1) -> float:
    """
    Energi bulanan satu entri peralatan (kWh/bulan).

    kWh = watt × jam × jumlah × 30 hari / 1000

    Parameter jumlah mengalikan konsumsi TOTAL (bukan fitur yang dikirim
    ke model klasifikasi DSM — model tetap menerima arus per-unit supaya
    prediksinya tidak keluar dari rentang data latih).
    """
    return round(watt * jam * jumlah * 30 / 1000, 4)


# ── Pengingat kapasitas watt vs daya tersambung (VA) ──────────────────────────

# Rasio VA -> Watt aman = faktor daya (PF) rumah tangga = 0,8. Sumber:
# infografis resmi PLN/Danantara Indonesia "Pahami Perbedaan VA & Watt"
# (Watt = VA x PF, PF rumah tangga contoh 0,8). Bukan angka rekayasa
# peneliti -- turunan langsung dari definisi VA sebagai daya semu yang
# disediakan PLN, dikonversi ke watt nyata pakai rasio resmi tersebut.
RASIO_WATT_AMAN = 0.8


def hitung_batas_watt_aman(daya_va: int) -> float:
    """
    Batas watt aman yang bisa disalurkan MCB, dari daya tersambung (VA).

    Watt aman = VA x PF (faktor daya rumah tangga = 0,8)
    Sumber: infografis resmi PLN/Danantara Indonesia, "Pahami Perbedaan
    VA & Watt" (Watt = VA x Faktor Daya, PF rumah tangga = 0,8).

    CATATAN PENTING: ini SEKADAR PENGINGAT edukatif, BUKAN deteksi
    anomali -- sistem TIDAK terhubung real-time ke MCB, dan TIDAK tahu
    alat mana yang benar-benar nyala bersamaan (form cuma menyimpan
    durasi jam/hari, bukan jadwal jam-ke-jam). Angka ini dihitung dari
    skenario ekstrem "kalau SEMUA alat yang didaftarkan nyala sekaligus"
    -- yang pada praktiknya jarang benar-benar terjadi. Karena itu,
    hasil pengecekan ini SENGAJA tidak digabung ke status_anomali (lihat
    action_analist/anomaly_evaluator.py) -- disajikan terpisah sebagai
    info santai, bukan peringatan yang menuduh ada yang salah.
    """
    return round(daya_va * RASIO_WATT_AMAN, 2)


def cek_kapasitas_watt(daftar_alat: list, daya_va: int) -> dict:
    """
    Jumlahkan watt SEMUA peralatan (skenario "kalau nyala sekaligus"),
    bandingkan dengan batas aman MCB.

    Parameters:
        daftar_alat : list of dict, tiap dict punya key 'watt' dan
                      'jumlah' (dari payload alat_valid Lapis 1)
        daya_va     : daya tersambung PLN (VA)

    Returns dict:
        total_watt      : jumlah watt semua alat kalau nyala bersamaan
        batas_watt_aman  : VA x 0,8
        melebihi         : bool, True kalau total_watt > batas_watt_aman
    """
    total_watt = round(
        sum(a['watt'] * a.get('jumlah', 1) for a in daftar_alat), 2
    )
    batas = hitung_batas_watt_aman(daya_va)
    return {
        "total_watt": total_watt,
        "batas_watt_aman": batas,
        "melebihi": total_watt > batas,
    }


# ── Batas pembelian token bulanan (khusus prabayar) ───────────────────────────

# Rasio VA -> kWh maksimal/bulan = 720 jam nyala / 1000 = 0,72. Sumber:
# PLN, dikutip CNN Indonesia, "Apa Sisa Token Listrik Diskon Bisa
# Hangus? Ini Kata PLN" (6 Jan 2025): "Diskon diberikan maksimum untuk
# pemakaian listrik selama 720 jam nyala ... Jika melakukan pembelian
# melebihi 720 jam nyala, maka pembelian token (kWh) akan tertolak
# oleh sistem."
#
# CATATAN KETERBATASAN (WAJIB diungkap di skripsi -- bukan disembunyikan):
# artikel sumber membahas KONTEKS DISKON ("Token Listrik Diskon"),
# sehingga ada kemungkinan angka 720 jam nyala ini spesifik berlaku
# untuk periode/skema diskon tertentu, bukan aturan pembelian token
# yang pasti berlaku permanen untuk SEMUA pelanggan prabayar sepanjang
# tahun. Belum ditemukan sumber independen yang memastikan cakupannya
# di luar konteks diskon. Peneliti tetap menerapkan angka ini karena
# ini satu-satunya sumber PLN yang berhasil ditemukan untuk mekanisme
# ini, dengan keterbatasan ini dinyatakan eksplisit -- bukan diklaim
# sebagai kepastian regulasi PLN yang berlaku umum.
RASIO_KWH_BULANAN_MAKSIMAL = 0.72


def hitung_batas_kwh_bulanan(daya_va: int) -> float:
    """
    Batas maksimal kWh yang bisa dibeli per bulan (khusus prabayar),
    dari daya tersambung (VA) -- setara 720 jam nyala.

    kWh maksimal = VA x 0,72

    Sumber & keterbatasan: lihat komentar RASIO_KWH_BULANAN_MAKSIMAL
    di atas.
    """
    return round(daya_va * RASIO_KWH_BULANAN_MAKSIMAL, 2)


# ── Fungsi kalkulasi tagihan & emisi (dipakai Lapis 1 DAN Lapis 3) ────────────

def hitung_biaya_materai(subtotal: float, is_prabayar: bool = False) -> float:
    """
    Bea Meterai pada tagihan listrik pascabayar. [4]

    Rp10.000 flat (UU No.10/2020 tentang Bea Meterai — nominal dan
    ambang berlaku nasional, bukan spesifik listrik), dikenakan HANYA
    kalau (biaya pemakaian + PBJT) > Rp5.000.000.

    HANYA berlaku untuk pascabayar — materai adalah bea atas dokumen
    tagihan resmi. estimasi_rp untuk prabayar bukan dokumen tagihan,
    cuma nilai informasional (setara Rupiah dari pemakaian), jadi tidak
    relevan dikenai bea materai.
    """
    if is_prabayar:
        return 0.0
    return 10000.0 if subtotal > 5_000_000 else 0.0


def hitung_tagihan(kwh: float, tarif: float, pbjt: float,
                   is_prabayar: bool = False) -> dict:
    """
    Menghitung rincian tagihan listrik dari total kWh.

    Struktur tagihan PLN pascabayar, sesuai infografis resmi PLN
    "Memahami Perhitungan Tagihan Listrik PLN Pascabayar" [4]:
        Tagihan = Biaya Pemakaian + PBJT-TL + Biaya Materai

    Returns dict:
        biaya_pemakaian, biaya_pbjt, biaya_materai, total
    """
    biaya_pemakaian = kwh * tarif
    biaya_pbjt      = biaya_pemakaian * pbjt
    subtotal        = biaya_pemakaian + biaya_pbjt
    biaya_materai   = hitung_biaya_materai(subtotal, is_prabayar)
    total           = subtotal + biaya_materai
    return {
        "biaya_pemakaian": round(biaya_pemakaian, 0),
        "biaya_pbjt"     : round(biaya_pbjt, 0),
        "biaya_materai"  : round(biaya_materai, 0),
        "total"          : round(total, 0),
    }


def hitung_emisi(total_kwh: float) -> dict:
    """
    Menghitung emisi CO₂ dari konsumsi listrik. [2]

    Returns dict: emisi_kg_bulan, emisi_kg_tahun, faktor_emisi, referensi
    """
    emisi_bulan = round(total_kwh * FAKTOR_EMISI_JAMALI_OM, 3)
    return {
        "emisi_kg_bulan": emisi_bulan,
        "emisi_kg_tahun": round(emisi_bulan * 12, 2),
        "faktor_emisi"  : FAKTOR_EMISI_JAMALI_OM,
        "referensi"     : "Faktor Emisi GRK Grid Jamali OM, ESDM 2019",
    }


# ── Fungsi kalkulasi profil rumah tangga ──────────────────────────────────────

def hitung_ike(total_kwh: float, luas_m2: float) -> float:
    """IKE (kWh/m²/bulan) = total_kwh / luas_m2."""
    return round(total_kwh / max(1.0, float(luas_m2)), 4)


def hitung_kwh_per_org(total_kwh: float, penghuni: int) -> float:
    """Konsumsi per penghuni (kWh/orang/bulan) = total_kwh / penghuni."""
    return round(total_kwh / max(1, int(penghuni)), 2)


# ── Fungsi kalkulasi token prabayar ───────────────────────────────────────────
# Ditambahkan untuk mendukung deteksi anomali berbasis saldo token, sebagai
# pengganti perbandingan tagihan Rupiah yang tidak relevan untuk prabayar
# (prabayar tidak punya siklus tagihan bulanan yang tetap).

def hitung_kwh_dari_token(nominal_token: float,
                          tarif_kwh: float,
                          pbjt: float) -> float:
    """
    Konversi nominal pembelian token prabayar menjadi kWh.

    Dasar pengenaan PBJT untuk prabayar adalah NOMINAL PEMBELIAN TOKEN
    itu sendiri (bukan hasil hitung-mundur dari kWh yang belum
    diketahui) — berbeda dengan pascabayar, di mana PBJT dihitung dari
    biaya_pemakaian yang sudah pasti karena kWh sudah diketahui dari
    meteran. [5]

        PBJT_Rp   = pbjt × nominal_token
        Energi_Rp = nominal_token − PBJT_Rp = nominal_token × (1 − pbjt)
        kWh       = Energi_Rp / tarif_kwh

    PENTING: parameter 'nominal_token' HARUS berupa harga stroom/token
    sesuai struk resmi PLN, TIDAK termasuk biaya admin channel
    pembayaran (bank/e-wallet/PPOB) — biaya admin adalah pungutan
    pihak ketiga, bukan bagian dari nilai token yang dikenai PBJT.

    Parameters:
        nominal_token : Rp, nominal token sesuai struk (bukan total bayar)
        tarif_kwh     : Rp/kWh, dari get_tarif(daya_va) [1]
        pbjt          : desimal, 0.024 untuk rumah tangga DKI Jakarta [3]

    Returns:
        float : kWh, dibulatkan 2 desimal (konvensi tampilan struk PLN)
    """
    energi_rp = float(nominal_token) * (1 - pbjt)
    return round(energi_rp / tarif_kwh, 2)


def hitung_hari_berjalan(tanggal_pembelian: date,
                         tanggal_referensi: Optional[date] = None) -> int:
    """
    Jumlah hari sejak tanggal pembelian token sampai tanggal referensi
    (default: hari ini).

    CATATAN: fungsi ini TIDAK melakukan clamping atau validasi — bisa
    mengembalikan nilai negatif kalau tanggal_pembelian di masa depan.
    Ini disengaja: kalkulasi.py murni aritmatika, validasi "apakah ini
    input yang valid" adalah tanggung jawab core/anomaly_detector.py
    (lihat evaluasi_anomali_prabayar).

    Returns:
        int : selisih hari (bisa negatif — divalidasi di layer pemanggil)
    """
    if tanggal_referensi is None:
        tanggal_referensi = date.today()
    return (tanggal_referensi - tanggal_pembelian).days


def hitung_estimasi_kwh_periode(total_kwh_bulanan: float,
                                hari_berjalan: int) -> float:
    """
    Skala estimasi konsumsi bulanan (asumsi 30 hari, dari perangkat yang
    diinput user) ke jumlah hari aktual yang sudah berjalan sejak
    pembelian token.

    Wajib dipakai untuk prabayar karena siklus top-up token TIDAK selalu
    30 hari seperti asumsi tagihan pascabayar — tanpa skala ini, sistem
    anomali akan salah tandai anomali murni karena panjang periode
    top-up, bukan karena ada kebocoran arus sungguhan.

        estimasi = total_kwh_bulanan × (hari_berjalan / 30)

    Returns:
        float : kWh, dibulatkan 3 desimal (konsisten dengan hitung_kwh_alat)
    """
    return round(total_kwh_bulanan * (hari_berjalan / 30.0), 3)


def hitung_saldo_token_awal(sisa_sebelum_beli: float,
                            kwh_dari_pembelian: float) -> float:
    """
    Saldo token tepat setelah top-up = sisa sebelumnya + kWh hasil
    konversi pembelian baru. Ini titik awal periode yang dipakai untuk
    menghitung konsumsi aktual (lihat hitung_token_terpakai_aktual).
    """
    return round(float(sisa_sebelum_beli) + kwh_dari_pembelian, 2)


def hitung_token_terpakai_aktual(saldo_awal: float,
                                 sisa_saat_ini: float) -> float:
    """
    Konsumsi aktual sejak top-up terakhir, dari selisih saldo — ini
    adalah GROUND TRUTH yang dipakai untuk deteksi anomali (dibandingkan
    dengan estimasi dari daftar perangkat).

    CATATAN: fungsi ini TIDAK memvalidasi hasil — bisa mengembalikan
    nilai negatif kalau sisa_saat_ini > saldo_awal (indikasi ada top-up
    susulan yang tidak tercatat, atau salah input). Validasi ini adalah
    tanggung jawab core/anomaly_detector.py.
    """
    return round(saldo_awal - float(sisa_saat_ini), 2)