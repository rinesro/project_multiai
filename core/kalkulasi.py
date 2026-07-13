"""
core/kalkulasi.py
==================
Sumber kebenaran tunggal (single source of truth) untuk seluruh konstanta
regulasi dan rumus kalkulasi dasar yang dipakai di lebih dari satu layer.

Sebelum modul ini dibuat, konstanta dan rumus yang sama ditulis ulang
secara terpisah di app.py, models/dsm_classifier.py, models/knn_recommender.py,
optimizer/brute_force.py, dan train_knn.py — berisiko saling tidak sinkron
kalau salah satu diubah tanpa mengubah yang lain.

Prinsip pemakaian:
    - Semua kalkulasi yang BISA dihitung dari data yang sudah tersedia
      di Lapis 1 (app.py) dihitung SEKALI di sana.
    - Layer 2 dan Layer 3 TIDAK menghitung ulang dari nol — mereka
      menerima hasil Lapis 1 sebagai input, atau memanggil fungsi yang
      SAMA dari modul ini kalau memang butuh menghitung skenario baru
      yang belum pernah dihitung Lapis 1 (mis. simulasi brute force).

CATATAN PEMISAHAN — deteksi anomali:
    Fungsi deteksi_anomali() dan konstanta BATAS_TOLERANSI_ANOMALI
    SUDAH DIPINDAH ke core/anomaly_detector.py. Modul ini (kalkulasi.py)
    HANYA berisi rumus aritmatika/regulasi murni, tanpa keputusan bisnis
    "apa yang dianggap anomali" — supaya kalkulasi.py tetap gampang diuji
    secara terisolasi, dan supaya semua aturan anomali (termasuk kasus
    khusus prabayar: data belum cukup, data tidak konsisten, tanggal
    tidak valid) terkumpul di satu tempat yang jelas tanggung jawabnya.

Referensi regulasi:
    [1] Tarif PLN April–Juni 2026 — PT PLN (Persero)
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO₂/kWh — ESDM 2019
    [3] PBJT 2,4% — Perda DKI Jakarta No.1/2024
    [4] Rumus Rekening Minimum — PT PLN (Persero)
    [5] Dasar pengenaan PBJT tenaga listrik prabayar dihitung dari
        nominal pembelian token (bukan hasil hitung-mundur dari kWh) —
        Badan Pendapatan Daerah Provinsi DKI Jakarta (2025),
        https://dpp.jakarta.go.id/berita/sobat-pajak-ini-dia-segala-
        hal-tentang-pbjt-tenaga-listrik
"""

from datetime import date
from typing import Optional

# ── Konstanta regulasi ────────────────────────────────────────────────────────

TARIF_PER_GOLONGAN = {
    (0,      900):  1352.00,   # R-1/TR 900 VA RTM       [1]
    (901,   2200):  1444.70,   # R-1/TR 1.300 & 2.200 VA [1]
    (2201,  5500):  1699.53,   # R-2/TR 3.500–5.500 VA   [1]
    (5501, 999999): 1699.53,   # R-3/TR 6.600 VA ke atas [1]
}

FAKTOR_EMISI_JAMALI_OM  = 0.80    # kgCO₂/kWh — Grid Jamali OM [2]
PBJT_RUMAH_TANGGA       = 0.024   # 2,4% [3]

# Golongan daya standar PLN (dipakai UI & generator data)
GOLONGAN_DAYA = [900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 13200]


# ── Fungsi dasar tarif & biaya beban ──────────────────────────────────────────

def get_tarif(daya_va: int) -> float:
    """Tarif Rp/kWh berdasarkan daya tersambung (VA). [1]"""
    for (lo, hi), tarif in TARIF_PER_GOLONGAN.items():
        if lo <= daya_va <= hi:
            return tarif
    return 1699.53


def hitung_biaya_beban(daya_va: int, is_prabayar: bool = False) -> float:
    """
    Rekening Minimum pascabayar.
    Rumus resmi PLN: RM1 = 40 jam × daya (kVA) × tarif (Rp/kWh) [4]
    Prabayar (token) tidak dikenakan biaya beban.
    """
    if is_prabayar:
        return 0.0
    return 40 * (daya_va / 1000.0) * get_tarif(daya_va)


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


# ── Fungsi kalkulasi tagihan & emisi (dipakai Lapis 1 DAN Lapis 3) ────────────

def hitung_tagihan(kwh: float, tarif: float, pbjt: float,
                   biaya_beban: float) -> dict:
    """
    Menghitung rincian tagihan listrik dari total kWh.

    Struktur tagihan PLN:
        Tagihan = Biaya Pemakaian + PBJT + Biaya Beban (jika pascabayar)

    Returns dict:
        biaya_pemakaian, biaya_pbjt, biaya_beban, total
    """
    biaya_pemakaian = kwh * tarif
    biaya_pbjt      = biaya_pemakaian * pbjt
    total           = biaya_pemakaian + biaya_pbjt + biaya_beban
    return {
        "biaya_pemakaian": round(biaya_pemakaian, 0),
        "biaya_pbjt"     : round(biaya_pbjt, 0),
        "biaya_beban"    : round(biaya_beban, 0),
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