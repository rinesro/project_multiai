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

Referensi regulasi:
    [1] Tarif PLN April–Juni 2026 — PT PLN (Persero)
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO₂/kWh — ESDM 2019
    [3] PBJT 2,4% — Perda DKI Jakarta No.1/2024
    [4] Rumus Rekening Minimum — PT PLN (Persero)
"""

# ── Konstanta regulasi ────────────────────────────────────────────────────────

TARIF_PER_GOLONGAN = {
    (0,      900):  1352.00,   # R-1/TR 900 VA RTM       [1]
    (901,   2200):  1444.70,   # R-1/TR 1.300 & 2.200 VA [1]
    (2201,  5500):  1699.53,   # R-2/TR 3.500–5.500 VA   [1]
    (5501, 999999): 1699.53,   # R-3/TR 6.600 VA ke atas [1]
}

FAKTOR_EMISI_JAMALI_OM  = 0.80    # kgCO₂/kWh — Grid Jamali OM [2]
PBJT_RUMAH_TANGGA       = 0.024   # 2,4% [3]
BATAS_TOLERANSI_ANOMALI = 0.15    # 15%

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


def deteksi_anomali(tagihan_asli: float, estimasi_tagihan: float) -> dict:
    """
    Deteksi anomali berbasis aturan (if-else).
    Anomali = selisih estimasi vs tagihan asli > BATAS_TOLERANSI_ANOMALI.

    Returns dict: is_anomali, selisih_pct
    """
    selisih_pct = (
        abs(tagihan_asli - estimasi_tagihan) / max(1.0, float(tagihan_asli))
    )
    return {
        "is_anomali" : selisih_pct > BATAS_TOLERANSI_ANOMALI,
        "selisih_pct": round(selisih_pct * 100, 1),
    }