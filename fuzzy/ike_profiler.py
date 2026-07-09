"""
fuzzy/ike_profiler.py
=====================
Modul Fuzzy Mamdani untuk klasifikasi Intensitas Konsumsi Energi (IKE)
rumah tangga Jakarta.

Referensi standar:
    Pedoman Teknis Konservasi Energi dan Penggunaan Energi Secara Efisien
    pada Bangunan Gedung, Departemen Pendidikan Nasional RI
    (digunakan sebagai acuan IKE rumah tangga, bukan gedung komersial)

    Catatan: Permen ESDM No.3 Tahun 2025 Lampiran II menggunakan satuan
    kWh/m²/tahun untuk gedung pemerintah. Modul ini menggunakan satuan
    kWh/m²/bulan sesuai pedoman Depdiknas untuk konteks rumah tangga.

Kategori IKE output (5 label, sesuai standar):
    Sangat Efisien | Efisien | Cukup Efisien | Boros | Sangat Boros

Input:
    luas_m2    (float) : luas bangunan dalam m²
    penghuni   (int)   : jumlah penghuni
    total_kwh  (float) : total konsumsi listrik bulanan (kWh)
                         dihitung dari Σ (V × I × jam × 30) / 1000
                         atas semua peralatan yang diinput user
    ada_ac     (bool)  : True jika ada peralatan kategori "Pendingin"

Output:
    str : label IKE — salah satu dari 5 kategori di atas
"""

import numpy as np


# ── Fungsi keanggotaan dasar ──────────────────────────────────────────────────

def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Fungsi keanggotaan segitiga (triangular membership function).

    Bentuk: naik dari a ke b, turun dari b ke c.
    Nilai 1.0 hanya di titik b, nilai 0.0 di luar [a, c].
    """
    x = np.asarray(x, dtype=float)
    left  = (x - a) / (b - a) if b != a else (x >= b).astype(float)
    right = (c - x) / (c - b) if c != b else (x <= b).astype(float)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def _trapmf(x: np.ndarray,
            a: float, b: float,
            c: float, d: float) -> np.ndarray:
    """
    Fungsi keanggotaan trapesium (trapezoidal membership function).

    Bentuk: naik dari a ke b, datar dari b ke c, turun dari c ke d.
    Nilai 1.0 di interval [b, c], nilai 0.0 di luar [a, d].
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    if b != a:
        y = np.maximum(y, np.minimum((x - a) / (b - a), 1.0))
    else:
        y = np.maximum(y, (x >= a).astype(float))
    right = (d - x) / (d - c) if d != c else (x <= d).astype(float)
    y = np.minimum(y, np.where(x <= c, 1.0, right))
    return np.clip(y, 0.0, 1.0)


def _derajat(func, nilai: float) -> float:
    """Menghitung derajat keanggotaan satu nilai skalar."""
    return float(func(np.array([nilai]))[0])


def _defuzzifikasi_centroid(output_sets: dict,
                             aktivasi: dict,
                             universe: np.ndarray = None) -> float:
    """
    Defuzzifikasi metode centroid (center of gravity).

    Mengagregasi semua output fuzzy yang aktif menggunakan
    operator MAX, lalu menghitung pusat massa area agregat.

    Parameters:
        output_sets : dict nama → fungsi keanggotaan output
        aktivasi    : dict nama → kekuatan aktivasi (0.0–1.0)
        universe    : array titik evaluasi (default: 0–100, 1001 titik)

    Returns:
        float : nilai crisp hasil defuzzifikasi
    """
    if universe is None:
        universe = np.linspace(0, 100, 1001)

    agregat = np.zeros_like(universe)
    for label, kekuatan in aktivasi.items():
        if kekuatan <= 0:
            continue
        terpotong = np.minimum(kekuatan, output_sets[label](universe))
        agregat = np.maximum(agregat, terpotong)

    if agregat.sum() == 0:
        return 0.0
    return float(np.sum(universe * agregat) / np.sum(agregat))


# ── Fungsi utama ──────────────────────────────────────────────────────────────

def profil_ike(ike: float,
               kwh_per_org: float,
               ada_ac: bool) -> str:
    """
    Mengklasifikasikan profil IKE rumah tangga menggunakan
    inferensi Fuzzy Mamdani dua variabel input.

    PENTING — perubahan sejak refactor konsolidasi kalkulasi:
    Fungsi ini TIDAK LAGI menghitung IKE dan kWh/penghuni sendiri.
    Kedua nilai itu sudah dihitung sekali di Lapis 1 (app.py, lewat
    core/kalkulasi.py) dan tinggal diterima di sini sebagai parameter,
    supaya tidak ada rumus yang sama ditulis ulang di dua tempat
    berbeda dan berisiko tidak sinkron.

    Variabel input:
        1. ike          : kWh/m²/bulan — dari core.hitung_ike()
        2. kwh_per_org  : kWh/orang/bulan — dari core.hitung_kwh_per_org()

    Batas IKE standar Depdiknas (kWh/m²/bulan):
        Tidak ber-AC:
            Sangat Efisien : IKE < 0.84
            Efisien        : 0.84 ≤ IKE < 1.67
            Cukup Efisien  : 1.67 ≤ IKE ≤ 2.50
            Boros          : 2.50 < IKE ≤ 3.34
            Sangat Boros   : IKE > 3.34

        Ber-AC:
            Sangat Efisien : IKE < 7.92
            Efisien        : 7.92 ≤ IKE < 12.08
            Cukup Efisien  : 12.08 ≤ IKE ≤ 14.58
            Boros          : 14.58 < IKE ≤ 23.75
            Sangat Boros   : IKE > 23.75

    Parameters:
        ike         : IKE (kWh/m²/bulan), sudah dihitung di Lapis 1
        kwh_per_org : konsumsi per penghuni (kWh/orang/bulan),
                      sudah dihitung di Lapis 1
        ada_ac      : True jika ada peralatan kategori "Pendingin"

    Returns:
        str : salah satu dari
              'Sangat Efisien' | 'Efisien' | 'Cukup Efisien' |
              'Boros' | 'Sangat Boros'
    """
    kwh_per_org = float(kwh_per_org)

    # ── Himpunan fuzzy input: IKE ─────────────────────────────────────────────
    # Dibedakan antara rumah ber-AC dan tidak ber-AC sesuai standar Depdiknas.
    # Tiga himpunan: rendah (hemat) · sedang · tinggi (boros)
    if not ada_ac:
        # Batas tidak ber-AC: SE<0.84 | E 0.84–1.67 | CE 1.67–2.50
        #                     B 2.50–3.34 | SB >3.34
        ike_sets = {
            "rendah": lambda x: _trapmf(x,  0.00, 0.00, 0.84, 1.67),
            "sedang": lambda x: _trimf( x,  0.84, 1.67, 3.34),
            "tinggi": lambda x: _trapmf(x,  2.50, 3.34, 15.0, 15.0),
        }
    else:
        # Batas ber-AC: SE<7.92 | E 7.92–12.08 | CE 12.08–14.58
        #               B 14.58–23.75 | SB >23.75
        ike_sets = {
            "rendah": lambda x: _trapmf(x,  0.00,  0.00,  7.92, 12.08),
            "sedang": lambda x: _trimf( x,  7.92, 12.08, 23.75),
            "tinggi": lambda x: _trapmf(x, 14.58, 23.75, 60.0,  60.0),
        }

    # ── Himpunan fuzzy input: kWh per penghuni ───────────────────────────────
    # Tiga himpunan: rendah · sedang · tinggi
    # Satuan: kWh/orang/bulan
    # Range berbeda antara ber-AC dan tidak ber-AC karena konsumsi total
    # rumah ber-AC secara wajar jauh lebih tinggi per penghuninya.
    if not ada_ac:
        kwh_org_sets = {
            "rendah": lambda x: _trapmf(x,   0,   0,  35,  55),
            "sedang": lambda x: _trimf( x,  40,  75, 110),
            "tinggi": lambda x: _trapmf(x,  90, 125, 400, 400),
        }
    else:
        # Ber-AC: konsumsi per orang wajar lebih tinggi
        kwh_org_sets = {
            "rendah": lambda x: _trapmf(x,   0,   0,  80, 130),
            "sedang": lambda x: _trimf( x,  100, 200, 350),
            "tinggi": lambda x: _trapmf(x,  280, 400, 1500, 1500),
        }

    # ── Himpunan fuzzy output (universe 0–100) ────────────────────────────────
    # 5 label sesuai kategori IKE standar Depdiknas
    output_sets = {
        "sangat_efisien": lambda x: _trapmf(x,  0,  0, 15, 30),
        "efisien":        lambda x: _trimf( x, 15, 30, 50),
        "cukup_efisien":  lambda x: _trimf( x, 40, 55, 70),
        "boros":          lambda x: _trimf( x, 60, 75, 90),
        "sangat_boros":   lambda x: _trapmf(x, 80, 90, 100, 100),
    }

    # ── Derajat keanggotaan input ─────────────────────────────────────────────
    mu_ike = {k: _derajat(f, ike)         for k, f in ike_sets.items()}
    mu_org = {k: _derajat(f, kwh_per_org) for k, f in kwh_org_sets.items()}

    # ── Evaluasi rules Mamdani (AND = min, OR = max) ──────────────────────────
    #
    # IKE adalah variabel primer (standar regulasi).
    # kWh/penghuni adalah variabel sekunder (konteks sosial).
    #
    # Desain rule: IKE yang sudah jelas (rendah/tinggi) langsung menentukan
    # output tanpa bisa "digeser" oleh kWh/orang. Hanya di zona sedang
    # kWh/orang berperan menggeser ke atas atau ke bawah.
    #
    # Rule base:
    #   R1:  IF ike=rendah                   → sangat_efisien  (IKE dominan)
    #   R2:  IF ike=rendah AND kwh_org=sedang → efisien
    #   R3:  IF ike=sedang AND kwh_org=rendah → efisien
    #   R4:  IF ike=sedang AND kwh_org=sedang → cukup_efisien
    #   R5:  IF ike=sedang AND kwh_org=tinggi → boros
    #   R6:  IF ike=tinggi AND kwh_org=rendah → cukup_efisien
    #   R7:  IF ike=tinggi AND kwh_org=sedang → boros
    #   R8:  IF ike=tinggi                   → boros           (IKE dominan)
    #   R9:  IF ike=tinggi AND kwh_org=tinggi → sangat_boros
    aktivasi = {
        "sangat_efisien": max([
            mu_ike["rendah"],                                       # R1
        ]),
        "efisien": max([
            min(mu_ike["rendah"], mu_org["sedang"]),                # R2
            min(mu_ike["sedang"], mu_org["rendah"]),                # R3
        ]),
        "cukup_efisien": max([
            min(mu_ike["sedang"], mu_org["sedang"]),                # R4
            min(mu_ike["tinggi"], mu_org["rendah"]),                # R6
        ]),
        "boros": max([
            min(mu_ike["sedang"], mu_org["tinggi"]),                # R5
            min(mu_ike["tinggi"], mu_org["sedang"]),                # R7
            mu_ike["tinggi"],                                       # R8
        ]),
        "sangat_boros": max([
            min(mu_ike["tinggi"], mu_org["tinggi"]),                # R9
        ]),
    }

    # ── Defuzzifikasi → nilai crisp ───────────────────────────────────────────
    nilai_crisp = _defuzzifikasi_centroid(output_sets, aktivasi)

    # ── Pemetaan crisp → label IKE standar ───────────────────────────────────
    # Threshold disesuaikan dengan posisi centroid tiap himpunan output
    if nilai_crisp < 22:  return "Sangat Efisien"
    if nilai_crisp < 42:  return "Efisien"
    if nilai_crisp < 62:  return "Cukup Efisien"
    if nilai_crisp < 82:  return "Boros"
    return "Sangat Boros"