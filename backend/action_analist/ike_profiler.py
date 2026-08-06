"""
action_analist/ike_profiler.py
================================
Klasifikasi fuzzy Intensitas Konsumsi Energi (IKE) rumah tangga Jakarta.

PEROMBAKAN BESAR (menggantikan sistem Mamdani 2-variabel lama):
    Sistem SEBELUMNYA memakai Fuzzy Mamdani dengan 2 variabel input
    (IKE + kWh/penghuni), 9 rule, dan skema terpisah untuk rumah
    ber-AC/tidak ber-AC, mengacu ke Permen ESDM 13/2012 (yang sudah
    DICABUT oleh Permen ESDM 9/2018, tanpa pengganti untuk rumah
    tangga — lihat catatan di bawah).

    Sistem SEKARANG memakai klasifikasi fuzzy LANGSUNG 1 variabel
    (IKE saja) — 5 himpunan trapesium yang derajat keanggotaannya
    dihitung langsung dari nilai IKE, kelas dengan derajat keanggotaan
    tertinggi yang dipilih (fuzzy argmax). Tidak ada lagi rule base,
    tidak ada lagi defuzzifikasi centroid, tidak ada lagi pembedaan
    ber-AC/tidak ber-AC, dan kWh/penghuni TIDAK LAGI dipakai sebagai
    input klasifikasi (tetap dihitung & ditampilkan sebagai metrik
    informasional terpisah di dashboard, lihat services/ingestion.py).

    Alasan penyederhanaan ini: threshold baru sudah melalui triangulasi
    5 lapis (lihat di bawah) yang jauh lebih tervalidasi daripada
    rule base Mamdani lama yang dirancang tanpa dasar empiris eksplisit
    untuk bobot tiap rule.

METODOLOGI KALIBRASI — triangulasi 5 lapis (notebook
Kalibrasi_IKE_Final.ipynb, disatukan jadi satu set ambang batas final):

    1. REGULATIF — Permen ESDM No.13/2012 tentang Pedoman Pelaksanaan
       Konservasi Energi (skema "non_ac"). PENTING: regulasi ini SUDAH
       DICABUT oleh Permen ESDM No.9/2018, dan sampai saat kalibrasi
       ini dibuat, TIDAK ADA regulasi pengganti yang menetapkan standar
       IKE khusus rumah tangga (Permen ESDM No.3/2025 Lampiran II yang
       lebih baru mengatur gedung PEMERINTAH, satuan kWh/m²/TAHUN,
       bukan rumah tangga per bulan). Permen 13/2012 tetap dipakai
       sebagai SALAH SATU acuan (bukan satu-satunya), dengan status
       "sudah dicabut" dinyatakan eksplisit -- bukan diklaim sebagai
       regulasi yang masih berlaku.
    2. EMPIRIS — Utomo, D. P., Purnama, A. A., & Adryan, R. (2021).
       [Judul lengkap terkait audit energi rumah tangga]. Prosiding
       IRWNS ke-12, Politeknik Negeri Bandung. Data primer n=20 rumah,
       diolah ulang dengan Jenks natural breaks (bukan equal-width
       binning seperti Tabel 6 aslinya, yang tidak sesuai untuk
       distribusi data yang skewed) dan uji normalitas Shapiro-Wilk.
    3. DEDUKTIF — profil peralatan rumah tangga representatif
       (spesifikasi & pola pakai wajar per kategori alat).
    4. KARBON — plafon konsumsi listrik rumah tangga diturunkan dari
       target 1,5°C Hot or Cool Institute (Akenji et al., 2021,
       "1.5-Degree Lifestyles"), porsi listrik dari total jatah karbon
       personal ASUMSI PENELITI (PORSI_LISTRIK=0.20, belum ada
       rujukan spesifik per rumah tangga Indonesia).
    5. AFORDABILITAS — ambang energy burden 6% dari ACEEE (Drehobl,
       Ross & Ayala, 2020), dikalikan estimasi pengeluaran rumah
       tangga (PENGELUARAN_RT — ASUMSI PENELITI Rp4.000.000/bulan,
       BELUM diverifikasi ke data Susenas BPS terbaru).

    Kelima lapis ini direkonsiliasi (natural breaks empiris dijadikan
    basis utama, dikoreksi oleh batas atas dari lapis karbon &
    afordabilitas yang lebih ketat) menjadi satu set batas kelas
    (band), lalu di-"fuzzifikasi" dengan zona transisi 10% di tiap
    batas untuk menghindari klasifikasi biner yang kaku di
    perbatasan kelas -- rumah dengan IKE persis di garis batas tidak
    dipaksa jatuh ke satu kelas secara tiba-tiba.

CATATAN JUJUR UNTUK BAB 3/5: layer 4 (karbon) dan 5 (afordabilitas)
masing-masing punya SATU angka yang masih berstatus asumsi peneliti,
belum divalidasi ke sumber resmi Indonesia (lihat PORSI_LISTRIK dan
PENGELUARAN_RT di atas). Ini WAJIB diungkap eksplisit di keterbatasan
penelitian, bukan disembunyikan.

Kategori IKE output (5 label):
    Sangat Efisien | Efisien | Cukup Efisien | Boros | Sangat Boros

Input:
    ike (float) : Intensitas Konsumsi Energi (kWh/m²/bulan),
                  dari core.kalkulasi.hitung_ike() -- SUDAH dihitung
                  sekali di Lapis 1, tidak dihitung ulang di sini.

Output:
    str : salah satu dari 5 kategori di atas
"""

# Parameter trapesium [a, b, c, d]: naik dari a ke b, datar dari b ke c,
# turun dari c ke d. Hasil kalibrasi 5-lapis (lihat docstring modul).
# Kelas PERTAMA (a==b==0): plateau dari titik 0, tidak ada IKE negatif.
# Kelas TERAKHIR (c==d): plateau terbuka ke atas (lihat _trapmf).
MF_IKE = {
    "Sangat Efisien": (0.0,   0.0,   1.586,  1.938),
    "Efisien"        : (1.669, 1.855, 2.597,  2.783),
    "Cukup Efisien"  : (2.5,   2.88,  4.396,  4.774),
    "Boros"          : (4.303, 4.867, 7.119,  7.682),
    "Sangat Boros"   : (6.956, 7.844, 11.84, 11.84),
}


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Fungsi keanggotaan trapesium (a<=b<=c<=d): naik dari a ke b, datar
    dari b ke c, turun dari c ke d.

    PERBAIKAN dari versi notebook: kasus c==d (dipakai khusus kelas
    TERAKHIR, "Sangat Boros") SENGAJA diperlakukan sebagai plateau
    TERBUKA ke atas -- versi notebook aslinya (dipakai cuma untuk
    plotting, universe dibatasi manual) memberi keanggotaan 0 untuk
    x melebihi d saat c==d, yang berarti IKE sangat tinggi bisa
    berakhir TIDAK masuk kelas manapun. Kelas terakhir tidak boleh
    punya batas atas -- IKE setinggi apa pun tetap "Sangat Boros".
    """
    if x <= a:
        return 1.0 if a == b else 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    if c == d:
        return 1.0  # plateau terbuka ke atas -- lihat catatan di atas
    if x < d:
        return (d - x) / (d - c)
    return 0.0


def profil_ike(ike: float) -> str:
    """
    Klasifikasi fuzzy langsung: hitung derajat keanggotaan IKE di
    kelima himpunan trapesium MF_IKE, kembalikan kelas dengan derajat
    keanggotaan TERTINGGI (fuzzy argmax) -- bukan lagi lewat rule base
    Mamdani + defuzzifikasi centroid seperti sistem lama.

    Parameters:
        ike : IKE (kWh/m²/bulan), dari core.kalkulasi.hitung_ike()

    Returns:
        str : salah satu dari 'Sangat Efisien' | 'Efisien' |
              'Cukup Efisien' | 'Boros' | 'Sangat Boros'
    """
    ike = float(ike)
    derajat = {label: _trapmf(ike, *params) for label, params in MF_IKE.items()}
    return max(derajat, key=derajat.get)