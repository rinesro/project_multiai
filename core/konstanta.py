"""
core/konstanta.py
=================
Lapis 1 — Sumber tunggal untuk SELURUH data dan konstanta yang dipakai sistem.

Modul ini hanya mengimpor `math` (untuk menggabungkan komponen ketidakpastian).
Tidak ada logika bisnis di sini, hanya data.

ATURAN MODUL INI
----------------
Setiap angka WAJIB punya sumber yang dapat ditelusuri. Angka tanpa sumber
ditandai eksplisit dengan awalan `_BELUM_TERVERIFIKASI_` dan TIDAK BOLEH
dipakai dalam perhitungan yang memengaruhi hasil, sampai sumbernya ditemukan.

Kode `[Pn]` di bawah merujuk ke entri Daftar Pustaka (format Harvard).


DAFTAR SUMBER
-------------
[P1] PT PLN (Persero)., 2026. Penetapan Penyesuaian Tarif Tenaga Listrik
     (Tariff Adjustment) Periode April–Juni 2026. Jakarta: PT PLN (Persero).
     -> Tabel tarif per golongan.
     -> Catatan *)   : Rekening Minimum RM1 = 40 (Jam Nyala) x Daya
                       tersambung (kVA) x Biaya Pemakaian.
     -> Catatan **)  : Jam nyala = kWh per bulan dibagi kVA tersambung.
     -> Catatan ****): Biaya kelebihan pemakaian daya reaktif (kVArh)
                       dikenakan dalam hal faktor daya rata-rata setiap bulan
                       kurang dari 0,85.

[P2] Pemerintah Daerah DKI Jakarta., 2024. Peraturan Daerah DKI Jakarta
     Nomor 1 Tahun 2024 tentang Pajak Daerah dan Retribusi Daerah. Jakarta:
     Pemerintah Provinsi DKI Jakarta.
     -> Pasal 53 ayat (3): tarif PBJT Tenaga Listrik.

[P3] Badan Pendapatan Daerah DKI Jakarta., 2024. Segala Hal tentang PBJT
     Tenaga Listrik. Jakarta: Bapenda DKI Jakarta.
     -> Dasar pengenaan PBJT: pascabayar = biaya beban tetap + biaya
        pemakaian kWh; prabayar = jumlah pembelian tenaga listrik, dan bila
        memakai voucer maka dasar pengenaan = nilai rupiah voucer tersebut.

[P4] Kementerian Energi dan Sumber Daya Mineral Republik Indonesia., 2019.
     Faktor Emisi GRK Sistem Ketenagalistrikan Tahun 2019. Jakarta:
     Direktorat Jenderal Ketenagalistrikan, Kementerian ESDM RI.
     -> Faktor emisi Grid Jamali, Operating Margin (OM).

[P5] Departemen Pendidikan Nasional Republik Indonesia., 2004. Pedoman Teknis
     Konservasi Energi dan Penggunaan Energi Secara Efisien pada Bangunan
     Gedung. Jakarta: Depdiknas RI.
     -> Klasifikasi IKE dalam kWh/m2/bulan, dibedakan ber-AC dan tidak ber-AC.

[P6] Kementerian Energi dan Sumber Daya Mineral Republik Indonesia., 2025.
     Peraturan Menteri ESDM Nomor 3 Tahun 2025 tentang Konservasi Energi oleh
     Pemerintah dan Pemerintah Daerah. Berita Negara RI Tahun 2025 Nomor 92.
     -> Lampiran II: klasifikasi IKE dalam kWh/m2/TAHUN, 4 kelas, pembeda
        luas bangunan. Pasal 3 & 4 membatasi ruang lingkup ke sarana dan
        prasarana Pemerintah — BUKAN rumah tangga.

[P7] International Electrotechnical Commission., 2020. IEC 62053-21:2020
     Electricity metering equipment — Particular requirements — Part 21:
     Static meters for AC active energy (classes 0,5, 1 and 2). Geneva: IEC.
     -> Kelas akurasi kWh meter statis. Kelas 1 = batas kesalahan +/- 1%.

[P8] Joint Committee for Guides in Metrology., 2008. Evaluation of Measurement
     Data — Guide to the Expression of Uncertainty in Measurement (GUM).
     JCGM 100:2008. Sevres: Bureau International des Poids et Mesures.
     -> Penggabungan komponen ketidakpastian yang saling bebas dilakukan
        secara kuadratur (root-sum-square), bukan dijumlahkan linear.

[P9] Wang, Y., Chen, Q., Hong, T. dan Kang, C., 2019. Review of smart meter
     data analytics: Applications, methodologies, and challenges. IEEE
     Transactions on Smart Grid, vol. 10, no. 3, pp. 3125–3148.
     DOI: https://doi.org/10.1109/TSG.2018.2818167
     -> Kerangka deteksi anomali konsumsi: bangun wilayah konsumsi normal
        (expected region), lalu tandai pengamatan yang jatuh di luarnya.
"""

import math

# ══════════════════════════════════════════════════════════════════════════════
# 1. TARIF TENAGA LISTRIK  [P1]
# ══════════════════════════════════════════════════════════════════════════════

# Kunci: (batas_bawah_VA, batas_atas_VA) inklusif. Nilai: Rp/kWh.
TARIF_PER_GOLONGAN = {
    (0,      900):  1352.00,   # R-1/TR 900 VA-RTM
    (901,   2200):  1444.70,   # R-1/TR 1.300 VA & 2.200 VA
    (2201,  5500):  1699.53,   # R-2/TR 3.500 VA s.d. 5.500 VA
    (5501, 999999): 1699.53,   # R-3/TR 6.600 VA ke atas
}
TARIF_DEFAULT = 1699.53

# Golongan daya tersambung yang ditawarkan PLN untuk rumah tangga.
GOLONGAN_DAYA = [900, 1300, 2200, 3500, 4400, 5500, 6600, 7700, 10600, 13200]


# ══════════════════════════════════════════════════════════════════════════════
# 2. REKENING MINIMUM  [P1, catatan *) dan **)]
# ══════════════════════════════════════════════════════════════════════════════
#
# RM1 = 40 (Jam Nyala) x Daya tersambung (kVA) x Biaya Pemakaian
# Jam nyala = kWh per bulan dibagi kVA tersambung.
#
# Pada lembar tarif [P1], kolom "BIAYA BEBAN (Rp/kVA/bulan)" untuk golongan
# R-1/TR, R-2/TR, dan R-3/TR TIDAK berisi angka rupiah, melainkan tanda *)
# yang merujuk ke catatan RM. Artinya RM adalah PENGGANTI biaya beban, dan
# berfungsi sebagai LANTAI tagihan — bukan komponen yang dijumlahkan.

JAM_NYALA_RM = 40


# ══════════════════════════════════════════════════════════════════════════════
# 3. PAJAK BARANG DAN JASA TERTENTU (PBJT) ATAS TENAGA LISTRIK  [P2, P3]
# ══════════════════════════════════════════════════════════════════════════════
#
# Perda DKI No.1/2024 Pasal 53 ayat (3).

PBJT_RUMAH_TANGGA = 0.024     # konsumen selain industri & pertambangan migas
PBJT_INDUSTRI = 0.030         # industri dan pertambangan minyak/gas alam
PBJT_SUMBER_SENDIRI = 0.015   # listrik yang dihasilkan sendiri

# Dasar pengenaan PBJT berbeda antara pascabayar dan prabayar [P3]:
#   pascabayar : biaya beban tetap + biaya pemakaian kWh
#                -> harga efektif = TDL x (1 + t)
#   prabayar   : nilai rupiah voucer
#                -> harga efektif = TDL / (1 - t)
# Selisihnya kecil (sekitar 0,06% untuk t = 2,4%) tetapi nyata, dan bukan bug.


# ══════════════════════════════════════════════════════════════════════════════
# 4. FAKTOR DAYA  [P1, catatan ****)]
# ══════════════════════════════════════════════════════════════════════════════
#
# P = V x I menghasilkan daya SEMU, satuannya VA. kWh meter PLN mencatat daya
# NYATA: W = V x I x cos(phi). Sistem ini hanya menerima input tegangan dan
# arus, jadi cos(phi) tidak diketahui.
#
# Alih-alih menebak nilai cos(phi) per kategori peralatan (yang tidak punya
# sumber), sistem memakai BATAS yang ditetapkan PLN sendiri:
#
#   catatan ****) lembar tarif [P1]:
#   "Biaya kelebihan pemakaian daya reaktif (kVArh) dikenakan dalam hal
#    faktor daya rata-rata setiap bulan kurang dari 0,85."
#
# Artinya PLN memperlakukan 0,85 sebagai faktor daya rata-rata bulanan minimum
# yang wajar untuk sebuah instalasi. Maka konsumsi nyata rumah tangga berada
# di dalam interval:
#
#       kWh dalam [ kVAh x 0,85 , kVAh x 1,00 ]
#
# Batas bawah  = seluruh beban induktif pada faktor daya minimum PLN.
# Batas atas   = seluruh beban resistif murni (cos phi = 1).
#
# Ini BUKAN estimasi titik. Ini interval yang batasnya berasal dari regulasi
# dan dari hukum fisika (cos phi tidak dapat melebihi 1).

FAKTOR_DAYA_MINIMUM_PLN = 0.85
FAKTOR_DAYA_MAKSIMUM = 1.00


# ══════════════════════════════════════════════════════════════════════════════
# 5. EMISI GAS RUMAH KACA  [P4]
# ══════════════════════════════════════════════════════════════════════════════

FAKTOR_EMISI_JAMALI_OM = 0.80   # kgCO2 per kWh, Grid Jamali, Operating Margin
REFERENSI_FAKTOR_EMISI = "Faktor Emisi GRK Sistem Ketenagalistrikan 2019, ESDM RI"


# ══════════════════════════════════════════════════════════════════════════════
# 6. WAKTU
# ══════════════════════════════════════════════════════════════════════════════

HARI_PER_BULAN = 30    # asumsi sistem. Bulan riil 28-31 hari.
BULAN_PER_TAHUN = 12


# ══════════════════════════════════════════════════════════════════════════════
# 7. INTENSITAS KONSUMSI ENERGI (IKE)
# ══════════════════════════════════════════════════════════════════════════════

# --- Depdiknas 2004 [P5] — satuan kWh/m2/BULAN, dipakai sistem ---------------
#
# Batas atas tiap kelas. Kelas terakhir ("sangat_boros") terbuka ke atas.
BATAS_IKE_DEPDIKNAS = {
    "tidak_ac": {
        "sangat_efisien": 0.84,
        "efisien":        1.67,
        "cukup_efisien":  2.50,
        "boros":          3.34,
        # IKE > 3,34 -> sangat boros
    },
    "ac": {
        "sangat_efisien":  7.92,
        "efisien":        12.08,
        "cukup_efisien":  14.58,
        "boros":          23.75,
        # IKE > 23,75 -> sangat boros
    },
}

# --- Permen ESDM No.3/2025 Lampiran II [P6] — satuan kWh/m2/TAHUN -----------
#
# TIDAK dipakai sebagai ambang sistem. Disimpan di sini agar dapat dikutip
# dan dibandingkan di dokumen penelitian.
#
# Empat kelas (tidak ada "Sangat Boros"), pembedanya luas bangunan, bukan
# ada/tidaknya AC. Pasal 3 & 4 membatasi ruang lingkup ke sarana/prasarana
# Pemerintah; formulir Lampiran I meminta fungsi bangunan berupa perkantoran,
# universitas, rumah sakit, hotel, atau mal.
#
# Konsekuensi bila dipaksakan ke rumah tangga: rumah 45 m2 baru masuk kelas
# "Boros" pada konsumsi sekitar 506 kWh/bulan, sehingga hampir seluruh rumah
# tangga Indonesia terklasifikasi "Sangat Efisien" dan optimizer tidak pernah
# menyala. Alasan inilah yang mendasari pemakaian ambang Depdiknas [P5].
BATAS_IKE_PERMEN_ESDM_3_2025 = {
    "luas_kurang_sama_5000_m2": {
        "sangat_efisien":  70,
        "efisien":         99,
        "cukup_efisien":  135,
        # IKE > 135 -> boros
    },
    "luas_lebih_5000_m2": {
        "sangat_efisien":  99,
        "efisien":        135,
        "cukup_efisien":  173,
        # IKE > 173 -> boros
    },
}

LABEL_IKE = ("Sangat Efisien", "Efisien", "Cukup Efisien", "Boros", "Sangat Boros")


# ══════════════════════════════════════════════════════════════════════════════
# 8. KATEGORI PERALATAN
# ══════════════════════════════════════════════════════════════════════════════
#
# Harus sama persis dengan kategori pada data latih model DSM. Mengubah,
# menambah, atau mengurangi salah satu akan membuat prediksi model tidak valid.

KATEGORI_PERALATAN = (
    "Pendingin",
    "Pemanas",
    "Laundry",
    "Memasak",
    "Hiburan/Elektronik",
    "Pencahayaan",
    "Pompa/Motor",
    "Lainnya",
)

LABEL_DSM = ("Fleksibel", "Tidak Fleksibel")


# ══════════════════════════════════════════════════════════════════════════════
# 9. TOLERANSI DETEKSI ANOMALI — DITURUNKAN, BUKAN DIASUMSIKAN
# ══════════════════════════════════════════════════════════════════════════════
#
# Tidak ada publikasi yang menetapkan ambang toleransi untuk membandingkan
# estimasi berbasis inventaris peralatan terhadap tagihan riil PLN. Angka 10%
# dan 20% yang beredar di literatur deteksi Non-Technical Loss adalah faktor
# skala serangan sintetis untuk membangkitkan data latih, bukan ambang deteksi.
# Mengutipnya sebagai ambang toleransi akan menjadi falsifikasi.
#
# Karena itu toleransi di sini DITURUNKAN dari anggaran ketidakpastian
# (uncertainty budget). Setiap komponen punya sumber, dan penggabungannya
# mengikuti JCGM 100:2008 [P8]: komponen yang saling bebas digabung secara
# kuadratur (root-sum-square), bukan dijumlahkan linear.
#
# Yang TIDAK dimasukkan ke sini, dan alasannya:
#
#   - Faktor daya. Sudah menjadi LEBAR RENTANG prediksi (bagian 4).
#     Memasukkannya lagi ke toleransi berarti menghitungnya dua kali.
#
#   - Siklus baca meter dan rekening rata-rata PLN. Efeknya tidak dapat
#     dikuantifikasi tanpa data internal PLN. Dicatat sebagai keterbatasan
#     penelitian (bagian 10), bukan disembunyikan ke dalam sebuah konstanta.
#
#   - Duty cycle dan galat perkiraan jam pakai pengguna. Sama seperti di atas:
#     tidak terkuantifikasi, dicatat sebagai keterbatasan.
#
# Konsekuensinya, toleransi ini adalah BATAS BAWAH dari ketidakpastian yang
# sesungguhnya. Sistem akan lebih mudah menyatakan "anomali" daripada
# seharusnya. Itu pilihan sadar: lebih baik memberi peringatan yang bisa
# ditindaklanjuti pengguna daripada mendiamkan kebocoran arus.

KOMPONEN_KETIDAKPASTIAN = (
    {
        "nama":   "akurasi_kwh_meter",
        "nilai":  0.010,
        "satuan": "relatif",
        "sumber": "IEC 62053-21:2020 [P7] — kWh meter statis kelas akurasi 1, "
                  "batas kesalahan +/- 1% pada kondisi acuan.",
    },
    {
        "nama":   "asumsi_panjang_bulan",
        "nilai":  2.0 / 30.0,          # |28 - 30| / 30 = 6,67%
        "satuan": "relatif",
        "sumber": "Aritmetika. Sistem memakai konstanta 30 hari; bulan riil "
                  "28-31 hari. Penyimpangan terbesar terjadi pada Februari.",
    },
)


def _gabung_kuadratur(komponen) -> float:
    """Gabungkan ketidakpastian relatif yang saling bebas. JCGM 100:2008 [P8]."""
    return math.sqrt(sum(k["nilai"] ** 2 for k in komponen))


# Nilai terhitung: sqrt(0,010^2 + 0,0667^2) = 0,0674 -> 6,74%
TOLERANSI_ANOMALI = round(_gabung_kuadratur(KOMPONEN_KETIDAKPASTIAN), 4)


def rincian_toleransi() -> dict:
    """
    Mengembalikan anggaran ketidakpastian secara utuh, supaya angka toleransi
    dapat dipertanggungjawabkan di dokumen penelitian dan di antarmuka.
    """
    return {
        "komponen":            [dict(k) for k in KOMPONEN_KETIDAKPASTIAN],
        "metode_penggabungan": "Root-sum-square (kuadratur), JCGM 100:2008",
        "toleransi":           TOLERANSI_ANOMALI,
        "toleransi_persen":    round(TOLERANSI_ANOMALI * 100, 2),
        "catatan":             "Batas bawah ketidakpastian. Siklus baca meter, "
                               "rekening rata-rata, duty cycle, dan galat "
                               "perkiraan jam pakai belum terkuantifikasi.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9b. DENOMINASI TOKEN PRABAYAR
# ══════════════════════════════════════════════════════════════════════════════
#
# Nominal token yang dijual PLN. Dapat diamati langsung pada aplikasi PLN Mobile
# dan pada kanal pembayaran resmi.
#
# CATATAN SITASI: sertakan tangkapan layar PLN Mobile sebagai Gambar di bab
# Metode Penelitian, lengkap dengan sitasi sumber sebagaimana disyaratkan
# Buku Pedoman bagian 4.5 huruf (e). Tanpa itu, daftar ini tidak boleh diklaim
# sebagai data resmi.
#
# Diurutkan menurun agar pemecahan greedy pada kalkulator berjalan benar.
DENOMINASI_TOKEN_PLN = (1_000_000, 500_000, 200_000, 100_000, 50_000, 20_000)

# Kelipatan nominal bebas yang umumnya diterima kanal pembayaran.
KELIPATAN_NOMINAL_TOKEN = 10_000


# ══════════════════════════════════════════════════════════════════════════════
# 10. KONSTANTA YANG BELUM PUNYA SUMBER
# ══════════════════════════════════════════════════════════════════════════════
#
# JANGAN dipakai dalam perhitungan yang memengaruhi hasil sampai sumbernya
# ditemukan dan dipindahkan ke bagian yang sesuai di atas.

# Batas maksimum pengisian token: daya (VA) x 720 jam / 1000 = kWh.
# Angka 720 beredar luas di artikel populer, namun belum saya temukan pada
# dokumen resmi PLN. Cari di SPLN atau Buku Pedoman Pelayanan Pelanggan PLN
# sebelum dipakai atau dikutip.
_BELUM_TERVERIFIKASI_JAM_NYALA_MAKS_TOKEN = 720

# Ambang kWh saat meteran prabayar membunyikan alarm. Bervariasi antar merek
# meter dan dapat dikonfigurasi. Sebaiknya dijadikan input pengguna.
_BELUM_TERVERIFIKASI_AMBANG_ALARM_KWH = 20.0


# ══════════════════════════════════════════════════════════════════════════════
# 11. KETERBATASAN PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
#
# Tampilkan di antarmuka, dan tulis di bab Keterbatasan Penelitian.
# Empat butir pertama adalah alasan mengapa keluaran sistem disebut PREDIKSI
# dan bukan tagihan.

KETERBATASAN_PREDIKSI = (
    "Periode rekening PLN mengikuti siklus baca meter, bukan bulan kalender. "
    "Sistem ini mengasumsikan bulan kalender.",

    "Bila meter tidak dapat dibaca, PLN dapat menerbitkan rekening berdasarkan "
    "rata-rata pemakaian periode sebelumnya lalu mengoreksinya pada periode "
    "berikutnya. Estimasi berbasis inventaris peralatan tidak akan cocok "
    "dengan rekening semacam itu.",

    "Sistem hanya menerima tegangan dan arus, sehingga faktor daya tiap "
    "peralatan tidak diketahui. Konsumsi disajikan sebagai rentang antara "
    "kVAh x 0,85 (faktor daya minimum menurut PT PLN) dan kVAh x 1,00 "
    "(beban resistif murni).",

    "Duty cycle diabaikan. Kulkas menyala 24 jam namun kompresornya hidup-mati; "
    "AC bertermostat tidak menarik arus penuh terus-menerus. Masukan jam per "
    "hari adalah jam operasi, bukan jam beban penuh.",

    "Jam pemakaian berasal dari perkiraan pengguna, bukan pengukuran.",

    "Sistem memakai konstanta 30 hari per bulan; bulan riil 28-31 hari.",

    "Tarif tenaga listrik disesuaikan setiap triwulan (tariff adjustment), "
    "sehingga tarif yang tertanam dalam sistem perlu diperbarui berkala.",

    "Ambang IKE mengacu pada Pedoman Depdiknas 2004 untuk bangunan gedung, "
    "bukan standar khusus rumah tinggal. Lampiran II Permen ESDM No.3/2025 "
    "tidak dapat diterapkan langsung karena ruang lingkupnya sarana dan "
    "prasarana Pemerintah, dan ambangnya membuat hampir seluruh rumah tangga "
    "terklasifikasi Sangat Efisien.",
)