"""
core/anomaly_detector.py
========================
Lapis 2 — Deteksi anomali tagihan listrik.

Membandingkan tagihan riil yang dilaporkan pengguna untuk bulan sebelumnya
terhadap rentang prediksi yang dihitung `core/kalkulator.py` dari inventaris
peralatan.

HANYA UNTUK PASCABAYAR
----------------------
Deteksi anomali tidak berlaku bagi pelanggan prabayar. Pelanggan prabayar
membeli token pada waktu yang mereka pilih sendiri, dalam nominal yang mereka
pilih sendiri, dan tidak seluruh token yang dibeli habis dalam bulan yang sama.
Total pembelian token bulan lalu karena itu bukan pengukuran konsumsi bulan
lalu, sehingga tidak dapat diperbandingkan terhadap prediksi konsumsi bulanan.
Memaksakannya akan menghasilkan positif palsu setiap kali pengguna kebetulan
membeli token besar di akhir bulan.

Fungsi di sini melempar `ValueError` bila dipanggil untuk pelanggan prabayar,
supaya kekeliruan itu tidak lolos diam-diam.


METODE
------
Mengikuti kerangka umum deteksi anomali pada data meter (Wang, Chen, Hong dan
Kang, 2019): bangun wilayah konsumsi yang diharapkan, lalu tandai pengamatan
yang jatuh di luar wilayah tersebut.

Wilayah yang diharapkan di sini adalah rentang tagihan
[tagihan_bawah, tagihan_atas] yang dihasilkan kalkulator, diperlebar oleh
toleransi ketidakpastian:

    batas_bawah = tagihan_bawah x (1 - toleransi)
    batas_atas  = tagihan_atas  x (1 + toleransi)

Anomali dinyatakan bila tagihan riil jatuh di luar [batas_bawah, batas_atas].


DARI MANA ANGKA TOLERANSI BERASAL
---------------------------------
Tidak ada publikasi yang menetapkan ambang toleransi untuk perbandingan jenis
ini. Angka 10% dan 20% yang muncul dalam literatur deteksi Non-Technical Loss
adalah faktor skala serangan sintetis untuk membangkitkan data latih, bukan
ambang deteksi. Mengutipnya sebagai ambang toleransi akan menjadi falsifikasi.

Karena itu toleransi diturunkan dari anggaran ketidakpastian di
`core/konstanta.py`, dengan komponen:

    akurasi kWh meter kelas 1   1,00%   IEC 62053-21:2020
    asumsi 30 hari per bulan    6,67%   aritmetika, |28-30|/30

digabung secara kuadratur sesuai JCGM 100:2008, menghasilkan 6,74%.

Faktor daya sengaja TIDAK dimasukkan ke toleransi, karena sudah menjadi lebar
rentang prediksi itu sendiri. Memasukkannya lagi berarti menghitung dua kali.

Panggil `core.konstanta.rincian_toleransi()` untuk menampilkan anggaran ini
kepada pengguna atau penguji.


ARAH ANOMALI
------------
Arah penyimpangan menentukan tafsirannya, dan sistem tidak boleh menampilkan
tafsiran yang keliru arah.

    tagihan_lebih_tinggi : tagihan riil di atas batas atas.
        Kemungkinan ada peralatan yang belum dimasukkan ke inventaris, atau
        terdapat kebocoran arus pada instalasi.

    tagihan_lebih_rendah : tagihan riil di bawah batas bawah.
        Kemungkinan jam pemakaian dilaporkan berlebih, atau PLN menerbitkan
        rekening rata-rata karena meter tidak terbaca pada periode itu.

    wajar : tagihan riil berada di dalam wilayah yang diharapkan.
"""

from core.konstanta import TOLERANSI_ANOMALI, rincian_toleransi

ARAH_WAJAR = "wajar"
ARAH_LEBIH_TINGGI = "tagihan_lebih_tinggi"
ARAH_LEBIH_RENDAH = "tagihan_lebih_rendah"

_PESAN = {
    ARAH_WAJAR: (
        "Tagihan riil berada di dalam rentang prediksi. Tidak ada indikasi "
        "ketidaksesuaian antara peralatan yang diinput dan tagihan."
    ),
    ARAH_LEBIH_TINGGI: (
        "Tagihan riil lebih tinggi daripada yang dapat dijelaskan oleh "
        "peralatan yang diinput. Kemungkinan ada peralatan yang belum "
        "dimasukkan, atau terdapat kebocoran arus pada instalasi."
    ),
    ARAH_LEBIH_RENDAH: (
        "Tagihan riil lebih rendah daripada yang dapat dijelaskan oleh "
        "peralatan yang diinput. Kemungkinan jam pemakaian dilaporkan "
        "berlebih, atau PLN menerbitkan rekening rata-rata karena meter tidak "
        "terbaca pada periode tersebut."
    ),
}


def _wilayah_diharapkan(tagihan_bawah: float, tagihan_atas: float,
                        toleransi: float) -> tuple:
    """Perlebar rentang prediksi dengan toleransi ketidakpastian."""
    if tagihan_bawah > tagihan_atas:
        raise ValueError(
            f"tagihan_bawah ({tagihan_bawah}) tidak boleh melebihi "
            f"tagihan_atas ({tagihan_atas})."
        )
    return (
        tagihan_bawah * (1.0 - toleransi),
        tagihan_atas * (1.0 + toleransi),
    )


def deteksi_anomali(tagihan_riil: float,
                    rentang_tagihan: dict,
                    is_prabayar: bool = False,
                    toleransi: float = TOLERANSI_ANOMALI) -> dict:
    """
    Parameters
    ----------
    tagihan_riil : float
        Tagihan bulan sebelumnya yang dilaporkan pengguna, dalam rupiah.
    rentang_tagihan : dict
        Keluaran `core.kalkulator.hitung_rentang_tagihan()`. Wajib memuat
        kunci `tagihan_bawah` dan `tagihan_atas`.
    is_prabayar : bool
        Bila True, fungsi melempar ValueError. Deteksi anomali tidak berlaku
        untuk prabayar.
    toleransi : float
        Ketidakpastian relatif. Default diturunkan dari anggaran ketidakpastian
        di core/konstanta.py, bukan ditetapkan sembarang.

    Returns
    -------
    dict dengan kunci:
        is_anomali, arah, pesan,
        tagihan_riil, batas_bawah, batas_atas,
        selisih_rp, selisih_pct,
        toleransi, rincian_toleransi
    """
    if is_prabayar:
        raise ValueError(
            "Deteksi anomali hanya berlaku untuk pelanggan pascabayar. "
            "Total pembelian token bulan lalu bukan pengukuran konsumsi bulan "
            "lalu, sehingga tidak dapat dibandingkan terhadap prediksi bulanan."
        )

    for kunci in ("tagihan_bawah", "tagihan_atas"):
        if kunci not in rentang_tagihan:
            raise ValueError(
                f"rentang_tagihan tidak memuat '{kunci}'. Gunakan keluaran "
                "core.kalkulator.hitung_rentang_tagihan()."
            )

    tagihan_riil = float(tagihan_riil)
    if tagihan_riil <= 0:
        raise ValueError("tagihan_riil harus lebih dari 0 rupiah.")

    bawah = float(rentang_tagihan["tagihan_bawah"])
    atas = float(rentang_tagihan["tagihan_atas"])
    batas_bawah, batas_atas = _wilayah_diharapkan(bawah, atas, toleransi)

    if batas_bawah <= tagihan_riil <= batas_atas:
        arah = ARAH_WAJAR
        selisih_rp = 0.0
        selisih_pct = 0.0
    elif tagihan_riil > batas_atas:
        arah = ARAH_LEBIH_TINGGI
        selisih_rp = tagihan_riil - batas_atas
        selisih_pct = selisih_rp / batas_atas * 100
    else:
        arah = ARAH_LEBIH_RENDAH
        selisih_rp = batas_bawah - tagihan_riil
        selisih_pct = selisih_rp / batas_bawah * 100

    return {
        "is_anomali":        arah != ARAH_WAJAR,
        "arah":              arah,
        "pesan":             _PESAN[arah],
        "tagihan_riil":      round(tagihan_riil, 0),
        "tagihan_bawah":     round(bawah, 0),
        "tagihan_atas":      round(atas, 0),
        "batas_bawah":       round(batas_bawah, 0),
        "batas_atas":        round(batas_atas, 0),
        "selisih_rp":        round(selisih_rp, 0),
        "selisih_pct":       round(selisih_pct, 1),
        "toleransi":         toleransi,
        "rincian_toleransi": rincian_toleransi(),
    }


def ringkas_untuk_pengguna(hasil: dict) -> str:
    """Satu kalimat yang dapat langsung ditampilkan di antarmuka."""
    if not hasil["is_anomali"]:
        return (
            f"Wajar. Tagihan Rp {hasil['tagihan_riil']:,.0f} berada di dalam "
            f"rentang prediksi Rp {hasil['batas_bawah']:,.0f} sampai "
            f"Rp {hasil['batas_atas']:,.0f}."
        )

    if hasil["arah"] == ARAH_LEBIH_TINGGI:
        pembanding = f"di atas batas atas Rp {hasil['batas_atas']:,.0f}"
    else:
        pembanding = f"di bawah batas bawah Rp {hasil['batas_bawah']:,.0f}"

    return (
        f"Anomali terdeteksi. Tagihan Rp {hasil['tagihan_riil']:,.0f} berada "
        f"{hasil['selisih_pct']}% {pembanding}. {hasil['pesan']}"
    )