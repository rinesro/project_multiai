"""
core/kalkulator.py
==================
Lapis 1 — Seluruh rumus kalkulasi sistem.

Mengimpor hanya `core.konstanta`. Tidak ada konstanta yang didefinisikan di
berkas ini, dan tidak ada keputusan (anomali/tidak, boros/tidak) yang diambil
di berkas ini. Keputusan adalah urusan Lapis 2.

ISI
---
    A. Validasi masukan
    B. Daya per peralatan            V x I -> VA
    C. Energi per peralatan          VA x jam x jumlah x 30 / 1000 -> kVAh
    D. Konversi kVAh -> rentang kWh  memakai batas faktor daya PLN
    E. Tagihan pascabayar            Rekening Minimum sebagai LANTAI
    F. Token prabayar                PPJ dikenakan atas nominal voucer,
                                     proyeksi habis, perencanaan pembelian
    G. Emisi CO2
    H. Profil rumah tangga           IKE dan kWh per penghuni

Berkas `core/token.py` yang lama sudah tidak ada. Seluruh isinya yang berupa
perhitungan dipindahkan ke bagian F. Bagian yang berupa KEPUTUSAN berambang
(label 'kritis'/'waspada'/'aman') tidak dipindahkan ke sini, karena kalkulator
tidak mengambil keputusan — sama seperti deteksi anomali yang kini berdiri
sendiri di core/anomaly_detector.py.

Dua fungsi lama sengaja TIDAK dipindahkan:
    batas_maks_kwh()     bergantung pada angka 720 jam yang belum bersumber
    validasi_pembelian() bergantung pada batas_maks_kwh()
Lihat bagian 10 core/konstanta.py.


DUA HAL YANG PALING MUDAH SALAH
-------------------------------
1. `hitung_daya_semu()` mengembalikan V x I dalam VA. Nilai inilah yang
   dikirim sebagai fitur `Daya_W` ke model DSM (LightGBM), karena model
   dilatih pada V x I. JANGAN dikalikan faktor daya sebelum masuk model.
   Kegagalannya tidak akan memunculkan error apa pun — hanya prediksi yang
   diam-diam salah.

2. Rekening Minimum adalah LANTAI tagihan, bukan komponen tambahan.
   Pada lembar tarif PT PLN, kolom "BIAYA BEBAN (Rp/kVA/bulan)" untuk
   golongan rumah tangga tidak berisi angka rupiah, melainkan tanda *) yang
   merujuk ke catatan Rekening Minimum. RM menggantikan biaya beban.

       SALAH : total = pemakaian + pbjt + biaya_beban
       BENAR : kwh_ditagih = max(kwh, 40 x kVA)
               total       = kwh_ditagih x tarif x (1 + pbjt)

   Untuk 1.300 VA / 200 kWh: cara lama menghasilkan Rp 370.999; yang benar
   Rp 295.875. Selisih 25,4%, dan selisih itulah yang selama ini merambat ke
   deteksi anomali dan membuatnya nyaris selalu positif palsu.


MENGAPA HASILNYA RENTANG, BUKAN SATU ANGKA
------------------------------------------
Sistem hanya menerima tegangan dan arus. Dari keduanya hanya daya SEMU (VA)
yang dapat dihitung. kWh meter PLN mencatat daya NYATA (W = V x I x cos phi),
dan cos phi tidak diketahui.

Alih-alih menebak cos phi tiap peralatan, sistem memakai batas yang ditetapkan
PLN sendiri pada catatan ****) lembar tarif: faktor daya rata-rata bulanan di
bawah 0,85 dikenakan biaya kelebihan daya reaktif. Maka:

    kWh berada di dalam [ kVAh x 0,85 , kVAh x 1,00 ]

Batas bawah dan batas atas keduanya berasal dari regulasi dan hukum fisika,
bukan dari asumsi penulis.
"""

import math
from datetime import date, timedelta

from core.konstanta import (
    TARIF_PER_GOLONGAN, TARIF_DEFAULT,
    JAM_NYALA_RM,
    PBJT_RUMAH_TANGGA,
    FAKTOR_DAYA_MINIMUM_PLN, FAKTOR_DAYA_MAKSIMUM,
    FAKTOR_EMISI_JAMALI_OM, REFERENSI_FAKTOR_EMISI,
    HARI_PER_BULAN, BULAN_PER_TAHUN,
    KATEGORI_PERALATAN,
    DENOMINASI_TOKEN_PLN, KELIPATAN_NOMINAL_TOKEN,
)


# ══════════════════════════════════════════════════════════════════════════════
# A. VALIDASI MASUKAN
# ══════════════════════════════════════════════════════════════════════════════

def validasi_daya_tersambung(daya_va) -> int:
    """
    Menolak daya tersambung yang tidak masuk akal.

    Penjaga kompatibilitas: signature lama `hitung_tagihan(kwh, tarif, pbjt,
    biaya_beban)` mengirim rupiah biaya beban (sering 0.0) di posisi ke-4.
    Di signature sekarang posisi itu adalah daya_va. Tanpa penjaga ini, Python
    menerimanya tanpa keluhan dan tagihan menjadi Rp 0 secara diam-diam.
    """
    try:
        daya = int(daya_va)
    except (TypeError, ValueError):
        raise ValueError(f"daya_va harus bilangan VA, bukan {daya_va!r}.") from None

    if daya <= 0:
        raise ValueError(
            f"daya_va = {daya_va!r} tidak valid. Nilai ini adalah daya "
            "tersambung dalam VA, misalnya 1300. Rekening Minimum adalah "
            "lantai tagihan, bukan komponen yang dijumlahkan."
        )
    return daya


def validasi_peralatan(alat: dict) -> dict:
    """
    Memeriksa satu entri peralatan. Tidak melempar exception — mengembalikan
    status supaya Lapis 2 dapat memutuskan apa yang dilakukan terhadap entri
    yang tidak valid.

    Entri peralatan: nama, kategori, tegangan (V), arus (A), jam (jam/hari),
    dan opsional jumlah (unit identik, default 1).
    """
    pesan = ""

    if alat.get("kategori") not in KATEGORI_PERALATAN:
        pesan = f"Kategori '{alat.get('kategori')}' tidak dikenal."
    elif float(alat.get("tegangan", 0)) <= 0 or float(alat.get("arus", 0)) <= 0:
        pesan = "Tegangan dan arus harus lebih dari 0."
    elif not (0 < float(alat.get("jam", 0)) <= 24):
        pesan = "Jam pemakaian harus lebih dari 0 dan tidak lebih dari 24."
    elif int(alat.get("jumlah", 1)) < 1:
        pesan = "Jumlah unit minimal 1."

    return {"valid": pesan == "", "pesan": pesan}


# ══════════════════════════════════════════════════════════════════════════════
# B. DAYA PER PERALATAN
# ══════════════════════════════════════════════════════════════════════════════

def hitung_daya_semu(tegangan: float, arus: float) -> float:
    """
    Daya semu satu unit peralatan, satuan VA.

        S = V x I

    Nilai ini dikirim sebagai fitur ke model DSM. JANGAN dikalikan cos phi.
    """
    return round(float(tegangan) * float(arus), 2)


# Alias. Model DSM dan kode lama memanggilnya `hitung_watt`. Namanya secara
# teknis keliru (satuannya VA, bukan W), tetapi nilainya benar untuk fitur
# model. Dipertahankan agar tidak memecah models/dsm_classifier.py.
hitung_watt = hitung_daya_semu


def hitung_daya_semu_total(alat: dict) -> float:
    """Daya semu seluruh unit pada satu entri peralatan (VA)."""
    satu_unit = hitung_daya_semu(alat["tegangan"], alat["arus"])
    return round(satu_unit * int(alat.get("jumlah", 1)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# C. ENERGI PER PERALATAN
# ══════════════════════════════════════════════════════════════════════════════

def hitung_kvah_alat(daya_semu_va: float, jam: float, jumlah: int = 1) -> float:
    """
    Energi semu bulanan satu entri peralatan, satuan kVAh per bulan.

        kVAh = VA x jam x jumlah x 30 / 1000

    `jumlah` mengalikan energi TOTAL. Fitur yang dikirim ke model DSM tetap
    per-unit, supaya nilainya tidak keluar dari rentang data latih.
    """
    return round(
        float(daya_semu_va) * float(jam) * int(jumlah) * HARI_PER_BULAN / 1000,
        4,
    )


# Alias untuk kode lama. Perhatikan: satuannya kVAh, bukan kWh.
hitung_kwh_alat = hitung_kvah_alat


def total_kvah(daftar_alat: list) -> float:
    """Menjumlahkan energi semu bulanan seluruh peralatan (kVAh/bulan)."""
    total = 0.0
    for alat in daftar_alat:
        va = hitung_daya_semu(alat["tegangan"], alat["arus"])
        total += hitung_kvah_alat(va, alat["jam"], alat.get("jumlah", 1))
    return round(total, 3)


def rincian_peralatan(daftar_alat: list) -> list:
    """
    Menghitung sekali untuk seluruh peralatan, lalu meneruskannya ke Lapis 2
    dan 3. Lapis lain TIDAK boleh menghitung ulang dari tegangan dan arus.

    Setiap keluaran memuat:
        nama, kategori, tegangan, arus, jam, jumlah,
        daya_semu_va      : per unit, fitur model DSM
        daya_semu_total_va: seluruh unit
        kvah_bulan        : energi semu bulanan
        kwh_bawah, kwh_atas : rentang energi nyata bulanan
        valid, pesan
    """
    hasil = []
    for i, alat in enumerate(daftar_alat):
        status = validasi_peralatan(alat)
        jumlah = int(alat.get("jumlah", 1))
        va = hitung_daya_semu(alat.get("tegangan", 0), alat.get("arus", 0))
        kvah = hitung_kvah_alat(va, alat.get("jam", 0), jumlah)

        hasil.append({
            "nama":               alat.get("nama", f"Peralatan {i + 1}"),
            "kategori":           alat.get("kategori", ""),
            "tegangan":           alat.get("tegangan", 0),
            "arus":               alat.get("arus", 0),
            "jam":                alat.get("jam", 0),
            "jumlah":             jumlah,
            "daya_semu_va":       va,
            "daya_semu_total_va": round(va * jumlah, 2),
            "kvah_bulan":         kvah,
            "kwh_bawah":          round(kvah * FAKTOR_DAYA_MINIMUM_PLN, 4),
            "kwh_atas":           round(kvah * FAKTOR_DAYA_MAKSIMUM, 4),
            "valid":              status["valid"],
            "pesan":              status["pesan"],
        })
    return hasil


# ══════════════════════════════════════════════════════════════════════════════
# D. KONVERSI kVAh -> RENTANG kWh
# ══════════════════════════════════════════════════════════════════════════════

def rentang_kwh(kvah: float) -> tuple:
    """
    (kwh_bawah, kwh_atas) dari energi semu.

        bawah = kVAh x 0,85   faktor daya minimum menurut PT PLN
        atas  = kVAh x 1,00   beban resistif murni, cos phi tidak bisa > 1
    """
    kvah = float(kvah)
    return (
        round(kvah * FAKTOR_DAYA_MINIMUM_PLN, 4),
        round(kvah * FAKTOR_DAYA_MAKSIMUM, 4),
    )


# ══════════════════════════════════════════════════════════════════════════════
# E. TAGIHAN PASCABAYAR
# ══════════════════════════════════════════════════════════════════════════════

def get_tarif(daya_va: int) -> float:
    """Tarif Rp/kWh menurut daya tersambung."""
    daya = validasi_daya_tersambung(daya_va)
    for (lo, hi), tarif in TARIF_PER_GOLONGAN.items():
        if lo <= daya <= hi:
            return tarif
    return TARIF_DEFAULT


def hitung_jam_nyala(kwh: float, daya_va: int) -> float:
    """Jam nyala = kWh per bulan dibagi kVA tersambung."""
    daya = validasi_daya_tersambung(daya_va)
    return round(float(kwh) / (daya / 1000.0), 1)


def hitung_kwh_minimum(daya_va: int) -> float:
    """
    Batas kWh minimum yang ditagihkan.

        RM1 = 40 jam nyala x daya tersambung (kVA)

    Contoh: 1.300 VA -> 40 x 1,3 = 52 kWh.

    Satuannya kWh, bukan rupiah. Ini disengaja: nilai rupiah RM tidak boleh
    dijumlahkan ke tagihan, dan mengembalikan kWh menutup peluang itu.
    """
    daya = validasi_daya_tersambung(daya_va)
    return JAM_NYALA_RM * (daya / 1000.0)


def hitung_rm_rupiah(daya_va: int) -> float:
    """
    Nilai rupiah Rekening Minimum. Hanya untuk ditampilkan sebagai informasi
    ("tagihan tidak akan di bawah Rp X"). Tidak pernah dijumlahkan.
    """
    daya = validasi_daya_tersambung(daya_va)
    return round(hitung_kwh_minimum(daya) * get_tarif(daya), 0)


def hitung_tagihan(kwh: float, daya_va: int,
                   pbjt: float = PBJT_RUMAH_TANGGA) -> dict:
    """
    Tagihan pascabayar untuk sejumlah kWh.

        jam_nyala   = kWh / kVA
        kwh_ditagih = kwh        bila jam_nyala >= 40
                    = 40 x kVA   bila jam_nyala <  40
        total       = kwh_ditagih x tarif x (1 + pbjt)

    Dasar pengenaan PBJT untuk pascabayar adalah biaya beban tetap ditambah
    biaya pemakaian kWh. Golongan rumah tangga tidak dikenakan biaya beban,
    sehingga dasarnya adalah biaya pemakaian saja.
    """
    daya = validasi_daya_tersambung(daya_va)
    kwh = float(kwh)
    tarif = get_tarif(daya)

    jam_nyala = hitung_jam_nyala(kwh, daya)
    kwh_minimum = hitung_kwh_minimum(daya)
    kena_rm = jam_nyala < JAM_NYALA_RM
    kwh_ditagih = kwh_minimum if kena_rm else kwh

    biaya_pemakaian = kwh_ditagih * tarif
    biaya_pbjt = biaya_pemakaian * pbjt

    return {
        "daya_va":         daya,
        "tarif":           tarif,
        "kwh":             round(kwh, 3),
        "jam_nyala":       jam_nyala,
        "kwh_minimum":     round(kwh_minimum, 2),
        "kena_rm":         kena_rm,
        "kwh_ditagih":     round(kwh_ditagih, 3),
        "biaya_pemakaian": round(biaya_pemakaian, 0),
        "biaya_pbjt":      round(biaya_pbjt, 0),
        "total":           round(biaya_pemakaian + biaya_pbjt, 0),
    }


def hitung_rentang_tagihan(daftar_alat: list, daya_va: int,
                           pbjt: float = PBJT_RUMAH_TANGGA) -> dict:
    """
    Prediksi tagihan pascabayar sebagai RENTANG. Inilah keluaran yang
    dikonsumsi core/anomaly_detector.py.

    Batas bawah memakai faktor daya 0,85; batas atas memakai 1,00.
    """
    kvah = total_kvah(daftar_alat)
    kwh_bawah, kwh_atas = rentang_kwh(kvah)

    bawah = hitung_tagihan(kwh_bawah, daya_va, pbjt)
    atas = hitung_tagihan(kwh_atas, daya_va, pbjt)

    return {
        "kvah":            kvah,
        "kwh_bawah":       kwh_bawah,
        "kwh_atas":        kwh_atas,
        "tagihan_bawah":   bawah["total"],
        "tagihan_atas":    atas["total"],
        "tagihan_tengah":  round((bawah["total"] + atas["total"]) / 2, 0),
        "lebar_pct":       round(
            (atas["total"] - bawah["total"]) / max(1.0, bawah["total"]) * 100, 1
        ),
        "rincian_bawah":   bawah,
        "rincian_atas":    atas,
    }


# ══════════════════════════════════════════════════════════════════════════════
# F. TOKEN PRABAYAR
# ══════════════════════════════════════════════════════════════════════════════
#
# Dasar pengenaan PBJT untuk prabayar adalah nilai rupiah voucer, bukan biaya
# pemakaian. Karena itu:
#
#       PPJ  = tarif_pbjt x Nominal Token
#       kWh  = (Nominal Token - PPJ) / TDL
#
# Prabayar tidak dikenakan Rekening Minimum.
#
# Asumsi yang dinyatakan terbuka:
#   Biaya admin kanal pembayaran tidak mengurangi nilai stroom, sehingga tidak
#   memengaruhi kWh yang diperoleh. Bila ingin ditampilkan, perlakukan sebagai
#   komponen terpisah.

def hitung_kwh_dari_token(nominal_rp: float, daya_va: int,
                          pbjt: float = PBJT_RUMAH_TANGGA,
                          biaya_transaksi: float = 0.0) -> dict:
    """
    kWh yang diperoleh dari sejumlah nominal token.

    `biaya_transaksi` adalah biaya admin kanal pembayaran. Ia TIDAK mengurangi
    nilai stroom, sehingga tidak memengaruhi kWh yang diperoleh. Dilaporkan
    terpisah sebagai uang yang keluar dari kantong pengguna.

    Tarif diturunkan dari daya tersambung, bukan diterima sebagai parameter.
    Ini menutup peluang pemanggil mengirim tarif yang tidak cocok dengan
    golongannya.
    """
    tarif = get_tarif(daya_va)
    nominal_rp = float(nominal_rp)

    ppj = nominal_rp * pbjt
    stroom = nominal_rp - ppj
    kwh_exact = stroom / tarif

    return {
        "nominal_rp":      round(nominal_rp, 0),
        "tarif":           tarif,
        "ppj_rp":          round(ppj, 0),
        "stroom_rp":       round(stroom, 0),
        "kwh":             round(kwh_exact, 2),   # untuk ditampilkan
        "kwh_exact":       kwh_exact,             # untuk perhitungan lanjutan
        "biaya_transaksi": round(float(biaya_transaksi), 0),
        "total_dibayar":   round(nominal_rp + float(biaya_transaksi), 0),
    }


def hitung_token_dari_kwh(kwh: float, daya_va: int,
                          pbjt: float = PBJT_RUMAH_TANGGA) -> float:
    """
    Nominal token yang harus dibeli untuk memperoleh sejumlah kWh.
    Merupakan inverse eksak dari hitung_kwh_dari_token().

        nominal = (kWh x TDL) / (1 - pbjt)
    """
    if float(kwh) <= 0:
        return 0.0
    return round((float(kwh) * get_tarif(daya_va)) / (1.0 - pbjt), 0)


def hitung_rentang_token(daftar_alat: list, daya_va: int,
                         pbjt: float = PBJT_RUMAH_TANGGA) -> dict:
    """
    Nominal token bulanan yang dibutuhkan peralatan, sebagai RENTANG.

    Dipakai Lapis 3 untuk menghitung penghematan token setelah optimasi.
    Deteksi anomali TIDAK memakai keluaran ini — anomali hanya berlaku pada
    pelanggan pascabayar.
    """
    kvah = total_kvah(daftar_alat)
    kwh_bawah, kwh_atas = rentang_kwh(kvah)

    token_bawah = hitung_token_dari_kwh(kwh_bawah, daya_va, pbjt)
    token_atas = hitung_token_dari_kwh(kwh_atas, daya_va, pbjt)

    return {
        "kvah":         kvah,
        "kwh_bawah":    kwh_bawah,
        "kwh_atas":     kwh_atas,
        "token_bawah":  token_bawah,
        "token_atas":   token_atas,
        "token_tengah": round((token_bawah + token_atas) / 2, 0),
    }


def kwh_per_hari(kwh_bulan: float) -> float:
    """Laju konsumsi harian dari konsumsi bulanan."""
    return round(float(kwh_bulan) / HARI_PER_BULAN, 4)


def hitung_hari_tersisa(sisa_kwh: float, laju_kwh_per_hari: float) -> float:
    """Berapa hari sisa kWh di meteran bertahan pada laju konsumsi tertentu."""
    if float(laju_kwh_per_hari) <= 0:
        raise ValueError("laju_kwh_per_hari harus lebih dari 0.")
    if float(sisa_kwh) < 0:
        raise ValueError("sisa_kwh tidak boleh negatif.")
    return round(float(sisa_kwh) / float(laju_kwh_per_hari), 1)


def proyeksi_habis_token(sisa_kwh: float, laju_kwh_per_hari: float,
                         tanggal_baca: date = None) -> dict:
    """
    Kapan sisa kWh habis bila pola pemakaian tidak berubah.

    Fungsi ini hanya MENGHITUNG. Ia tidak memberi label 'kritis', 'waspada',
    atau 'aman' — itu keputusan berambang, dan ambangnya belum punya sumber.
    Keputusan semacam itu adalah urusan Lapis 2, sebagaimana deteksi anomali.

    Hari dibulatkan ke BAWAH (math.floor) supaya peringatan muncul lebih awal,
    bukan terlambat. Sisa 2,9 hari dilaporkan sebagai tanggal habis di hari
    ke-2, bukan hari ke-3.
    """
    hari = hitung_hari_tersisa(sisa_kwh, laju_kwh_per_hari)
    tanggal_baca = tanggal_baca or date.today()

    return {
        "sisa_kwh":      round(float(sisa_kwh), 2),
        "laju_kwh_hari": round(float(laju_kwh_per_hari), 4),
        "hari_tersisa":  hari,
        "tanggal_habis": tanggal_baca + timedelta(days=math.floor(hari)),
    }


def hari_sampai_ambang(sisa_kwh: float, laju_kwh_per_hari: float,
                       ambang_kwh: float) -> float:
    """
    Berapa hari lagi sisa kWh menyentuh sebuah ambang (mis. saat meteran mulai
    berbunyi).

    `ambang_kwh` WAJIB diisi pemanggil. Ambang alarm berbeda antar merek meter
    dan dapat dikonfigurasi, sehingga tidak boleh menjadi konstanta sistem.
    """
    if float(ambang_kwh) < 0:
        raise ValueError("ambang_kwh tidak boleh negatif.")
    sisa_di_atas_ambang = max(0.0, float(sisa_kwh) - float(ambang_kwh))
    return hitung_hari_tersisa(sisa_di_atas_ambang, laju_kwh_per_hari)


def token_untuk_bertahan(laju_kwh_per_hari: float, jumlah_hari: int,
                         daya_va: int, sisa_kwh: float = 0.0,
                         pbjt: float = PBJT_RUMAH_TANGGA) -> dict:
    """Nominal token yang harus dibeli agar bertahan `jumlah_hari` hari."""
    kwh_butuh = max(0.0, float(laju_kwh_per_hari) * int(jumlah_hari) - float(sisa_kwh))
    return {
        "jumlah_hari": int(jumlah_hari),
        "kwh_butuh":   round(kwh_butuh, 2),
        "nominal_pas": hitung_token_dari_kwh(kwh_butuh, daya_va, pbjt),
    }


def bulatkan_nominal(nominal_rp: float,
                     kelipatan: int = KELIPATAN_NOMINAL_TOKEN) -> float:
    """
    Bulatkan KE ATAS ke kelipatan terdekat, supaya proyeksi hari bertahan tidak
    meleset ke bawah. Aritmetika murni.
    """
    if float(nominal_rp) <= 0:
        return 0.0
    return float(math.ceil(round(float(nominal_rp)) / kelipatan) * kelipatan)


def pecah_denominasi(nominal_rp: float,
                     denominasi: tuple = DENOMINASI_TOKEN_PLN) -> list:
    """
    Pecah kebutuhan menjadi kombinasi denominasi yang dijual PLN, secara greedy.
    Contoh: Rp 710.000 -> [500.000, 200.000, 20.000].

    Sisa di bawah denominasi terkecil dinaikkan ke denominasi terkecil, sehingga
    jumlah hasil selalu >= nominal yang diminta.
    """
    sisa = int(round(float(nominal_rp)))
    if sisa <= 0:
        return []

    hasil = []
    for d in sorted(denominasi, reverse=True):
        while sisa >= d:
            hasil.append(d)
            sisa -= d
    if sisa > 0:
        hasil.append(min(denominasi))
    return hasil


# ══════════════════════════════════════════════════════════════════════════════
# G. EMISI CO2
# ══════════════════════════════════════════════════════════════════════════════

def hitung_emisi(kwh: float) -> dict:
    """Emisi CO2 dari konsumsi listrik. Faktor emisi Grid Jamali OM."""
    emisi_bulan = round(float(kwh) * FAKTOR_EMISI_JAMALI_OM, 3)
    return {
        "emisi_kg_bulan": emisi_bulan,
        "emisi_kg_tahun": round(emisi_bulan * BULAN_PER_TAHUN, 2),
        "faktor_emisi":   FAKTOR_EMISI_JAMALI_OM,
        "referensi":      REFERENSI_FAKTOR_EMISI,
    }


def hitung_rentang_emisi(kwh_bawah: float, kwh_atas: float) -> dict:
    """Emisi sebagai rentang, mengikuti rentang konsumsi."""
    return {
        "emisi_bawah_kg_bulan": hitung_emisi(kwh_bawah)["emisi_kg_bulan"],
        "emisi_atas_kg_bulan":  hitung_emisi(kwh_atas)["emisi_kg_bulan"],
        "faktor_emisi":         FAKTOR_EMISI_JAMALI_OM,
        "referensi":            REFERENSI_FAKTOR_EMISI,
    }


# ══════════════════════════════════════════════════════════════════════════════
# H. PROFIL RUMAH TANGGA
# ══════════════════════════════════════════════════════════════════════════════

def hitung_ike(kwh: float, luas_m2: float) -> float:
    """
    Intensitas Konsumsi Energi, satuan kWh/m2/bulan.
    Pembagi dibatasi minimal 1,0 m2 agar tidak membagi nol.
    """
    return round(float(kwh) / max(1.0, float(luas_m2)), 4)


def hitung_kwh_per_org(kwh: float, penghuni: int) -> float:
    """Konsumsi per penghuni, satuan kWh/orang/bulan."""
    return round(float(kwh) / max(1, int(penghuni)), 2)


def hitung_ike_tahunan(kwh_bulan: float, luas_m2: float) -> float:
    """
    IKE dalam kWh/m2/TAHUN, satuan yang dipakai Lampiran II Permen ESDM
    No.3/2025. Disediakan untuk pembandingan di dokumen penelitian; sistem
    tidak memakainya sebagai ambang.
    """
    return round(
        float(kwh_bulan) * BULAN_PER_TAHUN / max(1.0, float(luas_m2)), 2
    )