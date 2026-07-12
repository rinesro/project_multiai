"""
app.py
======
Gerbang masuk sistem EnergiCerdas AI: menerima dan MEMVALIDASI input.

Berkas ini tidak mengimpor Streamlit, FastAPI, atau kerangka apa pun. Isinya
fungsi murni yang mengubah data mentah dari frontend menjadi data bersih yang
siap dihitung Lapis 1. Dengan begitu satu logika validasi yang sama dipakai
oleh:

    - streamlit_app.py   (antarmuka pengujian)
    - FastAPI            (nanti, saat deploy)
    - frontend mana pun  (lewat FastAPI)

Pembagian tanggung jawab:

    app.py                 memeriksa BENTUK data (tipe, rentang, kelengkapan)
    core/kalkulator.py     menghitung, dengan asumsi data sudah bersih
    core/anomaly_detector  memutuskan anomali (pascabayar)

Mengapa validasi di sini, bukan di kalkulator: kalkulator seharusnya boleh
mengasumsikan masukannya sudah benar. Menyaring data kotor adalah tugas gerbang,
sekali di depan, sebelum satu perhitungan pun berjalan. (Satu-satunya
pengecualian adalah validasi_daya_tersambung di kalkulator, yang merupakan
penjaga bug internal, bukan validasi input — lihat docstring-nya.)

Kontrak keluaran: setiap fungsi mengembalikan dict dengan kunci `ok` (bool).
Tidak ada exception yang dilempar untuk kesalahan input pengguna; kesalahan
dikembalikan sebagai data, supaya pemanggil (frontend) dapat menampilkannya.
Exception hanya untuk kesalahan pemrograman.
"""

from core.konstanta import KATEGORI_PERALATAN, GOLONGAN_DAYA


# ══════════════════════════════════════════════════════════════════════════════
# BATAS VALIDASI
# ══════════════════════════════════════════════════════════════════════════════
# Batas kewajaran input. Longgar dengan sengaja — tugasnya menolak yang mustahil
# (tegangan negatif, jam > 24), bukan menghakimi yang tidak biasa.

TEGANGAN_MIN, TEGANGAN_MAKS = 1.0, 1000.0      # volt
ARUS_MIN, ARUS_MAKS = 0.001, 1000.0            # ampere
JAM_MIN, JAM_MAKS = 0.0, 24.0                  # jam per hari (0 eksklusif)
JUMLAH_MIN, JUMLAH_MAKS = 1, 10_000            # unit identik

LUAS_MIN, LUAS_MAKS = 1.0, 100_000.0           # m2
PENGHUNI_MIN, PENGHUNI_MAKS = 1, 1_000         # orang
TAGIHAN_MIN, TAGIHAN_MAKS = 0.0, 1_000_000_000.0   # rupiah
SISA_KWH_MIN, SISA_KWH_MAKS = 0.0, 100_000.0   # kWh di meteran prabayar


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _angka(nilai, nama, minimum, maksimum, boleh_sama_minimum=True):
    """
    Mengubah `nilai` menjadi float dan memeriksa rentang.
    Mengembalikan (nilai_bersih, pesan_error). Salah satunya selalu None.
    """
    try:
        x = float(nilai)
    except (TypeError, ValueError):
        return None, f"{nama} harus berupa angka, bukan {nilai!r}."

    if x != x or x in (float("inf"), float("-inf")):   # NaN / inf
        return None, f"{nama} bukan angka yang valid."

    bawah_ok = x >= minimum if boleh_sama_minimum else x > minimum
    if not bawah_ok or x > maksimum:
        batas_bawah = "" if boleh_sama_minimum else " (eksklusif)"
        return None, (
            f"{nama} = {x:g} di luar rentang wajar "
            f"{minimum:g}{batas_bawah} sampai {maksimum:g}."
        )
    return x, None


def _bilangan_bulat(nilai, nama, minimum, maksimum):
    try:
        x = int(nilai)
    except (TypeError, ValueError):
        return None, f"{nama} harus berupa bilangan bulat, bukan {nilai!r}."
    if not (minimum <= x <= maksimum):
        return None, f"{nama} = {x} di luar rentang {minimum} sampai {maksimum}."
    return x, None


# ══════════════════════════════════════════════════════════════════════════════
# VALIDASI SATU PERALATAN
# ══════════════════════════════════════════════════════════════════════════════

def validasi_peralatan(alat: dict) -> dict:
    """
    Memeriksa dan membersihkan satu entri peralatan dari frontend.

    Masukan yang diharapkan:
        nama      (str, opsional)
        kategori  (str, wajib, salah satu KATEGORI_PERALATAN)
        tegangan  (angka, volt)
        arus      (angka, ampere)
        jam       (angka, jam per hari, 0 < jam <= 24)
        jumlah    (bilangan bulat, opsional, default 1)

    Returns dict:
        ok    : bool
        alat  : dict bersih (bila ok)  — nilai numerik sudah bertipe benar
        error : str (bila tidak ok)
    """
    if not isinstance(alat, dict):
        return {"ok": False, "error": "Entri peralatan harus berupa objek."}

    kategori = alat.get("kategori")
    if kategori not in KATEGORI_PERALATAN:
        return {
            "ok": False,
            "error": f"Kategori '{kategori}' tidak dikenal. Pilih salah satu: "
                     f"{', '.join(KATEGORI_PERALATAN)}.",
        }

    tegangan, e = _angka(alat.get("tegangan"), "Tegangan", TEGANGAN_MIN, TEGANGAN_MAKS)
    if e:
        return {"ok": False, "error": e}

    arus, e = _angka(alat.get("arus"), "Arus", ARUS_MIN, ARUS_MAKS)
    if e:
        return {"ok": False, "error": e}

    jam, e = _angka(alat.get("jam"), "Jam pemakaian", JAM_MIN, JAM_MAKS,
                    boleh_sama_minimum=False)
    if e:
        return {"ok": False, "error": e}

    jumlah, e = _bilangan_bulat(alat.get("jumlah", 1), "Jumlah unit",
                                JUMLAH_MIN, JUMLAH_MAKS)
    if e:
        return {"ok": False, "error": e}

    nama = str(alat.get("nama", "")).strip() or "Tanpa Nama"

    return {
        "ok": True,
        "alat": {
            "nama":     nama,
            "kategori": kategori,
            "tegangan": tegangan,
            "arus":     arus,
            "jam":      jam,
            "jumlah":   jumlah,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# VALIDASI SELURUH PERMINTAAN
# ══════════════════════════════════════════════════════════════════════════════

def validasi_permintaan(data: dict) -> dict:
    """
    Memeriksa satu permintaan analisis lengkap dari frontend.

    Masukan yang diharapkan:
        luas_m2       (angka, m2)
        penghuni      (bilangan bulat)
        daya_va       (bilangan bulat, salah satu GOLONGAN_DAYA)
        is_prabayar   (bool)
        peralatan     (list of dict, minimal 1)

        Untuk PASCABAYAR:
            tagihan_riil  (angka, rupiah)  — untuk deteksi anomali

        Untuk PRABAYAR:
            sisa_kwh      (angka, kWh di meteran)   — opsional
            target_hari   (bilangan bulat)          — opsional, default 30

    Returns dict:
        ok       : bool
        bersih   : dict input bersih (bila ok), siap diteruskan ke Lapis 1
        errors   : list[str] (bila tidak ok) — SEMUA kesalahan, bukan yang pertama
    """
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["Permintaan harus berupa objek."]}

    errors = []
    bersih = {}

    # --- Profil rumah ---
    luas, e = _angka(data.get("luas_m2"), "Luas bangunan", LUAS_MIN, LUAS_MAKS)
    if e:
        errors.append(e)
    else:
        bersih["luas_m2"] = luas

    penghuni, e = _bilangan_bulat(data.get("penghuni"), "Jumlah penghuni",
                                  PENGHUNI_MIN, PENGHUNI_MAKS)
    if e:
        errors.append(e)
    else:
        bersih["penghuni"] = penghuni

    # --- Daya & jenis meteran ---
    daya, e = _bilangan_bulat(data.get("daya_va"), "Daya tersambung", 1, 1_000_000)
    if e:
        errors.append(e)
    elif daya not in GOLONGAN_DAYA:
        errors.append(
            f"Daya {daya} VA bukan golongan resmi. Pilih salah satu: "
            f"{', '.join(str(d) for d in GOLONGAN_DAYA)}."
        )
    else:
        bersih["daya_va"] = daya

    is_prabayar = bool(data.get("is_prabayar", False))
    bersih["is_prabayar"] = is_prabayar

    # --- Peralatan ---
    peralatan = data.get("peralatan")
    if not isinstance(peralatan, list) or len(peralatan) == 0:
        errors.append("Minimal satu peralatan harus diinput.")
    else:
        alat_bersih = []
        for i, alat in enumerate(peralatan, start=1):
            hasil = validasi_peralatan(alat)
            if hasil["ok"]:
                alat_bersih.append(hasil["alat"])
            else:
                errors.append(f"Peralatan ke-{i}: {hasil['error']}")
        if alat_bersih:
            bersih["peralatan"] = alat_bersih

    # --- Bergantung jenis meteran ---
    if is_prabayar:
        # sisa_kwh opsional; hanya divalidasi bila diisi
        if data.get("sisa_kwh") is not None:
            sisa, e = _angka(data.get("sisa_kwh"), "Sisa kWh meteran",
                             SISA_KWH_MIN, SISA_KWH_MAKS)
            if e:
                errors.append(e)
            else:
                bersih["sisa_kwh"] = sisa

        target, e = _bilangan_bulat(data.get("target_hari", 30),
                                    "Target hari", 1, 366)
        if e:
            errors.append(e)
        else:
            bersih["target_hari"] = target
    else:
        # pascabayar: tagihan riil wajib, untuk deteksi anomali
        tagihan, e = _angka(data.get("tagihan_riil"), "Tagihan bulan lalu",
                            TAGIHAN_MIN, TAGIHAN_MAKS, boleh_sama_minimum=False)
        if e:
            errors.append(e)
        else:
            bersih["tagihan_riil"] = tagihan

    if errors:
        return {"ok": False, "errors": errors}
    return {"ok": True, "bersih": bersih}


# ══════════════════════════════════════════════════════════════════════════════
# METADATA UNTUK FRONTEND
# ══════════════════════════════════════════════════════════════════════════════

def opsi_input() -> dict:
    """
    Data pilihan yang dibutuhkan frontend untuk membangun formulir: daftar
    kategori peralatan dan golongan daya. Supaya frontend tidak menuliskannya
    ulang dan berisiko tidak sinkron dengan sistem.
    """
    return {
        "kategori_peralatan": list(KATEGORI_PERALATAN),
        "golongan_daya":      list(GOLONGAN_DAYA),
        "batas": {
            "tegangan": [TEGANGAN_MIN, TEGANGAN_MAKS],
            "arus":     [ARUS_MIN, ARUS_MAKS],
            "jam":      [JAM_MIN, JAM_MAKS],
            "jumlah":   [JUMLAH_MIN, JUMLAH_MAKS],
            "luas_m2":  [LUAS_MIN, LUAS_MAKS],
            "penghuni": [PENGHUNI_MIN, PENGHUNI_MAKS],
        },
    }