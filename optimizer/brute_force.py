"""
optimizer/brute_force.py
=========================
Brute Force Optimizer untuk EnergiCerdas AI — Lapis 3.

Fokus: IKE. Mencari komposisi jam pemakaian peralatan yang menurunkan IKE
sampai menyentuh zona Efisien (atau Cukup Efisien sebagai fallback), lalu
menghitung tagihan/token yang perlu dibayar untuk komposisi hasil optimasi.

Cara kerja (TIDAK berubah dari versi sebelumnya):
    1. Hanya aktif jika IKE user berada di zona Cukup Efisien ke atas
       (IKE >= batas Efisien). Zona Sangat Efisien dan Efisien dilewati.
    2. Hanya memproses peralatan berlabel 'Fleksibel'.
    3. Greedy: kurangi jam pemakaian satu langkah per iterasi, mulai dari
       peralatan dengan energi terbesar.
    4. Target pertama : zona Efisien. Fallback: zona Cukup Efisien.
    5. Output: langkah per peralatan + selisih biaya + selisih emisi.

Strategi pengurangan:
    Step pengurangan : 0,5 jam per iterasi
    Batas minimum    : 0,5 jam (tidak boleh nol)
    Maks pengurangan : 50% dari jam asal per peralatan


YANG BERUBAH dari versi sebelumnya, dan alasannya
-------------------------------------------------
Hanya empat hal, semuanya karena Lapis 1 sudah dirombak. Algoritma greedy,
ambang IKE, urutan target, dan strategi pengurangan TETAP SAMA.

1. Impor dari core.kalkulator (dulu core.kalkulasi). Modul berganti nama.
   Hack sys.path dihapus; paket sudah proper.

2. hitung_tagihan(kwh, daya_va, pbjt) menggantikan
   hitung_tagihan(kwh, tarif, pbjt, biaya_beban). Rekening Minimum kini
   LANTAI tagihan, bukan komponen tambahan, sehingga biaya_beban tidak ada.

3. Field peralatan mengikuti keluaran Lapis 1 baru:
       watt      -> daya_semu_va   (V x I, satuan VA)
       kwh_bulan -> kvah_bulan     (energi semu bulanan, kVAh)

4. TAMBAHAN yang diminta: biaya yang perlu dibayar untuk komposisi hasil
   optimasi. Pascabayar -> tagihan; prabayar -> nominal token. Keduanya
   disajikan sebagai RENTANG, karena sistem hanya tahu daya semu sedangkan
   kWh meter mencatat daya nyata (lihat catatan faktor daya di bawah).


CATATAN FAKTOR DAYA (mengapa biaya berupa rentang)
--------------------------------------------------
Optimizer bekerja pada energi SEMU (kVAh = V x I x jam), sama seperti versi
asli. IKE dan pencarian greedy memakai basis ini — ini pilihan konservatif:
energi semu adalah batas atas energi nyata (karena kWh nyata = kVAh x cos phi,
dan cos phi <= 1), sehingga IKE tidak pernah diremehkan dan optimizer tidak
berhenti terlalu dini.

Untuk BIAYA yang perlu dibayar, energi semu dikonversi ke rentang kWh nyata
memakai batas faktor daya PLN (catatan ****) lembar tarif): kWh nyata berada
di antara kVAh x 0,85 dan kVAh x 1,00. Batas atas rentang biaya (cos phi = 1)
sama dengan angka tunggal yang dihitung versi asli.


Referensi:
    [1] Penetapan Penyesuaian Tarif Tenaga Listrik April-Juni 2026 — PT PLN.
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO2/kWh — ESDM 2019.
    [3] PBJT 2,4% — Perda DKI Jakarta No.1/2024.
    [5] Standar IKE — Pedoman Konservasi Energi Depdiknas RI 2004.
"""

from copy import deepcopy

from core.konstanta import (
    BATAS_IKE_DEPDIKNAS,
    FAKTOR_EMISI_JAMALI_OM,
    HARI_PER_BULAN,
)
from core.kalkulator import (
    hitung_tagihan, hitung_token_dari_kwh, hitung_emisi,
    rentang_kwh, get_tarif,
)

STEP_JAM        = 0.5    # step pengurangan jam
MAKS_KURANG_PCT = 0.50   # maksimum pengurangan 50% dari jam asal
MIN_JAM         = 0.5    # minimum jam operasi per peralatan


# ── Helper: ambang & label IKE ────────────────────────────────────────────────

def _get_batas(ada_ac: bool, zona: str) -> float:
    """Ambil batas IKE Depdiknas berdasarkan kondisi AC dan nama zona."""
    return BATAS_IKE_DEPDIKNAS["ac" if ada_ac else "tidak_ac"][zona]


def _label_zona(ike: float, ada_ac: bool) -> str:
    """
    Tentukan label zona IKE dari nilai numerik. Ambang Depdiknas.
    Logika sama persis dengan versi sebelumnya; ambang kini bersumber dari
    core.konstanta agar tidak ada dua salinan angka.
    """
    b = BATAS_IKE_DEPDIKNAS["ac" if ada_ac else "tidak_ac"]
    if ike < b["sangat_efisien"]: return "Sangat Efisien"
    if ike < b["efisien"]:        return "Efisien"
    if ike <= b["cukup_efisien"]: return "Cukup Efisien"
    if ike <= b["boros"]:         return "Boros"
    return "Sangat Boros"


# ── Helper: energi ────────────────────────────────────────────────────────────

def _kvah_flex(alat: dict) -> float:
    """Energi semu bulanan satu peralatan fleksibel pada jam_saat_ini (kVAh)."""
    return round(
        alat["daya_semu_va"] * alat["jam_saat_ini"] * alat.get("jumlah", 1)
        * HARI_PER_BULAN / 1000,
        4,
    )


def _total_kvah(alat_tetap: list, alat_flex: list) -> float:
    """
    Total energi semu bulanan (kVAh) dari semua peralatan.

    Peralatan fleksibel dihitung ulang tiap iterasi karena Lapis 3 mencoba
    skenario jam HIPOTETIS (mis. "kalau AC jadi 7 jam?"). Peralatan tetap
    memakai kvah_bulan yang sudah dihitung Lapis 1 — tidak dihitung ulang.
    """
    kvah_tetap = sum(a["kvah_bulan"] for a in alat_tetap)
    kvah_flex = sum(_kvah_flex(a) for a in alat_flex)
    return round(kvah_tetap + kvah_flex, 3)


# ── Helper: biaya (tagihan pascabayar / token prabayar) ──────────────────────

def _biaya_titik(kwh: float, daya_va: int, pbjt: float, is_prabayar: bool) -> float:
    """Biaya untuk satu nilai kWh. Prabayar -> nominal token; pascabayar -> tagihan."""
    if is_prabayar:
        return hitung_token_dari_kwh(kwh, daya_va, pbjt)
    return hitung_tagihan(kwh, daya_va, pbjt)["total"]


def _biaya_rentang(kvah: float, daya_va: int, pbjt: float,
                   is_prabayar: bool) -> dict:
    """
    Biaya yang perlu dibayar untuk energi semu tertentu, sebagai RENTANG.

        batas bawah : cos phi = 0,85 (faktor daya minimum PLN)
        batas atas  : cos phi = 1,00 (beban resistif; = angka versi asli)
    """
    kwh_bawah, kwh_atas = rentang_kwh(kvah)
    bawah = _biaya_titik(kwh_bawah, daya_va, pbjt, is_prabayar)
    atas = _biaya_titik(kwh_atas, daya_va, pbjt, is_prabayar)
    return {
        "jenis":     "token" if is_prabayar else "tagihan",
        "kwh_bawah": kwh_bawah,
        "kwh_atas":  kwh_atas,
        "bawah":     bawah,
        "atas":      atas,
        "tengah":    round((bawah + atas) / 2, 0),
    }


def _emisi_rentang(kvah: float) -> dict:
    """Emisi CO2 sebagai rentang, mengikuti rentang kWh nyata."""
    kwh_bawah, kwh_atas = rentang_kwh(kvah)
    return {
        "bawah":  hitung_emisi(kwh_bawah)["emisi_kg_bulan"],
        "atas":   hitung_emisi(kwh_atas)["emisi_kg_bulan"],
        "tengah": round(
            (hitung_emisi(kwh_bawah)["emisi_kg_bulan"]
             + hitung_emisi(kwh_atas)["emisi_kg_bulan"]) / 2, 3),
    }


def _rate_per_kvah(daya_va: int, pbjt: float, is_prabayar: bool) -> float:
    """
    Biaya marginal per 1 kVAh (basis cos phi = 1) untuk perhitungan per-langkah.
    Prabayar memakai /(1-pbjt) karena PBJT atas nominal voucer; pascabayar
    memakai x(1+pbjt) karena PBJT atas biaya pemakaian.
    """
    tarif = get_tarif(daya_va)
    return tarif / (1 - pbjt) if is_prabayar else tarif * (1 + pbjt)


# ── Greedy optimizer (logika TIDAK berubah) ──────────────────────────────────

def _greedy(alat_tetap: list, alat_flex: list,
            luas_m2: float, ada_ac: bool, zona_target: str) -> tuple:
    """
    Greedy reduction: kurangi jam peralatan satu langkah per iterasi, mulai
    dari energi terbesar, sampai IKE target tercapai atau semua peralatan
    sudah di batas minimum.

    Returns (berhasil: bool, state_akhir: list)
    """
    batas = _get_batas(ada_ac, zona_target)
    state = deepcopy(alat_flex)

    # Urutkan dari energi terbesar — dampak terbesar dulu
    state.sort(key=lambda x: x["kvah_bulan"], reverse=True)

    while True:
        ike_kini = _total_kvah(alat_tetap, state) / max(1.0, luas_m2)
        if ike_kini <= batas:
            return True, state

        bisa_kurang = [a for a in state if a["jam_saat_ini"] > a["jam_minimum"]]
        if not bisa_kurang:
            return False, state

        for alat in state:
            if alat["jam_saat_ini"] > alat["jam_minimum"]:
                alat["jam_saat_ini"] = max(
                    round(alat["jam_saat_ini"] - STEP_JAM, 1),
                    alat["jam_minimum"],
                )
                break


# ── Builder hasil (dipakai jalur aktif maupun tidak aktif) ───────────────────

def _bungkus_hasil(aktif, status, zona_awal, zona_akhir, ike_awal, ike_akhir,
                   target_ike, kvah_awal, kvah_akhir, daya_va, pbjt,
                   is_prabayar, langkah, pesan) -> dict:
    biaya_awal = _biaya_rentang(kvah_awal, daya_va, pbjt, is_prabayar)
    biaya_akhir = _biaya_rentang(kvah_akhir, daya_va, pbjt, is_prabayar)
    emisi_awal = _emisi_rentang(kvah_awal)
    emisi_akhir = _emisi_rentang(kvah_akhir)

    # Penghematan dihitung pada basis cos phi = 1 (batas atas), sehingga
    # konsisten dengan penjumlahan langkah per peralatan.
    hemat_kvah = round(kvah_awal - kvah_akhir, 3)
    hemat_rp = round(biaya_awal["atas"] - biaya_akhir["atas"], 0)
    hemat_emisi = round(emisi_awal["atas"] - emisi_akhir["atas"], 3)

    pct_rp = round(hemat_rp / max(1.0, biaya_awal["atas"]) * 100, 1)
    pct_emisi = round(hemat_emisi / max(1e-9, emisi_awal["atas"]) * 100, 1)

    return {
        "aktif":              aktif,
        "status":             status,
        "is_prabayar":        is_prabayar,
        "zona_awal":          zona_awal,
        "zona_akhir":         zona_akhir,
        "ike_awal":           ike_awal,
        "ike_akhir":          ike_akhir,
        "target_ike":         target_ike,
        "kvah_awal":          kvah_awal,
        "kvah_akhir":         kvah_akhir,
        "biaya_awal":         biaya_awal,
        "biaya_akhir":        biaya_akhir,
        "emisi_awal":         emisi_awal,
        "emisi_akhir":        emisi_akhir,
        "hemat_kvah":         hemat_kvah,
        "hemat_rp":           int(hemat_rp),
        "hemat_emisi_kg":     hemat_emisi,
        "persen_hemat_rp":    pct_rp,
        "persen_hemat_emisi": pct_emisi,
        "langkah":            langkah,
        "pesan":              pesan,
    }


# ── Fungsi utama ──────────────────────────────────────────────────────────────

def optimasi(ringkasan_dsm: dict,
             luas_m2: float,
             ada_ac: bool,
             daya_va: int,
             pbjt: float,
             is_prabayar: bool) -> dict:
    """
    Brute force optimizer berbasis target IKE Depdiknas.

    Parameters
    ----------
    ringkasan_dsm : dict
        Keluaran DSMClassifier.ringkasan_dsm(), memuat 'fleksibel' dan
        'tidak_fleksibel'. Tiap peralatan wajib punya: nama, kategori, jam,
        jumlah, daya_semu_va, kvah_bulan.
    luas_m2 : float
        Luas bangunan (m2).
    ada_ac : bool
        True jika ada peralatan kategori Pendingin.
    daya_va : int
        Daya tersambung, untuk tarif dan Rekening Minimum.
    pbjt : float
        Tarif PBJT (0,024 untuk rumah tangga Jakarta).
    is_prabayar : bool
        True -> biaya dihitung sebagai nominal token; False -> tagihan.

    Returns
    -------
    dict. Kunci utama:
        aktif, status, zona_awal, zona_akhir, ike_awal, ike_akhir, target_ike,
        kvah_awal, kvah_akhir,
        biaya_awal, biaya_akhir : {jenis, bawah, atas, tengah, kwh_bawah, kwh_atas}
        emisi_awal, emisi_akhir : {bawah, atas, tengah}
        hemat_kvah, hemat_rp, hemat_emisi_kg, persen_hemat_rp, persen_hemat_emisi
        langkah : list rekomendasi per peralatan
        pesan

    status: 'sudah_efisien' | 'efisien' | 'cukup_efisien' | 'tidak_tercapai'
    """
    alat_tetap = ringkasan_dsm.get("tidak_fleksibel", [])
    alat_raw = ringkasan_dsm.get("fleksibel", [])

    # Energi & IKE awal (basis energi semu, seperti versi asli)
    kvah_awal = round(
        sum(a["kvah_bulan"] for a in alat_tetap)
        + sum(a["kvah_bulan"] for a in alat_raw),
        3,
    )
    ike_awal = round(kvah_awal / max(1.0, luas_m2), 4)
    zona_awal = _label_zona(ike_awal, ada_ac)

    batas_cukup = _get_batas(ada_ac, "cukup_efisien")
    batas_efisien = _get_batas(ada_ac, "efisien")

    # ── Gerbang aktivasi: SKIP jika sudah Efisien / Sangat Efisien ───────────
    if ike_awal < batas_efisien:
        return _bungkus_hasil(
            False, "sudah_efisien", zona_awal, zona_awal, ike_awal, ike_awal,
            batas_cukup, kvah_awal, kvah_awal, daya_va, pbjt, is_prabayar,
            [], f"Konsumsi sudah dalam zona {zona_awal}. Tidak perlu optimasi.",
        )

    # ── Tidak ada peralatan fleksibel ────────────────────────────────────────
    if not alat_raw:
        return _bungkus_hasil(
            True, "tidak_tercapai", zona_awal, zona_awal, ike_awal, ike_awal,
            batas_cukup, kvah_awal, kvah_awal, daya_va, pbjt, is_prabayar,
            [], "Semua peralatan Tidak Fleksibel. Tidak ada yang bisa "
                "dioptimasi secara otomatis.",
        )

    # ── Siapkan peralatan fleksibel ──────────────────────────────────────────
    alat_flex = []
    for a in alat_raw:
        item = deepcopy(a)
        item["jam_awal"] = a["jam"]
        item["jam_saat_ini"] = a["jam"]
        item["jam_minimum"] = max(MIN_JAM, round(a["jam"] * (1 - MAKS_KURANG_PCT), 1))
        alat_flex.append(item)

    # ── Iterasi 1: target Efisien ────────────────────────────────────────────
    berhasil, state = _greedy(alat_tetap, alat_flex, luas_m2, ada_ac, "efisien")
    status, target_zona = "efisien", "efisien"

    # ── Iterasi 2: fallback Cukup Efisien ────────────────────────────────────
    if not berhasil:
        berhasil, state = _greedy(alat_tetap, alat_flex, luas_m2, ada_ac, "cukup_efisien")
        status = "cukup_efisien" if berhasil else "tidak_tercapai"
        target_zona = "cukup_efisien"

    # ── Hasil akhir ──────────────────────────────────────────────────────────
    kvah_akhir = _total_kvah(alat_tetap, state)
    ike_akhir = round(kvah_akhir / max(1.0, luas_m2), 4)
    zona_akhir = _label_zona(ike_akhir, ada_ac)

    # ── Langkah per peralatan ────────────────────────────────────────────────
    rate = _rate_per_kvah(daya_va, pbjt, is_prabayar)
    langkah = []
    for alat in state:
        kurang = round(alat["jam_awal"] - alat["jam_saat_ini"], 1)
        if kurang <= 0:
            continue
        jumlah = alat.get("jumlah", 1)
        kvah_hemat = round(alat["daya_semu_va"] * kurang * jumlah * HARI_PER_BULAN / 1000, 3)
        langkah.append({
            "nama":            alat["nama"],
            "kategori":        alat["kategori"],
            "jam_awal":        alat["jam_awal"],
            "jam_rekomendasi": alat["jam_saat_ini"],
            "kurang_jam":      kurang,
            "hemat_kvah":      kvah_hemat,
            "hemat_rp":        int(round(kvah_hemat * rate, 0)),
            "hemat_emisi_kg":  round(kvah_hemat * FAKTOR_EMISI_JAMALI_OM, 3),
        })
    langkah.sort(key=lambda x: x["hemat_kvah"], reverse=True)

    pesan_map = {
        "efisien": (
            f"Berhasil. Konsumsi turun dari zona {zona_awal} ke zona Efisien "
            f"(IKE <= {batas_efisien} kWh/m2/bulan)."
        ),
        "cukup_efisien": (
            f"Target Efisien tidak tercapai dari zona {zona_awal}. Sistem "
            "menurunkan target ke Cukup Efisien dan berhasil."
        ),
        "tidak_tercapai": (
            "Pengurangan maksimal (50% jam per peralatan) sudah diterapkan "
            "namun IKE masih di atas Cukup Efisien. Pertimbangkan mengganti "
            "peralatan dengan label SKEM bintang tinggi."
        ),
    }

    return _bungkus_hasil(
        True, status, zona_awal, zona_akhir, ike_awal, ike_akhir,
        _get_batas(ada_ac, target_zona), kvah_awal, kvah_akhir,
        daya_va, pbjt, is_prabayar, langkah, pesan_map[status],
    )