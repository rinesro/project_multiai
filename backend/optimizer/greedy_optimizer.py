"""
optimizer/greedy_optimizer.py
=========================
Greedy Optimizer untuk EnergiCerdas AI — Lapis 3.

Cara kerja:
    1. Hanya aktif jika IKE user di zona Boros atau Sangat Boros
       (IKE > batas Cukup Efisien)
    2. Hanya memproses peralatan berlabel 'Fleksibel'
    3. Mencoba kombinasi pengurangan jam pemakaian secara greedy:
       mulai dari peralatan kWh terbesar
    4. Target pertama : zona Efisien
       Target fallback: zona Cukup Efisien (jika Efisien tidak tercapai)
    5. Output berisi langkah per peralatan + selisih biaya + selisih emisi

Strategi pengurangan:
    - Step pengurangan : 0.5 jam per iterasi
    - Batas minimum    : 0.5 jam (tidak boleh nol)
    - Maks pengurangan : 50% dari jam asal per peralatan

PEROMBAKAN BESAR — ambang IKE TIDAK LAGI disalin ulang di sini:
    Modul ini SEBELUMNYA punya salinan sendiri ambang IKE Depdiknas
    (BATAS_IKE, dibedakan ber-AC/tidak) yang independen dari
    action_analist/ike_profiler.py — risiko dua sumber kebenaran
    tidak sinkron kalau salah satu diubah tanpa yang lain (persis
    yang terjadi: ike_profiler.py sudah diperbarui pakai kalibrasi
    5-lapis baru, tapi file ini masih Depdiknas lama sampai perombakan
    ini). Sekarang ambang batas & label zona diambil LANGSUNG dari
    action_analist/ike_profiler.py (MF_IKE, profil_ike()) — satu-
    satunya sumber kebenaran untuk klasifikasi IKE di seluruh sistem.
    Parameter ada_ac juga DIHAPUS TOTAL dari modul ini karena sistem
    baru tidak lagi membedakan rumah ber-AC/tidak (skala tunggal).

Referensi regulasi:
    [1] Tarif PLN — PT PLN (Persero)
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO₂/kWh — ESDM 2019
    [3] PBJT 2,4% — Perda DKI Jakarta No.1/2024
    [5] Ambang IKE — lihat action_analist/ike_profiler.py untuk
        metodologi kalibrasi 5-lapis lengkap & sitasi.
"""

from copy import deepcopy
from pathlib import Path
import sys as _sys

_BASE = Path(__file__).parent.parent
if str(_BASE) not in _sys.path:
    _sys.path.insert(0, str(_BASE))
from core.kalkulasi import hitung_tagihan as _core_hitung_tagihan
from core.kalkulasi import hitung_emisi as _core_hitung_emisi
from core.kalkulasi import FAKTOR_EMISI_JAMALI_OM
from core.format_id import format_angka_id
from action_analist.ike_profiler import profil_ike, MF_IKE

# Ambang IKE dipakai optimizer (target ceiling akhir zona hasil optimasi)
# -- diturunkan LANGSUNG dari MF_IKE yang sama dipakai ike_profiler.py,
# BUKAN disalin ulang. "c" (elemen indeks ke-2 tiap tuple trapesium)
# adalah ujung plateau zona itu -- titik di mana keanggotaan zona
# tersebut masih penuh (1.0) sebelum mulai turun ke zona berikutnya.
BATAS_EFISIEN       = MF_IKE["Efisien"][2]        # ~2.597 kWh/m²/bulan
BATAS_CUKUP_EFISIEN = MF_IKE["Cukup Efisien"][2]  # ~4.396 kWh/m²/bulan

# MAKS_KURANG_PCT & MIN_JAM: sekarang berperan sebagai PAGAR PENGAMAN
# untuk hasil kalkulasi Δt=ΔE/P (lihat _greedy()), bukan lagi penentu
# UTAMA besar pengurangan seperti versi sebelumnya (yang iteratif step
# 0,5 jam). STEP_JAM (step iteratif lama) sudah dihapus total --
# tidak diperlukan lagi karena kalkulasi sekarang langsung, bukan
# iteratif. Status sitasi kedua angka di bawah ini TIDAK berubah dari
# sebelumnya -- masih perlu diselesaikan terpisah (lihat CATATAN
# JUJUR di docstring _greedy()).
MAKS_KURANG_PCT = 0.50   # maksimum pengurangan 50% dari jam asal
MIN_JAM         = 0.5    # minimum jam operasi per peralatan

# Toleransi floating-point untuk perbandingan "IKE <= batas_ike" -- BUKAN
# angka rekayasa/asumsi peneliti seperti MAKS_KURANG_PCT di atas, ini
# murni standar praktik numerik (perbandingan float exact-equality
# rawan meleset karena representasi biner, lihat IEEE 754). Ditemukan
# perlu lewat uji banding empiris terhadap 200 skenario acak: kalkulasi
# Δt=ΔE/P kadang mendarat PERSIS di titik target secara matematis, tapi
# representasi float-nya meleset -- awalnya dicoba 1e-9, TERBUKTI tidak
# cukup (galat menumpuk sampai ~1e-6 di skenario banyak alat berurutan,
# tiap alat menambah sedikit galat pembagian/perkalian). 1e-4 dipilih
# sebagai margin aman (masih 10-1000x lebih kecil dari beda IKE yang
# BERMAKNA secara praktis -- lihat jarak antar ambang MF_IKE di
# action_analist/ike_profiler.py, semuanya berjarak >0.1).
_EPSILON_IKE = 1e-4


# ── Helper functions ──────────────────────────────────────────────────────────

def _total_kwh(alat_tetap: list, alat_fleksibel: list) -> float:
    """
    Hitung total kWh bulanan dari semua peralatan.

    Peralatan fleksibel dihitung ulang tiap iterasi karena Lapis 3
    mencoba skenario jam HIPOTETIS yang belum pernah dihitung Lapis 1
    (mis. "kalau AC dikurangi jadi 7 jam, bagaimana?"). Peralatan
    tetap (tidak fleksibel) memakai kwh_bulan yang SUDAH dihitung
    Lapis 1 lewat DSM classifier — tidak dihitung ulang di sini.

    Kedua kelompok ikut mengalikan 'jumlah' unit per entri.
    """
    kwh_tetap = sum(a['kwh_bulan'] for a in alat_tetap)
    kwh_flex  = sum(
        round(a['watt'] * a['jam_saat_ini'] * a.get('jumlah', 1) * 30 / 1000, 4)
        for a in alat_fleksibel
    )
    return round(kwh_tetap + kwh_flex, 3)


def _hitung_tagihan(kwh: float, tarif: float,
                    pbjt: float, is_prabayar: bool) -> float:
    """Wrapper tipis ke core.kalkulasi.hitung_tagihan — rumus sama, tidak ditulis ulang."""
    return _core_hitung_tagihan(kwh, tarif, pbjt, is_prabayar)["total"]


def _hitung_emisi(kwh: float) -> float:
    """Wrapper tipis ke core.kalkulasi.hitung_emisi — rumus sama, tidak ditulis ulang."""
    return _core_hitung_emisi(kwh)["emisi_kg_bulan"]


# ── Greedy optimizer ──────────────────────────────────────────────────────────

def _greedy(alat_tetap: list, alat_flex: list,
            luas_m2: float, batas_ike: float) -> tuple:
    """
    Greedy DENGAN KALKULASI LANGSUNG: untuk tiap alat (urutan kWh
    terbesar dulu), hitung LANGSUNG berapa jam perlu dikurangi dari
    alat itu memakai rumus dasar kelistrikan Δt = ΔE / P (turunan
    langsung dari E = P × t, bukan konstanta yang perlu disitasi) --
    BUKAN lagi tebak-tebakan step 0,5 jam berulang seperti sebelumnya.

    PERUBAHAN dari versi sebelumnya: dulu tiap alat dikurangi 0,5 jam
    per iterasi (bisa belasan iterasi untuk 1 alat saja) sampai target
    tercapai atau alat itu mentok batas minimum. Sekarang, begitu giliran
    sebuah alat diproses, langsung dihitung PERSIS berapa jam alat itu
    perlu dikurangi untuk menutup sisa selisih ΔE yang masih dibutuhkan
    -- satu kalkulasi per alat, bukan banyak langkah kecil.

    CATATAN JUJUR: batas MAKS_KURANG_PCT (50%) TETAP ada di sini,
    sebagai PAGAR PENGAMAN -- bukan lagi penentu UTAMA seberapa besar
    pengurangan (itu sekarang peran rumus Δt=ΔE/P), tapi tetap
    mencegah hasil kalkulasi yang tidak masuk akal dibebankan ke satu
    alat saja (mis. "AC harus 0 jam karena dialah satu-satunya alat
    fleksibel"). Status sitasi angka 50% ini TIDAK berubah dari
    sebelumnya -- masih perlu diselesaikan terpisah.

    Returns:
        (berhasil: bool, state_akhir: list)
    """
    state = deepcopy(alat_flex)

    # Urutkan dari kWh terbesar — dampak terbesar dulu (struktur greedy
    # dipertahankan persis seperti sebelumnya)
    state.sort(key=lambda x: x['kwh_bulan'], reverse=True)

    kwh_tetap = sum(a['kwh_bulan'] for a in alat_tetap)

    for alat in state:
        kwh_flex_sekarang = sum(
            a['watt'] * a['jam_saat_ini'] * a.get('jumlah', 1) * 30 / 1000
            for a in state
        )
        ike_kini = (kwh_tetap + kwh_flex_sekarang) / max(1.0, luas_m2)

        if ike_kini <= batas_ike + _EPSILON_IKE:
            return True, state

        # ΔE (kWh/bulan) yang MASIH perlu dipangkas supaya IKE capai target
        delta_e_kwh = (ike_kini - batas_ike) * luas_m2

        # Δt = ΔE / P — rumus dasar E=P×t, P dalam kW, disesuaikan skala
        # bulanan (30 hari) supaya satuannya konsisten dengan ΔE (kWh/bulan)
        p_kw = alat['watt'] * alat.get('jumlah', 1) / 1000
        if p_kw <= 0:
            continue  # alat 0 watt (data tidak valid), lewati
        delta_t_dibutuhkan = delta_e_kwh / (p_kw * 30)  # jam/hari

        # Pagar pengaman: jangan sampai satu alat menanggung SEMUA ΔE
        # kalau itu bikin dia turun di bawah batas minimumnya
        delta_t_maksimal = alat['jam_saat_ini'] - alat['jam_minimum']
        delta_t_dipakai = min(delta_t_dibutuhkan, delta_t_maksimal)
        delta_t_dipakai = max(delta_t_dipakai, 0.0)

        # SENGAJA belum dibulatkan di sini -- pembulatan di tengah
        # kalkulasi (mis. round(..., 2) tiap alat) terbukti menumpuk jadi
        # galat kecil antar-alat, dan di kasus yang pas-pasan di ambang
        # batas, itu cukup menggeser hasil dari "tercapai" jadi "tidak
        # tercapai" secara keliru (ditemukan lewat uji banding 30
        # skenario acak vs versi lama). Presisi penuh dipertahankan
        # sampai akhir; pembulatan buat tampilan ada di optimasi()
        # saat menyusun 'langkah' untuk direspons ke user.
        alat['jam_saat_ini'] = alat['jam_saat_ini'] - delta_t_dipakai

    # Setelah semua alat diproses, cek final
    kwh_flex_akhir = sum(
        a['watt'] * a['jam_saat_ini'] * a.get('jumlah', 1) * 30 / 1000
        for a in state
    )
    ike_akhir = (kwh_tetap + kwh_flex_akhir) / max(1.0, luas_m2)
    return (ike_akhir <= batas_ike + _EPSILON_IKE), state


# ── Fungsi utama ──────────────────────────────────────────────────────────────

def optimasi(ringkasan_dsm : dict,
             luas_m2       : float,
             tarif_kwh     : float,
             pbjt          : float,
             is_prabayar   : bool,
             kwh_awal      : float,
             tagihan_awal  : float,
             emisi_awal    : float) -> dict:
    """
    Greedy optimizer berbasis target IKE hasil kalibrasi 5-lapis
    (lihat action_analist/ike_profiler.py untuk metodologi lengkap).

    Parameters:
        ringkasan_dsm  : output DSMClassifier.ringkasan_dsm()
        luas_m2        : luas bangunan user (m²)
        tarif_kwh      : tarif PLN sesuai golongan (Rp/kWh)
        pbjt           : tarif PBJT (0.024 untuk rumah tangga Jakarta)
        is_prabayar    : True jika prabayar — menentukan apakah Biaya
                         Materai dikenakan saat simulasi tagihan_akhir
                         (materai hanya berlaku pascabayar; lihat
                         core/kalkulasi.py::hitung_biaya_materai)
        kwh_awal       : total kWh sebelum optimasi
        tagihan_awal   : tagihan sebelum optimasi (Rp)
        emisi_awal     : emisi CO₂ sebelum optimasi (kgCO₂/bulan)

    Returns dict berisi:
        aktif           : bool — apakah optimizer berjalan
        status          : 'sudah_efisien' | 'efisien' | 'cukup_efisien' |
                          'tidak_tercapai'
        zona_awal       : label zona IKE sebelum optimasi
        zona_akhir      : label zona IKE setelah optimasi
        ike_awal        : IKE sebelum (kWh/m²/bulan)
        ike_akhir       : IKE setelah (kWh/m²/bulan)
        target_ike      : nilai batas IKE yang dicapai
        total_kwh_akhir : total kWh setelah optimasi
        tagihan_akhir   : tagihan setelah optimasi (Rp)
        emisi_akhir     : emisi setelah optimasi (kgCO₂/bulan)
        hemat_kwh       : selisih kWh
        hemat_rp        : selisih tagihan (Rp)
        hemat_emisi_kg  : selisih emisi (kgCO₂/bulan)
        persen_hemat_rp : persentase penghematan biaya (%)
        persen_hemat_emisi: persentase pengurangan emisi (%)
        langkah         : list rekomendasi per peralatan
        pesan           : pesan ringkas status optimasi
    """
    ike_awal  = round(kwh_awal / max(1.0, luas_m2), 4)
    zona_awal = profil_ike(ike_awal)

    # ── Cek apakah optimizer perlu berjalan ───────────────────────────────────
    # SKIP : IKE masih di zona Sangat Efisien / Efisien / Cukup Efisien
    #        (IKE < batas Cukup Efisien) -- zona ini dianggap sudah cukup
    #        baik, optimizer tidak perlu memaksa turun lebih jauh.
    # AKTIF: IKE >= batas Cukup Efisien (masuk zona Boros / Sangat Boros)
    #   → coba turunkan ke Efisien, fallback ke Cukup Efisien kalau gagal.
    if ike_awal < BATAS_CUKUP_EFISIEN:
        return {
            "aktif"             : False,
            "status"            : "sudah_efisien",
            "zona_awal"         : zona_awal,
            "zona_akhir"        : zona_awal,
            "ike_awal"          : ike_awal,
            "ike_akhir"         : ike_awal,
            "target_ike"        : BATAS_CUKUP_EFISIEN,
            "total_kwh_akhir"   : kwh_awal,
            "tagihan_akhir"     : int(tagihan_awal),
            "emisi_akhir"       : emisi_awal,
            "hemat_kwh"         : 0.0,
            "hemat_rp"          : 0,
            "hemat_emisi_kg"    : 0.0,
            "persen_hemat_rp"   : 0.0,
            "persen_hemat_emisi": 0.0,
            "langkah"           : [],
            "pesan"             : (
                f"Konsumsi sudah dalam zona {zona_awal}. "
                "Tidak perlu optimasi."
            ),
        }

    # ── Siapkan peralatan ─────────────────────────────────────────────────────
    alat_tetap = ringkasan_dsm.get('tidak_fleksibel', [])
    alat_raw   = ringkasan_dsm.get('fleksibel', [])

    if not alat_raw:
        return {
            "aktif"             : True,
            "status"            : "tidak_tercapai",
            "zona_awal"         : zona_awal,
            "zona_akhir"        : zona_awal,
            "ike_awal"          : ike_awal,
            "ike_akhir"         : ike_awal,
            "target_ike"        : BATAS_CUKUP_EFISIEN,
            "total_kwh_akhir"   : kwh_awal,
            "tagihan_akhir"     : int(tagihan_awal),
            "emisi_akhir"       : emisi_awal,
            "hemat_kwh"         : 0.0,
            "hemat_rp"          : 0,
            "hemat_emisi_kg"    : 0.0,
            "persen_hemat_rp"   : 0.0,
            "persen_hemat_emisi": 0.0,
            "langkah"           : [],
            "pesan"             : (
                "Semua peralatan Tidak Fleksibel. "
                "Tidak ada yang bisa dioptimasi secara otomatis."
            ),
        }

    # Tambah field jam_saat_ini dan jam_minimum
    alat_flex = []
    for a in alat_raw:
        item = deepcopy(a)
        item['jam_awal']     = a['jam']
        item['jam_saat_ini'] = a['jam']
        item['jam_minimum']  = max(
            MIN_JAM,
            round(a['jam'] * (1 - MAKS_KURANG_PCT), 1)
        )
        alat_flex.append(item)

    # ── Iterasi 1: target Efisien ─────────────────────────────────────────────
    berhasil, state = _greedy(alat_tetap, alat_flex, luas_m2, BATAS_EFISIEN)
    status      = "efisien"
    target_ike  = BATAS_EFISIEN

    # ── Iterasi 2: fallback ke Cukup Efisien ──────────────────────────────────
    if not berhasil:
        berhasil, state = _greedy(alat_tetap, alat_flex, luas_m2, BATAS_CUKUP_EFISIEN)
        status     = "cukup_efisien" if berhasil else "tidak_tercapai"
        target_ike = BATAS_CUKUP_EFISIEN

    # ── Hitung hasil akhir ────────────────────────────────────────────────────
    kwh_akhir     = _total_kwh(alat_tetap, state)
    ike_akhir     = round(kwh_akhir / max(1.0, luas_m2), 4)
    zona_akhir    = profil_ike(ike_akhir)
    tagihan_akhir = _hitung_tagihan(kwh_akhir, tarif_kwh, pbjt, is_prabayar)
    emisi_akhir   = _hitung_emisi(kwh_akhir)

    hemat_kwh   = round(kwh_awal     - kwh_akhir,     3)
    hemat_rp    = round(tagihan_awal - tagihan_akhir,  0)
    hemat_emisi = round(emisi_awal   - emisi_akhir,    3)

    pct_rp    = round(hemat_rp    / max(1, tagihan_awal) * 100, 1)
    pct_emisi = round(hemat_emisi / max(1, emisi_awal)   * 100, 1)

    # ── Susun langkah rekomendasi ─────────────────────────────────────────────
    langkah = []
    for alat in state:
        kurang = round(alat['jam_awal'] - alat['jam_saat_ini'], 1)
        if kurang <= 0:
            continue

        jumlah = alat.get('jumlah', 1)
        # jam_rekomendasi dihitung dari jam_awal - kurang (kurang yang
        # SUDAH dibulatkan), bukan langsung dari alat['jam_saat_ini']
        # yang presisi penuh -- supaya jam_awal - jam_rekomendasi di
        # tampilan selalu PERSIS sama dengan kurang_jam, tidak ada
        # selisih desimal kecil yang janggal kalau diperiksa manual.
        jam_rekomendasi = round(alat['jam_awal'] - kurang, 1)
        kwh_hemat  = round(alat['watt'] * kurang * jumlah * 30 / 1000, 3)
        rp_hemat   = round(kwh_hemat * tarif_kwh * (1 + pbjt), 0)
        emisi_hemat= round(kwh_hemat * FAKTOR_EMISI_JAMALI_OM, 3)

        langkah.append({
            "nama"           : alat['nama'],
            "kategori"       : alat['kategori'],
            "jam_awal"       : alat['jam_awal'],
            "jam_rekomendasi": jam_rekomendasi,
            "kurang_jam"     : kurang,
            "hemat_kwh"      : kwh_hemat,
            "hemat_rp"       : int(rp_hemat),
            "hemat_emisi_kg" : emisi_hemat,
        })

    # Urutkan dari penghematan terbesar
    langkah.sort(key=lambda x: x['hemat_kwh'], reverse=True)

    # ── Pesan status ──────────────────────────────────────────────────────────
    pesan_map = {
        "efisien"       : (
            f"Berhasil! Konsumsi turun dari zona {zona_awal} "
            f"ke zona Efisien (IKE ≤ {format_angka_id(BATAS_EFISIEN, 3)} kWh/m²/bulan)."
        ),
        "cukup_efisien" : (
            f"Target Efisien tidak tercapai dari zona {zona_awal}. "
            "Sistem menurunkan target ke Cukup Efisien dan berhasil."
        ),
        "tidak_tercapai": (
            "Pengurangan maksimal sudah diterapkan namun IKE masih "
            "di atas Cukup Efisien. Pertimbangkan mengganti peralatan "
            "dengan yang lebih hemat energi (label SKEM bintang tinggi)."
        ),
    }

    return {
        "aktif"             : True,
        "status"            : status,
        "zona_awal"         : zona_awal,
        "zona_akhir"        : zona_akhir,
        "ike_awal"          : ike_awal,
        "ike_akhir"         : ike_akhir,
        "target_ike"        : target_ike,
        "total_kwh_akhir"   : kwh_akhir,
        "tagihan_akhir"     : int(tagihan_akhir),
        "emisi_akhir"       : emisi_akhir,
        "hemat_kwh"         : hemat_kwh,
        "hemat_rp"          : int(hemat_rp),
        "hemat_emisi_kg"    : hemat_emisi,
        "persen_hemat_rp"   : pct_rp,
        "persen_hemat_emisi": pct_emisi,
        "langkah"           : langkah,
        "pesan"             : pesan_map[status],
    }