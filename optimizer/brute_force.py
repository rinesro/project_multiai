"""
optimizer/brute_force.py
=========================
Brute Force Optimizer untuk EnergiCerdas AI — Lapis 3.

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

Referensi standar IKE Depdiknas (kWh/m²/bulan):
    Tidak ber-AC:
        Sangat Efisien : IKE < 0.84
        Efisien        : 0.84 ≤ IKE < 1.67
        Cukup Efisien  : 1.67 ≤ IKE ≤ 2.50
        Boros          : 2.50 < IKE ≤ 3.34   ← optimizer aktif
        Sangat Boros   : IKE > 3.34           ← optimizer aktif
    Ber-AC:
        Sangat Efisien : IKE < 7.92
        Efisien        : 7.92 ≤ IKE < 12.08
        Cukup Efisien  : 12.08 ≤ IKE ≤ 14.58
        Boros          : 14.58 < IKE ≤ 23.75 ← optimizer aktif
        Sangat Boros   : IKE > 23.75          ← optimizer aktif

Referensi regulasi:
    [1] Tarif PLN April–Juni 2026 — PT PLN (Persero)
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO₂/kWh — ESDM 2019
    [3] PBJT 2,4% — Perda DKI Jakarta No.1/2024
    [5] Standar IKE — Pedoman Konservasi Energi Depdiknas RI
"""

from copy import deepcopy

# ── Konstanta IKE standar Depdiknas (kWh/m²/bulan) ───────────────────────────
# Optimizer SKIP jika IKE ≤ batas Cukup Efisien (sudah dalam zona aman)
# Optimizer AKTIF jika IKE > batas Cukup Efisien (zona Boros/Sangat Boros)
BATAS_IKE = {
    "tidak_ac": {
        "sangat_efisien": 0.84,
        "efisien"       : 1.67,
        "cukup_efisien" : 2.50,
    },
    "ac": {
        "sangat_efisien": 7.92,
        "efisien"       : 12.08,
        "cukup_efisien" : 14.58,
    },
}

FAKTOR_EMISI    = 0.80   # kgCO₂/kWh — Grid Jamali OM [2]
STEP_JAM        = 0.5    # step pengurangan jam
MAKS_KURANG_PCT = 0.50   # maksimum pengurangan 50% dari jam asal
MIN_JAM         = 0.5    # minimum jam operasi per peralatan


# ── Helper functions ──────────────────────────────────────────────────────────

def _get_batas(ada_ac: bool, zona: str) -> float:
    """Ambil batas IKE berdasarkan kondisi AC dan nama zona."""
    kunci = "ac" if ada_ac else "tidak_ac"
    return BATAS_IKE[kunci][zona]


def _label_zona(ike: float, ada_ac: bool) -> str:
    """
    Tentukan label zona IKE dari nilai numerik.

    Batas zona ber-AC (kWh/m²/bulan) — standar Depdiknas:
        Sangat Efisien : IKE < 7.92
        Efisien        : 7.92 ≤ IKE < 12.08
        Cukup Efisien  : 12.08 ≤ IKE ≤ 14.58
        Boros          : 14.58 < IKE ≤ 23.75
        Sangat Boros   : IKE > 23.75

    Batas zona tidak ber-AC (kWh/m²/bulan):
        Sangat Efisien : IKE < 0.84
        Efisien        : 0.84 ≤ IKE < 1.67
        Cukup Efisien  : 1.67 ≤ IKE ≤ 2.50
        Boros          : 2.50 < IKE ≤ 3.34
        Sangat Boros   : IKE > 3.34
    """
    if ada_ac:
        if ike < 7.92:   return "Sangat Efisien"
        if ike < 12.08:  return "Efisien"
        if ike <= 14.58: return "Cukup Efisien"
        if ike <= 23.75: return "Boros"
        return "Sangat Boros"
    else:
        if ike < 0.84:  return "Sangat Efisien"
        if ike < 1.67:  return "Efisien"
        if ike <= 2.50: return "Cukup Efisien"
        if ike <= 3.34: return "Boros"
        return "Sangat Boros"


def _total_kwh(alat_tetap: list, alat_fleksibel: list) -> float:
    """Hitung total kWh bulanan dari semua peralatan."""
    kwh_tetap = sum(a['kwh_bulan'] for a in alat_tetap)
    kwh_flex  = sum(
        round(a['watt'] * a['jam_saat_ini'] * 30 / 1000, 4)
        for a in alat_fleksibel
    )
    return round(kwh_tetap + kwh_flex, 3)


def _hitung_tagihan(kwh: float, tarif: float,
                    pbjt: float, biaya_beban: float) -> float:
    biaya = kwh * tarif
    return round(biaya + biaya * pbjt + biaya_beban, 0)


def _hitung_emisi(kwh: float) -> float:
    return round(kwh * FAKTOR_EMISI, 3)


# ── Greedy optimizer ──────────────────────────────────────────────────────────

def _greedy(alat_tetap: list, alat_flex: list,
            luas_m2: float, ada_ac: bool,
            zona_target: str) -> tuple:
    """
    Greedy reduction: kurangi jam peralatan satu langkah per iterasi,
    mulai dari yang kWh-nya terbesar, sampai IKE target tercapai
    atau semua peralatan sudah di batas minimum.

    Returns:
        (berhasil: bool, state_akhir: list)
    """
    batas = _get_batas(ada_ac, zona_target)
    state = deepcopy(alat_flex)

    # Urutkan dari kWh terbesar — dampak terbesar dulu
    state.sort(key=lambda x: x['kwh_bulan'], reverse=True)

    while True:
        ike_kini = _total_kwh(alat_tetap, state) / max(1.0, luas_m2)

        if ike_kini <= batas:
            return True, state

        # Cek apakah masih ada yang bisa dikurangi
        bisa_kurang = [
            a for a in state
            if a['jam_saat_ini'] > a['jam_minimum']
        ]
        if not bisa_kurang:
            return False, state

        # Kurangi satu langkah pada peralatan kWh terbesar yang masih bisa
        for alat in state:
            if alat['jam_saat_ini'] > alat['jam_minimum']:
                alat['jam_saat_ini'] = max(
                    round(alat['jam_saat_ini'] - STEP_JAM, 1),
                    alat['jam_minimum']
                )
                break


# ── Fungsi utama ──────────────────────────────────────────────────────────────

def optimasi(ringkasan_dsm : dict,
             luas_m2       : float,
             ada_ac        : bool,
             tarif_kwh     : float,
             pbjt          : float,
             biaya_beban   : float,
             kwh_awal      : float,
             tagihan_awal  : float,
             emisi_awal    : float) -> dict:
    """
    Brute force optimizer berbasis target IKE Depdiknas.

    Parameters:
        ringkasan_dsm  : output DSMClassifier.ringkasan_dsm()
        luas_m2        : luas bangunan user (m²)
        ada_ac         : True jika ada peralatan kategori Pendingin
        tarif_kwh      : tarif PLN sesuai golongan (Rp/kWh)
        pbjt           : tarif PBJT (0.024 untuk rumah tangga Jakarta)
        biaya_beban    : biaya beban/RM bulanan (Rp), 0 jika prabayar
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
    zona_awal = _label_zona(ike_awal, ada_ac)

    # ── Cek apakah optimizer perlu berjalan ───────────────────────────────────
    # SKIP : IKE sudah < batas Efisien (zona Sangat Efisien / Efisien)
    # AKTIF: IKE >= batas Efisien
    #   → Cukup Efisien : coba turunkan ke Efisien
    #   → Boros         : coba turunkan ke Efisien, fallback Cukup Efisien
    #   → Sangat Boros  : coba turunkan ke Efisien, fallback Cukup Efisien
    batas_cukup   = _get_batas(ada_ac, "cukup_efisien")
    batas_efisien = _get_batas(ada_ac, "efisien")

    if ike_awal < batas_efisien:
        return {
            "aktif"             : False,
            "status"            : "sudah_efisien",
            "zona_awal"         : zona_awal,
            "zona_akhir"        : zona_awal,
            "ike_awal"          : ike_awal,
            "ike_akhir"         : ike_awal,
            "target_ike"        : batas_cukup,
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
            "target_ike"        : _get_batas(ada_ac, "cukup_efisien"),
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
    berhasil, state = _greedy(
        alat_tetap, alat_flex, luas_m2, ada_ac, "efisien"
    )
    status     = "efisien"
    target_zona= "efisien"

    # ── Iterasi 2: fallback ke Cukup Efisien ──────────────────────────────────
    if not berhasil:
        berhasil, state = _greedy(
            alat_tetap, alat_flex, luas_m2, ada_ac, "cukup_efisien"
        )
        status     = "cukup_efisien" if berhasil else "tidak_tercapai"
        target_zona= "cukup_efisien"

    # ── Hitung hasil akhir ────────────────────────────────────────────────────
    kwh_akhir     = _total_kwh(alat_tetap, state)
    ike_akhir     = round(kwh_akhir / max(1.0, luas_m2), 4)
    zona_akhir    = _label_zona(ike_akhir, ada_ac)
    tagihan_akhir = _hitung_tagihan(kwh_akhir, tarif_kwh, pbjt, biaya_beban)
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

        kwh_hemat  = round(alat['watt'] * kurang * 30 / 1000, 3)
        rp_hemat   = round(kwh_hemat * tarif_kwh * (1 + pbjt), 0)
        emisi_hemat= round(kwh_hemat * FAKTOR_EMISI, 3)

        langkah.append({
            "nama"           : alat['nama'],
            "kategori"       : alat['kategori'],
            "jam_awal"       : alat['jam_awal'],
            "jam_rekomendasi": alat['jam_saat_ini'],
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
            f"ke zona Efisien (IKE ≤ {_get_batas(ada_ac, 'efisien')} "
            "kWh/m²/bulan)."
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
        "target_ike"        : _get_batas(ada_ac, target_zona),
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