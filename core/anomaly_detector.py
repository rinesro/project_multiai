"""
core/anomaly_detector.py
=========================
Interpretasi hasil perbandingan konsumsi aktual vs estimasi menjadi
status yang bisa ditampilkan ke user.

Sengaja dipisah dari core/kalkulasi.py: kalkulasi.py murni berisi rumus
aritmatika/regulasi tanpa aturan bisnis, sedangkan modul ini fokus
menjawab "apa yang dianggap anomali, data belum cukup, atau data tidak
konsisten" — untuk KEDUA domain (Rp pascabayar, kWh/token prabayar).

Kenapa satu fungsi generik bisa dipakai dua domain:
    Persentase selisih (selisih_pct) tidak bergantung satuan — Rp dan
    kWh sama-sama besaran linear, jadi ambang toleransi 15% yang sama
    berlaku valid untuk keduanya tanpa perlu dua rumus terpisah.

Prabayar butuh penanganan kasus khusus yang tidak dimiliki pascabayar
(karena pascabayar punya siklus tagihan tetap 30 hari, prabayar tidak):
    - Tanggal pembelian di masa depan → input tidak valid
    - Periode berjalan < 1 hari       → belum cukup data untuk dibandingkan
    - Konsumsi aktual negatif         → data tidak konsisten (kemungkinan
                                         ada top-up susulan yang tak tercatat)
"""

BATAS_TOLERANSI_ANOMALI = 0.15  # 15% — ambang rekayasa sistem, bukan dari regulasi


# ── Fungsi inti (generik, dua domain) ─────────────────────────────────────────

def deteksi_anomali(nilai_aktual: float, nilai_estimasi: float) -> dict:
    """
    Deteksi anomali berbasis aturan (if-else) generik.
    Anomali = selisih estimasi vs nilai aktual > BATAS_TOLERANSI_ANOMALI.

    Dipakai untuk Rp (pascabayar: tagihan_asli vs estimasi_tagihan) MAUPUN
    kWh (prabayar: token_terpakai_aktual vs estimasi_terpakai_perangkat).
    Satuan tidak memengaruhi logika — lihat catatan modul di atas.

    Returns dict: is_anomali, selisih_pct
    """
    selisih_pct = (
        abs(nilai_aktual - nilai_estimasi) / max(1.0, float(nilai_aktual))
    )
    return {
        "is_anomali" : selisih_pct > BATAS_TOLERANSI_ANOMALI,
        "selisih_pct": round(selisih_pct * 100, 1),
    }


# ── Wrapper status seragam per domain ─────────────────────────────────────────
# Kedua wrapper di bawah selalu mengembalikan skema dict yang sama
# (status, selisih_pct, pesan) supaya Lapis 1 (app.py) bisa merender hasil
# tanpa perlu tahu apakah user prabayar atau pascabayar.

def evaluasi_anomali_pascabayar(tagihan_asli: float,
                                estimasi_tagihan: float) -> dict:
    """
    Wrapper status untuk pascabayar (domain Rupiah).

    Returns dict:
        status       : 'anomali' | 'normal'
        selisih_pct  : float
        pesan        : str, siap ditampilkan ke user
    """
    hasil  = deteksi_anomali(tagihan_asli, estimasi_tagihan)
    status = "anomali" if hasil["is_anomali"] else "normal"

    if status == "anomali":
        pesan = (
            f"Anomali terdeteksi — selisih estimasi vs tagihan asli "
            f"{hasil['selisih_pct']}% (ambang batas "
            f"{int(BATAS_TOLERANSI_ANOMALI * 100)}%). Kemungkinan ada "
            "perangkat yang belum diinput atau indikasi kebocoran arus."
        )
    else:
        pesan = f"Tagihan wajar (selisih {hasil['selisih_pct']}%)."

    return {
        "status"     : status,
        "selisih_pct": hasil["selisih_pct"],
        "pesan"      : pesan,
    }


def evaluasi_anomali_prabayar(token_terpakai_aktual: float,
                              estimasi_terpakai_perangkat: float,
                              hari_berjalan: int) -> dict:
    """
    Wrapper status untuk prabayar (domain kWh/token).

    Menangani tiga kondisi khusus SEBELUM menjalankan deteksi anomali
    murni — masing-masing punya akar masalah berbeda dan harus
    ditampilkan dengan pesan berbeda ke user, bukan disamakan sebagai
    "anomali":

        1. hari_berjalan < 0          → tanggal pembelian tidak valid
        2. hari_berjalan < 1          → periode terlalu singkat
        3. token_terpakai_aktual < 0  → data tidak konsisten

    Returns dict:
        status       : 'anomali' | 'normal' | 'tanggal_tidak_valid' |
                       'data_belum_cukup' | 'data_tidak_konsisten'
        selisih_pct  : float | None (None untuk 3 status khusus di atas)
        pesan        : str, siap ditampilkan ke user
    """
    if hari_berjalan < 0:
        return {
            "status"     : "tanggal_tidak_valid",
            "selisih_pct": None,
            "pesan"      : (
                "Tanggal pembelian token tidak boleh di masa depan. "
                "Periksa kembali tanggal yang diinput."
            ),
        }

    if hari_berjalan < 1:
        return {
            "status"     : "data_belum_cukup",
            "selisih_pct": None,
            "pesan"      : (
                "Belum cukup waktu berlalu sejak pembelian token untuk "
                "deteksi anomali yang bermakna. Cek kembali minimal "
                "1 hari setelah pembelian."
            ),
        }

    if token_terpakai_aktual < 0:
        return {
            "status"     : "data_tidak_konsisten",
            "selisih_pct": None,
            "pesan"      : (
                "Sisa token saat ini lebih besar dari saldo awal periode "
                "(sisa sebelum beli + hasil konversi pembelian). "
                "Kemungkinan ada top-up susulan yang belum tercatat, "
                "atau kesalahan input sisa token."
            ),
        }

    hasil  = deteksi_anomali(token_terpakai_aktual, estimasi_terpakai_perangkat)
    status = "anomali" if hasil["is_anomali"] else "normal"

    if status == "anomali":
        pesan = (
            f"Anomali terdeteksi — konsumsi aktual "
            f"{token_terpakai_aktual} kWh vs estimasi dari perangkat "
            f"{estimasi_terpakai_perangkat} kWh dalam {hari_berjalan} hari "
            f"(selisih {hasil['selisih_pct']}%, ambang batas "
            f"{int(BATAS_TOLERANSI_ANOMALI * 100)}%). Kemungkinan ada "
            "kebocoran arus atau perangkat yang belum diinput."
        )
    else:
        pesan = (
            f"Konsumsi wajar (selisih {hasil['selisih_pct']}% dari "
            f"estimasi perangkat, dalam {hari_berjalan} hari)."
        )

    return {
        "status"     : status,
        "selisih_pct": hasil["selisih_pct"],
        "pesan"      : pesan,
    }