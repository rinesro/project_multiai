"""
action_analist/anomaly_evaluator.py
=================================
(Riwayat lokasi: core/anomaly_detector.py -> core/anomaly_predictor.py
-> sempat core/anomaly_evaluator.py -> lokasi final di sini, satu
folder dengan ike_profiler.py. Keduanya modul "evaluasi/klasifikasi
hasil", dipisah dari core/kalkulasi.py yang murni rumus regulasi.)

Interpretasi hasil perbandingan konsumsi aktual vs estimasi menjadi
status yang bisa ditampilkan ke user.

(Riwayat nama: anomaly_detector.py/deteksi_anomali -> anomaly_predictor.py/
prediksi_anomali -> anomaly_evaluator.py/evaluasi_anomali, nama final ini.
"Deteksi" dilepas karena sistem ini tidak memastikan anomali secara
definitif. "Prediksi" JUGA dilepas: istilah itu secara akademis
menyiratkan model statistik/ML yang dilatih dan divalidasi (butuh
akurasi, presisi, cross-validation) — padahal mekanisme di modul ini
murni ATURAN AMBANG BATAS (bandingkan selisih % terhadap satu angka
tetap dari rujukan eksternal), bukan model yang di-fit ke data.
"Evaluasi" tepat karena itu persis yang terjadi: MENILAI sebuah nilai
terhadap kriteria ambang batas, tanpa mengklaim kepastian (bukan
"deteksi") maupun validitas prediktif statistik (bukan "prediksi").)

Sengaja dipisah dari core/kalkulasi.py: kalkulasi.py murni berisi rumus
aritmatika/regulasi tanpa aturan bisnis, sedangkan modul ini fokus
menjawab "apa yang dinilai sebagai anomali, data belum cukup, atau
data tidak konsisten" — untuk KEDUA domain (Rp pascabayar, kWh/token
prabayar).

Dasar ambang batas 29% (BATAS_TOLERANSI_ANOMALI):
    Parry, D. A., Davidson, B. I., Sewall, C. J. R., Fisher, J. T.,
    Mieczkowski, H., & Quintana, D. S. (2021). A systematic review and
    meta-analysis of discrepancies between logged and self-reported
    digital media use. Nature Human Behaviour.
    https://doi.org/10.1038/s41562-021-01117-5

    Table 2 (Reporting accuracy in subgroup analyses), kategori
    "Usage duration": R = 1,29 (rasio rata-rata self-report/logged),
    95% CI [1,01–1,66], P = 0,044, k = 35 studi. Kategori "duration"
    dipilih karena paling sesuai dengan input sistem ini (estimasi jam
    pemakaian per hari, bukan volume/jumlah interaksi), dan hasilnya
    signifikan secara statistik — berbeda dari R keseluruhan (semua
    kategori) yang CI-nya mencakup 1,0 (tidak signifikan).

    CATATAN METODOLOGIS PENTING: penelitian ini tentang penggunaan
    media digital (screen time, media sosial, telepon), BUKAN tentang
    estimasi pemakaian listrik. R=1,29 diadopsi sebagai ANALOGI lintas
    domain — pola umum "self-report cenderung meleset dari data
    logged/aktual" — karena belum ditemukan penelitian spesifik
    tentang akurasi estimasi pemakaian listrik rumah tangga. Baik
    prabayar (estimasi jam pakai vs selisih saldo token) maupun
    pascabayar (estimasi tagihan vs tagihan asli) sama-sama
    membandingkan estimasi self-report dengan data logged/aktual,
    sehingga dasar empiris yang sama berlaku untuk keduanya.

Prabayar butuh penanganan kasus khusus yang tidak dimiliki pascabayar
(karena pascabayar punya siklus tagihan tetap 30 hari, prabayar tidak):
    - Tanggal pembelian di masa depan → input tidak valid
    - Periode berjalan < 1 hari       → belum cukup data untuk dibandingkan
    - Konsumsi aktual negatif         → data tidak konsisten (kemungkinan
                                         ada top-up susulan yang tak tercatat)
"""

from core.kalkulasi import hitung_batas_kwh_bulanan
from core.format_id import format_angka_id

# 29% — Parry et al. (2021), Table 2, kategori "Usage duration": R=1,29,
# 95% CI [1,01-1,66], P=0,044. Lihat docstring modul untuk penjelasan
# lengkap kenapa angka ini dipakai dan catatan analogi lintas domainnya.
BATAS_TOLERANSI_ANOMALI = 0.29


# ── Fungsi inti (generik, dua domain) ─────────────────────────────────────────

def evaluasi_anomali(nilai_aktual: float, nilai_estimasi: float) -> dict:
    """
    Evaluasi anomali berbasis aturan (if-else) generik — MENILAI selisih
    terhadap ambang batas tetap, bukan model statistik yang di-fit/dilatih.
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
    hasil  = evaluasi_anomali(tagihan_asli, estimasi_tagihan)
    status = "anomali" if hasil["is_anomali"] else "normal"

    if status == "anomali":
        pesan = (
            f"Anomali terindikasi, selisih estimasi vs tagihan asli "
            f"{format_angka_id(hasil['selisih_pct'], 1)}% (ambang batas "
            f"{round(BATAS_TOLERANSI_ANOMALI * 100)}%). Kemungkinan ada "
            "perangkat yang belum diinput atau indikasi kebocoran arus."
        )
    else:
        pesan = f"Tagihan wajar (selisih {format_angka_id(hasil['selisih_pct'], 1)}%)."

    return {
        "status"     : status,
        "selisih_pct": hasil["selisih_pct"],
        "pesan"      : pesan,
    }


def evaluasi_anomali_prabayar(token_terpakai_aktual: float,
                              estimasi_terpakai_perangkat: float,
                              hari_berjalan: int,
                              daya_va: int,
                              total_kwh: float) -> dict:
    """
    Wrapper status untuk prabayar (domain kWh/token).

    Menangani EMPAT kondisi khusus SEBELUM menjalankan evaluasi anomali
    murni — masing-masing punya akar masalah berbeda dan harus
    ditampilkan dengan pesan berbeda ke user, bukan disamakan sebagai
    "anomali" konsumsi biasa:

        1. hari_berjalan < 0                → tanggal pembelian tidak valid
        2. hari_berjalan < 1                → periode terlalu singkat
        3. token_terpakai_aktual < 0        → data tidak konsisten
        4. total_kwh > batas kWh bulanan    → indikasi meteran rusak
           (lihat core/kalkulasi.py::hitung_batas_kwh_bulanan) --
           DICEK SEBELUM perbandingan 29% biasa dan MENGGANTIKAN
           pesannya kalau dua-duanya sama-sama terpicu, karena
           kemungkinan penyebabnya (meteran rusak) jauh lebih serius
           dan actionable daripada sekadar "konsumsi tidak wajar".

    Parameters:
        daya_va   : daya tersambung PLN (VA), untuk Cek B
        total_kwh : total konsumsi bulanan dari daftar alat (kWh),
                    dipakai sebagai proksi kemungkinan pembelian token
                    bulanan -- SATU-SATUNYA nilai kWh bulanan yang
                    tersedia di sistem ini (tidak ada pencatatan
                    akumulasi pembelian token sepanjang bulan berjalan)

    Returns dict:
        status       : 'anomali' | 'normal' | 'tanggal_tidak_valid' |
                       'data_belum_cukup' | 'data_tidak_konsisten'
        selisih_pct  : float | None (None untuk status khusus di atas)
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
                "evaluasi anomali yang bermakna. Cek kembali minimal "
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

    # Cek B: batas maksimal pembelian token bulanan (~720 jam nyala).
    # Dicek SEBELUM evaluasi 29% biasa -- kalau terpicu, pesan ini yang
    # tampil, BUKAN pesan anomali konsumsi biasa (lihat docstring).
    batas_kwh_bulanan = hitung_batas_kwh_bulanan(daya_va)
    if total_kwh > batas_kwh_bulanan:
        return {
            "status"     : "anomali",
            "selisih_pct": None,
            "pesan"      : (
                f"Total konsumsi bulanan Anda ({format_angka_id(total_kwh, 2)} kWh) melebihi "
                f"batas maksimal pembelian token untuk daya {daya_va} VA "
                f"(~{format_angka_id(batas_kwh_bulanan, 2)} kWh/bulan, setara 720 jam nyala). "
                "Kemungkinan mesin meteran Anda bermasalah — harap segera "
                "hubungi PLN 123 untuk menghindari hal-hal yang tidak "
                "diinginkan."
            ),
        }

    hasil  = evaluasi_anomali(token_terpakai_aktual, estimasi_terpakai_perangkat)
    status = "anomali" if hasil["is_anomali"] else "normal"

    if status == "anomali":
        pesan = (
            f"Anomali terindikasi, konsumsi aktual "
            f"{format_angka_id(token_terpakai_aktual, 2)} kWh vs estimasi dari perangkat "
            f"{format_angka_id(estimasi_terpakai_perangkat, 2)} kWh dalam {hari_berjalan} hari "
            f"(selisih {format_angka_id(hasil['selisih_pct'], 1)}%, ambang batas "
            f"{round(BATAS_TOLERANSI_ANOMALI * 100)}%). Kemungkinan ada "
            "kebocoran arus atau perangkat yang belum diinput."
        )
    else:
        pesan = (
            f"Konsumsi wajar (selisih {format_angka_id(hasil['selisih_pct'], 1)}% dari "
            f"estimasi perangkat, dalam {hari_berjalan} hari)."
        )

    return {
        "status"     : status,
        "selisih_pct": hasil["selisih_pct"],
        "pesan"      : pesan,
    }
