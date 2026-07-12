"""
orchestrator.py
===============
Perangkai (orchestrator) sistem EnergiCerdas AI.

Menyambungkan tujuh komponen menjadi satu fungsi: dari input mentah frontend
sampai hasil analisis lengkap. Tidak mengandung rumus sendiri — hanya memanggil
komponen dalam urutan yang benar dan merakit hasilnya.

    input mentah
        -> app.validasi_permintaan            (gerbang)
        -> kalkulator.rincian_peralatan        (Lapis 1)
        -> kalkulator.hitung_rentang_tagihan   (Lapis 1)
        -> anomaly_detector.deteksi_anomali    (Lapis 2, pascabayar saja)
        -> DSMClassifier.prediksi_batch        (Lapis 2)
        -> brute_force.optimasi                (Lapis 3)
    hasil analisis

Modul ini tidak mengimpor Streamlit, sehingga bisa dipakai ulang oleh FastAPI
saat deploy. streamlit_app.py memanggil analisis() lalu menampilkan hasilnya.
"""

from app import validasi_permintaan
from core.kalkulator import (
    rincian_peralatan, hitung_rentang_tagihan, hitung_rentang_token,
    hitung_ike, hitung_kwh_per_org, hitung_emisi,
    total_kvah, rentang_kwh,
)
from core.anomaly_detector import deteksi_anomali
from models.dsm_classifier import DSMClassifier
from optimizer.brute_force import optimasi
from core.konstanta import PBJT_RUMAH_TANGGA


# Model dimuat sekali saja (mahal). streamlit_app membungkusnya dengan cache.
_DSM = None


def _dsm() -> DSMClassifier:
    global _DSM
    if _DSM is None:
        _DSM = DSMClassifier()
    return _DSM


def analisis(permintaan: dict, dsm_clf: DSMClassifier = None,
             pbjt: float = PBJT_RUMAH_TANGGA) -> dict:
    """
    Menjalankan seluruh pipeline atas satu permintaan MENTAH dari frontend.

    Parameters
    ----------
    permintaan : dict
        Persis bentuk yang dikirim frontend (belum divalidasi). Divalidasi
        di dalam fungsi ini.
    dsm_clf : DSMClassifier, opsional
        Bila None, memakai instance bersama yang dimuat sekali.

    Returns
    -------
    dict:
        ok : bool
        Jika False -> {'ok': False, 'errors': [...]}
        Jika True  -> hasil analisis lengkap (lihat kunci di bawah).
    """
    # ── Gerbang: validasi input ──────────────────────────────────────────────
    v = validasi_permintaan(permintaan)
    if not v["ok"]:
        return {"ok": False, "errors": v["errors"]}

    b = v["bersih"]
    daftar_alat = b["peralatan"]
    daya_va = b["daya_va"]
    is_prabayar = b["is_prabayar"]

    clf = dsm_clf or _dsm()

    # ── Lapis 1: kalkulasi ───────────────────────────────────────────────────
    alat = rincian_peralatan(daftar_alat)
    kvah = total_kvah(daftar_alat)
    kwh_bawah, kwh_atas = rentang_kwh(kvah)

    ike = hitung_ike(kwh_atas, b["luas_m2"])          # basis cos phi = 1
    kwh_org = hitung_kwh_per_org(kwh_atas, b["penghuni"])
    ada_ac = any(a["kategori"] == "Pendingin" for a in daftar_alat)

    if is_prabayar:
        biaya = hitung_rentang_token(daftar_alat, daya_va, pbjt)
        biaya_ringkas = {
            "jenis": "token",
            "bawah": biaya["token_bawah"],
            "atas":  biaya["token_atas"],
            "tengah": biaya["token_tengah"],
        }
    else:
        biaya = hitung_rentang_tagihan(daftar_alat, daya_va, pbjt)
        biaya_ringkas = {
            "jenis": "tagihan",
            "bawah": biaya["tagihan_bawah"],
            "atas":  biaya["tagihan_atas"],
            "tengah": biaya["tagihan_tengah"],
        }

    emisi = {
        "bawah":  hitung_emisi(kwh_bawah)["emisi_kg_bulan"],
        "atas":   hitung_emisi(kwh_atas)["emisi_kg_bulan"],
    }

    # ── Lapis 2: deteksi anomali (PASCABAYAR SAJA) ───────────────────────────
    anomali = None
    if not is_prabayar:
        anomali = deteksi_anomali(b["tagihan_riil"], biaya)

    # ── Lapis 2: DSM ─────────────────────────────────────────────────────────
    hasil_dsm = clf.prediksi_batch(alat)
    ringkasan = clf.ringkasan_dsm(hasil_dsm)

    # ── Lapis 3: optimasi ────────────────────────────────────────────────────
    hasil_opt = optimasi(
        ringkasan_dsm=ringkasan,
        luas_m2=b["luas_m2"],
        ada_ac=ada_ac,
        daya_va=daya_va,
        pbjt=pbjt,
        is_prabayar=is_prabayar,
    )

    return {
        "ok":            True,
        "input":         b,
        "ada_ac":        ada_ac,
        "kvah":          kvah,
        "kwh_bawah":     kwh_bawah,
        "kwh_atas":      kwh_atas,
        "ike":           ike,
        "kwh_per_org":   kwh_org,
        "biaya":         biaya_ringkas,
        "emisi":         emisi,
        "anomali":       anomali,
        "dsm":           hasil_dsm,
        "ringkasan_dsm": ringkasan,
        "optimasi":      hasil_opt,
        "model_siap":    clf.siap,
        "model_error":   clf.pesan_error,
    }