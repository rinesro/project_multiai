"""
backend/test_sistem.py
========================
Script tes kesiapan sistem — jalankan mandiri kapan saja untuk cek
apakah ada error, TANPA perlu deploy dulu atau minta bantuan orang lain.

Cara pakai:
    cd backend
    python test_sistem.py

Yang diuji (semua skenario yang pernah kita temukan & perbaiki bareng):
    1. Semua modul bisa diimpor (core, action_analist, services, models, optimizer)
    2. Model DSM asli bisa dimuat dan dipakai prediksi
    3. Endpoint /api/health dan /api/referensi
    4. Analisis penuh — pascabayar (skenario normal)
    5. Analisis penuh — prabayar (skenario normal)
    6. Bea Materai terpicu benar (>Rp5 juta, cuma pascabayar)
    7. Ambang toleransi anomali 29% diterapkan konsisten
    8. Tiga status khusus prabayar (tanggal_tidak_valid, data_belum_cukup,
       data_tidak_konsisten)
    9. DSM classifier bedakan AC vs Kulkas (sama-sama "Pendingin")
    10. Validasi input 24 jam maksimal (backend/schemas.py)

Narasi Gemini di-MOCK secara default (tidak makan kuota API, tidak butuh
internet) — supaya tes ini cepat dan bisa diulang-ulang. Kalau mau tes
Gemini API KEY asli beneran nyambung, jalankan dengan:
    python test_sistem.py --gemini-asli
(Butuh GEMINI_API_KEY valid di backend/.env)
"""

import os
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

GUNAKAN_GEMINI_ASLI = "--gemini-asli" in sys.argv

# PENTING: kalau tidak pakai Gemini asli, tetap WAJIB set dummy key di sini
# SEBELUM backend/main.py di-import — karena main.py membaca GEMINI_API_KEY
# dari environment SAAT MODUL DIIMPOR (baris atas file), bukan saat endpoint
# dipanggil. Tanpa ini, endpoint /api/analisis akan selalu balas 503
# "GEMINI_API_KEY belum dikonfigurasi" sebelum sempat mencapai kode yang
# di-mock di bawah.
if not GUNAKAN_GEMINI_ASLI and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = "dummy-key-untuk-tes-otomatis"

_lulus = []
_gagal = []


def cek(nama, fn):
    """Jalankan satu tes, catat hasilnya, JANGAN hentikan tes lain kalau gagal."""
    try:
        fn()
        print(f"  ✅ {nama}")
        _lulus.append(nama)
    except AssertionError as e:
        print(f"  ❌ {nama}")
        print(f"     -> Gagal: {e}")
        _gagal.append((nama, str(e)))
    except Exception as e:
        print(f"  ❌ {nama}")
        print(f"     -> Error tak terduga: {type(e).__name__}: {e}")
        print(f"     {traceback.format_exc().splitlines()[-2].strip()}")
        _gagal.append((nama, str(e)))


print("=" * 60)
print("TES KESIAPAN SISTEM — EnergiCerdas AI")
print("=" * 60)

# ── 1. Semua modul bisa diimpor ────────────────────────────────────────────
print("\n[1] Impor semua modul...")


def tes_impor():
    global optimasi, DSMClassifier, DataIngestionValidatorAgent
    global profil_ike, evaluasi_anomali_pascabayar, evaluasi_anomali_prabayar
    global hitung_tagihan, hitung_biaya_materai, BATAS_TOLERANSI_ANOMALI
    from core.kalkulasi import hitung_tagihan, hitung_biaya_materai
    from action_analist.ike_profiler import profil_ike
    from action_analist.anomaly_evaluator import (
        evaluasi_anomali_pascabayar, evaluasi_anomali_prabayar,
        BATAS_TOLERANSI_ANOMALI,
    )
    from services.ingestion import DataIngestionValidatorAgent
    from models.dsm_classifier import DSMClassifier
    from optimizer.brute_force import optimasi
    assert BATAS_TOLERANSI_ANOMALI == 0.29, "Ambang toleransi harusnya 0.29"


cek("Semua modul (core, action_analist, services, models, optimizer)", tes_impor)

# ── 2. Model DSM asli ────────────────────────────────────────────────────────
print("\n[2] Model DSM...")

_dsm = None


def tes_dsm_load():
    global _dsm
    _dsm = DSMClassifier()
    assert _dsm.siap, f"Model gagal dimuat: {_dsm.pesan_error}"


def tes_dsm_bedakan_kategori_sama():
    hasil = _dsm.prediksi_batch([
        {"nama": "AC", "kategori": "Pendingin", "tegangan": 220, "arus": 2.5, "jam": 8, "jumlah": 1},
        {"nama": "Kulkas", "kategori": "Pendingin", "tegangan": 220, "arus": 0.68, "jam": 24, "jumlah": 1},
    ])
    label = {h["nama"]: h["label_dsm"] for h in hasil}
    assert label["AC"] == "Fleksibel", f"AC harusnya Fleksibel, dapat {label['AC']}"
    assert label["Kulkas"] == "Tidak Fleksibel", f"Kulkas harusnya Tidak Fleksibel, dapat {label['Kulkas']}"


cek("Model DSM asli berhasil dimuat", tes_dsm_load)
cek("DSM bedakan AC vs Kulkas (kategori sama, label beda)", tes_dsm_bedakan_kategori_sama)

# ── 3-9: Setup mock Gemini + TestClient ───────────────────────────────────────
print("\n[3-9] Endpoint & skenario penuh...")

if not GUNAKAN_GEMINI_ASLI:
    import services.narasi as _narasi_mod

    class _FakeResponse:
        text = "[NARASI MOCK — tes kesiapan sistem, bukan Gemini asli]"

    class _FakeModel:
        def __init__(self, *a, **k):
            pass

        def generate_content(self, prompt):
            assert isinstance(prompt, str) and len(prompt) > 50
            return _FakeResponse()

    _narasi_mod.genai.configure = lambda **k: None
    _narasi_mod.genai.GenerativeModel = _FakeModel
    print("  (Gemini di-mock — pakai --gemini-asli untuk tes API key sungguhan)")
else:
    print("  (Memakai Gemini API key ASLI dari backend/.env)")

from fastapi.testclient import TestClient
from main import app

_client = TestClient(app)
_client.__enter__()  # trigger lifespan (load model)

ALAT_DASAR = [
    {"nama": "AC", "kategori": "Pendingin", "tegangan": 220, "arus": 2.5, "jam": 8, "jumlah": 1},
]


def tes_health():
    r = _client.get("/api/health")
    assert r.status_code == 200, f"status {r.status_code}"
    assert r.json()["dsm_model_siap"] is True


def tes_referensi():
    r = _client.get("/api/referensi")
    assert r.status_code == 200, f"status {r.status_code}"
    d = r.json()
    assert len(d["golongan_daya"]) > 0
    assert len(d["kategori_alat"]) == 8


def tes_pascabayar_normal():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 150000, "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    d = r.json()
    assert "biaya_materai" in d, "biaya_materai harus ada di response"
    assert "biaya_beban" not in d, "biaya_beban seharusnya sudah tidak ada"
    assert d["status_anomali"] in ("normal", "anomali")


def tes_prabayar_normal():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": True, "luas_rumah": 45, "penghuni": 3,
        "token_context": {
            "tanggal_pembelian": str(date.today() - timedelta(days=10)),
            "sisa_sebelum_beli": 5.0, "nominal_dibeli": 150000, "sisa_saat_ini": 20.0,
        },
        "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    d = r.json()
    assert d["biaya_materai"] == 0, "Prabayar tidak boleh kena materai"
    assert d["token_context"] is not None


def tes_materai_terpicu():
    # ~3500 kWh -> subtotal > Rp5 juta -> materai harus Rp10.000
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 5000000,
        "daftar_alat": [
            {"nama": "AC Besar", "kategori": "Pendingin", "tegangan": 220,
             "arus": 66.0, "jam": 24, "jumlah": 1},
        ],
        "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    d = r.json()
    assert d["biaya_materai"] == 10000, f"Materai harusnya Rp10.000, dapat {d['biaya_materai']}"


def tes_ambang_29_persen():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 50000,  # jauh beda dari estimasi -> pasti anomali
        "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    d = r.json()
    if d["selisih_pct"] is not None:
        expected = d["selisih_pct"] > 29.0
        assert d["is_anomali"] == expected, (
            f"Ambang tidak konsisten: selisih={d['selisih_pct']}, is_anomali={d['is_anomali']}"
        )


def tes_status_tanggal_tidak_valid():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": True, "luas_rumah": 45, "penghuni": 3,
        "token_context": {
            "tanggal_pembelian": str(date.today() + timedelta(days=5)),  # masa depan
            "sisa_sebelum_beli": 5.0, "nominal_dibeli": 150000, "sisa_saat_ini": 20.0,
        },
        "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    assert r.json()["status_anomali"] == "tanggal_tidak_valid", r.json().get("status_anomali")


def tes_status_data_belum_cukup():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": True, "luas_rumah": 45, "penghuni": 3,
        "token_context": {
            "tanggal_pembelian": str(date.today()),  # hari ini -> H+0
            "sisa_sebelum_beli": 5.0, "nominal_dibeli": 150000, "sisa_saat_ini": 20.0,
        },
        "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    assert r.json()["status_anomali"] == "data_belum_cukup", r.json().get("status_anomali")


def tes_status_data_tidak_konsisten():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": True, "luas_rumah": 45, "penghuni": 3,
        "token_context": {
            "tanggal_pembelian": str(date.today() - timedelta(days=5)),
            "sisa_sebelum_beli": 5.0, "nominal_dibeli": 10000,  # dikit
            "sisa_saat_ini": 999.0,  # sisa sekarang > saldo awal -> tidak masuk akal
        },
        "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    assert r.json()["status_anomali"] == "data_tidak_konsisten", r.json().get("status_anomali")


def tes_validasi_24_jam():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 150000,
        "daftar_alat": [
            {"nama": "AC", "kategori": "Pendingin", "tegangan": 220,
             "arus": 2.5, "jam": 30, "jumlah": 1},  # 30 jam -> tidak valid
        ],
        "intent_user": ["Biaya"],
    })
    assert r.status_code == 422, f"Harusnya ditolak (422), dapat {r.status_code}"


def tes_kombinasi_tidak_konsisten_ditolak():
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": True, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 150000,  # prabayar tapi kirim tagihan_asli, bukan token_context
        "daftar_alat": ALAT_DASAR, "intent_user": ["Biaya"],
    })
    assert r.status_code == 422, f"Harusnya ditolak (422), dapat {r.status_code}"


cek("GET /api/health", tes_health)
cek("GET /api/referensi", tes_referensi)
cek("POST /api/analisis — pascabayar normal", tes_pascabayar_normal)
cek("POST /api/analisis — prabayar normal", tes_prabayar_normal)
cek("Bea Materai terpicu benar (>Rp5 juta)", tes_materai_terpicu)
cek("Ambang toleransi 29% konsisten", tes_ambang_29_persen)
cek("Status 'tanggal_tidak_valid' (tanggal masa depan)", tes_status_tanggal_tidak_valid)
cek("Status 'data_belum_cukup' (H+0)", tes_status_data_belum_cukup)
cek("Status 'data_tidak_konsisten' (saldo tidak masuk akal)", tes_status_data_tidak_konsisten)
cek("Validasi jam > 24 ditolak (422)", tes_validasi_24_jam)
cek("Kombinasi is_prabayar/token_context tidak konsisten ditolak (422)", tes_kombinasi_tidak_konsisten_ditolak)

_client.__exit__(None, None, None)

# ── Ringkasan ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"HASIL: {len(_lulus)} lulus, {len(_gagal)} gagal")
print("=" * 60)

if _gagal:
    print("\nYang gagal:")
    for nama, err in _gagal:
        print(f"  - {nama}: {err}")
    print("\n❌ SISTEM BELUM SIAP — perbaiki dulu yang di atas sebelum deploy.")
    sys.exit(1)
else:
    print("\n✅ SEMUA TES LULUS — sistem siap untuk langkah berikutnya (deploy/demo).")
    sys.exit(0)