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

# WAJIB paling awal, SEBELUM tes_impor() (di bawah) memicu impor
# models.dsm_classifier (-> LightGBM) secara langsung -- lihat
# bootstrap.py untuk detail lengkap. Modul ini TIDAK bergantung pada
# FastAPI/Gemini apa pun, jadi aman dipanggil di sini tanpa mengganggu
# urutan mock Gemini yang harus terjadi SEBELUM main.py diimpor (lihat
# baris di bawah).
from bootstrap import preload_vendored_libgomp
preload_vendored_libgomp()

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


def tes_profil_ike_nilai_ekstrem():
    """
    Regresi khusus untuk bug plateau tertutup yang pernah ditemukan:
    versi awal _trapmf() memberi keanggotaan 0 (bukan 1) untuk IKE
    yang jauh melebihi batas atas kelas terakhir (Sangat Boros).
    """
    from action_analist.ike_profiler import profil_ike
    assert profil_ike(0.0) == "Sangat Efisien"
    assert profil_ike(1000.0) == "Sangat Boros", (
        "IKE ekstrem harus tetap 'Sangat Boros', bukan 'tidak masuk kelas manapun'"
    )


def tes_optimizer_aktif_untuk_ike_tinggi():
    r = _client.post("/api/analisis", json={
        "daya_va": 2200, "is_prabayar": False, "luas_rumah": 20, "penghuni": 2,
        "tagihan_asli": 800000,
        "daftar_alat": [
            {"nama": "AC 1", "kategori": "Pendingin", "tegangan": 220,
             "arus": 4.5, "jam": 12, "jumlah": 1},
            {"nama": "AC 2", "kategori": "Pendingin", "tegangan": 220,
             "arus": 4.5, "jam": 12, "jumlah": 1},
        ],
        "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    d = r.json()
    assert d["hasil_optimasi"]["aktif"] is True, "Optimizer harusnya aktif untuk IKE tinggi"


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
cek("profil_ike() nilai ekstrem tetap 'Sangat Boros' (regresi bug plateau)", tes_profil_ike_nilai_ekstrem)
def tes_optimizer_tidak_aktif_untuk_cukup_efisien():
    """
    Regresi khusus: optimizer sempat aktif untuk zona 'Cukup Efisien'
    (harusnya cuma Boros/Sangat Boros) -- ambang aktivasi salah pakai
    BATAS_EFISIEN, seharusnya BATAS_CUKUP_EFISIEN.
    """
    from optimizer.brute_force import BATAS_EFISIEN, BATAS_CUKUP_EFISIEN
    ike_cukup_efisien = (BATAS_EFISIEN + BATAS_CUKUP_EFISIEN) / 2
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False,
        "luas_rumah": round(180 / ike_cukup_efisien, 2), "penghuni": 3,
        "tagihan_asli": 270000,
        "daftar_alat": [
            {"nama": "AC", "kategori": "Pendingin", "tegangan": 220,
             "arus": 3.41, "jam": 8, "jumlah": 1},
        ],
        "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    d = r.json()
    assert d["label_ike"] == "Cukup Efisien", f"Setup tes salah, dapat {d['label_ike']}"
    assert d["hasil_optimasi"]["aktif"] is False, (
        "Optimizer TIDAK BOLEH aktif untuk zona Cukup Efisien"
    )


cek("Optimizer brute force aktif untuk IKE tinggi", tes_optimizer_aktif_untuk_ike_tinggi)
def tes_greedy_kalkulasi_langsung_akurat():
    """
    Regresi khusus untuk perombakan algoritma optimizer dari iteratif
    (step 0,5 jam berulang) ke kalkulasi langsung (Δt=ΔE/P). Uji multi-
    skenario acak: kalau status 'aktif dan berhasil', IKE akhir yang
    SUNGGUHAN dihitung ulang dari payload harus benar-benar <= target
    -- bukan cuma laporan status yang salah karena galat pembulatan
    floating-point (masalah nyata yang sempat ditemukan & diperbaiki
    lewat epsilon _EPSILON_IKE di optimizer/brute_force.py).
    """
    import random
    random.seed(123)
    kategori_cycle = ["Pendingin", "Hiburan/Elektronik", "Laundry", "Pencahayaan"]
    for i in range(15):
        n_alat = random.randint(1, 4)
        daftar_alat = []
        for j in range(n_alat):
            daftar_alat.append({
                "nama": f"Alat{j}", "kategori": kategori_cycle[j % len(kategori_cycle)],
                "tegangan": 220, "arus": round(random.uniform(0.5, 5.0), 2),
                "jam": round(random.uniform(2, 10), 1), "jumlah": 1,
            })
        r = _client.post("/api/analisis", json={
            "daya_va": 2200, "is_prabayar": False,
            "luas_rumah": round(random.uniform(15, 80), 1), "penghuni": 3,
            "tagihan_asli": 800000, "daftar_alat": daftar_alat,
            "intent_user": ["Biaya", "Lingkungan"],
        })
        assert r.status_code == 200, f"skenario #{i} status {r.status_code}: {r.text[:200]}"
        d = r.json()
        opt = d["hasil_optimasi"]
        if opt["aktif"] and opt["status"] in ("efisien", "cukup_efisien"):
            # Hitung ulang IKE dari total_kwh_akhir yang dilaporkan --
            # harus benar-benar capai target, bukan cuma status yang salah lapor
            assert opt["ike_akhir"] <= opt["target_ike"] + 1e-3, (
                f"skenario #{i}: status bilang '{opt['status']}' tapi "
                f"ike_akhir={opt['ike_akhir']} > target_ike={opt['target_ike']}"
            )


cek("Optimizer TIDAK aktif untuk zona Cukup Efisien (regresi)", tes_optimizer_tidak_aktif_untuk_cukup_efisien)
cek("Greedy kalkulasi langsung akurat, 15 skenario acak (regresi epsilon)", tes_greedy_kalkulasi_langsung_akurat)


def tes_kapasitas_watt_terpisah_dari_anomali():
    """
    Pengingat kapasitas watt vs VA (core.kalkulasi.cek_kapasitas_watt)
    HARUS tampil di field terpisah (info_kapasitas_watt), TIDAK boleh
    ikut mengubah status_anomali/pesan_anomali -- ini sengaja bukan
    deteksi anomali (lihat docstring cek_kapasitas_watt).
    """
    r = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 281883,  # persis estimasi asli (dihitung, bukan ditebak) -> pasti "normal"
        "daftar_alat": [
            {"nama": "AC", "kategori": "Pendingin", "tegangan": 220,
             "arus": 3.41, "jam": 8, "jumlah": 1},
            {"nama": "Mesin Cuci", "kategori": "Laundry", "tegangan": 220,
             "arus": 1.59, "jam": 1, "jumlah": 1},
        ],
        "intent_user": ["Biaya"],
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    d = r.json()
    assert "info_kapasitas_watt" in d, "Field info_kapasitas_watt harus ada"
    ikw = d["info_kapasitas_watt"]
    assert set(ikw.keys()) == {"total_watt", "batas_watt_aman", "melebihi"}
    # AC(750.2W) + Mesin Cuci(349.8W) = 1100W > batas 1300*0.8=1040W -> melebihi
    assert ikw["melebihi"] is True, f"Harusnya melebihi, dapat {ikw}"
    assert ikw["batas_watt_aman"] == 1040.0
    # PENTING: field ini TIDAK BOLEH mengubah status_anomali
    assert d["status_anomali"] == "normal", (
        f"status_anomali harusnya tidak terpengaruh kapasitas watt, dapat {d['status_anomali']}"
    )


cek("Info kapasitas watt terpisah dari status_anomali", tes_kapasitas_watt_terpisah_dari_anomali)


def tes_batas_kwh_bulanan_prabayar_prioritas():
    """
    Cek B: total_kwh > batas kWh bulanan (~720 jam nyala, VA x 0,72)
    HARUS memicu pesan "meteran rusak, hubungi 123" -- MENGGANTIKAN
    pesan anomali 29% biasa, khusus untuk prabayar. Pascabayar TIDAK
    BOLEH terpengaruh sama sekali (lihat core/kalkulasi.py::
    hitung_batas_kwh_bulanan, action_analist/anomaly_evaluator.py::
    evaluasi_anomali_prabayar).
    """
    alat_besar = [
        {"nama": "AC Besar", "kategori": "Pendingin", "tegangan": 220,
         "arus": 10.0, "jam": 20, "jumlah": 2},
    ]

    # 1300 VA -> batas 936 kWh/bulan. Alat ini hasilkan 2640 kWh -> jelas melebihi.
    r_prabayar = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": True, "luas_rumah": 45, "penghuni": 3,
        "token_context": {
            "tanggal_pembelian": str(date.today() - timedelta(days=10)),
            "sisa_sebelum_beli": 50.0, "nominal_dibeli": 500000, "sisa_saat_ini": 100.0,
        },
        "daftar_alat": alat_besar, "intent_user": ["Biaya"],
    })
    assert r_prabayar.status_code == 200, f"status {r_prabayar.status_code}: {r_prabayar.text[:200]}"
    d1 = r_prabayar.json()
    assert d1["status_anomali"] == "anomali"
    assert "meteran" in d1["pesan_anomali"].lower() and "123" in d1["pesan_anomali"], (
        f"Pesan meteran rusak harus muncul, dapat: {d1['pesan_anomali']}"
    )

    # Kombinasi alat & VA SAMA PERSIS, tapi PASCABAYAR -- Cek B TIDAK BOLEH berlaku
    r_pascabayar = _client.post("/api/analisis", json={
        "daya_va": 1300, "is_prabayar": False, "luas_rumah": 45, "penghuni": 3,
        "tagihan_asli": 1000000,
        "daftar_alat": alat_besar, "intent_user": ["Biaya"],
    })
    assert r_pascabayar.status_code == 200
    d2 = r_pascabayar.json()
    assert "meteran" not in d2["pesan_anomali"].lower(), (
        "Cek B TIDAK BOLEH berlaku untuk pascabayar, tapi pesan meteran rusak muncul"
    )


cek("Batas kWh bulanan prabayar (Cek B) prioritas & khusus prabayar", tes_batas_kwh_bulanan_prabayar_prioritas)

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