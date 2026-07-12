"""
preflight.py
============
Cek cepat sebelum menjalankan test atau deploy.

Menjawab satu pertanyaan: apakah semua file di folder ini sudah versi baru?

Kalau ada file lama yang tertinggal, pytest akan gagal dengan ImportError yang
tidak menjelaskan file mana yang salah. Skrip ini menyebutkan namanya.

Jalankan dari ROOT proyek (sejajar app.py):

    python preflight.py
"""

import importlib
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

OK = "  OK  "
GAGAL = " GAGAL"

# (modul, [simbol yang HARUS ada], [simbol yang HARUS SUDAH HILANG])
KONTRAK = [
    ("core.konstanta", [
        "TARIF_PER_GOLONGAN", "PBJT_RUMAH_TANGGA", "FAKTOR_EMISI_JAMALI_OM",
        "JAM_NYALA_RM", "JAM_NYALA_MAKS_TOKEN", "AMBANG_ALARM_KWH",
        "COS_PHI", "COS_PHI_DEFAULT", "KATEGORI_VALID", "BATAS_IKE",
    ], []),

    ("core.token", [
        "hitung_kwh_dari_token", "hitung_token_dari_kwh", "batas_maks_kwh",
        "validasi_pembelian", "proyeksi_token", "token_untuk_bertahan",
        "jadwal_isi_ulang", "bulatkan_nominal", "kombinasi_denominasi",
    ], []),

    ("core.kalkulasi", [
        "get_tarif", "hitung_kwh_minimum", "hitung_rm_rupiah",
        "hitung_watt", "hitung_watt_nyata", "hitung_kwh_alat",
        "hitung_tagihan", "hitung_biaya_bulanan", "hitung_rentang_biaya",
        "hitung_emisi", "hitung_ike", "hitung_kwh_per_org",
        "deteksi_anomali", "KETERBATASAN_PREDIKSI",
    ], [
        "hitung_biaya_beban",   # RM adalah lantai, bukan komponen tambahan
    ]),

    ("fuzzy.ike_profiler", ["profil_ike"], []),

    ("models.dsm_classifier", ["DSMClassifier"], []),

    ("optimizer.brute_force", [
        "optimasi", "ZONA_PERLU_OPTIMASI", "BATAS_IKE",
    ], [
        "_label_zona",          # label crisp diganti profil_ike()
    ]),

    ("services.pipeline", ["analisis", "DataIngestionValidatorAgent"], []),
    ("services.narasi", ["generate_narasi"], []),
]

BERKAS_WAJIB = [
    "app.py",
    "requirements.txt",
    "core/__init__.py", "core/konstanta.py", "core/kalkulasi.py", "core/token.py",
    "fuzzy/__init__.py", "fuzzy/ike_profiler.py",
    "models/__init__.py", "models/dsm_classifier.py",
    "optimizer/__init__.py", "optimizer/brute_force.py",
    "services/__init__.py", "services/pipeline.py", "services/narasi.py",
    "data/dsm_lightgbm_model.pkl", "data/dsm_target_encoder.pkl",
]


def cek_berkas(root: Path) -> int:
    print("\n[1] Keberadaan berkas")
    print("-" * 72)
    gagal = 0
    for rel in BERKAS_WAJIB:
        ada = (root / rel).exists()
        print(f"  [{OK if ada else GAGAL}] {rel}")
        if not ada:
            gagal += 1
    return gagal


def cek_kontrak() -> int:
    print("\n[2] Isi modul (mendeteksi berkas versi lama)")
    print("-" * 72)
    gagal = 0
    for nama, harus_ada, harus_hilang in KONTRAK:
        try:
            mod = importlib.import_module(nama)
        except Exception as e:                       # noqa: BLE001
            print(f"  [{GAGAL}] {nama}: tidak bisa diimpor -> {e}")
            gagal += 1
            continue

        hilang = [s for s in harus_ada if not hasattr(mod, s)]
        tersisa = [s for s in harus_hilang if hasattr(mod, s)]

        if not hilang and not tersisa:
            print(f"  [{OK}] {nama}")
            continue

        gagal += 1
        print(f"  [{GAGAL}] {nama}  <-- BERKAS INI MASIH VERSI LAMA")
        for s in hilang:
            print(f"           simbol belum ada  : {s}")
        for s in tersisa:
            print(f"           simbol harus hilang: {s}")
    return gagal


def cek_model() -> int:
    print("\n[3] Model DSM benar-benar dimuat")
    print("-" * 72)
    try:
        from models.dsm_classifier import DSMClassifier
        clf = DSMClassifier()
    except Exception as e:                           # noqa: BLE001
        print(f"  [{GAGAL}] DSMClassifier tidak bisa dibuat: {e}")
        return 1

    if clf.siap:
        print(f"  [{OK}] LightGBM aktif")
        return 0

    print(f"  [{GAGAL}] Model TIDAK dimuat: {clf.pesan_error}")
    print("           Sistem akan berjalan dengan fallback if-else, bukan LightGBM.")
    print("           Pickle butuh numpy >= 2.0 dan scikit-learn == 1.8.0.")
    return 1


def cek_rumus_kunci() -> int:
    """Dua rumus yang paling sering salah. Cek tanpa pytest."""
    print("\n[4] Rumus kunci")
    print("-" * 72)
    gagal = 0
    from core.kalkulasi import hitung_tagihan, get_tarif
    from core.konstanta import PBJT_RUMAH_TANGGA

    tarif = get_tarif(1300)

    r = hitung_tagihan(200.0, tarif, PBJT_RUMAH_TANGGA, 1300)
    harapan = 200.0 * tarif * (1 + PBJT_RUMAH_TANGGA)
    if abs(r["total"] - harapan) < 1.0:
        print(f"  [{OK}] Rekening Minimum diperlakukan sebagai lantai "
              f"(200 kWh -> Rp {r['total']:,.0f})")
    else:
        gagal += 1
        print(f"  [{GAGAL}] RM tampaknya masih dijumlahkan ke tagihan.")
        print(f"           dapat Rp {r['total']:,.0f}, seharusnya Rp {harapan:,.0f}")

    r30 = hitung_tagihan(30.0, tarif, PBJT_RUMAH_TANGGA, 1300)
    if r30["kena_rm"] and abs(r30["kwh_ditagih"] - 52.0) < 0.01:
        print(f"  [{OK}] Pemakaian 30 kWh ditagih pada RM 52 kWh")
    else:
        gagal += 1
        print(f"  [{GAGAL}] RM tidak diterapkan pada pemakaian di bawah 40 jam nyala")

    return gagal


def main() -> int:
    root = Path(__file__).resolve().parent
    print("=" * 72)
    print(f"PREFLIGHT EnergiCerdas AI  —  {root}")
    print("=" * 72)

    total = cek_berkas(root)
    total += cek_kontrak()
    if total == 0:
        total += cek_model()
        total += cek_rumus_kunci()
    else:
        print("\n[3] & [4] dilewati karena ada berkas yang belum diperbarui.")

    print("\n" + "=" * 72)
    if total == 0:
        print("SEMUA LOLOS. Silakan jalankan: python -m pytest test_energicerdas.py -v")
    else:
        print(f"{total} MASALAH. Salin ulang berkas yang ditandai GAGAL di atas,")
        print("lalu hapus semua folder __pycache__ dan ulangi.")
    print("=" * 72)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())