"""
test_energicerdas.py
====================
Test suite untuk EnergiCerdas AI (pytest, class-based).

Modul yang diuji
----------------
    core/konstanta.py       konstanta regulasi
    core/kalkulasi.py       rumus inti (RM sebagai lantai, daya nyata/semu)
    core/token.py           prabayar: token <-> kWh, proyeksi, isi ulang
    fuzzy/ike_profiler.py   Fuzzy Mamdani IKE — SATU-SATUNYA sumber label
    models/dsm_classifier.py LightGBM DSM classifier
    optimizer/brute_force.py Greedy penghematan marginal
    services/pipeline.py    Lapis 1 + orkestrasi Lapis 1-2-3

PERUBAHAN BESAR dari versi test sebelumnya
------------------------------------------
1. `hitung_biaya_beban()` SUDAH TIDAK ADA. Rekening Minimum bukan komponen
   yang dijumlahkan ke tagihan, melainkan LANTAI tagihan. Penggantinya
   `hitung_kwh_minimum()` dan `hitung_rm_rupiah()`. Test lama
   `test_estimasi_total_benar` yang mengasumsikan
   total = pemakaian + pbjt + biaya_beban justru MENGUNCI bug tersebut,
   jadi dihapus dan diganti test kontinuitas di titik 40 jam nyala.

2. `_label_zona()` di brute_force SUDAH DIHAPUS. Seluruh label berasal dari
   `profil_ike()`. TestHelperBruteForce lama tidak lagi relevan.

3. `optimasi()` berganti signature: menerima `penghuni`, `label_ike`,
   `daya_va`, `is_prabayar`, `biaya_awal`; tidak lagi menerima
   `biaya_beban`/`tagihan_awal`. Mengembalikan `biaya_akhir`
   (bukan `tagihan_akhir`), karena untuk prabayar nilainya nominal token.

4. `langkah` tidak lagi memuat `hemat_rp`. Di bawah Rekening Minimum,
   penghematan rupiah tidak linear terhadap kWh per peralatan, sehingga
   angka per-langkah akan menyesatkan. Penghematan rupiah hanya dilaporkan
   di tingkat total (`hemat_rp`).

5. Daya SEMU vs NYATA dipisah. `watt` (= V x I) tetap fitur model DSM;
   `watt_nyata` (= V x I x cos phi) dipakai untuk energi/tagihan/emisi.
   Ada test regresi yang menjaga agar keduanya tidak tertukar.

6. `deteksi_anomali()` sekarang berbasis RENTANG prediksi dan punya ARAH.

7. Test baru untuk `core/token.py` (prabayar) dan `services/pipeline.py`.

Jalankan
--------
    python -m pytest test_energicerdas.py -v
atau
    python test_energicerdas.py
"""

import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.kalkulasi as kalkulasi_mod
import optimizer.brute_force as bf_mod

from core.konstanta import (
    PBJT_RUMAH_TANGGA, FAKTOR_EMISI_JAMALI_OM, JAM_NYALA_RM,
    KATEGORI_VALID, COS_PHI, BATAS_IKE,
)
from core.kalkulasi import (
    get_tarif, hitung_kwh_minimum, hitung_rm_rupiah,
    hitung_watt, hitung_watt_nyata, hitung_kwh_alat,
    hitung_tagihan, hitung_biaya_bulanan, hitung_rentang_biaya,
    hitung_emisi, hitung_ike, hitung_kwh_per_org, deteksi_anomali,
    KETERBATASAN_PREDIKSI,
)
from core.token import (
    hitung_kwh_dari_token, hitung_token_dari_kwh, batas_maks_kwh,
    validasi_pembelian, proyeksi_token, token_untuk_bertahan,
    jadwal_isi_ulang, bulatkan_nominal, kombinasi_denominasi,
)
from fuzzy.ike_profiler import profil_ike
from models.dsm_classifier import DSMClassifier
from optimizer.brute_force import optimasi, ZONA_PERLU_OPTIMASI
from services.pipeline import analisis, DataIngestionValidatorAgent


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

ALAT_AC = {"nama": "AC 1 PK", "kategori": "Pendingin",
           "tegangan": 220.0, "arus": 4.0, "jam": 8.0}
ALAT_KULKAS = {"nama": "Kulkas 2 Pintu", "kategori": "Pendingin",
               "tegangan": 220.0, "arus": 1.5, "jam": 24.0}
ALAT_TV = {"nama": "TV LED", "kategori": "Hiburan/Elektronik",
           "tegangan": 220.0, "arus": 0.5, "jam": 6.0}
ALAT_LAMPU = {"nama": "Lampu LED", "kategori": "Pencahayaan",
              "tegangan": 220.0, "arus": 0.05, "jam": 10.0}
ALAT_MESIN_CUCI = {"nama": "Mesin Cuci", "kategori": "Laundry",
                   "tegangan": 220.0, "arus": 3.0, "jam": 2.0}
ALAT_LAMPU_12_UNIT = {"nama": "Lampu Taman", "kategori": "Pencahayaan",
                      "tegangan": 220.0, "arus": 0.05, "jam": 8.0, "jumlah": 12}

ALAT_INVALID_KATEGORI = {"nama": "Alat X", "kategori": "KategoriAsal",
                         "tegangan": 220.0, "arus": 1.0, "jam": 4.0}
ALAT_WATT_NOL = {"nama": "Alat Y", "kategori": "Pencahayaan",
                 "tegangan": 0.0, "arus": 0.0, "jam": 4.0}
ALAT_JAM_NOL = {"nama": "Alat Z", "kategori": "Pendingin",
                "tegangan": 220.0, "arus": 2.0, "jam": 0.0}
ALAT_JAM_LEBIH = {"nama": "Alat W", "kategori": "Pendingin",
                  "tegangan": 220.0, "arus": 2.0, "jam": 25.0}

# Pasangan khusus untuk menguji urutan greedy:
#   POMPA : kwh_bulan LEBIH BESAR (57,02) tapi marginal KECIL   (105,6 W)
#   AC_BSR: kwh_bulan lebih kecil  (50,49) tapi marginal BESAR   (841,5 W)
# Greedy berbasis kWh (versi lama) akan menguras POMPA lebih dulu — SALAH.
# Greedy berbasis penghematan marginal akan menguras AC_BSR lebih dulu — BENAR.
ALAT_POMPA = {"nama": "Pompa Sirkulasi", "kategori": "Pompa/Motor",
              "tegangan": 220.0, "arus": 0.6, "jam": 18.0}
ALAT_AC_BESAR = {"nama": "AC Besar", "kategori": "Pendingin",
                 "tegangan": 220.0, "arus": 4.5, "jam": 2.0}

TARIF_1300 = 1444.70


@pytest.fixture(scope="module")
def clf():
    return DSMClassifier()


# ══════════════════════════════════════════════════════════════════════════════
# 0. REGRESI — mengunci bug yang sudah diperbaiki agar tidak kembali
# ══════════════════════════════════════════════════════════════════════════════

class TestRegresi:

    def test_model_dsm_benar_benar_dimuat(self, clf):
        """
        PALING PENTING. Kalau joblib.load gagal (versi numpy/sklearn tidak
        cocok), DSMClassifier diam-diam jatuh ke fallback if-else dan
        aplikasi TETAP JALAN. Test ini membuat kegagalan itu terlihat.
        """
        assert clf.siap, (
            f"Model LightGBM GAGAL dimuat: {clf.pesan_error}\n"
            "Sistem akan berjalan dengan fallback if-else, BUKAN LightGBM. "
            "Cek versi: pickle butuh numpy >= 2.0 dan scikit-learn == 1.8.0."
        )

    def test_hitung_biaya_beban_sudah_dihapus(self):
        """
        `hitung_biaya_beban()` lama mengembalikan rupiah RM yang lalu
        DIJUMLAHKAN ke tagihan. Itu salah: RM adalah lantai, bukan tambahan.
        Fungsinya sengaja dihapus agar tidak ada yang memanggilnya lagi.
        """
        assert not hasattr(kalkulasi_mod, "hitung_biaya_beban"), (
            "hitung_biaya_beban() muncul kembali. RM adalah LANTAI tagihan, "
            "bukan komponen yang dijumlahkan. Pakai hitung_kwh_minimum()."
        )

    def test_label_zona_crisp_sudah_dihapus(self):
        """
        `_label_zona()` menghasilkan label crisp sendiri, sehingga UI bisa
        menampilkan dua label berbeda untuk IKE yang sama. Seluruh label
        sekarang berasal dari profil_ike().
        """
        assert not hasattr(bf_mod, "_label_zona"), (
            "_label_zona() muncul kembali. Satu-satunya sumber label IKE "
            "adalah fuzzy.profil_ike()."
        )

    def test_rm_tidak_dijumlahkan_ke_tagihan(self):
        """total = kwh_ditagih x tarif x (1 + pbjt). Tidak ada suku tambahan."""
        r = hitung_tagihan(200.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        expected = 200.0 * TARIF_1300 * (1 + PBJT_RUMAH_TANGGA)
        assert abs(r["total"] - expected) < 1.0
        assert r["kena_rm"] is False

    def test_tagihan_kontinu_di_titik_40_jam_nyala(self):
        """
        Di bawah 40 jam nyala tagihan datar di RM; di atasnya mengikuti
        pemakaian. Tidak boleh ada lompatan di titik peralihan.
        """
        kva = 1300 / 1000.0
        kwh_batas = JAM_NYALA_RM * kva          # 52,0 kWh
        eps = 1e-6
        bawah = hitung_tagihan(kwh_batas - eps, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        tepat = hitung_tagihan(kwh_batas,       TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        atas  = hitung_tagihan(kwh_batas + eps, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        assert abs(bawah["total"] - tepat["total"]) < 1.0
        assert abs(atas["total"] - tepat["total"]) < 1.0

    def test_watt_fitur_model_tetap_daya_semu(self, clf):
        """
        `watt` yang dikirim ke LightGBM WAJIB V x I (daya semu), karena model
        dilatih pada nilai itu. Kalau tertukar dengan watt_nyata, prediksi
        DSM tidak valid dan TIDAK akan memunculkan error apa pun.
        """
        hasil = clf.prediksi_batch([ALAT_AC])[0]
        assert hasil["watt"] == pytest.approx(220.0 * 4.0)          # 880 VA
        assert hasil["watt_nyata"] == pytest.approx(880.0 * COS_PHI["Pendingin"])
        assert hasil["watt"] > hasil["watt_nyata"]

    def test_energi_memakai_daya_nyata_bukan_semu(self, clf):
        """kwh_bulan harus dihitung dari watt_nyata, bukan watt."""
        h = clf.prediksi_batch([ALAT_AC])[0]
        expected = hitung_kwh_alat(h["watt_nyata"], ALAT_AC["jam"], 1)
        assert abs(h["kwh_bulan"] - expected) < 0.01

    def test_daftar_keterbatasan_prediksi_terisi(self):
        """Dosen pembimbing meminta sistem menegaskan bahwa ini prediksi."""
        assert len(KETERBATASAN_PREDIKSI) >= 5
        gabungan = " ".join(KETERBATASAN_PREDIKSI).lower()
        assert "cos phi" in gabungan or "daya semu" in gabungan
        assert "rata-rata" in gabungan          # rekening rata-rata PLN


# ══════════════════════════════════════════════════════════════════════════════
# 1. core/kalkulasi.py
# ══════════════════════════════════════════════════════════════════════════════

class TestCoreKalkulasi:

    @pytest.mark.parametrize("daya,tarif", [
        (900, 1352.00), (1300, 1444.70), (2200, 1444.70),
        (3500, 1699.53), (6600, 1699.53), (999999, 1699.53),
    ])
    def test_tarif_per_golongan(self, daya, tarif):
        assert get_tarif(daya) == tarif

    def test_hitung_watt_adalah_v_kali_i(self):
        assert hitung_watt(220, 3.5) == 770.0

    def test_hitung_watt_nyata_resistif_tidak_berubah(self):
        """cos phi = 1,0 untuk beban resistif (Memasak, Pemanas)."""
        assert hitung_watt_nyata(770.0, "Memasak") == 770.0
        assert hitung_watt_nyata(770.0, "Pemanas") == 770.0

    def test_hitung_watt_nyata_induktif_turun(self):
        assert hitung_watt_nyata(1000.0, "Pompa/Motor") == 800.0
        assert hitung_watt_nyata(1000.0, "Pendingin") == 850.0

    def test_hitung_watt_nyata_kategori_asing_pakai_default(self):
        assert hitung_watt_nyata(1000.0, "TidakAda") == 900.0

    def test_hitung_kwh_alat_jumlah_satu(self):
        assert hitung_kwh_alat(770, 8, 1) == round(770 * 8 * 30 / 1000, 4)

    def test_hitung_kwh_alat_jumlah_banyak(self):
        assert hitung_kwh_alat(100, 5, 10) == round(100 * 5 * 10 * 30 / 1000, 4)

    # --- Rekening Minimum ----------------------------------------------------

    def test_kwh_minimum_formula(self):
        """RM1 = 40 jam nyala x daya (kVA)."""
        assert hitung_kwh_minimum(1300) == pytest.approx(52.0)
        assert hitung_kwh_minimum(900) == pytest.approx(36.0)

    def test_kwh_minimum_prabayar_nol(self):
        assert hitung_kwh_minimum(1300, is_prabayar=True) == 0.0

    def test_rm_rupiah_1300va(self):
        """40 x 1,3 kVA x Rp 1.444,70 = Rp 75.124."""
        assert hitung_rm_rupiah(1300) == pytest.approx(75124.0, abs=1.0)

    def test_pemakaian_di_bawah_rm_ditagih_rm(self):
        r = hitung_tagihan(30.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        assert r["kena_rm"] is True
        assert r["kwh_ditagih"] == pytest.approx(52.0)
        assert r["jam_nyala"] < JAM_NYALA_RM

    def test_pemakaian_di_atas_rm_ditagih_apa_adanya(self):
        r = hitung_tagihan(200.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        assert r["kena_rm"] is False
        assert r["kwh_ditagih"] == pytest.approx(200.0)

    def test_tagihan_struktur_key(self):
        r = hitung_tagihan(100.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        assert set(r) == {"mode", "jam_nyala", "kena_rm", "kwh_minimum",
                          "kwh_ditagih", "biaya_pemakaian", "biaya_pbjt", "total"}
        assert "biaya_beban" not in r

    def test_pbjt_2_4_persen_dari_biaya_pemakaian(self):
        r = hitung_tagihan(200.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        assert abs(r["biaya_pbjt"] - r["biaya_pemakaian"] * 0.024) < 1.0

    # --- Biaya bulanan: prabayar vs pascabayar -------------------------------

    def test_biaya_bulanan_pascabayar_sama_dengan_hitung_tagihan(self):
        a = hitung_biaya_bulanan(200.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, False)
        b = hitung_tagihan(200.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300)
        assert a["total"] == b["total"]

    def test_biaya_bulanan_prabayar_adalah_nominal_token(self):
        kwh = 200.0
        a = hitung_biaya_bulanan(kwh, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, True)
        assert a["total"] == hitung_token_dari_kwh(kwh, TARIF_1300, PBJT_RUMAH_TANGGA)
        assert a["kena_rm"] is False

    def test_prabayar_sedikit_lebih_mahal_per_kwh(self):
        """
        Basis pajak berbeda menurut Perda DKI No.1/2024:
            pascabayar : PBJT atas biaya pemakaian -> TDL x (1 + t)
            prabayar   : PBJT atas nominal voucer  -> TDL / (1 - t)
        Selisihnya kecil tapi nyata (~0,06%).
        """
        pasca = TARIF_1300 * (1 + PBJT_RUMAH_TANGGA)
        pra = TARIF_1300 / (1 - PBJT_RUMAH_TANGGA)
        assert pra > pasca
        assert (pra / pasca - 1) < 0.001

    def test_pascabayar_dan_prabayar_selisih_tipis_di_atas_rm(self):
        """
        Guard silang untuk bug RM aditif. Di atas 40 jam nyala, tagihan
        pascabayar dan nominal token prabayar untuk kWh yang sama hanya boleh
        berbeda < 0,1% (karena basis PBJT). Kalau RM dijumlahkan kembali,
        selisihnya melonjak puluhan persen dan test ini gagal.
        """
        kwh = 200.0
        pasca = hitung_biaya_bulanan(kwh, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, False)["total"]
        pra = hitung_biaya_bulanan(kwh, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, True)["total"]
        assert abs(pra / pasca - 1) < 0.001, (
            f"Selisih prabayar vs pascabayar {abs(pra/pasca-1):.1%} — "
            "kemungkinan Rekening Minimum dijumlahkan lagi ke tagihan."
        )

    def test_prabayar_tidak_pernah_kena_rm(self):
        r = hitung_biaya_bulanan(10.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, True)
        assert r["kena_rm"] is False
        assert r["kwh_ditagih"] == pytest.approx(10.0)

    # --- Rentang prediksi ----------------------------------------------------

    def test_rentang_biaya_bawah_lebih_kecil_dari_atas(self):
        r = hitung_rentang_biaya(100.0, 120.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, False)
        assert r["biaya_bawah"] < r["biaya_atas"]
        assert r["biaya_bawah"] <= r["biaya_tengah"] <= r["biaya_atas"]
        assert r["lebar_pct"] > 0

    def test_rentang_biaya_menyempit_bila_semua_resistif(self):
        """kwh nyata == kwh semu -> rentang selebar nol."""
        r = hitung_rentang_biaya(100.0, 100.0, TARIF_1300, PBJT_RUMAH_TANGGA, 1300, False)
        assert r["lebar_pct"] == 0.0

    # --- Emisi, IKE, kWh/orang ----------------------------------------------

    def test_faktor_emisi_jamali(self):
        assert FAKTOR_EMISI_JAMALI_OM == 0.80

    def test_emisi_formula(self):
        assert abs(hitung_emisi(100.0)["emisi_kg_bulan"] - 80.0) < 0.01

    def test_emisi_tahunan_12x_bulanan(self):
        e = hitung_emisi(100.0)
        assert abs(e["emisi_kg_tahun"] - e["emisi_kg_bulan"] * 12) < 0.01

    def test_emisi_nol_kwh(self):
        assert hitung_emisi(0.0)["emisi_kg_bulan"] == 0.0

    def test_hitung_ike(self):
        assert hitung_ike(300, 45) == round(300 / 45, 4)

    def test_hitung_ike_luas_nol_tidak_error(self):
        assert hitung_ike(300, 0) == 300.0     # dibatasi max(1.0, luas)

    def test_hitung_kwh_per_org(self):
        assert hitung_kwh_per_org(300, 3) == round(300 / 3, 2)

    def test_hitung_kwh_per_org_penghuni_nol_tidak_error(self):
        assert hitung_kwh_per_org(300, 0) == 300.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. deteksi_anomali — berbasis RENTANG dan punya ARAH
# ══════════════════════════════════════════════════════════════════════════════

class TestDeteksiAnomali:

    BAWAH, ATAS = 400_000.0, 500_000.0

    def test_di_dalam_rentang_wajar(self):
        r = deteksi_anomali(450_000, self.BAWAH, self.ATAS)
        assert r["is_anomali"] is False
        assert r["arah"] == "wajar"
        assert r["selisih_pct"] == 0.0

    def test_batas_rentang_inklusif(self):
        assert deteksi_anomali(self.BAWAH, self.BAWAH, self.ATAS)["arah"] == "wajar"
        assert deteksi_anomali(self.ATAS, self.BAWAH, self.ATAS)["arah"] == "wajar"

    def test_di_atas_rentang_arah_benar(self):
        r = deteksi_anomali(700_000, self.BAWAH, self.ATAS)
        assert r["is_anomali"] is True
        assert r["arah"] == "tagihan_lebih_tinggi"
        assert "kebocoran" in r["pesan"].lower()

    def test_di_bawah_rentang_arah_benar(self):
        """
        BUG LAMA: pesan selalu berbunyi 'kebocoran arus' padahal itu hanya
        berlaku bila tagihan lebih TINGGI dari prediksi.
        """
        r = deteksi_anomali(200_000, self.BAWAH, self.ATAS)
        assert r["is_anomali"] is True
        assert r["arah"] == "tagihan_lebih_rendah"
        assert "kebocoran" not in r["pesan"].lower()
        assert "rata-rata" in r["pesan"].lower()

    def test_di_luar_rentang_tapi_masih_toleransi(self):
        """510.000 hanya 2% di atas batas atas -> belum anomali."""
        r = deteksi_anomali(510_000, self.BAWAH, self.ATAS)
        assert r["is_anomali"] is False
        assert r["arah"] == "tagihan_lebih_tinggi"

    def test_selisih_diukur_dari_tepi_rentang_bukan_titik(self):
        r = deteksi_anomali(600_000, self.BAWAH, self.ATAS)
        expected = (600_000 - self.ATAS) / self.ATAS * 100
        assert abs(r["selisih_pct"] - expected) < 0.1

    def test_key_lengkap(self):
        r = deteksi_anomali(600_000, self.BAWAH, self.ATAS)
        assert set(r) == {"is_anomali", "arah", "selisih_pct", "pesan"}


# ══════════════════════════════════════════════════════════════════════════════
# 3. core/token.py — prabayar
# ══════════════════════════════════════════════════════════════════════════════

class TestCoreToken:

    def test_ppj_dihitung_dari_nominal_voucer(self):
        """
        Perda DKI No.1/2024: bila pembayaran memakai voucer, dasar pengenaan
        PBJT adalah nilai rupiah voucer. Jadi PPJ = t x nominal, bukan
        di-gross-up dari nilai stroom.
        """
        r = hitung_kwh_dari_token(100_000, TARIF_1300)
        assert r["ppj_rp"] == pytest.approx(2400.0)
        assert r["stroom_rp"] == pytest.approx(97_600.0)
        assert r["kwh"] == pytest.approx(97_600.0 / TARIF_1300, abs=0.01)

    @pytest.mark.parametrize("nominal", [20_000, 50_000, 100_000, 200_000, 500_000])
    def test_round_trip_token_presisi(self, nominal):
        """kwh_exact (tanpa pembulatan) harus balik persis ke nominal."""
        kwh = hitung_kwh_dari_token(nominal, TARIF_1300)["kwh_exact"]
        assert hitung_token_dari_kwh(kwh, TARIF_1300) == pytest.approx(nominal, abs=1.0)

    def test_kwh_tampil_dibulatkan_dua_desimal(self):
        r = hitung_kwh_dari_token(100_000, TARIF_1300)
        assert r["kwh"] == round(r["kwh_exact"], 2)

    def test_biaya_admin_tidak_mengurangi_stroom(self):
        """Sejak 2015 biaya admin dipungut di luar nominal voucer."""
        tanpa = hitung_kwh_dari_token(100_000, TARIF_1300, biaya_transaksi=0)
        dengan = hitung_kwh_dari_token(100_000, TARIF_1300, biaya_transaksi=3_000)
        assert tanpa["kwh"] == dengan["kwh"]
        assert dengan["total_dibayar"] == 103_000

    def test_token_dari_kwh_nol(self):
        assert hitung_token_dari_kwh(0, TARIF_1300) == 0.0

    def test_batas_maks_kwh(self):
        """batas = daya(VA) x 720 jam / 1000."""
        assert batas_maks_kwh(1300) == pytest.approx(936.0)
        assert batas_maks_kwh(900) == pytest.approx(648.0)

    def test_validasi_pembelian_diterima(self):
        r = validasi_pembelian(100_000, 1300, TARIF_1300)
        assert r["diterima"] is True
        assert r["pesan"] == ""

    def test_validasi_pembelian_ditolak_melebihi_batas_meteran(self):
        r = validasi_pembelian(5_000_000, 900, TARIF_1300)
        assert r["diterima"] is False
        assert "melebihi batas" in r["pesan"]

    # --- Pembulatan & denominasi --------------------------------------------

    def test_bulatkan_nominal_ke_atas(self):
        assert bulatkan_nominal(706_926) == 710_000
        assert bulatkan_nominal(15_000) == 20_000
        assert bulatkan_nominal(0) == 0

    def test_bulatkan_nominal_tidak_pernah_kurang(self):
        for n in (1, 9_999, 100_001, 362_581):
            assert bulatkan_nominal(n) >= n

    def test_kombinasi_denominasi(self):
        assert kombinasi_denominasi(710_000) == [500_000, 200_000, 20_000]
        assert kombinasi_denominasi(20_000) == [20_000]
        assert kombinasi_denominasi(0) == []

    def test_kombinasi_denominasi_menutupi_nominal(self):
        for n in (30_000, 370_000, 1_250_000):
            assert sum(kombinasi_denominasi(n)) >= n

    # --- Proyeksi sisa token -------------------------------------------------

    def test_proyeksi_hari_tersisa(self):
        r = proyeksi_token(45.0, 9.0, date(2026, 7, 10))
        assert r["valid"] is True
        assert r["hari_tersisa"] == pytest.approx(5.0)
        assert r["tanggal_habis"] == date(2026, 7, 15)

    def test_proyeksi_status_kritis_waspada_aman(self):
        t = date(2026, 7, 10)
        assert proyeksi_token(10.0, 5.0, t)["status"] == "kritis"      # 2 hari
        assert proyeksi_token(30.0, 5.0, t)["status"] == "waspada"     # 6 hari
        assert proyeksi_token(100.0, 5.0, t)["status"] == "aman"       # 20 hari

    def test_proyeksi_tanggal_alarm_sebelum_tanggal_habis(self):
        r = proyeksi_token(45.0, 9.0, date(2026, 7, 10))
        assert r["tanggal_alarm"] <= r["tanggal_habis"]
        assert r["hari_ke_alarm"] <= r["hari_tersisa"]

    def test_proyeksi_konsumsi_nol_tidak_membagi_nol(self):
        r = proyeksi_token(45.0, 0.0, date(2026, 7, 10))
        assert r["valid"] is False

    def test_proyeksi_sisa_negatif_ditolak(self):
        r = proyeksi_token(-5.0, 9.0, date(2026, 7, 10))
        assert r["valid"] is False

    def test_token_untuk_bertahan(self):
        r = token_untuk_bertahan(10.0, 30, TARIF_1300, sisa_kwh=45.0)
        assert r["kwh_butuh"] == pytest.approx(255.0)   # 300 - 45
        assert r["nominal_rp"] == hitung_token_dari_kwh(255.0, TARIF_1300)

    def test_token_untuk_bertahan_sisa_sudah_cukup(self):
        r = token_untuk_bertahan(1.0, 10, TARIF_1300, sisa_kwh=500.0)
        assert r["kwh_butuh"] == 0.0
        assert r["nominal_rp"] == 0.0

    def test_jadwal_isi_ulang_key_lengkap(self):
        r = jadwal_isi_ulang(45.0, 9.0, date(2026, 7, 10), TARIF_1300, 1300)
        for k in ("hari_tersisa", "tanggal_habis", "tanggal_alarm", "status",
                  "kwh_perlu_dibeli", "nominal_pas", "nominal_disarankan",
                  "kombinasi", "pembelian_valid"):
            assert k in r, f"key '{k}' hilang dari jadwal_isi_ulang()"

    def test_jadwal_isi_ulang_nominal_disarankan_bulat(self):
        r = jadwal_isi_ulang(45.0, 9.0, date(2026, 7, 10), TARIF_1300, 1300)
        assert r["nominal_disarankan"] % 10_000 == 0
        assert r["nominal_disarankan"] >= r["nominal_pas"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. fuzzy/ike_profiler.py
# ══════════════════════════════════════════════════════════════════════════════

class TestIkeProfiler:

    VALID = {"Sangat Efisien", "Efisien", "Cukup Efisien", "Boros", "Sangat Boros"}

    @staticmethod
    def _ike_kpo(luas, penghuni, kwh):
        return hitung_ike(kwh, luas), hitung_kwh_per_org(kwh, penghuni)

    def test_output_selalu_label_valid(self):
        for kwh in (0.1, 10, 50, 100, 200, 500, 5000):
            ike, kpo = self._ike_kpo(50.0, 3, float(kwh))
            assert profil_ike(ike, kpo, False) in self.VALID
            assert profil_ike(ike, kpo, True) in self.VALID

    def test_sangat_efisien_tanpa_ac(self):
        ike, kpo = self._ike_kpo(50.0, 5, 30.0)      # IKE 0,6
        assert profil_ike(ike, kpo, False) == "Sangat Efisien"

    def test_boros_tanpa_ac(self):
        ike, kpo = self._ike_kpo(50.0, 3, 200.0)     # IKE 4,0
        assert profil_ike(ike, kpo, False) in {"Boros", "Sangat Boros"}

    def test_sangat_efisien_dengan_ac(self):
        ike, kpo = self._ike_kpo(100.0, 4, 500.0)    # IKE 5,0
        assert profil_ike(ike, kpo, True) == "Sangat Efisien"

    def test_boros_dengan_ac(self):
        ike, kpo = self._ike_kpo(50.0, 2, 1000.0)    # IKE 20,0
        assert profil_ike(ike, kpo, True) in {"Boros", "Sangat Boros"}

    def test_ike_ekstrem_tidak_kembali_ke_sangat_efisien(self):
        """
        Regresi bug trapmf: batas atas himpunan 'tinggi' yang terlalu sempit
        membuat nilai ekstrem jatuh ke keanggotaan 0 dan salah dilabeli
        'Sangat Efisien'. Batas atas sekarang 1.000.000.
        """
        ike, kpo = self._ike_kpo(50.0, 3, 5000.0)    # IKE 100,0
        assert profil_ike(ike, kpo, False) == "Sangat Boros"

    def test_kwh_mendekati_nol(self):
        ike, kpo = self._ike_kpo(50.0, 3, 0.1)
        assert profil_ike(ike, kpo, False) == "Sangat Efisien"

    def test_deterministik(self):
        ike, kpo = self._ike_kpo(60.0, 3, 120.0)
        assert profil_ike(ike, kpo, False) == profil_ike(ike, kpo, False)

    def test_ac_mengubah_ambang(self):
        ike, kpo = self._ike_kpo(50.0, 3, 150.0)     # IKE 3,0
        assert profil_ike(ike, kpo, False) != profil_ike(ike, kpo, True)

    def test_monoton_naik_terhadap_ike(self):
        """Menaikkan IKE tidak boleh membuat label jadi lebih efisien."""
        urutan = ["Sangat Efisien", "Efisien", "Cukup Efisien", "Boros", "Sangat Boros"]
        sebelumnya = -1
        for ike in [0.2, 1.0, 2.0, 3.0, 5.0, 20.0]:
            idx = urutan.index(profil_ike(ike, ike * 15, False))
            assert idx >= sebelumnya, f"Label turun peringkat pada IKE={ike}"
            sebelumnya = idx


# ══════════════════════════════════════════════════════════════════════════════
# 5. models/dsm_classifier.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDSMClassifier:

    LABELS = {"Fleksibel", "Tidak Fleksibel"}
    METODE = {"model", "fallback"}
    KEYS = {"nama", "kategori", "tegangan", "arus", "watt", "watt_nyata",
            "jam", "jumlah", "kwh_bulan", "label_dsm", "metode", "valid", "pesan"}

    def test_output_key_lengkap(self, clf):
        h = clf.prediksi_batch([ALAT_AC])
        assert len(h) == 1
        assert self.KEYS.issubset(h[0].keys()), f"Key hilang: {self.KEYS - h[0].keys()}"

    def test_label_dan_metode_valid(self, clf):
        for item in clf.prediksi_batch([ALAT_AC, ALAT_TV, ALAT_LAMPU, ALAT_MESIN_CUCI]):
            assert item["label_dsm"] in self.LABELS
            assert item["metode"] in self.METODE

    def test_batch_kosong(self, clf):
        assert clf.prediksi_batch([]) == []

    def test_batch_panjang(self, clf):
        daftar = [ALAT_AC, ALAT_KULKAS, ALAT_TV, ALAT_LAMPU, ALAT_MESIN_CUCI]
        assert len(clf.prediksi_batch(daftar)) == 5

    def test_prediksi_batch_konsisten_dengan_baris_tunggal(self, clf):
        """
        LightGBM menyimpan urutan kategori data latih (pandas_categorical) dan
        me-remap saat predict. Batch yang tidak memuat seluruh 8 kategori
        harus menghasilkan label yang sama dengan prediksi satu-satu.
        """
        daftar = [ALAT_AC, ALAT_TV, ALAT_MESIN_CUCI]
        batch = [r["label_dsm"] for r in clf.prediksi_batch(daftar)]
        satuan = [clf.prediksi_batch([a])[0]["label_dsm"] for a in daftar]
        assert batch == satuan

    def test_semua_kategori_valid_dapat_diprediksi(self, clf):
        daftar = [{"nama": f"Alat {k}", "kategori": k,
                   "tegangan": 220.0, "arus": 1.0, "jam": 4.0}
                  for k in KATEGORI_VALID]
        hasil = clf.prediksi_batch(daftar)
        assert len(hasil) == len(KATEGORI_VALID)
        for item in hasil:
            assert item["valid"] is True
            assert item["label_dsm"] in self.LABELS

    # --- Validasi input ------------------------------------------------------

    def test_kategori_invalid_pakai_fallback(self, clf):
        h = clf.prediksi_batch([ALAT_INVALID_KATEGORI])[0]
        assert h["valid"] is False
        assert h["metode"] == "fallback"
        assert h["label_dsm"] in self.LABELS

    def test_watt_nol_tidak_valid(self, clf):
        assert clf.prediksi_batch([ALAT_WATT_NOL])[0]["valid"] is False

    def test_jam_nol_tidak_valid(self, clf):
        assert clf.prediksi_batch([ALAT_JAM_NOL])[0]["valid"] is False

    def test_jam_lebih_24_tidak_valid(self, clf):
        assert clf.prediksi_batch([ALAT_JAM_LEBIH])[0]["valid"] is False

    def test_nama_dipertahankan(self, clf):
        assert clf.prediksi_batch([ALAT_AC])[0]["nama"] == ALAT_AC["nama"]

    # --- Field 'jumlah' ------------------------------------------------------

    def test_jumlah_default_satu(self, clf):
        assert clf.prediksi_batch([ALAT_AC])[0]["jumlah"] == 1

    def test_jumlah_mengalikan_kwh_bukan_watt(self, clf):
        """
        Fitur yang dikirim ke model tetap PER-UNIT supaya tidak keluar dari
        rentang data latih. Hanya kWh yang dikalikan jumlah.
        """
        h = clf.prediksi_batch([ALAT_LAMPU_12_UNIT])[0]
        watt_unit = 220.0 * 0.05
        assert h["watt"] == pytest.approx(watt_unit)
        assert h["jumlah"] == 12
        expected = hitung_kwh_alat(h["watt_nyata"], 8.0, 12)
        assert abs(h["kwh_bulan"] - expected) < 0.01

    # --- ringkasan_dsm -------------------------------------------------------

    def test_ringkasan_key(self, clf):
        r = clf.ringkasan_dsm(clf.prediksi_batch([ALAT_AC, ALAT_TV]))
        assert set(r) == {"fleksibel", "tidak_fleksibel", "total_kwh_fleksibel"}

    def test_ringkasan_partisi_lengkap(self, clf):
        daftar = [ALAT_AC, ALAT_TV, ALAT_LAMPU, ALAT_MESIN_CUCI]
        r = clf.ringkasan_dsm(clf.prediksi_batch(daftar))
        assert len(r["fleksibel"]) + len(r["tidak_fleksibel"]) == len(daftar)

    def test_ringkasan_total_kwh_fleksibel_konsisten(self, clf):
        hasil = clf.prediksi_batch([ALAT_AC, ALAT_KULKAS, ALAT_TV, ALAT_MESIN_CUCI])
        r = clf.ringkasan_dsm(hasil)
        manual = sum(a["kwh_bulan"] for a in hasil if a["label_dsm"] == "Fleksibel")
        assert abs(r["total_kwh_fleksibel"] - manual) < 0.01

    # --- Fallback rule -------------------------------------------------------

    @pytest.mark.parametrize("kategori,label", [
        ("Pendingin", "Fleksibel"), ("Pemanas", "Fleksibel"),
        ("Laundry", "Fleksibel"), ("Pompa/Motor", "Fleksibel"),
        ("Memasak", "Tidak Fleksibel"), ("Hiburan/Elektronik", "Tidak Fleksibel"),
        ("Pencahayaan", "Tidak Fleksibel"), ("Lainnya", "Tidak Fleksibel"),
    ])
    def test_fallback_rule(self, clf, kategori, label):
        assert clf._fallback(kategori) == label

    def test_fallback_kategori_asing(self, clf):
        assert clf._fallback("TidakAda") == "Tidak Fleksibel"


# ══════════════════════════════════════════════════════════════════════════════
# 6. optimizer/brute_force.py
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizer:

    KEYS = {"aktif", "status", "zona_awal", "zona_akhir", "ike_awal", "ike_akhir",
            "target_ike", "total_kwh_akhir", "biaya_akhir", "emisi_akhir",
            "hemat_kwh", "hemat_rp", "hemat_emisi_kg", "persen_hemat_rp",
            "persen_hemat_emisi", "langkah", "pesan"}

    def _jalankan(self, clf, daftar, luas, penghuni=3, daya_va=1300, is_prabayar=False):
        """Meniru persis alur services/pipeline.analisis()."""
        hasil = clf.prediksi_batch(daftar)
        ringkasan = clf.ringkasan_dsm(hasil)
        kwh = round(sum(a["kwh_bulan"] for a in hasil), 3)
        ada_ac = any(a["kategori"] == "Pendingin" for a in daftar)
        tarif = get_tarif(daya_va)
        label = profil_ike(hitung_ike(kwh, luas), hitung_kwh_per_org(kwh, penghuni), ada_ac)
        biaya = hitung_biaya_bulanan(kwh, tarif, PBJT_RUMAH_TANGGA, daya_va, is_prabayar)["total"]
        return optimasi(
            ringkasan_dsm=ringkasan, luas_m2=luas, penghuni=penghuni, ada_ac=ada_ac,
            label_ike=label, tarif_kwh=tarif, pbjt=PBJT_RUMAH_TANGGA,
            daya_va=daya_va, is_prabayar=is_prabayar, kwh_awal=kwh,
            biaya_awal=biaya, emisi_awal=hitung_emisi(kwh)["emisi_kg_bulan"],
        ), label

    # --- Gerbang aktivasi ----------------------------------------------------

    def test_zona_perlu_optimasi_terdefinisi(self):
        assert ZONA_PERLU_OPTIMASI == {"Cukup Efisien", "Boros", "Sangat Boros"}

    def test_tidak_aktif_saat_sudah_efisien(self, clf):
        o, label = self._jalankan(clf, [ALAT_TV, ALAT_LAMPU], luas=50.0)
        assert label == "Sangat Efisien"
        assert o["aktif"] is False
        assert o["status"] == "sudah_efisien"
        assert o["langkah"] == []

    def test_aktif_saat_boros(self, clf):
        o, label = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        assert label == "Boros"
        assert o["aktif"] is True
        assert o["langkah"]

    def test_gerbang_memakai_label_fuzzy(self, clf):
        """
        Optimizer menyala berdasarkan LABEL FUZZY, bukan ambang crisp.
        Konsekuensinya, rumah dengan IKE 15,4 (crisp Depdiknas: Boros) tetap
        dilewati karena fuzzy melabelinya 'Efisien'. Perilaku ini disengaja
        agar tidak ada dua label berbeda di satu layar.

        Kalau test ini gagal, berarti gerbang aktivasi diubah kembali ke
        ambang numerik dan konsistensi label perlu ditinjau ulang.
        """
        o, label = self._jalankan(clf, [ALAT_POMPA, ALAT_AC_BESAR], luas=7.0)
        assert label == "Efisien"
        assert o["ike_awal"] > BATAS_IKE["ac"]["efisien"]   # crisp: sudah Boros
        assert o["aktif"] is False

    def test_output_key_lengkap_saat_tidak_aktif(self, clf):
        o, _ = self._jalankan(clf, [ALAT_TV, ALAT_LAMPU], luas=50.0)
        assert self.KEYS.issubset(o.keys()), f"Key hilang: {self.KEYS - o.keys()}"

    def test_output_key_lengkap_saat_aktif(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        assert self.KEYS.issubset(o.keys())

    def test_tagihan_akhir_sudah_diganti_biaya_akhir(self, clf):
        """Untuk prabayar nilainya nominal token, bukan tagihan."""
        o, _ = self._jalankan(clf, [ALAT_AC] * 3, luas=30.0)
        assert "biaya_akhir" in o
        assert "tagihan_akhir" not in o

    def test_tanpa_peralatan_fleksibel_tidak_tercapai(self, clf):
        o, label = self._jalankan(clf, [ALAT_TV, ALAT_LAMPU], luas=5.0)
        assert label == "Boros"
        assert o["aktif"] is True
        assert o["status"] == "tidak_tercapai"
        assert o["langkah"] == []

    # --- Urutan greedy: penghematan MARGINAL, bukan kWh bulanan --------------

    def test_greedy_memilih_penghematan_marginal_terbesar(self, clf):
        """
        POMPA  : kwh_bulan 57,02  marginal 105,6 W
        AC_BSR : kwh_bulan 50,49  marginal 841,5 W

        Greedy lama (urut kwh_bulan) akan menguras POMPA lebih dulu.
        Greedy marginal harus menguras AC_BSR sampai batas minimumnya
        sebelum menyentuh POMPA, karena satu jam AC_BSR menghemat 8x lipat.
        """
        o, label = self._jalankan(clf, [ALAT_POMPA, ALAT_AC_BESAR], luas=6.4)
        assert label in ZONA_PERLU_OPTIMASI
        assert o["aktif"] is True

        langkah = {l["nama"]: l for l in o["langkah"]}
        assert "AC Besar" in langkah, "AC Besar seharusnya dikurangi lebih dulu"

        ac = langkah["AC Besar"]
        assert ac["jam_rekomendasi"] == pytest.approx(1.0), (
            "AC Besar harus dikuras sampai batas minimum (50% dari 2 jam) "
            "sebelum POMPA disentuh."
        )
        if "Pompa Sirkulasi" in langkah:
            pompa = langkah["Pompa Sirkulasi"]
            assert pompa["jam_rekomendasi"] > 9.0, (
                "POMPA tidak boleh dikuras habis selama AC Besar masih bisa "
                "dikurangi — itu tanda greedy masih memakai kwh_bulan."
            )

    def test_langkah_diurutkan_penghematan_terbesar(self, clf):
        o, _ = self._jalankan(clf, [ALAT_POMPA, ALAT_AC_BESAR], luas=6.4)
        hemat = [l["hemat_kwh"] for l in o["langkah"]]
        assert hemat == sorted(hemat, reverse=True)

    # --- Invarian hasil ------------------------------------------------------

    def test_langkah_format(self, clf):
        REQUIRED = {"nama", "kategori", "jam_awal", "jam_rekomendasi",
                    "kurang_jam", "hemat_kwh", "hemat_emisi_kg"}
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        for l in o["langkah"]:
            assert REQUIRED.issubset(l.keys())
            assert "hemat_rp" not in l, (
                "hemat_rp per langkah dihapus: di bawah Rekening Minimum "
                "penghematan rupiah tidak linear terhadap kWh."
            )

    def test_jam_rekomendasi_selalu_turun(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        for l in o["langkah"]:
            assert l["jam_rekomendasi"] < l["jam_awal"]

    def test_jam_rekomendasi_tidak_kurang_dari_setengah_jam(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=20.0)
        for l in o["langkah"]:
            assert l["jam_rekomendasi"] >= 0.5

    def test_pengurangan_maksimal_50_persen(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=10.0)
        for l in o["langkah"]:
            assert l["jam_rekomendasi"] >= l["jam_awal"] * 0.5 - 1e-9

    def test_penghematan_tidak_negatif(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        assert o["hemat_kwh"] >= 0
        assert o["hemat_rp"] >= 0
        assert o["hemat_emisi_kg"] >= 0

    def test_ike_akhir_konsisten_dengan_kwh_akhir(self, clf):
        luas = 30.0
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=luas)
        assert abs(o["ike_akhir"] - o["total_kwh_akhir"] / luas) < 0.01

    def test_ike_akhir_tidak_lebih_besar_dari_awal(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        assert o["ike_akhir"] <= o["ike_awal"]

    def test_persen_hemat_dalam_batas_wajar(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        assert 0.0 <= o["persen_hemat_rp"] <= 100.0
        assert 0.0 <= o["persen_hemat_emisi"] <= 100.0

    def test_status_valid(self, clf):
        VALID = {"sudah_efisien", "efisien", "cukup_efisien", "tidak_tercapai"}
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], luas=30.0)
        assert o["status"] in VALID

    def test_zona_akhir_berasal_dari_fuzzy(self, clf):
        """zona_akhir harus persis = profil_ike(ike_akhir, kwh_org_akhir, ada_ac)."""
        luas, penghuni = 30.0, 3
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI],
                              luas=luas, penghuni=penghuni)
        expected = profil_ike(o["ike_akhir"], o["total_kwh_akhir"] / penghuni, True)
        assert o["zona_akhir"] == expected

    def test_biaya_akhir_prabayar_adalah_nominal_token(self, clf):
        o, _ = self._jalankan(clf, [ALAT_AC] * 3 + [ALAT_MESIN_CUCI],
                              luas=30.0, is_prabayar=True)
        expected = hitung_token_dari_kwh(o["total_kwh_akhir"], get_tarif(1300))
        assert o["biaya_akhir"] == pytest.approx(expected, abs=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 7. services/pipeline.py — Lapis 1 & orkestrasi
# ══════════════════════════════════════════════════════════════════════════════

class TestLapis1:

    def test_alat_valid_punya_watt_dan_watt_nyata(self):
        agent = DataIngestionValidatorAgent(1300, False)
        p = agent.proses_data(50.0, 3, 600_000, [ALAT_AC])
        a = p["alat_valid"][0]
        assert a["watt"] == pytest.approx(880.0)
        assert a["watt_nyata"] == pytest.approx(748.0)
        assert a["kwh_bulan"] < a["kwh_bulan_semu"]

    def test_total_kwh_primer_adalah_daya_nyata(self):
        agent = DataIngestionValidatorAgent(1300, False)
        p = agent.proses_data(50.0, 3, 600_000, [ALAT_AC, ALAT_TV])
        assert p["total_kwh"] < p["total_kwh_semu"]

    def test_kwh_harian_adalah_total_dibagi_30(self):
        agent = DataIngestionValidatorAgent(1300, False)
        p = agent.proses_data(50.0, 3, 600_000, [ALAT_AC])
        assert abs(p["kwh_harian"] - p["total_kwh"] / 30) < 0.001

    def test_jumlah_terkali_di_total_kwh(self):
        agent = DataIngestionValidatorAgent(1300, True)
        p = agent.proses_data(50.0, 3, 400_000, [ALAT_LAMPU_12_UNIT])
        watt_ny = hitung_watt_nyata(220.0 * 0.05, "Pencahayaan")
        assert abs(p["total_kwh"] - hitung_kwh_alat(watt_ny, 8.0, 12)) < 0.01

    def test_jumlah_default_satu(self):
        agent = DataIngestionValidatorAgent(1300, True)
        p = agent.proses_data(50.0, 3, 400_000, [ALAT_AC])
        assert p["alat_valid"][0]["jumlah"] == 1

    def test_output_key_lengkap(self):
        agent = DataIngestionValidatorAgent(1300, False)
        p = agent.proses_data(50.0, 3, 600_000, [ALAT_AC])
        for k in ("mode", "total_kwh", "total_kwh_semu", "kwh_harian",
                  "biaya_bawah", "biaya_atas", "biaya_tengah", "ike",
                  "kwh_per_org", "emisi_sebelum", "is_anomali", "arah_anomali",
                  "selisih_pct", "alat_valid", "rm_rupiah"):
            assert k in p, f"key '{k}' hilang dari payload Lapis 1"

    def test_mode_prabayar_dan_pascabayar(self):
        assert DataIngestionValidatorAgent(1300, True).proses_data(
            50.0, 3, 400_000, [ALAT_AC])["mode"] == "prabayar"
        assert DataIngestionValidatorAgent(1300, False).proses_data(
            50.0, 3, 600_000, [ALAT_AC])["mode"] == "pascabayar"


class TestIntegrasi:

    def test_pipeline_pascabayar_berjalan(self, clf):
        h = analisis(luas_rumah=45, penghuni=3, biaya_asli=600_000, daya_va=1300,
                     is_prabayar=False, daftar_alat=[ALAT_AC, ALAT_TV, ALAT_LAMPU],
                     dsm_clf=clf)
        assert h["payload"]["total_kwh"] > 0
        assert h["label_ike"] in TestIkeProfiler.VALID
        assert h["token"] is None
        assert h["model_siap"] is True

    def test_pipeline_prabayar_menghasilkan_proyeksi_token(self, clf):
        h = analisis(luas_rumah=45, penghuni=3, biaya_asli=400_000, daya_va=1300,
                     is_prabayar=True, daftar_alat=[ALAT_AC, ALAT_TV],
                     dsm_clf=clf, sisa_kwh=45.0, tanggal_baca=date(2026, 7, 10))
        tk = h["token"]
        assert tk is not None and tk["valid"] is True
        assert tk["hari_tersisa"] > 0
        assert tk["tanggal_habis"] >= date(2026, 7, 10)
        assert tk["token_bulanan_sekarang"] > 0

    def test_prabayar_tanpa_sisa_kwh_tidak_error(self, clf):
        h = analisis(luas_rumah=45, penghuni=3, biaya_asli=400_000, daya_va=1300,
                     is_prabayar=True, daftar_alat=[ALAT_AC], dsm_clf=clf)
        assert h["token"] is None

    def test_deteksi_ada_ac(self, clf):
        assert analisis(45, 3, 600_000, 1300, False, [ALAT_AC, ALAT_TV], clf)["ada_ac"] is True
        assert analisis(45, 3, 600_000, 1300, False, [ALAT_TV, ALAT_LAMPU], clf)["ada_ac"] is False

    def test_total_kwh_lapis1_sama_dengan_jumlah_dsm(self, clf):
        """
        Lapis 2 TIDAK boleh menghitung ulang kWh. Selisih apa pun berarti ada
        rumus yang ditulis dua kali dan berpotensi tidak sinkron.
        """
        h = analisis(45, 3, 600_000, 1300, False,
                     [ALAT_AC, ALAT_TV, ALAT_LAMPU, ALAT_LAMPU_12_UNIT], clf)
        dari_dsm = sum(a["kwh_bulan"] for a in h["hasil_dsm"])
        assert abs(h["payload"]["total_kwh"] - dari_dsm) < 0.01

    def test_dsm_mempartisi_semua_alat(self, clf):
        daftar = [ALAT_AC, ALAT_TV, ALAT_LAMPU, ALAT_MESIN_CUCI]
        h = analisis(45, 3, 600_000, 1300, False, daftar, clf)
        r = h["ringkasan"]
        assert len(r["fleksibel"]) + len(r["tidak_fleksibel"]) == len(daftar)

    def test_rumah_hemat_optimizer_tidur(self, clf):
        h = analisis(50, 3, 100_000, 1300, False, [ALAT_TV, ALAT_LAMPU], clf)
        assert h["label_ike"] in {"Sangat Efisien", "Efisien"}
        assert h["hasil_opt"]["aktif"] is False

    def test_rumah_boros_optimizer_menghemat(self, clf):
        h = analisis(30, 4, 2_000_000, 2200, False,
                     [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], clf)
        o = h["hasil_opt"]
        assert h["label_ike"] in ZONA_PERLU_OPTIMASI
        assert o["aktif"] is True
        assert o["hemat_kwh"] > 0
        assert o["hemat_rp"] > 0
        assert "emisi_sesudah" in o

    def test_optimasi_prabayar_menurunkan_biaya_token(self, clf):
        h = analisis(30, 4, 800_000, 2200, True,
                     [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], clf,
                     sisa_kwh=38.0, tanggal_baca=date(2026, 7, 10))
        tk = h["token"]
        assert tk["hemat_token_rp"] > 0
        assert tk["token_bulanan_optimal"] < tk["token_bulanan_sekarang"]
        assert tk["hari_tersisa_optimal"] > tk["hari_tersisa"], (
            "Pola pemakaian yang lebih hemat harus memperpanjang umur sisa token."
        )

    def test_hemat_token_sama_dengan_hemat_rp_optimizer(self, clf):
        h = analisis(30, 4, 800_000, 2200, True,
                     [ALAT_AC] * 3 + [ALAT_MESIN_CUCI], clf,
                     sisa_kwh=38.0, tanggal_baca=date(2026, 7, 10))
        assert abs(h["token"]["hemat_token_rp"] - h["hasil_opt"]["hemat_rp"]) <= 1.0

    def test_anomali_prabayar_membandingkan_pembelian_token(self, clf):
        """biaya_asli untuk prabayar = total pembelian token bulan lalu."""
        h = analisis(45, 3, 50_000, 1300, True, [ALAT_AC], clf)
        assert h["payload"]["is_anomali"] is True
        assert h["payload"]["arah_anomali"] == "tagihan_lebih_rendah"

    def test_biaya_asli_di_dalam_rentang_tidak_anomali(self, clf):
        h1 = analisis(45, 3, 1, 1300, False, [ALAT_AC, ALAT_TV], clf)
        tengah = h1["payload"]["biaya_tengah"]
        h2 = analisis(45, 3, tengah, 1300, False, [ALAT_AC, ALAT_TV], clf)
        assert h2["payload"]["is_anomali"] is False
        assert h2["payload"]["arah_anomali"] == "wajar"


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    ).returncode)