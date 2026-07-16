"""
test_energicerdas.py
=========================
Pengujian Unit (Unit Test) dan Sistem (System Test) untuk EnergiCerdas AI.

Jalankan pengujian ini di terminal dengan perintah:
    python -m unittest test_energicerdas.py -v
"""

import unittest
import sys
import traceback
from pathlib import Path

# Memasukkan direktori root agar modul core, fuzzy, models, optimizer terbaca
sys.path.insert(0, '.')

# ── Menggunakan Import Baru Sesuai __init__.py ────────────────────────────
try:
    from core.kalkulasi import (
        get_tarif, hitung_biaya_beban, hitung_watt, hitung_kwh_alat,
        hitung_tagihan, hitung_emisi, hitung_ike, hitung_kwh_per_org,
        deteksi_anomali, PBJT_RUMAH_TANGGA,
    )
    from fuzzy import profil_ike
    from models import DSMClassifier
    from optimizer import optimasi
    
    # Membangun replika DataIngestionValidatorAgent dari app.py untuk testing
    # (Menghindari import app.py langsung agar Streamlit tidak ikut tereksekusi di background)
    class DataIngestionValidatorAgent:
        def __init__(self, daya_va=1300, is_prabayar=False):
            self.daya_va     = daya_va
            self.TARIF_KWH   = get_tarif(daya_va)
            self.PBJT        = PBJT_RUMAH_TANGGA
            self.BIAYA_BEBAN = hitung_biaya_beban(daya_va, is_prabayar)

        def proses_data(self, luas_rumah, penghuni, tagihan_asli, daftar_alat):
            total_kwh, alat_valid = 0.0, []
            for alat in daftar_alat:
                jumlah = alat.get('jumlah', 1)
                watt   = hitung_watt(alat['tegangan'], alat['arus'])
                kwh    = hitung_kwh_alat(watt, alat['jam'], jumlah)
                total_kwh += kwh
                alat_valid.append({
                    'nama': alat['nama'], 'kategori': alat['kategori'],
                    'tegangan': alat['tegangan'], 'arus': alat['arus'],
                    'watt': watt, 'jam': alat['jam'], 'jumlah': jumlah,
                    'kwh_bulan': kwh,
                })
            total_kwh = round(total_kwh, 3)
            rincian = hitung_tagihan(total_kwh, self.TARIF_KWH, self.PBJT, self.BIAYA_BEBAN)
            anomali = deteksi_anomali(tagihan_asli, rincian['total'])
            return {
                "total_kwh": total_kwh, "estimasi_rp": rincian['total'],
                "biaya_pemakaian": rincian['biaya_pemakaian'],
                "biaya_pbjt": rincian['biaya_pbjt'], "biaya_beban": rincian['biaya_beban'],
                "ike": hitung_ike(total_kwh, luas_rumah),
                "kwh_per_org": hitung_kwh_per_org(total_kwh, penghuni),
                "emisi_sebelum": hitung_emisi(total_kwh),
                "is_anomali": anomali['is_anomali'], "selisih_pct": anomali['selisih_pct'],
                "tarif_digunakan": self.TARIF_KWH, "golongan_daya": f"{self.daya_va} VA",
                "alat_valid": alat_valid,
            }
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================
# 1. PENGUJIAN UNIT (UNIT TESTS)
# Menguji setiap modul secara terisolasi.
# ============================================================
class TestUnitLapisSistem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        if not IMPORT_SUCCESS:
            raise unittest.SkipTest(f"Gagal mengimpor modul: {IMPORT_ERROR}")

    def test_lapis1_kalkulator(self):
        """Unit Test: Lapis 1 Data Ingestion & Validator"""
        agent = DataIngestionValidatorAgent(1300, False)
        daftar_alat = [
            {'nama': 'Kipas', 'kategori': 'Pendingin', 'tegangan': 220, 'arus': 0.2, 'jam': 10} # 44 Watt
        ]
        payload = agent.proses_data(luas_rumah=50, penghuni=2, tagihan_asli=20000, daftar_alat=daftar_alat)
        
        self.assertIn('ike', payload, "Payload harus memiliki key 'ike'")
        self.assertIn('is_anomali', payload, "Payload harus mendeteksi anomali")
        self.assertEqual(len(payload['alat_valid']), 1, "Alat valid harus berjumlah 1")
        self.assertEqual(payload['alat_valid'][0]['watt'], 44.0, "Kalkulasi Watt tidak tepat")

    def test_lapis2_fuzzy(self):
        """Unit Test: Lapis 2a Fuzzy IKE Profiler"""
        label = profil_ike(ike=80, kwh_per_org=50, ada_ac=True)
        self.assertIn(label, ['Sangat Efisien', 'Efisien', 'Cukup Efisien', 'Boros', 'Sangat Boros'], 
                      "Label IKE tidak sesuai format")

    def test_lapis2_dsm_classifier(self):
        """Unit Test: Lapis 2b DSM Classifier (Akses Model)"""
        clf = DSMClassifier()
        if not clf.siap:
            self.skipTest(f"Model DSM tidak siap: {clf.pesan_error}")
            
        alat_dummy = [{'nama': 'AC', 'kategori': 'Pendingin', 'tegangan': 220, 'arus': 4.0, 'watt': 880, 'jam': 10, 'kwh_bulan': 264}]
        hasil = clf.prediksi_batch(alat_dummy)
        
        self.assertEqual(len(hasil), 1)
        self.assertIn('label_dsm', hasil[0])
        self.assertIn(hasil[0]['label_dsm'], ['Fleksibel', 'Tidak Fleksibel'])


# ============================================================
# 2. PENGUJIAN SISTEM (SYSTEM TESTS / END-TO-END)
# Menguji integrasi data mengalir dari Lapis 1 -> Lapis 2 -> Lapis 3.
# ============================================================
class TestSistemEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not IMPORT_SUCCESS:
            raise unittest.SkipTest("Skip system test karena gagal import modul")
        cls.clf = DSMClassifier()

    def setUp(self):
        if not self.clf.siap:
            self.skipTest("Skip test: Model DSM belum tersedia (belum ditrain/file tidak ada)")

    def test_pipeline_normal(self):
        """System Test: Skenario Rumah Menengah Ber-AC"""
        skenario = {
            "luas": 45, "penghuni": 4, "daya_va": 2200, "prabayar": False,
            "alat": [
                {'nama':'AC 1','kategori':'Pendingin','tegangan':220,'arus':5.0,'jam':12},
                {'nama':'Kulkas','kategori':'Pendingin','tegangan':220,'arus':0.8,'jam':24},
            ]
        }
        
        # Eksekusi Pipeline
        agent = DataIngestionValidatorAgent(skenario['daya_va'], skenario['prabayar'])
        payload = agent.proses_data(skenario['luas'], skenario['penghuni'], 500000, skenario['alat'])
        
        ada_ac = any(a['kategori'] == 'Pendingin' for a in skenario['alat'])
        
        hasil_dsm = self.clf.prediksi_batch(payload['alat_valid'])
        ringkasan = self.clf.ringkasan_dsm(hasil_dsm)
        
        hasil_opt = optimasi(
            ringkasan_dsm = ringkasan, luas_m2 = skenario['luas'],
            ada_ac = ada_ac, tarif_kwh = agent.TARIF_KWH,
            pbjt = agent.PBJT, biaya_beban = agent.BIAYA_BEBAN,
            kwh_awal = payload['total_kwh'], tagihan_awal = payload['estimasi_rp'],
            emisi_awal = payload['emisi_sebelum']['emisi_kg_bulan'],
        )
        
        # Validasi Sistem
        self.assertIsNotNone(hasil_opt)
        self.assertTrue(hasil_opt['ike_akhir'] <= hasil_opt['ike_awal'], 
                        "Sistem Optimasi gagal: IKE akhir lebih besar dari IKE awal")
        self.assertIn('langkah', hasil_opt, "Output Lapis 3 harus memiliki 'langkah' rekomendasi")

    def test_pipeline_anomali_kosong(self):
        """System Test: Stabilitas terhadap input alat kosong"""
        agent = DataIngestionValidatorAgent(1300, True)
        payload = agent.proses_data(30, 2, 50000, []) # Alat Kosong
        
        hasil_dsm = self.clf.prediksi_batch(payload['alat_valid'])
        ringkasan = self.clf.ringkasan_dsm(hasil_dsm)
        
        self.assertEqual(len(hasil_dsm), 0, "Sistem gagal menangani daftar alat kosong")
        self.assertEqual(ringkasan['total_kwh_fleksibel'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)