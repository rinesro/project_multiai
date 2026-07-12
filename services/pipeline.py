"""
services/pipeline.py
====================
Lapis 1 (ingestion + validasi + kalkulasi) dan orkestrasi Lapis 1 -> 2 -> 3.

Modul ini sengaja TIDAK mengimpor Streamlit. Semua logika analisis berada di
sini supaya bisa dipanggil dari app.py (Streamlit) maupun dari FastAPI saat
di-deploy nanti, tanpa menyalin kode.

    dict masuk  ->  analisis()  ->  dict keluar

Prinsip: semua kalkulasi yang bisa dihitung dari data yang tersedia di Lapis 1
dihitung SEKALI di sini. Lapis 2 dan 3 menerima hasilnya, tidak menghitung ulang.
"""

from datetime import date

from core.kalkulasi import (
    get_tarif, hitung_watt, hitung_watt_nyata, hitung_kwh_alat,
    hitung_biaya_bulanan, hitung_rentang_biaya, hitung_emisi,
    hitung_ike, hitung_kwh_per_org, deteksi_anomali, hitung_rm_rupiah,
)
from core.konstanta import PBJT_RUMAH_TANGGA
from core.token import jadwal_isi_ulang, hitung_token_dari_kwh
from fuzzy.ike_profiler import profil_ike
from optimizer.brute_force import optimasi


# --- Lapis 1 -----------------------------------------------------------------

class DataIngestionValidatorAgent:
    """
    Menghitung, sekali saja:
        watt semu & nyata per alat, kWh per alat, total kWh (nyata & semu),
        rentang prediksi biaya, IKE, kWh/penghuni, emisi, status anomali.
    """

    def __init__(self, daya_va: int = 1300, is_prabayar: bool = False):
        self.daya_va = int(daya_va)
        self.is_prabayar = bool(is_prabayar)
        self.tarif = get_tarif(self.daya_va)
        self.pbjt = PBJT_RUMAH_TANGGA
        self.rm_rupiah = hitung_rm_rupiah(self.daya_va)

    def proses_data(self, luas_rumah, penghuni, biaya_asli, daftar_alat) -> dict:
        """
        `biaya_asli` = tagihan bulan lalu (pascabayar) ATAU total pembelian
        token bulan lalu (prabayar).
        """
        total_kwh_nyata = 0.0
        total_kwh_semu = 0.0
        alat_valid = []

        for alat in daftar_alat:
            jumlah = int(alat.get("jumlah", 1))
            watt = hitung_watt(alat["tegangan"], alat["arus"])          # VA - fitur DSM
            watt_nyata = hitung_watt_nyata(watt, alat["kategori"])       # W  - energi

            kwh_nyata = hitung_kwh_alat(watt_nyata, alat["jam"], jumlah)
            kwh_semu = hitung_kwh_alat(watt, alat["jam"], jumlah)

            total_kwh_nyata += kwh_nyata
            total_kwh_semu += kwh_semu

            alat_valid.append({
                "nama":          alat["nama"],
                "kategori":      alat["kategori"],
                "tegangan":      alat["tegangan"],
                "arus":          alat["arus"],
                "watt":          watt,          # daya semu (V x I)
                "watt_nyata":    watt_nyata,    # daya nyata (V x I x cos phi)
                "jam":           alat["jam"],
                "jumlah":        jumlah,
                "kwh_bulan":     kwh_nyata,     # basis energi = daya nyata
                "kwh_bulan_semu": kwh_semu,
            })

        total_kwh_nyata = round(total_kwh_nyata, 3)
        total_kwh_semu = round(total_kwh_semu, 3)

        rentang = hitung_rentang_biaya(
            total_kwh_nyata, total_kwh_semu,
            self.tarif, self.pbjt, self.daya_va, self.is_prabayar,
        )
        anomali = deteksi_anomali(
            biaya_asli, rentang["biaya_bawah"], rentang["biaya_atas"]
        )

        return {
            "mode":            "prabayar" if self.is_prabayar else "pascabayar",
            "golongan_daya":   f"{self.daya_va} VA",
            "daya_va":         self.daya_va,
            "is_prabayar":     self.is_prabayar,
            "tarif":           self.tarif,
            "pbjt":            self.pbjt,
            "rm_rupiah":       self.rm_rupiah,

            "total_kwh":       total_kwh_nyata,   # nilai primer di seluruh sistem
            "total_kwh_semu":  total_kwh_semu,
            "kwh_harian":      round(total_kwh_nyata / 30, 3),

            "biaya_bawah":     rentang["biaya_bawah"],
            "biaya_atas":      rentang["biaya_atas"],
            "biaya_tengah":    rentang["biaya_tengah"],
            "lebar_pct":       rentang["lebar_pct"],
            "rincian":         rentang["rincian_bawah"],

            "ike":             hitung_ike(total_kwh_nyata, luas_rumah),
            "kwh_per_org":     hitung_kwh_per_org(total_kwh_nyata, penghuni),
            "emisi_sebelum":   hitung_emisi(total_kwh_nyata),

            "biaya_asli":      biaya_asli,
            "is_anomali":      anomali["is_anomali"],
            "arah_anomali":    anomali["arah"],
            "selisih_pct":     anomali["selisih_pct"],
            "pesan_anomali":   anomali["pesan"],

            "alat_valid":      alat_valid,
        }


# --- Orkestrasi Lapis 1 -> 2 -> 3 -------------------------------------------

def analisis(luas_rumah: float,
             penghuni: int,
             biaya_asli: float,
             daya_va: int,
             is_prabayar: bool,
             daftar_alat: list,
             dsm_clf,
             sisa_kwh: float = None,
             tanggal_baca: date = None,
             target_hari: int = 30) -> dict:
    """
    Menjalankan seluruh pipeline dan mengembalikan satu dict hasil.

    `sisa_kwh` dan `tanggal_baca` hanya dipakai bila is_prabayar=True.
    """
    # Lapis 1
    agent = DataIngestionValidatorAgent(daya_va, is_prabayar)
    payload = agent.proses_data(luas_rumah, penghuni, biaya_asli, daftar_alat)

    # Lapis 2a - Fuzzy IKE (satu-satunya sumber label)
    ada_ac = any(a["kategori"] == "Pendingin" for a in daftar_alat)
    label_ike = profil_ike(payload["ike"], payload["kwh_per_org"], ada_ac)

    # Lapis 2b - DSM
    hasil_dsm = dsm_clf.prediksi_batch(payload["alat_valid"])
    ringkasan = dsm_clf.ringkasan_dsm(hasil_dsm)

    # Lapis 3 - Optimizer
    hasil_opt = optimasi(
        ringkasan_dsm=ringkasan,
        luas_m2=float(luas_rumah),
        penghuni=int(penghuni),
        ada_ac=ada_ac,
        label_ike=label_ike,
        tarif_kwh=payload["tarif"],
        pbjt=payload["pbjt"],
        daya_va=payload["daya_va"],
        is_prabayar=is_prabayar,
        kwh_awal=payload["total_kwh"],
        biaya_awal=payload["biaya_bawah"],
        emisi_awal=payload["emisi_sebelum"]["emisi_kg_bulan"],
    )
    if hasil_opt["aktif"]:
        hasil_opt["emisi_sesudah"] = hitung_emisi(hasil_opt["total_kwh_akhir"])

    hasil = {
        "payload":      payload,
        "label_ike":    label_ike,
        "ada_ac":       ada_ac,
        "hasil_dsm":    hasil_dsm,
        "ringkasan":    ringkasan,
        "hasil_opt":    hasil_opt,
        "model_siap":   dsm_clf.siap,
        "model_error":  dsm_clf.pesan_error,
        "token":        None,
    }

    # Lapis 1b - Proyeksi token (prabayar saja)
    if is_prabayar and sisa_kwh is not None:
        tanggal_baca = tanggal_baca or date.today()
        proyeksi = jadwal_isi_ulang(
            sisa_kwh=sisa_kwh,
            kwh_harian=payload["kwh_harian"],
            tanggal_baca=tanggal_baca,
            tarif=payload["tarif"],
            daya_va=payload["daya_va"],
            target_hari=target_hari,
            pbjt=payload["pbjt"],
        )

        token_sekarang = hitung_token_dari_kwh(
            payload["total_kwh"], payload["tarif"], payload["pbjt"]
        )
        if hasil_opt["aktif"] and hasil_opt["langkah"]:
            kwh_opt = hasil_opt["total_kwh_akhir"]
            token_optimal = hitung_token_dari_kwh(
                kwh_opt, payload["tarif"], payload["pbjt"]
            )
            kwh_harian_opt = kwh_opt / 30
            proyeksi_opt = jadwal_isi_ulang(
                sisa_kwh=sisa_kwh,
                kwh_harian=kwh_harian_opt,
                tanggal_baca=tanggal_baca,
                tarif=payload["tarif"],
                daya_va=payload["daya_va"],
                target_hari=target_hari,
                pbjt=payload["pbjt"],
            )
        else:
            token_optimal = token_sekarang
            proyeksi_opt = proyeksi

        proyeksi.update({
            "token_bulanan_sekarang": token_sekarang,
            "token_bulanan_optimal":  token_optimal,
            "hemat_token_rp":         round(token_sekarang - token_optimal, 0),
            "hari_tersisa_optimal":   proyeksi_opt["hari_tersisa"],
            "tanggal_habis_optimal":  proyeksi_opt["tanggal_habis"],
            "nominal_optimal":        proyeksi_opt["nominal_disarankan"],
        })
        hasil["token"] = proyeksi

    return hasil