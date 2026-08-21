"""
services/ingestion.py
======================
DATA INGESTION & VALIDATOR AGENT — LAPIS 1

Diekstrak dari app.py supaya Streamlit (app.py) DAN backend FastAPI
sama-sama impor kelas yang SAMA — tidak ada logika kalkulasi/validasi
yang ditulis ulang di dua tempat berbeda dengan risiko tidak sinkron.

Kelas ini murni Python (tidak ada dependensi Streamlit), jadi aman
dipakai di kedua konteks.

Prediksi anomali: dilempar ke action_analist/anomaly_evaluator.py — kelas ini
hanya orkestrasi kalkulasi.

Semua kalkulasi yang BISA dihitung dari data yang tersedia di sini
(kWh per alat, total kWh, IKE, kWh per penghuni, tagihan, emisi)
dihitung SEKALI di kelas ini. Layer 2 (fuzzy IKE, DSM classifier)
dan Layer 3 (greedy) menerima hasilnya lewat 'payload' dan
'alat_valid', tidak menghitung ulang dari nol.
"""

from core.kalkulasi import (
    get_tarif, hitung_watt, hitung_kwh_alat,
    hitung_tagihan, hitung_emisi, hitung_ike, hitung_kwh_per_org,
    hitung_kwh_dari_token, hitung_hari_berjalan,
    hitung_estimasi_kwh_periode, hitung_saldo_token_awal,
    hitung_token_terpakai_aktual, PBJT_RUMAH_TANGGA,
)
from action_analist.anomaly_evaluator import (
    evaluasi_anomali_pascabayar, evaluasi_anomali_prabayar,
)


class DataIngestionValidatorAgent:
    def __init__(self, daya_va: int = 1300, is_prabayar: bool = False,
                is_subsidi: bool = False):
        self.daya_va     = daya_va
        self.is_prabayar = is_prabayar
        self.is_subsidi  = is_subsidi  # cuma relevan untuk 900 VA (450VA/>=1300VA abaikan ini)
        self.TARIF_KWH   = get_tarif(daya_va, is_subsidi)
        self.PBJT        = PBJT_RUMAH_TANGGA  # alias, dipakai pipeline & optimasi()
        # CATATAN PEROMBAKAN: self.BIAYA_BEBAN (Rekening Minimum) sudah
        # dihapus total — rumus resmi PLN Persero untuk tagihan pascabayar
        # tidak menyertakan komponen ini (lihat core/kalkulasi.py). Rumus
        # lama ternyata bersumber dari skema PLN Batam, bukan nasional.

    def proses_data(self, luas_rumah, penghuni, daftar_alat,
                    tagihan_asli: float = None,
                    token_context: dict = None):
        """
        Parameters:
            daftar_alat : list of dict, tiap dict berisi
                          nama, kategori, tegangan, arus, jam, jumlah
                          ('jumlah' opsional, default 1 kalau tidak ada)
            tagihan_asli : Rp — WAJIB diisi kalau self.is_prabayar False.
            token_context : dict — WAJIB diisi kalau self.is_prabayar True.
                {
                  'tanggal_pembelian' : date,
                  'sisa_sebelum_beli' : float (kWh),
                  'nominal_dibeli'    : float (Rp, sesuai struk),
                  'sisa_saat_ini'     : float (kWh),
                }
        """
        total_kwh  = 0.0
        alat_valid = []

        for alat in daftar_alat:
            jumlah = alat.get('jumlah', 1)
            watt   = hitung_watt(alat['tegangan'], alat['arus'])
            kwh    = hitung_kwh_alat(watt, alat['jam'], jumlah)
            total_kwh += kwh
            alat_valid.append({
                'nama'     : alat['nama'],
                'kategori' : alat['kategori'],
                'tegangan' : alat['tegangan'],
                'arus'     : alat['arus'],
                'watt'     : watt,
                'jam'      : alat['jam'],
                'jumlah'   : jumlah,
                'kwh_bulan': kwh,
            })

        total_kwh = round(total_kwh, 3)

        # Tagihan = Biaya Pemakaian + PBJT + Biaya Materai (kalau relevan).
        # is_prabayar dikirim ke hitung_tagihan() supaya materai TIDAK
        # dikenakan untuk prabayar (estimasi_rp bukan dokumen tagihan resmi).
        rincian_tagihan  = hitung_tagihan(
            total_kwh, self.TARIF_KWH, PBJT_RUMAH_TANGGA, self.is_prabayar
        )
        ike           = hitung_ike(total_kwh, luas_rumah)
        kwh_per_org   = hitung_kwh_per_org(total_kwh, penghuni)
        emisi_sebelum = hitung_emisi(total_kwh)

        payload = {
            "total_kwh"      : total_kwh,
            "biaya_pemakaian": rincian_tagihan['biaya_pemakaian'],
            "biaya_pbjt"     : rincian_tagihan['biaya_pbjt'],
            "biaya_materai"  : rincian_tagihan['biaya_materai'],
            "estimasi_rp"    : rincian_tagihan['total'],
            "ike"            : ike,
            "kwh_per_org"    : kwh_per_org,
            "emisi_sebelum"  : emisi_sebelum,
            "tarif_digunakan": self.TARIF_KWH,
            "golongan_daya"  : f"{self.daya_va} VA",
            "alat_valid"     : alat_valid,
            "is_prabayar"    : self.is_prabayar,
        }

        # ── Cabang prediksi anomali: prabayar (token/kWh) vs pascabayar (Rp) ──
        # Basisnya beda total: pascabayar bandingkan Rp vs Rp memakai siklus
        # tagihan tetap 30 hari; prabayar bandingkan kWh vs kWh dari selisih
        # saldo token, di-skala ke jumlah hari aktual sejak top-up (siklus
        # top-up tidak selalu 30 hari). Lihat action_analist/anomaly_evaluator.py.
        if self.is_prabayar:
            if token_context is None:
                raise ValueError(
                    "token_context wajib diisi untuk meteran prabayar"
                )

            kwh_dari_pembelian = hitung_kwh_dari_token(
                token_context['nominal_dibeli'], self.TARIF_KWH, self.PBJT
            )
            saldo_awal = hitung_saldo_token_awal(
                token_context['sisa_sebelum_beli'], kwh_dari_pembelian
            )
            hari_berjalan = hitung_hari_berjalan(
                token_context['tanggal_pembelian']
            )
            token_terpakai_aktual = hitung_token_terpakai_aktual(
                saldo_awal, token_context['sisa_saat_ini']
            )
            # max(hari_berjalan, 0) — kalau tanggal tidak valid (negatif),
            # estimasi tetap ditampilkan sebagai 0, pesan errornya sudah
            # ditangani terpisah oleh evaluasi_anomali_prabayar().
            estimasi_terpakai = hitung_estimasi_kwh_periode(
                total_kwh, max(hari_berjalan, 0)
            )

            anomali = evaluasi_anomali_prabayar(
                token_terpakai_aktual, estimasi_terpakai, hari_berjalan,
                daya_va=self.daya_va, total_kwh=total_kwh,
            )

            payload["token_context"] = {
                "tanggal_pembelian"          : token_context['tanggal_pembelian'],
                "sisa_sebelum_beli"          : token_context['sisa_sebelum_beli'],
                "nominal_dibeli"             : token_context['nominal_dibeli'],
                "sisa_saat_ini"              : token_context['sisa_saat_ini'],
                "kwh_dari_pembelian"         : kwh_dari_pembelian,
                "saldo_awal"                 : saldo_awal,
                "hari_berjalan"              : hari_berjalan,
                "token_terpakai_aktual"      : token_terpakai_aktual,
                "estimasi_terpakai_perangkat": estimasi_terpakai,
            }
        else:
            if tagihan_asli is None:
                raise ValueError(
                    "tagihan_asli wajib diisi untuk meteran pascabayar"
                )
            anomali = evaluasi_anomali_pascabayar(
                tagihan_asli, rincian_tagihan['total']
            )
            payload["tagihan_asli"] = tagihan_asli

        # Skema status seragam untuk kedua cabang — Lapis 1 (render) tidak
        # perlu tahu jenis meteran untuk menampilkan status anomali.
        payload["status_anomali"] = anomali["status"]
        payload["selisih_pct"]    = anomali["selisih_pct"]
        payload["pesan_anomali"]  = anomali["pesan"]
        payload["is_anomali"]     = anomali["status"] == "anomali"

        return payload