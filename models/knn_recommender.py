"""
models/knn_recommender.py
==========================
Pembungkus KNN Recommender untuk mencari rumah tangga "role model"
yang profilnya mirip dengan user tapi konsumsi energinya lebih efisien.

Cara kerja:
    1. User direpresentasikan oleh fitur fisik (luas, penghuni)
    2. KNN mencari K=5 tetangga terdekat dari database role model
    3. Dari 5 kandidat, dipilih 1 terbaik berdasarkan intent user:
         - Fokus Biaya       → pilih tagihan terendah
         - Fokus Lingkungan  → pilih emisi terendah
         - Fokus keduanya    → kombinasi skor biaya + emisi
         - Tidak ada fokus   → pilih IKE terendah (paling efisien)
    4. Role model terpilih dikirim ke generate_narasi() di app.py
       sebagai target referensi untuk Gen AI

Input dari app.py:
    luas_rumah  (float) : luas bangunan (m²)
    penghuni    (int)   : jumlah penghuni
    intent_user (list)  : ['Biaya'] | ['Lingkungan'] | keduanya | []

Output ke app.py:
    dict berisi data role model terpilih

File yang dibutuhkan (generate dengan train_knn.py):
    data/knn_biaya_lingkungan_recommender.pkl
    data/knn_scaler.pkl
    data/database_role_model.csv

Referensi regulasi (konsisten dengan app.py):
    [1] Tarif PLN April–Juni 2026
    [2] Faktor Emisi GRK Grid Jamali OM 0,80 kgCO₂/kWh
    [3] PBJT 2,4% Perda DKI Jakarta No.1/2024
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ── Path file (relatif dari root proyek) ──────────────────────────────────────
_BASE         = Path(__file__).parent.parent
_PATH_KNN     = _BASE / "data" / "knn_biaya_lingkungan_recommender.pkl"
_PATH_SCALER  = _BASE / "data" / "knn_scaler.pkl"
_PATH_DATABASE= _BASE / "data" / "database_role_model.csv"


class KNNRecommender:
    """
    Pembungkus KNN untuk mencari role model rumah tangga efisien.

    Menangani:
        - Load model, scaler, dan database dari disk
        - Normalisasi fitur input user
        - Pencarian K tetangga terdekat
        - Pemilihan role model terbaik berdasarkan intent
        - Fallback jika model tidak tersedia
    """

    def __init__(self):
        self._knn      = None
        self._scaler   = None
        self._database = None
        self._loaded   = False
        self._error    = None
        self._muat_model()

    def _muat_model(self):
        """Memuat KNN, scaler, dan database dari disk."""
        try:
            for path in [_PATH_KNN, _PATH_SCALER, _PATH_DATABASE]:
                if not path.exists():
                    raise FileNotFoundError(
                        f"File tidak ditemukan: {path}\n"
                        "Jalankan train_knn.py terlebih dahulu."
                    )
            self._knn      = joblib.load(_PATH_KNN)
            self._scaler   = joblib.load(_PATH_SCALER)
            self._database = pd.read_csv(_PATH_DATABASE)
            self._loaded   = True
        except Exception as e:
            self._error  = str(e)
            self._loaded = False

    @property
    def siap(self) -> bool:
        return self._loaded

    @property
    def pesan_error(self) -> str:
        return self._error or ""

    def _fallback(self, luas: float, penghuni: int,
                  intent: list) -> dict:
        """
        Fallback jika model tidak tersedia.
        Menghitung target ideal berdasarkan IKE efisien Depdiknas
        dan regulasi tarif PLN + PBJT Jakarta.

        Asumsi fallback:
            IKE target : 1.5 kWh/m²/bulan (zona Efisien, tidak ber-AC)
            Daya       : 1300 VA (golongan paling umum)
            Meteran    : prabayar (tidak ada biaya beban)
        """
        TARIF_FALLBACK = 1444.70   # R-1/TR 1300 VA
        PBJT_FALLBACK  = 0.024
        FAKTOR_EMISI   = 0.80

        ike_target  = 1.5
        kwh_target  = round(ike_target * luas, 2)
        biaya_p     = kwh_target * TARIF_FALLBACK
        total_tg    = round(biaya_p + biaya_p * PBJT_FALLBACK, 0)
        emisi       = round(kwh_target * FAKTOR_EMISI, 3)

        return {
            "Luas_Rumah_m2"   : luas,
            "Jumlah_Penghuni" : penghuni,
            "Daya_VA"         : 1300,
            "Is_Prabayar"     : 1,
            "Ada_AC"          : 0,
            "IKE_kWh_per_m2"  : ike_target,
            "Total_kWh_Bulan" : kwh_target,
            "Tarif_per_kWh"   : TARIF_FALLBACK,
            "Total_Tagihan_Rp": total_tg,
            "Emisi_CO2_kg"    : emisi,
            "metode"          : "fallback",
        }

    def cari_role_model(self,
                        luas_rumah : float,
                        penghuni   : int,
                        intent_user: list) -> dict:
        """
        Mencari role model terbaik untuk user.

        Parameters:
            luas_rumah  : luas bangunan user (m²)
            penghuni    : jumlah penghuni
            intent_user : list fokus — kombinasi 'Biaya' dan/atau 'Lingkungan'

        Returns:
            dict berisi data role model terpilih beserta metadata:
                Luas_Rumah_m2    : luas role model (m²)
                Jumlah_Penghuni  : penghuni role model
                Daya_VA          : daya tersambung role model
                IKE_kWh_per_m2   : IKE role model
                Total_kWh_Bulan  : konsumsi role model (kWh/bulan)
                Total_Tagihan_Rp : tagihan role model (Rp/bulan)
                Emisi_CO2_kg     : emisi role model (kgCO₂/bulan)
                metode           : 'knn' atau 'fallback'
        """
        # Gunakan fallback jika model tidak tersedia
        if not self._loaded:
            return self._fallback(luas_rumah, penghuni, intent_user)

        try:
            # ── Normalisasi input user ────────────────────────────────────────
            X_user   = self._scaler.transform([[luas_rumah, penghuni]])

            # ── Cari K tetangga terdekat ──────────────────────────────────────
            _, indeks = self._knn.kneighbors(X_user)
            kandidat  = self._database.iloc[indeks[0]].copy()

            # ── Hitung skor per intent ────────────────────────────────────────
            # Normalisasi min-max dalam kandidat supaya skor tidak bias
            # ke salah satu kolom karena satuan berbeda
            biaya_min = kandidat["Total_Tagihan_Rp"].min()
            biaya_max = kandidat["Total_Tagihan_Rp"].max()
            emisi_min = kandidat["Emisi_CO2_kg"].min()
            emisi_max = kandidat["Emisi_CO2_kg"].max()

            kandidat["Skor_Biaya"] = (
                (kandidat["Total_Tagihan_Rp"] - biaya_min)
                / (biaya_max - biaya_min + 1e-9)
            )
            kandidat["Skor_Lingkungan"] = (
                (kandidat["Emisi_CO2_kg"] - emisi_min)
                / (emisi_max - emisi_min + 1e-9)
            )
            kandidat["Skor_Akhir"] = 0.0

            if "Biaya"      in intent_user:
                kandidat["Skor_Akhir"] += kandidat["Skor_Biaya"]
            if "Lingkungan" in intent_user:
                kandidat["Skor_Akhir"] += kandidat["Skor_Lingkungan"]
            if not intent_user:
                # Tidak ada intent → pilih yang IKE-nya paling rendah
                kandidat["Skor_Akhir"] = kandidat["IKE_kWh_per_m2"]

            # ── Pilih role model terbaik (skor terendah = paling efisien) ────
            terbaik = (
                kandidat
                .sort_values("Skor_Akhir", ascending=True)
                .iloc[0]
                .to_dict()
            )
            terbaik["metode"] = "knn"
            return terbaik

        except Exception as e:
            self._error = f"KNN gagal: {e}"
            return self._fallback(luas_rumah, penghuni, intent_user)

    def format_untuk_narasi(self,
                             role_model : dict,
                             payload    : dict) -> dict:
        """
        Memformat perbandingan user vs role model untuk dikirim ke Gen AI.

        Parameters:
            role_model : output cari_role_model()
            payload    : output DataIngestionValidatorAgent.proses_data()

        Returns:
            dict siap kirim ke generate_narasi() di app.py:
                target_tagihan_rp  : tagihan role model (Rp)
                target_emisi_kg    : emisi role model (kgCO₂/bulan)
                target_ike         : IKE role model (kWh/m²/bulan)
                selisih_tagihan_rp : selisih tagihan user vs role model
                selisih_emisi_kg   : selisih emisi user vs role model
                persen_hemat_biaya : potensi hemat biaya (%)
                persen_hemat_emisi : potensi hemat emisi (%)
                metode             : 'knn' atau 'fallback'
        """
        tagihan_user  = payload.get("estimasi_rp", 0)
        emisi_user    = payload.get("emisi_sebelum", {}).get("emisi_kg_bulan", 0)
        tagihan_target= role_model.get("Total_Tagihan_Rp", 0)
        emisi_target  = role_model.get("Emisi_CO2_kg", 0)

        selisih_tagihan = round(tagihan_user  - tagihan_target, 0)
        selisih_emisi   = round(emisi_user    - emisi_target,   3)

        persen_biaya = 0.0
        persen_emisi = 0.0
        if tagihan_user > 0:
            persen_biaya = round(selisih_tagihan / tagihan_user * 100, 1)
        if emisi_user > 0:
            persen_emisi = round(selisih_emisi   / emisi_user   * 100, 1)

        return {
            "target_tagihan_rp" : int(tagihan_target),
            "target_emisi_kg"   : emisi_target,
            "target_ike"        : role_model.get("IKE_kWh_per_m2", 0),
            "selisih_tagihan_rp": int(selisih_tagihan),
            "selisih_emisi_kg"  : selisih_emisi,
            "persen_hemat_biaya": persen_biaya,
            "persen_hemat_emisi": persen_emisi,
            "metode"            : role_model.get("metode", "knn"),
        }