"""
models/dsm_classifier.py
=========================
Pembungkus LightGBM untuk klasifikasi DSM peralatan listrik.

Label DSM (2 kelas):
    Fleksibel       — daya/durasi bisa dikurangi → masuk brute force
    Tidak Fleksibel — harus menyala saat dibutuhkan → skip optimizer

Input per peralatan:
    nama      (str)   : nama bebas dari user
    kategori  (str)   : salah satu dari 8 pilihan dropdown
    tegangan  (float) : tegangan (V)
    arus      (float) : arus (A)
    jam       (float) : jam pemakaian per hari

Output per peralatan:
    dict berisi nama, label_dsm, watt, kwh_bulan, metode, valid
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys as _sys

_BASE = Path(__file__).parent.parent
if str(_BASE) not in _sys.path:
    _sys.path.insert(0, str(_BASE))
from core.kalkulasi import hitung_watt, hitung_kwh_alat, KATEGORI_ALAT

# ── Label model → label sistem ────────────────────────────────────────────────
LABEL_MAP = {
    "Fleksibel"      : "Fleksibel",
    "Tidak Fleksibel": "Tidak Fleksibel",
}

# Kategori valid — diturunkan dari core.kalkulasi.KATEGORI_ALAT (sumber
# kebenaran tunggal), bukan hardcode terpisah lagi. Dulu ada 2 salinan
# (di sini dan app.py::KATEGORI_OPTIONS) yang berisiko tidak sinkron.
KATEGORI_VALID = set(KATEGORI_ALAT)

# ── Path model (relatif dari root proyek) ─────────────────────────────────────
_PATH_MODEL   = _BASE / "data" / "dsm_lightgbm_model.pkl"
_PATH_ENCODER = _BASE / "data" / "dsm_target_encoder.pkl"


class DSMClassifier:
    """Pembungkus LightGBM DSM classifier — 2 label."""

    def __init__(self):
        self._model   = None
        self._encoder = None
        self._loaded  = False
        self._error   = None
        self._muat_model()

    def _muat_model(self):
        try:
            if not _PATH_MODEL.exists():
                raise FileNotFoundError(f"Model tidak ditemukan: {_PATH_MODEL}")
            if not _PATH_ENCODER.exists():
                raise FileNotFoundError(f"Encoder tidak ditemukan: {_PATH_ENCODER}")
            self._model   = joblib.load(_PATH_MODEL)
            self._encoder = joblib.load(_PATH_ENCODER)
            self._loaded  = True
        except Exception as e:
            self._error  = str(e)
            self._loaded = False

    @property
    def siap(self) -> bool:
        return self._loaded

    @property
    def pesan_error(self) -> str:
        return self._error or ""

    # ── Fallback rule-based ───────────────────────────────────────────────────
    _FALLBACK_MAP = {
        "Pendingin"          : "Fleksibel",
        "Pemanas"            : "Fleksibel",
        "Laundry"            : "Fleksibel",
        "Pompa/Motor"        : "Fleksibel",
        "Memasak"            : "Tidak Fleksibel",
        "Hiburan/Elektronik" : "Tidak Fleksibel",
        "Pencahayaan"        : "Tidak Fleksibel",
        "Lainnya"            : "Tidak Fleksibel",
    }

    def _fallback(self, kategori: str) -> str:
        return self._FALLBACK_MAP.get(kategori, "Tidak Fleksibel")

    def _validasi_alat(self, alat: dict) -> dict:
        hasil = alat.copy()

        # Pakai 'watt' yang sudah dihitung Lapis 1 (app.py) kalau tersedia
        # di payload['alat_valid'] — tidak dihitung ulang di sini.
        # Fallback ke core.kalkulasi kalau DSMClassifier dipanggil
        # berdiri sendiri (mis. dari test) tanpa lewat Lapis 1 dulu.
        if 'watt' not in hasil:
            hasil['watt'] = hitung_watt(
                alat.get('tegangan', 0), alat.get('arus', 0)
            )

        hasil['valid'] = True
        hasil['pesan'] = ""

        if alat.get('kategori') not in KATEGORI_VALID:
            hasil['valid'] = False
            hasil['pesan'] = f"Kategori '{alat.get('kategori')}' tidak dikenal."

        if hasil['watt'] <= 0:
            hasil['valid'] = False
            hasil['pesan'] = "Tegangan dan arus harus lebih dari 0."

        if not (0 < alat.get('jam', 0) <= 24):
            hasil['valid'] = False
            hasil['pesan'] = "Jam pemakaian harus antara 0 dan 24."

        return hasil

    def prediksi_batch(self, daftar_alat: list) -> list:
        """
        Memprediksi label DSM untuk semua peralatan sekaligus.

        Idealnya dipanggil dengan payload['alat_valid'] hasil Lapis 1
        (app.py) yang sudah punya 'watt' dan 'kwh_bulan' terhitung —
        supaya kwh_bulan TIDAK dihitung ulang di sini dan tetap satu
        sumber kebenaran. Kalau dipanggil dengan data mentah (tanpa
        'watt'/'kwh_bulan'), fungsi ini tetap bisa jalan dengan fallback
        ke core.kalkulasi (dipakai untuk pengujian berdiri sendiri).

        Fitur yang dikirim ke model tetap arus PER-UNIT (bukan dikali
        jumlah), supaya prediksi tidak keluar dari rentang data latih.

        Returns list of dict per peralatan:
            nama, kategori, tegangan, arus, watt, jam, jumlah,
            kwh_bulan, label_dsm, metode, valid, pesan
        """
        if not daftar_alat:
            return []

        hasil_list      = []
        alat_valid_idx  = []
        rows_untuk_pred = []

        alat_diproses = [self._validasi_alat(a) for a in daftar_alat]

        for i, alat in enumerate(alat_diproses):
            if alat['valid']:
                alat_valid_idx.append(i)
                rows_untuk_pred.append({
                    'Kategori'    : alat['kategori'],
                    'Tegangan_V'  : alat['tegangan'],
                    'Arus_A'      : alat['arus'],   # per-unit, bukan × jumlah
                    'Daya_W'      : alat['watt'],   # per-unit
                    'Jam_Per_Hari': alat['jam'],
                })

        # Prediksi batch
        prediksi_model = {}
        if self._loaded and rows_untuk_pred:
            try:
                df_pred = pd.DataFrame(rows_untuk_pred)
                df_pred['Kategori'] = df_pred['Kategori'].astype('category')
                y_enc   = self._model.predict(
                    df_pred[['Kategori','Tegangan_V','Arus_A',
                              'Daya_W','Jam_Per_Hari']]
                )
                y_label = self._encoder.inverse_transform(y_enc)
                for j, idx in enumerate(alat_valid_idx):
                    prediksi_model[idx] = LABEL_MAP.get(
                        y_label[j], "Tidak Fleksibel"
                    )
            except Exception as e:
                self._error    = f"Prediksi gagal: {e}"
                prediksi_model = {}

        # Rakit hasil
        for i, alat in enumerate(alat_diproses):
            jumlah = alat.get('jumlah', 1)

            # Pakai kwh_bulan dari Lapis 1 kalau sudah ada (sumber
            # kebenaran) — TIDAK dihitung ulang. Fallback ke core
            # kalau dipanggil tanpa lewat Lapis 1.
            if 'kwh_bulan' in alat:
                kwh_bulan = alat['kwh_bulan']
            else:
                kwh_bulan = hitung_kwh_alat(
                    alat['watt'], alat.get('jam', 0), jumlah
                )

            if not alat['valid']:
                label  = self._fallback(alat.get('kategori', 'Lainnya'))
                metode = 'fallback'
            elif i in prediksi_model:
                label  = prediksi_model[i]
                metode = 'model'
            else:
                label  = self._fallback(alat.get('kategori', 'Lainnya'))
                metode = 'fallback'

            hasil_list.append({
                'nama'     : alat.get('nama', f'Peralatan {i+1}'),
                'kategori' : alat.get('kategori', ''),
                'tegangan' : alat.get('tegangan', 0),
                'arus'     : alat.get('arus', 0),
                'watt'     : alat['watt'],
                'jam'      : alat.get('jam', 0),
                'jumlah'   : jumlah,
                'kwh_bulan': kwh_bulan,
                'label_dsm': label,
                'metode'   : metode,
                'valid'    : alat['valid'],
                'pesan'    : alat['pesan'],
            })

        return hasil_list

    def ringkasan_dsm(self, hasil_prediksi: list) -> dict:
        """
        Memisahkan peralatan berdasarkan label DSM.
        Output dikirim ke optimizer/brute_force.py.

        Returns:
            dict:
                fleksibel        : list peralatan yang bisa dioptimasi
                tidak_fleksibel  : list peralatan yang tidak disentuh
                total_kwh_fleksibel : total kWh dari peralatan fleksibel
        """
        fleksibel       = [a for a in hasil_prediksi
                           if a['label_dsm'] == 'Fleksibel']
        tidak_fleksibel = [a for a in hasil_prediksi
                           if a['label_dsm'] == 'Tidak Fleksibel']

        return {
            'fleksibel'           : fleksibel,
            'tidak_fleksibel'     : tidak_fleksibel,
            'total_kwh_fleksibel' : round(
                sum(a['kwh_bulan'] for a in fleksibel), 3
            ),
        }