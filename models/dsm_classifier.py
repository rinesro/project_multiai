"""
models/dsm_classifier.py
=========================
Pembungkus LightGBM untuk klasifikasi DSM peralatan listrik — Lapis 2.

Label DSM (2 kelas):
    Fleksibel       — daya/durasi bisa dikurangi   -> masuk brute force
    Tidak Fleksibel — harus menyala saat dibutuhkan -> dilewati optimizer


YANG BERUBAH dari versi sebelumnya, dan alasannya
-------------------------------------------------
Logika model, fallback, dan konsistensi kategori LightGBM TIDAK berubah.
Hanya empat hal yang menyesuaikan Lapis 1 yang sudah dirombak.

1. Impor dari core.kalkulator (dulu core.kalkulasi). Modul berganti nama.
   Hack sys.path dihapus; paket sudah proper.

2. Field peralatan mengikuti keluaran Lapis 1 baru:
       watt      -> daya_semu_va   (V x I, satuan VA)
       kwh_bulan -> kvah_bulan     (energi semu bulanan, kVAh)
   Nilai yang dikirim ke model TIDAK berubah; hanya nama fieldnya.

3. Validasi input dihapus dari sini. Bentuk/rentang data sudah diperiksa
   app.py sebelum sampai ke Lapis 2. Modul ini mengasumsikan data bersih.

4. Kategori valid diambil dari core.konstanta.KATEGORI_PERALATAN, bukan
   salinan lokal, agar satu sumber kebenaran.


CATATAN PENTING soal fitur model
--------------------------------
Kolom fitur `Daya_W` yang dikirim ke model diisi dengan DAYA SEMU (V x I,
satuan VA), karena model dilatih pada nilai itu. Nama kolomnya tetap `Daya_W`
(nama saat pelatihan) walau isinya sebenarnya VA — mengganti nama kolom akan
membuat model tidak mengenalinya. JANGAN mengirim daya nyata (V x I x cos phi)
ke sini; distribusi fiturnya akan berbeda dari data latih dan prediksi menjadi
tidak valid tanpa memunculkan error apa pun.

Fitur Arus_A dan Daya_W dikirim PER-UNIT (tidak dikali jumlah), supaya nilainya
tetap berada di dalam rentang data latih.

Kategori pandas 'category' aman: booster LightGBM menyimpan urutan kategori
data latih (pandas_categorical) dan me-remap saat predict, sehingga batch yang
tidak memuat seluruh 8 kategori tetap menghasilkan prediksi yang benar.
"""

from pathlib import Path

import joblib
import pandas as pd

from core.konstanta import KATEGORI_PERALATAN
from core.kalkulator import hitung_daya_semu, hitung_kvah_alat

# ── Label model -> label sistem ───────────────────────────────────────────────
LABEL_MAP = {
    "Fleksibel":       "Fleksibel",
    "Tidak Fleksibel": "Tidak Fleksibel",
}

# ── Path model (relatif dari root proyek) ─────────────────────────────────────
_BASE = Path(__file__).resolve().parents[1]
_PATH_MODEL = _BASE / "data" / "dsm_lightgbm_model.pkl"
_PATH_ENCODER = _BASE / "data" / "dsm_target_encoder.pkl"

# Urutan kolom fitur — harus sama persis dengan saat pelatihan.
_FITUR = ["Kategori", "Tegangan_V", "Arus_A", "Daya_W", "Jam_Per_Hari"]


class DSMClassifier:
    """Pembungkus LightGBM DSM classifier — 2 label."""

    # Fallback berbasis aturan bila model gagal dimuat.
    _FALLBACK_MAP = {
        "Pendingin":          "Fleksibel",
        "Pemanas":            "Fleksibel",
        "Laundry":            "Fleksibel",
        "Pompa/Motor":        "Fleksibel",
        "Memasak":            "Tidak Fleksibel",
        "Hiburan/Elektronik": "Tidak Fleksibel",
        "Pencahayaan":        "Tidak Fleksibel",
        "Lainnya":            "Tidak Fleksibel",
    }

    def __init__(self):
        self._model = None
        self._encoder = None
        self._loaded = False
        self._error = None
        self._muat_model()

    def _muat_model(self):
        try:
            if not _PATH_MODEL.exists():
                raise FileNotFoundError(f"Model tidak ditemukan: {_PATH_MODEL}")
            if not _PATH_ENCODER.exists():
                raise FileNotFoundError(f"Encoder tidak ditemukan: {_PATH_ENCODER}")
            self._model = joblib.load(_PATH_MODEL)
            self._encoder = joblib.load(_PATH_ENCODER)
            self._loaded = True
        except Exception as e:      # noqa: BLE001 — dilaporkan lewat properti
            self._error = str(e)
            self._loaded = False

    @property
    def siap(self) -> bool:
        """False artinya sistem berjalan dengan fallback if-else, BUKAN LightGBM."""
        return self._loaded

    @property
    def pesan_error(self) -> str:
        return self._error or ""

    def _fallback(self, kategori: str) -> str:
        return self._FALLBACK_MAP.get(kategori, "Tidak Fleksibel")

    def _lengkapi(self, alat: dict) -> dict:
        """
        Melengkapi field turunan bila DSMClassifier dipanggil dengan data mentah
        (mis. dari pengujian) tanpa melewati Lapis 1. Bila data datang dari
        Lapis 1, field daya_semu_va dan kvah_bulan sudah ada dan dipakai apa
        adanya — tidak dihitung ulang, tetap satu sumber kebenaran.
        """
        hasil = dict(alat)
        if "daya_semu_va" not in hasil:
            hasil["daya_semu_va"] = hitung_daya_semu(
                alat.get("tegangan", 0), alat.get("arus", 0)
            )
        if "kvah_bulan" not in hasil:
            hasil["kvah_bulan"] = hitung_kvah_alat(
                hasil["daya_semu_va"], alat.get("jam", 0), alat.get("jumlah", 1)
            )
        return hasil

    def prediksi_batch(self, daftar_alat: list) -> list:
        """
        Memprediksi label DSM untuk semua peralatan sekaligus.

        Idealnya dipanggil dengan keluaran core.kalkulator.rincian_peralatan()
        yang sudah punya daya_semu_va dan kvah_bulan — supaya tidak ada rumus
        yang dihitung ulang di sini.

        Data dianggap sudah divalidasi app.py. Kategori tak dikenal tetap
        ditangani lewat fallback sebagai jaring pengaman, bukan sebagai
        validasi utama.

        Returns list of dict per peralatan:
            nama, kategori, tegangan, arus, jam, jumlah,
            daya_semu_va, kvah_bulan, label_dsm, metode
        """
        if not daftar_alat:
            return []

        alat_diproses = [self._lengkapi(a) for a in daftar_alat]

        # Baris fitur hanya untuk peralatan berkategori dikenal.
        idx_dikenal, rows = [], []
        for i, alat in enumerate(alat_diproses):
            if alat.get("kategori") in KATEGORI_PERALATAN:
                idx_dikenal.append(i)
                rows.append({
                    "Kategori":     alat["kategori"],
                    "Tegangan_V":   alat["tegangan"],
                    "Arus_A":       alat["arus"],          # per-unit
                    "Daya_W":       alat["daya_semu_va"],  # daya SEMU, sesuai data latih
                    "Jam_Per_Hari": alat["jam"],
                })

        prediksi_model = {}
        if self._loaded and rows:
            try:
                df = pd.DataFrame(rows)
                df["Kategori"] = df["Kategori"].astype("category")
                y_enc = self._model.predict(df[_FITUR])
                y_label = self._encoder.inverse_transform(y_enc)
                for j, idx in enumerate(idx_dikenal):
                    prediksi_model[idx] = LABEL_MAP.get(y_label[j], "Tidak Fleksibel")
            except Exception as e:      # noqa: BLE001
                self._error = f"Prediksi gagal: {e}"
                prediksi_model = {}

        hasil_list = []
        for i, alat in enumerate(alat_diproses):
            if i in prediksi_model:
                label, metode = prediksi_model[i], "model"
            else:
                label, metode = self._fallback(alat.get("kategori", "Lainnya")), "fallback"

            hasil_list.append({
                "nama":         alat.get("nama", f"Peralatan {i + 1}"),
                "kategori":     alat.get("kategori", ""),
                "tegangan":     alat.get("tegangan", 0),
                "arus":         alat.get("arus", 0),
                "jam":          alat.get("jam", 0),
                "jumlah":       alat.get("jumlah", 1),
                "daya_semu_va": alat["daya_semu_va"],
                "kvah_bulan":   alat["kvah_bulan"],
                "label_dsm":    label,
                "metode":       metode,
            })

        return hasil_list

    def ringkasan_dsm(self, hasil_prediksi: list) -> dict:
        """
        Memisahkan peralatan berdasarkan label DSM. Output dikirim ke
        optimizer/brute_force.py, yang mengharapkan tiap peralatan punya
        daya_semu_va dan kvah_bulan.
        """
        fleksibel = [a for a in hasil_prediksi if a["label_dsm"] == "Fleksibel"]
        tidak_fleksibel = [a for a in hasil_prediksi if a["label_dsm"] == "Tidak Fleksibel"]
        return {
            "fleksibel":            fleksibel,
            "tidak_fleksibel":      tidak_fleksibel,
            "total_kvah_fleksibel": round(sum(a["kvah_bulan"] for a in fleksibel), 3),
        }
