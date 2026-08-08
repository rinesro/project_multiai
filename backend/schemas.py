"""
backend/schemas.py
====================
Pydantic models untuk request/response POST /api/analisis dan
GET /api/referensi — kontrak API yang disepakati di Tahap 1.

Field response (hasil_dsm, hasil_optimasi) sengaja bertipe longgar
(list[dict] / dict) alih-alih di-strict-kan jadi nested model —
supaya tidak menduplikasi/fabrikasi definisi field di dua tempat
(di sini dan di models/dsm_classifier.py, optimizer/greedy_optimizer.py).
Kalau frontend nanti butuh tipe yang lebih ketat, definisikan nested
model di sini SETELAH field-nya diverifikasi dari source asli.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

KategoriAlat = Literal[
    "Pendingin", "Pemanas", "Laundry", "Memasak",
    "Hiburan/Elektronik", "Pencahayaan", "Pompa/Motor", "Lainnya",
]
FokusOptimasi = Literal["Biaya", "Lingkungan"]


# ── Request ──────────────────────────────────────────────────────────────────

class PeralatanInput(BaseModel):
    nama: str = Field(min_length=1, max_length=100)
    kategori: KategoriAlat
    tegangan: float = Field(gt=0, description="Volt")
    arus: float = Field(gt=0, description="Ampere, per-unit")
    jam: float = Field(gt=0, le=24, description="Jam nyala per hari")
    jumlah: int = Field(default=1, ge=1)


class TokenContextInput(BaseModel):
    tanggal_pembelian: date
    sisa_sebelum_beli: float = Field(ge=0, description="kWh, sebelum top-up terakhir")
    nominal_dibeli: float = Field(gt=0, description="Rp, sesuai struk — BUKAN termasuk biaya admin")
    sisa_saat_ini: float = Field(ge=0, description="kWh, saat ini")


class AnalisisRequest(BaseModel):
    daya_va: int = Field(gt=0)
    is_prabayar: bool
    is_subsidi: bool = Field(
        default=False,
        description="Cuma relevan untuk 900 VA (dua tarif berbeda). Diabaikan untuk 450VA (selalu subsidi, 1 tarif) maupun >=1300VA (1 tarif per golongan)",
    )
    luas_rumah: float = Field(gt=0, description="m²")
    penghuni: int = Field(ge=1)
    tagihan_asli: Optional[float] = Field(default=None, description="Rp — wajib kalau is_prabayar=False")
    token_context: Optional[TokenContextInput] = Field(default=None, description="wajib kalau is_prabayar=True")
    daftar_alat: list[PeralatanInput] = Field(min_length=1)
    intent_user: list[FokusOptimasi] = Field(default_factory=lambda: ["Biaya"])

    @model_validator(mode="after")
    def cek_konsistensi_jenis_meteran(self):
        """
        Validasi silang WAJIB dilakukan di server — tidak boleh percaya
        begitu saja gabungan is_prabayar + tagihan_asli/token_context
        yang dikirim client, karena client (frontend) bisa saja kirim
        kombinasi yang tidak konsisten (bug UI atau request dimodifikasi
        manual). Ini setara dengan ValueError yang sudah ada di
        services/ingestion.py::DataIngestionValidatorAgent.proses_data(),
        cuma ditegakkan lebih awal di lapisan validasi request.
        """
        if self.is_prabayar and self.token_context is None:
            raise ValueError(
                "token_context wajib diisi kalau is_prabayar=True"
            )
        if not self.is_prabayar and self.tagihan_asli is None:
            raise ValueError(
                "tagihan_asli wajib diisi kalau is_prabayar=False"
            )
        return self


# ── Response ─────────────────────────────────────────────────────────────────

class AnalisisResponse(BaseModel):
    # Lapis 1 — kalkulasi dasar
    total_kwh: float
    biaya_pemakaian: float
    biaya_pbjt: float
    biaya_materai: float
    estimasi_rp: float
    ike: float
    label_ike: str
    kwh_per_org: float
    emisi_sebelum: dict
    tarif_digunakan: float
    golongan_daya: str
    is_prabayar: bool
    alat_valid: list[dict]

    # Lapis 1 — konteks tagihan/token (salah satu None tergantung is_prabayar)
    tagihan_asli: Optional[float] = None
    token_context: Optional[dict] = None

    # Lapis 1 — status prediksi anomali (skema seragam, lihat action_analist/anomaly_evaluator.py)
    status_anomali: str
    selisih_pct: Optional[float] = None
    pesan_anomali: str
    is_anomali: bool

    # Lapis 2 — DSM classifier (list of dict, lihat models/dsm_classifier.py)
    hasil_dsm: list[dict]

    # Lapis 3 — brute force optimizer (lihat optimizer/greedy_optimizer.py)
    hasil_optimasi: dict

    # Narasi Gemini
    narasi: str


class ReferensiResponse(BaseModel):
    golongan_daya: list[int]
    tarif_per_golongan: dict[str, float]  # >=1.300 VA, satu tarif per golongan (key string krn JSON tidak izinkan int key)
    tarif_daya_rendah: dict[str, dict[str, float]]  # 450 & 900 VA -- tergantung status subsidi
    kategori_alat: list[str]
    fokus_optimasi: list[str]
    pbjt_rumah_tangga: float


class ErrorResponse(BaseModel):
    detail: str