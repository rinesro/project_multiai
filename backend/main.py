"""
backend/main.py
=================
FastAPI backend untuk EnergiCerdas AI — dipakai frontend Next.js
(Vercel). Deploy target: Vercel (Python serverless function).

Orkestrasi Lapis 1 (kalkulasi & anomali) → Lapis 2 (fuzzy IKE + DSM
classifier) → Lapis 3 (brute force optimizer) → narasi Gemini,
memakai modul yang SAMA dengan app.py Streamlit (core/, services/,
models/, optimizer/, action_analist/) — tidak ada logika yang ditulis ulang.
"""

import os
import sys
import warnings
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# core/, action_analist/, services/, models/, optimizer/, data/ sekarang ada DI
# DALAM backend/ (bukan sepupu di repo root) — struktur ini dipertahankan
# supaya folder backend/ tetap mandiri, gampang di-COPY utuh di Dockerfile.
sys.path.insert(0, str(Path(__file__).parent))

# Muat backend/.env kalau ada (development lokal). Di platform Docker
# (HF Spaces/Render/Cloud Run/dll), secrets biasanya sudah otomatis jadi
# environment variable — load_dotenv() tidak menimpa env var yang sudah
# ada, jadi aman dipakai di kedua konteks.
load_dotenv(Path(__file__).parent / ".env")

# WAJIB sebelum import apa pun yang menyeret LightGBM (models/dsm_classifier
# di bawah) -- lihat bootstrap.py untuk detail lengkap kenapa ini perlu di
# Vercel. Dipisah jadi modul sendiri supaya entry point LAIN (test_sistem.py)
# juga bisa memanggilnya sedini mungkin, sebelum main.py sempat diimpor.
from bootstrap import preload_vendored_libgomp
preload_vendored_libgomp()

from core.kalkulasi import (
    GOLONGAN_DAYA, KATEGORI_ALAT, PBJT_RUMAH_TANGGA, get_tarif,
    TARIF_DAYA_RENDAH, cek_kapasitas_watt,
)
from action_analist.ike_profiler import profil_ike
from models.dsm_classifier import DSMClassifier
from optimizer.greedy_optimizer import optimasi
from services.ingestion import DataIngestionValidatorAgent
from services.narasi import generate_gemini_narasi

from schemas import (
    AnalisisRequest, AnalisisResponse, ReferensiResponse, ErrorResponse,
)

logger = logging.getLogger("uvicorn.error")

# ── Konfigurasi dari environment (HF Spaces secrets) ──────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Domain Vercel diisi di env var ALLOWED_ORIGINS, dipisah koma.
# Contoh: "https://energicerdas.vercel.app,http://localhost:3000"
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# ── Model DSM dimuat SEKALI saat startup (bukan per-request) ──────────────────
# Setara dengan @st.cache_resource di app.py Streamlit.
_dsm_clf: DSMClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _dsm_clf
    logger.info("Memuat model DSM Classifier...")
    _dsm_clf = DSMClassifier()
    if not _dsm_clf.siap:
        # Sengaja tidak menghentikan startup server — endpoint /analisis
        # akan menolak request dengan 503 kalau model belum siap, supaya
        # /api/referensi (yang tidak butuh model) tetap bisa diakses.
        logger.error("DSM Classifier GAGAL dimuat: %s", _dsm_clf.pesan_error)
    else:
        logger.info("DSM Classifier siap.")
    yield


app = FastAPI(
    title="EnergiCerdas AI API",
    description="Backend untuk analisis konsumsi listrik rumah tangga Jakarta",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/referensi", response_model=ReferensiResponse)
def get_referensi():
    """
    Data statis untuk dropdown/preview di frontend — golongan daya,
    tarif per golongan, kategori alat, fokus optimasi. Diambil dari
    core/kalkulasi.py supaya frontend tidak hardcode nilai yang bisa
    berubah kalau regulasi tarif PLN di-update.
    """
    return ReferensiResponse(
        golongan_daya=GOLONGAN_DAYA,
        # >=1.300 VA: satu tarif per golongan, tidak ada distingsi subsidi.
        tarif_per_golongan={
            str(v): get_tarif(v) for v in GOLONGAN_DAYA if v >= 1300
        },
        # 450 & 900 VA: dua opsi (subsidi/non-subsidi) untuk 900VA, 450VA
        # cuma satu tarif tapi tetap dibungkus struktur yang sama supaya
        # frontend tidak perlu kasus khusus per VA.
        tarif_daya_rendah={
            "450": {"subsidi": TARIF_DAYA_RENDAH[450], "non_subsidi": TARIF_DAYA_RENDAH[450]},
            "900": dict(TARIF_DAYA_RENDAH[900]),
        },
        kategori_alat=KATEGORI_ALAT,
        fokus_optimasi=["Biaya", "Lingkungan"],
        pbjt_rumah_tangga=PBJT_RUMAH_TANGGA,
    )


@app.get("/api/health")
def health_check():
    """Endpoint sederhana untuk cek Space hidup & model siap — dipakai
    monitoring/uptime check, bukan dipanggil frontend saat operasi normal."""
    return {
        "status": "ok",
        "dsm_model_siap": _dsm_clf.siap if _dsm_clf else False,
        "gemini_configured": GEMINI_API_KEY is not None,
    }


@app.post(
    "/api/analisis",
    response_model=AnalisisResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def post_analisis(req: AnalisisRequest):
    """
    Endpoint utama — setara tombol "🚀 Mulai Analisis" di app.py
    Streamlit. Menjalankan Lapis 1 (kalkulasi + anomali) → Lapis 2
    (fuzzy IKE + DSM classifier) → Lapis 3 (brute force optimizer) →
    narasi Gemini, lalu mengembalikan semuanya sebagai satu response.

    Validasi konsistensi is_prabayar vs tagihan_asli/token_context
    SUDAH dilakukan di schemas.py (Pydantic model_validator) sebelum
    endpoint ini dipanggil — request yang tidak konsisten otomatis
    ditolak FastAPI dengan 422 sebelum sampai ke sini.
    """
    if _dsm_clf is None or not _dsm_clf.siap:
        raise HTTPException(
            status_code=503,
            detail=f"Model DSM Classifier belum siap: {_dsm_clf.pesan_error if _dsm_clf else 'belum dimuat'}",
        )
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY belum dikonfigurasi di server (HF Spaces secrets).",
        )

    # ── Lapis 1 ────────────────────────────────────────────────────────────
    agent = DataIngestionValidatorAgent(req.daya_va, req.is_prabayar, req.is_subsidi)

    daftar_alat = [a.model_dump() for a in req.daftar_alat]
    token_context = (
        req.token_context.model_dump() if req.token_context is not None else None
    )

    try:
        payload = agent.proses_data(
            req.luas_rumah, req.penghuni, daftar_alat,
            tagihan_asli=req.tagihan_asli,
            token_context=token_context,
        )
    except ValueError as e:
        # ValueError dari services/ingestion.py — seharusnya sudah ditangkap
        # validator di schemas.py, ini jaring pengaman kedua.
        raise HTTPException(status_code=400, detail=str(e))

    # ── Lapis 2 ────────────────────────────────────────────────────────────
    # ada_ac (dulu dipakai profil_ike() & optimasi() untuk skema ber-AC/
    # tidak) sudah dihapus total dari sistem sejak perombakan kalibrasi
    # 5-lapis — tidak dihitung lagi karena tidak dipakai di mana pun lagi.
    label_ike = profil_ike(payload['ike'])

    hasil_dsm = _dsm_clf.prediksi_batch(payload['alat_valid'])
    ringkasan_dsm = _dsm_clf.ringkasan_dsm(hasil_dsm)

    # ── Lapis 3 ────────────────────────────────────────────────────────────
    hasil_opt = optimasi(
        ringkasan_dsm=ringkasan_dsm,
        luas_m2=float(req.luas_rumah),
        tarif_kwh=agent.TARIF_KWH,
        pbjt=agent.PBJT,
        is_prabayar=req.is_prabayar,
        kwh_awal=payload['total_kwh'],
        tagihan_awal=payload['estimasi_rp'],
        emisi_awal=payload['emisi_sebelum']['emisi_kg_bulan'],
    )

    # ── Narasi Gemini ─────────────────────────────────────────────────────
    try:
        narasi = generate_gemini_narasi(
            api_key=GEMINI_API_KEY,
            label_ike=label_ike,
            payload=payload,
            hasil_dsm=hasil_dsm,
            hasil_opt=hasil_opt,
            intent_user=req.intent_user,
        )
    except Exception as e:
        # Gemini API bisa gagal (rate limit, network) — jangan gagalkan
        # SELURUH analisis cuma karena narasi gagal dibuat. Dashboard
        # angka tetap valid & berguna tanpa narasi.
        logger.warning("Gagal membuat narasi Gemini: %s", e)
        narasi = (
            "Narasi rekomendasi tidak dapat dibuat saat ini. "
            "Dashboard angka di atas tetap valid — silakan coba lagi "
            "beberapa saat lagi."
        )

    # ── Pengingat kapasitas watt vs VA (BUKAN anomali) ──────────────────────
    # Skenario ekstrem "kalau semua alat nyala bersamaan" -- info edukatif
    # santai, bukan klaim mendeteksi kejadian nyata (sistem tidak realtime
    # terhubung MCB). Lihat core/kalkulasi.py::cek_kapasitas_watt.
    info_kapasitas_watt = cek_kapasitas_watt(payload['alat_valid'], req.daya_va)

    return AnalisisResponse(
        **payload,
        label_ike=label_ike,
        hasil_dsm=hasil_dsm,
        hasil_optimasi=hasil_opt,
        narasi=narasi,
        info_kapasitas_watt=info_kapasitas_watt,
    )