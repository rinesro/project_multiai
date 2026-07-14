# ============================================================
# EnergiCerdas AI — Backend Dockerfile (Hugging Face Spaces)
# ============================================================
# Image ini HANYA berisi backend FastAPI — TIDAK termasuk app.py
# (Streamlit), yang deploy-nya terpisah (Streamlit Community Cloud).

FROM python:3.11-slim

WORKDIR /app

# libgomp1 dibutuhkan LightGBM (OpenMP runtime) — tanpa ini,
# import lightgbm bisa gagal di image Debian-based yang minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (terpisah dari kode) supaya Docker layer cache
# tidak invalidasi instalasi dependency tiap kali cuma kode yang berubah.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code — core/, services/, models/, optimizer/, fuzzy/, data/
# sekarang semua nested di dalam backend/ (bukan folder sepupu terpisah),
# supaya struktur yang sama juga bisa di-deploy sebagai Vercel Service
# (yang cuma bundling isi satu folder root, tidak menjangkau folder di
# luar itu). Jadi cukup satu COPY untuk semuanya.
COPY backend/ ./backend/

# Hugging Face Spaces (Docker SDK) secara default expose port 7860 dan
# menjalankan container sebagai user non-root — pastikan /app writable.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Port dinamis: Render inject env var PORT saat runtime (biasanya 10000),
# Hugging Face Spaces pakai port tetap 7860. Fallback ${PORT:-7860} bikin
# satu Dockerfile ini portable untuk keduanya tanpa perlu diubah.
# CATATAN: shell form (bukan exec form/JSON array) SENGAJA dipakai di sini
# supaya variabel $PORT di-substitusi oleh shell saat container start —
# exec form tidak bisa melakukan substitusi environment variable.
EXPOSE 7860

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}