"""
backend/bootstrap.py
======================
Utilitas start-up yang HARUS dipanggil paling awal, sebelum modul apa
pun yang bergantung pada LightGBM diimpor (models/dsm_classifier.py,
dan apa pun yang mengimpornya).

Sengaja dipisah dari main.py sebagai modul kecil TANPA dependensi lain
(tidak ada FastAPI, tidak ada Gemini, dst.) -- supaya bisa dipanggil
dari ENTRY POINT MANA PUN yang butuh LightGBM (main.py untuk produksi
Vercel, test_sistem.py untuk pengujian) tanpa menyeret urutan impor
modul lain yang punya dependensi urutan sendiri (mis. mock Gemini di
test_sistem.py yang harus terjadi SEBELUM main.py diimpor penuh).

Ditemukan lewat pengujian langsung (bukan asumsi teoretis): kalau
sebuah entry point mengimpor models/dsm_classifier.py TANPA lebih
dulu memanggil preload_vendored_libgomp() di sini, LightGBM akan
gagal dimuat di lingkungan yang tidak punya libgomp sistem (mis.
Vercel Python serverless) -- persis walau main.py SENDIRI sudah
benar memanggil ini di modul levelnya, KALAU entry point lain
(seperti test_sistem.py) sempat mengimpor DSM classifier lebih dulu
sebelum main.py sempat dimuat.
"""

import ctypes
import logging
from pathlib import Path

_sudah_dipanggil = False


def preload_vendored_libgomp() -> None:
    """
    Vercel Python serverless function TIDAK menyediakan libgomp.so.1
    (runtime OpenMP) sebagai library sistem, padahal LightGBM
    membutuhkannya untuk memuat extension terkompilasinya -- tanpa ini,
    'import lightgbm' gagal dengan OSError: libgomp.so.1: cannot open
    shared object file.

    Solusinya: preload salinan libgomp.so.1 yang di-vendor manual di
    backend/vendor/ memakai ctypes SEBELUM LightGBM diimpor. Dengan
    RTLD_GLOBAL, symbol OpenMP di dalamnya langsung tersedia untuk
    proses yang sedang berjalan -- saat LightGBM dynamic-link ke
    "libgomp.so.1", linker menemukan salinan ini sudah termuat di
    memori by SONAME MATCH, tidak perlu dicari lagi di sistem.

    ASAL FILE & LANGKAH PENTING (kalau perlu regenerasi):
    1. Diambil dari wheel resmi scikit-learn==1.6.1 manylinux_2_17
       x86_64 (cocok arsitektur Vercel): file
       scikit_learn.libs/libgomp-a34b3233.so.1.0.0 di dalam wheel --
       BUKAN dari libgomp sistem lokal manapun (berisiko beda glibc
       ABI/arsitektur dari runtime Vercel).
    2. WAJIB di-patch dulu pakai `patchelf --set-soname libgomp.so.1
       vendor/libgomp.so.1` -- SONAME asli file ini adalah
       "libgomp-a34b3233.so.1.0.0" (auditwheel manylinux packaging
       vendor tiap wheel dengan nama unik supaya tidak bentrok simbol
       antar-wheel). TANPA langkah ini, RTLD_GLOBAL preload TETAP
       GAGAL menyelamatkan LightGBM -- dynamic linker mencocokkan
       dependency by SONAME PERSIS, bukan nama file di disk.

    Idempoten -- aman dipanggil berkali-kali dari entry point berbeda
    (main.py DAN test_sistem.py sama-sama memanggil ini), cuma efektif
    di panggilan pertama.

    TIDAK diperlukan di lingkungan lain yang sudah punya libgomp
    sebagai library sistem (mis. Docker berbasis Debian/Ubuntu dengan
    apt-get install libgomp1) -- fungsi ini aman dipanggil di mana pun
    karena try/except silent kalau vendor file tidak ada atau gagal
    dimuat (biar tidak menghalangi start-up di lingkungan lain).
    """
    global _sudah_dipanggil
    if _sudah_dipanggil:
        return
    _sudah_dipanggil = True

    vendor_path = Path(__file__).parent / "vendor" / "libgomp.so.1"
    if not vendor_path.exists():
        return
    try:
        ctypes.CDLL(str(vendor_path), mode=ctypes.RTLD_GLOBAL)
        logging.info(f"libgomp.so.1 vendored berhasil di-preload dari {vendor_path}")
    except OSError as e:
        # Sengaja tidak melempar error -- kalau environment ini sudah
        # punya libgomp sistem sendiri (mis. Docker), preload vendor
        # ini opsional, bukan wajib. Kegagalan di sini bukan berarti
        # LightGBM pasti gagal dimuat setelahnya.
        logging.warning(f"Gagal preload libgomp.so.1 vendored: {e}")
