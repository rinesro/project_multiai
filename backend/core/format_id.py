"""
backend/core/format_id.py
===========================
Format angka gaya Indonesia (koma untuk desimal, titik untuk ribuan)
buat pesan teks yang langsung ditampilkan ke user (pesan_anomali,
narasi Gemini, dll).

Kenapa modul terpisah, bukan pakai locale bawaan Python: locale
sistem (mis. `locale.setlocale(locale.LC_ALL, 'id_ID')`) TIDAK
selalu tersedia di lingkungan serverless (Vercel dkk. seringkali
tidak menyediakan locale selain 'C'/'POSIX' secara default), dan
mengandalkannya bisa diam-diam salah format kalau locale itu gagal
di-set tanpa error yang jelas. Pendekatan manual (format standar lalu
tukar simbol) di sini menjamin hasil konsisten di lingkungan apa pun.
"""


def format_angka_id(nilai: float, desimal: int = 1) -> str:
    """
    Format angka: koma untuk desimal, titik untuk ribuan (kebalikan
    default Python/Inggris).

    Contoh: format_angka_id(90.2, 1) -> "90,2"
            format_angka_id(1234.5, 1) -> "1.234,5"
            format_angka_id(936, 0) -> "936"
    """
    # Format standar dulu (koma=ribuan, titik=desimal, gaya Inggris),
    # baru tukar via placeholder supaya tidak tabrakan pas swap.
    teks = f"{nilai:,.{desimal}f}"
    teks = teks.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return teks


def format_rupiah_id(nilai: float) -> str:
    """
    Format Rupiah gaya Indonesia: "Rp " + titik sebagai pemisah ribuan,
    tanpa desimal (Rupiah dibulatkan ke satuan penuh).

    PENTING: Python f"{nilai:,}" pakai KOMA sebagai pemisah ribuan
    (gaya Inggris) -- kalau dipakai langsung di pesan berbahasa
    Indonesia, itu justru salah arah (harusnya titik). Fungsi ini
    memperbaikinya.

    Contoh: format_rupiah_id(139135) -> "Rp 139.135"
    """
    return f"Rp {format_angka_id(nilai, 0)}"