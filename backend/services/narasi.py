"""
services/narasi.py
====================
Generator narasi rekomendasi berbasis Gemini — diekstrak dari app.py
supaya Streamlit (app.py) DAN backend FastAPI sama-sama impor dari
sini, bukan menulis ulang prompt/logika di dua tempat.

Tidak ada dependensi Streamlit di modul ini.
"""

import re

import google.generativeai as genai


def _bersihkan_markdown(teks: str) -> str:
    """
    Jaring pengaman KEDUA selain instruksi prompt (aturan #11 di bawah)
    -- LLM tidak 100% patuh instruksi setiap saat. Hapus sisa sintaks
    markdown yang mungkin lolos supaya tidak muncul literal di UI
    (kedua interface merender teks ini sebagai paragraf polos: Next.js
    lewat interpolasi teks biasa <p>{narasi}</p>, Streamlit lewat
    st.markdown() yang justru MERENDER markdown -- inkonsistensi ini
    persis kenapa markdown harus dihindari total dari sumbernya, bukan
    cukup diandalkan salah satu sisi UI untuk menanganinya).

    Regex ini SENGAJA tidak menyentuh tanda hubung/pisah yang sah dalam
    Bahasa Indonesia (kata ulang seperti "anak-anak", rentang angka
    seperti "2020-2024") -- sudah diuji eksplisit untuk kasus ini.
    """
    teks = re.sub(r'\*\*(.+?)\*\*', r'\1', teks)          # **tebal**
    teks = re.sub(r'__(.+?)__', r'\1', teks)               # __tebal__
    teks = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'\1', teks)  # *miring*
    teks = re.sub(r'^#{1,6}\s*', '', teks, flags=re.MULTILINE)          # # Judul
    teks = re.sub(r'^[\-\*]\s+', '', teks, flags=re.MULTILINE)          # - bullet
    return teks.strip()


# ============================================================
# HELPER FORMAT HASIL OPTIMASI — token (prabayar) vs Rp (pascabayar)
# ============================================================
# Optimizer (optimizer/brute_force.py) TIDAK berubah — tetap menghitung
# hemat_kwh & hemat_rp sekaligus untuk kedua jenis meteran. Yang beda
# cuma cara MENAMPILKANNYA. Dipusatkan di sini supaya narasi Gemini dan
# render UI (Streamlit maupun frontend Next.js lewat response API)
# selalu konsisten — tidak ditulis ulang di banyak tempat.

def format_hemat_langkah(l: dict, is_prabayar: bool) -> str:
    """Format penghematan satu langkah rekomendasi peralatan."""
    if is_prabayar:
        return f"hemat {l['hemat_kwh']} kWh token"
    return f"hemat Rp {l['hemat_rp']:,}/bulan"


def format_hemat_total(hasil_opt: dict, is_prabayar: bool) -> str:
    """Format ringkasan total penghematan hasil optimasi."""
    if is_prabayar:
        return (
            f"{hasil_opt['hemat_kwh']} kWh/bulan "
            f"(≈ Rp {hasil_opt['hemat_rp']:,}, "
            f"{hasil_opt['persen_hemat_rp']}%)"
        )
    return f"Rp {hasil_opt['hemat_rp']:,}/bulan ({hasil_opt['persen_hemat_rp']}%)"


def bucket_rekomendasi(hasil_dsm: list, hasil_opt: dict) -> dict:
    """
    Versi Python dari logika bucketing yang sama dengan
    frontend/components/RekomendasiPerangkat.tsx — sengaja diduplikasi
    lintas bahasa (Python di sini, TypeScript di frontend) karena
    keduanya merender UI masing-masing dari JSON yang sama, tidak ada
    cara praktis berbagi satu implementasi lintas runtime. Kalau logika
    bucketing ini berubah, update DUA tempat ini.

    Returns dict:
        ganti   : list alat 'Tidak Fleksibel' (nama, tegangan, arus, watt)
        kurangi : list alat dari hasil_opt['langkah'] (kalau optimasi aktif)
    """
    ganti = [
        {'nama': a['nama'], 'tegangan': a['tegangan'], 'arus': a['arus'], 'watt': a['watt']}
        for a in hasil_dsm if a['label_dsm'] == 'Tidak Fleksibel'
    ]
    kurangi = hasil_opt.get('langkah', []) if hasil_opt and hasil_opt.get('aktif') else []
    return {'ganti': ganti, 'kurangi': kurangi}



def generate_gemini_narasi(api_key       : str,
                            label_ike     : str,
                            payload       : dict,
                            hasil_dsm     : list,
                            hasil_opt     : dict,
                            intent_user   : list) -> str:
    """
    Menghasilkan narasi rekomendasi menggunakan Gemini.

    PENTING — pembagian tanggung jawab (sejak restrukturisasi UX):
    Narasi ini sekarang jadi KONTEN UTAMA yang ditampilkan di ATAS
    dashboard angka (bukan lagi pelengkap di bawah). Supaya tetap
    akurat dan ringkas, Gemini HANYA menulis narasi strategi/motivasi
    tingkat tinggi — TIDAK diminta menyebut nama alat, tegangan, arus,
    atau angka spesifik per-alat. Daftar aksi konkret per alat (ganti/
    kurangi) dihitung deterministik oleh kode (lihat
    components/RekomendasiPerangkat.tsx di frontend, atau rendering
    setara di app.py) dan ditampilkan TEPAT DI BAWAH narasi ini —
    supaya nama alat & angka teknis selalu presisi, tidak berisiko
    salah ketik/halusinasi oleh LLM.

    Konteks yang dikirim ke Gemini (ringkasan, bukan daftar mentah):
        - Status anomali (Rp untuk pascabayar, kWh/token untuk prabayar)
        - Profil IKE (zona efisiensi)
        - RINGKASAN jumlah alat per kategori (bukan nama alat)
        - RINGKASAN hasil optimasi (total hemat, jumlah langkah)
        - Fokus user (Biaya / Lingkungan) — memengaruhi penekanan narasi
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Ringkasan jumlah alat per kategori — BUKAN daftar nama, supaya
    # Gemini tidak tergoda menyebut alat spesifik (itu tugas UI).
    jumlah_fleksibel     = sum(1 for a in hasil_dsm if a['label_dsm'] == 'Fleksibel')
    jumlah_tdk_fleksibel = sum(1 for a in hasil_dsm if a['label_dsm'] == 'Tidak Fleksibel')

    # Susun ringkasan konteks optimasi — angka total saja, tanpa rincian per-alat.
    if hasil_opt and hasil_opt.get('aktif'):
        opt_status = hasil_opt['status']
        jumlah_langkah = len(hasil_opt.get('langkah', []))
        if opt_status in ('efisien', 'cukup_efisien'):
            label_hemat = "Hemat token" if payload['is_prabayar'] else "Hemat biaya"
            konteks_opt = f"""
Hasil Optimasi: Berhasil, ada {jumlah_langkah} alat yang direkomendasikan \
untuk dikurangi jam pakainya.
  {label_hemat} total : {format_hemat_total(hasil_opt, payload['is_prabayar'])}
  Kurang emisi total  : {hasil_opt['hemat_emisi_kg']} kgCO₂/bulan \
({hasil_opt['persen_hemat_emisi']}%)"""
        else:
            label_hemat = "Hemat token" if payload['is_prabayar'] else "Hemat biaya"
            konteks_opt = f"""
Hasil Optimasi: Target belum tercapai meski {jumlah_langkah} alat fleksibel \
sudah dimaksimalkan pengurangannya — sisanya cuma bisa dihemat lewat ganti alat.
  {label_hemat} total : {format_hemat_total(hasil_opt, payload['is_prabayar'])}"""
    else:
        konteks_opt = "Optimasi tidak diperlukan — konsumsi sudah dalam zona efisien."

    # pesan_anomali sudah lengkap & sesuai domain (Rp/kWh) dari
    # action_analist/anomaly_evaluator.py — tidak perlu disusun ulang di sini.
    anomali_str       = payload['pesan_anomali']
    jenis_meteran_str = (
        "Prabayar (Token)" if payload['is_prabayar'] else "Pascabayar (Tagihan)"
    )
    # PENTING: sebelumnya cuma nyantumin fokus_str sebagai DATA pasif
    # ("Fokus user: Biaya") tanpa instruksi perilaku yang tegas —
    # akibatnya Gemini tidak benar-benar membedakan narasi antara
    # "Biaya saja" vs "Biaya + Lingkungan" (dua-duanya kebaca sama).
    # Sekarang dibuat instruksi EKSPLISIT per kombinasi, bukan cuma
    # data yang diserahkan ke interpretasi bebas Gemini.
    fokus_biaya      = "Biaya" in intent_user
    fokus_lingkungan = "Lingkungan" in intent_user
    if fokus_biaya and fokus_lingkungan:
        instruksi_fokus = (
            "User peduli BIAYA dan LINGKUNGAN sekaligus — bahas dua-duanya "
            "secara seimbang, jangan berat sebelah ke salah satu."
        )
    elif fokus_biaya:
        instruksi_fokus = (
            "User HANYA memilih fokus BIAYA — fokus penuh ke penghematan "
            "tagihan/token. JANGAN singgung dampak lingkungan atau emisi "
            "sama sekali, walau datanya tersedia di atas."
        )
    elif fokus_lingkungan:
        instruksi_fokus = (
            "User HANYA memilih fokus LINGKUNGAN — fokus penuh ke "
            "pengurangan emisi/dampak lingkungan. JANGAN singgung "
            "penghematan biaya/tagihan/token sama sekali, walau datanya "
            "tersedia di atas."
        )
    else:
        instruksi_fokus = (
            "User tidak menentukan fokus spesifik — bahas efisiensi secara "
            "umum, boleh singgung biaya maupun lingkungan secukupnya."
        )
    emisi_sblm   = payload['emisi_sebelum']

    prompt = f"""
Kamu adalah EnergiCerdas AI — konsultan energi rumah tangga Jakarta yang ramah, \
suportif, dan berbasis data regulasi resmi Indonesia.

Teks ini adalah KONTEN UTAMA yang dibaca user PALING PERTAMA, di ATAS \
dashboard angka teknis. Tepat di BAWAH teks kamu, sudah ada daftar aksi \
konkret per alat (nama, ganti atau kurangi jam pakai) yang dibuat sistem \
secara terpisah — jadi kamu TIDAK PERLU dan TIDAK BOLEH menyebutkan nama \
alat, tegangan, arus, atau angka spesifik per-alat. Tugasmu murni \
menjelaskan STRATEGI dan MEMOTIVASI, dengan bahasa yang bisa dipahami \
orang awam yang tidak paham istilah kelistrikan teknis.

DATA ANALISIS (ringkasan, bukan rincian):
- Jenis meteran         : {jenis_meteran_str}
- Status anomali        : {anomali_str}
- Tingkat efisiensi     : {label_ike} (skor IKE {payload['ike']:.2f} kWh/m²/bulan — \
JANGAN sebut angka atau istilah "IKE" ini ke user)
- Emisi sekarang        : {emisi_sblm['emisi_kg_bulan']} kgCO₂/bulan
- Jumlah alat bisa dikurangi jam pakainya : {jumlah_fleksibel}
- Jumlah alat sebaiknya diganti           : {jumlah_tdk_fleksibel}

INSTRUKSI FOKUS (WAJIB DIIKUTI KETAT, ini bukan sekadar informasi): \
{instruksi_fokus}

{konteks_opt}

ATURAN PENULISAN:
1. JANGAN gunakan kalimat pembuka seperti "Berikut adalah..." atau \
"Berdasarkan analisis...". Langsung ke inti, contoh: "Kondisi kelistrikan \
Anda saat ini..."
2. JANGAN sebutkan nama alat spesifik, tegangan, arus, atau angka \
per-alat — itu sudah ditampilkan sebagai daftar terpisah tepat di bawah \
tulisanmu. Cukup rujuk secara umum, contoh: "beberapa alat bisa dikurangi \
jam pakainya, dan ada juga yang lebih baik diganti — lihat daftar di bawah."
3. Jelaskan STRATEGI secara umum dengan antusias: kenapa mengurangi jam \
pakai beda dengan mengganti alat, dan mana yang lebih prioritas.
4. Akhiri dengan kalimat penyemangat yang hangat dan personal.
5. Jika konsumsi sudah efisien, berikan pujian tulus dan satu tips umum \
tingkat lanjut (tanpa menyebut alat spesifik).
6. Ikuti INSTRUKSI FOKUS di atas secara KETAT — itu aturan wajib, bukan \
sekadar konteks tambahan. Kalau dampak lingkungan/emisi memang boleh \
disinggung (sesuai instruksi fokus), JANGAN sebut istilah kebijakan/ \
akademis seperti "SDG", "Sustainable Development Goals", "pembangunan \
berkelanjutan", "tujuan pembangunan global", "aksi iklim", atau \
semacamnya — cukup manfaat konkret yang LANGSUNG dirasakan user sendiri.
7. Gunakan bahasa Indonesia yang hangat, sangat mudah dipahami orang \
awam, tidak teknis sama sekali.
8. Panjang respons: 2–3 paragraf singkat — ini konten UTAMA yang dibaca \
duluan, harus padat dan enak di-scan, bukan wall of text.
9. Kalau jenis meteran Prabayar (Token), JANGAN gunakan istilah \
"tagihan" — pakai "saldo token" atau "penggunaan listrik". Kalau \
Pascabayar, tetap pakai istilah "tagihan" seperti biasa.
10. JANGAN gunakan istilah teknis "IKE" atau "Intensitas Konsumsi \
Energi" sama sekali — cukup sebut kategorinya ("Efisien", "Boros", dst).
11. JANGAN pakai format markdown SAMA SEKALI — tidak ada **tebal**, \
*miring*, daftar berpoin dengan "-"/"*", maupun tanda pagar "#" untuk \
judul. Teks ini dirender APA ADANYA sebagai paragraf polos (bukan lewat \
parser markdown), jadi simbol markdown akan muncul literal dan terlihat \
rusak di layar user. Tulis paragraf naratif biasa dari awal sampai akhir.
12. Ikuti kaidah tanda baca Bahasa Indonesia (PUEBI) dengan benar, \
khususnya tanda hubung/pisah: JANGAN pakai tanda "-" sebagai penyisip \
klausa informal di tengah kalimat (gaya itu tidak baku dalam Bahasa \
Indonesia meski umum di teks Inggris). Kalau perlu menyisipkan \
keterangan tambahan, gunakan tanda koma, titik dua, atau susun ulang \
jadi dua kalimat — bukan tanda hubung/pisah di tengah kalimat.
"""
    response = model.generate_content(prompt)
    return _bersihkan_markdown(response.text)