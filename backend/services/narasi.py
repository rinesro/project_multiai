"""
services/narasi.py
====================
Generator narasi rekomendasi berbasis Gemini — diekstrak dari app.py
supaya Streamlit (app.py) DAN backend FastAPI sama-sama impor dari
sini, bukan menulis ulang prompt/logika di dua tempat.

Tidak ada dependensi Streamlit di modul ini.
"""

import google.generativeai as genai


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
    # action_analyst/anomaly_evaluator.py — tidak perlu disusun ulang di sini.
    anomali_str       = payload['pesan_anomali']
    jenis_meteran_str = (
        "Prabayar (Token)" if payload['is_prabayar'] else "Pascabayar (Tagihan)"
    )
    fokus_str    = " + ".join(intent_user) if intent_user else "Efisiensi Umum"
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
- Fokus user             : {fokus_str}
- Jumlah alat bisa dikurangi jam pakainya : {jumlah_fleksibel}
- Jumlah alat sebaiknya diganti           : {jumlah_tdk_fleksibel}

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
6. Kalau relevan dengan fokus user (terutama fokus "Lingkungan"), boleh \
sebutkan dampak lingkungan (pengurangan emisi) secara ringan sebagai \
bonus motivasi — TAPI JANGAN sebut istilah kebijakan/akademis seperti \
"SDG", "Sustainable Development Goals", "pembangunan berkelanjutan", \
"tujuan pembangunan global", "aksi iklim", atau semacamnya. Fokus pada \
manfaat konkret yang LANGSUNG dirasakan user sendiri — rumah lebih \
hemat, tagihan lebih ringan, kontribusi kecil untuk lingkungan sekitar \
rumahnya — bukan narasi kebijakan atau agenda global.
7. Gunakan bahasa Indonesia yang hangat, sangat mudah dipahami orang \
awam, tidak teknis sama sekali.
8. Panjang respons: 2–3 paragraf singkat — ini konten UTAMA yang dibaca \
duluan, harus padat dan enak di-scan, bukan wall of text.
9. Kalau jenis meteran Prabayar (Token), JANGAN gunakan istilah \
"tagihan" — pakai "saldo token" atau "penggunaan listrik". Kalau \
Pascabayar, tetap pakai istilah "tagihan" seperti biasa.
10. JANGAN gunakan istilah teknis "IKE" atau "Intensitas Konsumsi \
Energi" sama sekali — cukup sebut kategorinya ("Efisien", "Boros", dst).
"""
    response = model.generate_content(prompt)
    return response.text
