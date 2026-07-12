"""
services/narasi.py
==================
Pembangkit narasi rekomendasi dengan Gemini. Dipisah dari app.py supaya
FastAPI bisa memanggilnya tanpa Streamlit.

SDK: `google-generativeai` (paket lama), sesuai requirements.txt proyek ini.

    Paket ini berstatus legacy. Penggantinya adalah `google-genai`. Bila suatu
    saat dimigrasikan, HANYA tiga baris di berkas ini yang berubah:

        from google import genai                       # bukan google.generativeai
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(model=MODEL, contents=prompt).text

    dan `google-generativeai` di requirements.txt diganti `google-genai`.

CATATAN DEPLOYMENT:
    API key HARUS berada di sisi server (environment variable / secret).
    Jangan pernah mengirimkannya ke browser. Pasang rate limit sebelum
    endpoint ini dibuka ke publik.
"""

import google.generativeai as genai

MODEL = "gemini-2.5-flash"


def _ringkas_alat(hasil_dsm: list, label: str) -> str:
    items = [a for a in hasil_dsm if a["label_dsm"] == label]
    if not items:
        return "tidak ada"
    return ", ".join(
        f"{a['nama']} ({a['watt_nyata']:.0f}W, {a['jam']}j/hari)" for a in items
    )


def _konteks_optimasi(o: dict) -> str:
    if not o["aktif"]:
        return "Optimasi tidak diperlukan — konsumsi sudah dalam zona efisien."

    if not o["langkah"]:
        return (
            f"Optimasi tidak menghasilkan langkah. {o['pesan']}\n"
            f"IKE terbaik yang dicapai: {o['ike_akhir']} kWh/m2/bulan."
        )

    langkah = "\n".join(
        f"  - {l['nama']}: kurangi dari {l['jam_awal']} jam menjadi "
        f"{l['jam_rekomendasi']} jam (hemat {l['hemat_kwh']} kWh, "
        f"{l['hemat_emisi_kg']} kgCO2 per bulan)"
        for l in o["langkah"]
    )
    return (
        f"Hasil Optimasi:\n"
        f"  Status      : {o['status']}\n"
        f"  IKE         : {o['ike_awal']} -> {o['ike_akhir']} kWh/m2/bulan\n"
        f"  Zona        : {o['zona_awal']} -> {o['zona_akhir']}\n"
        f"  Hemat biaya : Rp {o['hemat_rp']:,}/bulan ({o['persen_hemat_rp']}%)\n"
        f"  Kurang emisi: {o['hemat_emisi_kg']} kgCO2/bulan\n"
        f"Langkah spesifik:\n{langkah}"
    )


def _konteks_token(tk: dict) -> str:
    if not tk or not tk.get("valid"):
        return ""
    baris = [
        "Status Token Prabayar:",
        f"  Sisa kWh di meteran : {tk['sisa_kwh']} kWh",
        f"  Konsumsi harian     : {tk['kwh_harian']} kWh/hari",
        f"  Habis dalam         : {tk['hari_tersisa']} hari ({tk['tanggal_habis']})",
        f"  Alarm meteran       : sekitar {tk['tanggal_alarm']}",
        f"  Isi ulang disarankan: Rp {tk['nominal_disarankan']:,.0f} "
        f"untuk {tk['target_hari']} hari",
        f"  Token per bulan     : Rp {tk['token_bulanan_sekarang']:,.0f} sekarang, "
        f"Rp {tk['token_bulanan_optimal']:,.0f} bila pola dioptimalkan "
        f"(hemat Rp {tk['hemat_token_rp']:,.0f})",
    ]
    if tk["hemat_token_rp"] > 0:
        baris.append(
            f"  Dengan pola optimal, sisa token bertahan "
            f"{tk['hari_tersisa_optimal']} hari (bukan {tk['hari_tersisa']} hari)."
        )
    return "\n".join(baris)


def generate_narasi(api_key: str, hasil: dict, intent_user: list) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    p, o, tk = hasil["payload"], hasil["hasil_opt"], hasil["token"]
    is_prabayar = p["is_prabayar"]
    istilah = "biaya token" if is_prabayar else "tagihan"

    anomali_str = (
        f"TERDETEKSI ({p['arah_anomali']}, selisih {p['selisih_pct']}%). "
        f"{p['pesan_anomali']}"
        if p["is_anomali"] else "Normal, biaya riil di dalam rentang prediksi."
    )

    prompt = f"""
Kamu adalah EnergiCerdas AI, konsultan energi rumah tangga yang ramah dan
berbasis data regulasi resmi Indonesia. Teks ini muncul TEPAT DI BAWAH dasbor
metrik angka di aplikasi web.

DATA ANALISIS
- Jenis meteran     : {'Prabayar (token)' if is_prabayar else 'Pascabayar (tagihan)'}
- Anomali {istilah} : {anomali_str}
- Profil IKE        : {hasil['label_ike']} (IKE {p['ike']:.4f} kWh/m2/bulan)
- Prediksi {istilah}: Rp {p['biaya_bawah']:,.0f} sampai Rp {p['biaya_atas']:,.0f}
- Emisi sekarang    : {p['emisi_sebelum']['emisi_kg_bulan']} kgCO2/bulan
- Fokus pengguna    : {' + '.join(intent_user) if intent_user else 'Efisiensi Umum'}
- Peralatan fleksibel : {_ringkas_alat(hasil['hasil_dsm'], 'Fleksibel')}
- Peralatan tetap     : {_ringkas_alat(hasil['hasil_dsm'], 'Tidak Fleksibel')}

{_konteks_optimasi(o)}

{_konteks_token(tk)}

ATURAN PENULISAN
1. Jangan gunakan kalimat pembuka seperti "Berikut adalah" atau "Berdasarkan
   analisis". Langsung merujuk ke dasbor di atas.
2. Selalu sebut angka {istilah} sebagai RENTANG PREDIKSI, bukan angka pasti,
   dan ingatkan sekali secara halus bahwa PLN menghitung dengan cara berbeda.
3. Bila ada langkah optimasi, sebut nama peralatan, jam pengurangannya, dan
   dampaknya secara spesifik.
{"4. Jelaskan kapan token akan habis dan berapa yang sebaiknya dibeli, dengan nada membantu bukan menakuti." if is_prabayar else "4. Jelaskan bahwa Rekening Minimum hanya berlaku bila pemakaian di bawah 40 jam nyala."}
5. Hubungkan penghematan energi dengan SDG 7 dan SDG 13 secara natural.
6. Bahasa Indonesia yang hangat dan tidak terlalu teknis. 3-4 paragraf,
   tanpa bullet point.
7. Akhiri dengan kalimat penyemangat yang personal.
"""
    return model.generate_content(prompt).text