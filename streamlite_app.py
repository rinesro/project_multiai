"""
streamlit_app.py
================
Antarmuka Streamlit untuk menguji sistem EnergiCerdas AI end-to-end.

Mengumpulkan input dari widget, menyusunnya jadi satu dict mentah, memanggil
orchestrator.analisis(), lalu menampilkan hasilnya. Logika validasi dan
kalkulasi tidak ditulis ulang di sini.

Jalankan dari ROOT proyek:  streamlit run streamlit_app.py
"""

import streamlit as st

from app import opsi_input
from orchestrator import analisis
from models.dsm_classifier import DSMClassifier

st.set_page_config(page_title="EnergiCerdas AI", page_icon="⚡", layout="wide")

OPSI = opsi_input()


@st.cache_resource(show_spinner="Memuat model DSM...")
def muat_dsm():
    return DSMClassifier()


dsm_clf = muat_dsm()

with st.sidebar:
    st.header("Status Sistem")
    if dsm_clf.siap:
        st.success("Model DSM: LightGBM aktif")
    else:
        st.error(
            "Model DSM GAGAL dimuat. Sistem memakai fallback if-else, "
            "**bukan** LightGBM.\n\n"
            f"`{dsm_clf.pesan_error}`\n\n"
            "Pastikan dua berkas .pkl ada di folder data/."
        )

st.title("⚡ EnergiCerdas AI")

if "peralatan" not in st.session_state:
    st.session_state.peralatan = []

st.header("1. Profil Rumah & Meteran")
c1, c2, c3 = st.columns(3)
with c1:
    luas_m2 = st.number_input("Luas Bangunan (m²)", 1, 100_000, 45)
with c2:
    penghuni = st.number_input("Jumlah Penghuni", 1, 1_000, 3)
with c3:
    daya_va = st.selectbox("Daya Tersambung (VA)", OPSI["golongan_daya"], index=1)

jenis = st.radio("Jenis Meteran", ["Pascabayar", "Prabayar"], horizontal=True)
is_prabayar = jenis == "Prabayar"

tagihan_riil = None
if is_prabayar:
    st.info("Deteksi anomali tidak dijalankan untuk prabayar.")
else:
    tagihan_riil = st.number_input("Tagihan Bulan Lalu (Rp)", 0.0, 1e9, 500_000.0,
                                   step=10_000.0,
                                   help="Pembanding untuk deteksi anomali.")

st.header("2. Inventaris Peralatan")
with st.form("tambah", clear_on_submit=True):
    f1, f2 = st.columns(2)
    with f1:
        in_nama = st.text_input("Nama Alat", placeholder="cth: AC Kamar")
        in_kat = st.selectbox("Kategori", OPSI["kategori_peralatan"])
        in_jam = st.number_input("Jam / Hari", 0.1, 24.0, 4.0, 0.5)
    with f2:
        in_teg = st.number_input("Tegangan (V)", 1.0, 1000.0, 220.0)
        in_arus = st.number_input("Arus per Unit (A)", 0.001, 1000.0, 1.5, format="%.3f")
        in_jml = st.number_input("Jumlah Unit", 1, 10_000, 1)
    if st.form_submit_button("Tambahkan", width="stretch"):
        st.session_state.peralatan.append({
            "nama": in_nama, "kategori": in_kat, "tegangan": in_teg,
            "arus": in_arus, "jam": in_jam, "jumlah": in_jml,
        })

if st.session_state.peralatan:
    for i, a in enumerate(st.session_state.peralatan):
        col, btn = st.columns([8, 1])
        jml = f" × {a['jumlah']}" if a.get("jumlah", 1) > 1 else ""
        col.write(f"🔌 **{a['nama'] or '(tanpa nama)'}** — {a['kategori']} · "
                  f"{a['tegangan']}V × {a['arus']}A · {a['jam']} j/hari{jml}")
        if btn.button("🗑️", key=f"d{i}"):
            st.session_state.peralatan.pop(i)
            st.rerun()
    if st.button("Hapus Semua"):
        st.session_state.peralatan = []
        st.rerun()
else:
    st.warning("Belum ada peralatan.")

st.divider()

if st.button("🚀 Analisis", type="primary", width="stretch"):
    permintaan = {
        "luas_m2": luas_m2, "penghuni": penghuni, "daya_va": daya_va,
        "is_prabayar": is_prabayar, "peralatan": st.session_state.peralatan,
    }
    if not is_prabayar:
        permintaan["tagihan_riil"] = tagihan_riil

    hasil = analisis(permintaan, dsm_clf=dsm_clf)

    if not hasil["ok"]:
        st.error(f"{len(hasil['errors'])} kesalahan input:")
        for e in hasil["errors"]:
            st.markdown(f"- {e}")
        st.stop()

    st.divider()
    st.subheader("Hasil Analisis")

    jenis_biaya = hasil["biaya"]["jenis"].title()
    st.warning(
        f"**Prediksi — bukan {jenis_biaya.lower()} resmi PLN.** Sistem hanya "
        f"menerima tegangan dan arus, sehingga faktor daya tidak diketahui. "
        f"Nilai berupa rentang antara asumsi induktif (0,85) dan resistif (1,00)."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Prediksi {jenis_biaya}", f"Rp {hasil['biaya']['bawah']:,.0f}",
              f"s/d Rp {hasil['biaya']['atas']:,.0f}", delta_color="off")
    m2.metric("IKE", f"{hasil['ike']} kWh/m²/bln",
              hasil["optimasi"]["zona_awal"], delta_color="off")
    m3.metric("Konsumsi", f"{hasil['kwh_bawah']} – {hasil['kwh_atas']} kWh",
              f"{hasil['kwh_per_org']} kWh/orang", delta_color="off")
    m4.metric("Emisi CO₂", f"{hasil['emisi']['bawah']} – {hasil['emisi']['atas']} kg",
              delta_color="off")

    if hasil["anomali"] is not None:
        an = hasil["anomali"]
        if an["is_anomali"]:
            st.error(f"**Anomali terdeteksi** ({an['selisih_pct']}%). {an['pesan']}")
        else:
            st.success(f"Tagihan wajar. {an['pesan']}")

    o = hasil["optimasi"]
    st.divider()
    st.subheader("Rekomendasi Optimasi")

    if not o["aktif"]:
        st.success(o["pesan"])
    elif not o["langkah"]:
        st.warning(o["pesan"])
    else:
        st.info(f"{o['pesan']}\n\nIKE {o['ike_awal']} → {o['ike_akhir']} · "
                f"Zona {o['zona_awal']} → {o['zona_akhir']}")
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Hemat {jenis_biaya}/bln", f"Rp {o['hemat_rp']:,}",
                  f"{o['persen_hemat_rp']}%")
        k2.metric("Kurang Emisi/bln", f"{o['hemat_emisi_kg']} kgCO₂",
                  f"{o['persen_hemat_emisi']}%")
        k3.metric(f"{jenis_biaya} Optimal", f"Rp {o['biaya_akhir']['bawah']:,.0f}",
                  f"s/d Rp {o['biaya_akhir']['atas']:,.0f}", delta_color="off")
        st.write("**Langkah:**")
        for l in o["langkah"]:
            st.markdown(
                f"- 🔌 **{l['nama']}** — kurangi {l['jam_awal']} → "
                f"**{l['jam_rekomendasi']} jam/hari** "
                f"(hemat {l['hemat_kvah']} kVAh · Rp {l['hemat_rp']:,} · "
                f"{l['hemat_emisi_kg']} kgCO₂/bln)"
            )

    with st.expander("Klasifikasi DSM per Peralatan"):
        for a in hasil["dsm"]:
            icon = "🟢" if a["label_dsm"] == "Fleksibel" else "🔴"
            tag = " ⚠️ *fallback*" if a["metode"] == "fallback" else ""
            st.markdown(f"{icon} **{a['nama']}** — {a['label_dsm']} "
                        f"({a['daya_semu_va']:.0f} VA · {a['kvah_bulan']} kVAh/bln){tag}")

    with st.expander("Data Input Bersih (JSON)"):
        st.json(hasil["input"])
