"""
streamlit_app.py
================
Antarmuka Streamlit untuk MENGUJI sistem penerimaan dan validasi input.

Berkas ini SATU-SATUNYA yang mengimpor Streamlit. Ia tidak menulis ulang
logika validasi — ia mengumpulkan input dari widget, menyusunnya menjadi satu
dict, lalu memanggil `app.validasi_permintaan()`. Dengan begitu logika yang
diuji di sini adalah logika yang sama persis yang nanti dipakai FastAPI dan
frontend, bukan salinannya.

    widget Streamlit  ->  dict mentah  ->  app.validasi_permintaan()  ->  hasil

Jalankan dari ROOT proyek:

    streamlit run streamlit_app.py

Catatan: pada tahap ini antarmuka baru sampai validasi. Ia menampilkan data
bersih yang lolos, atau daftar kesalahan. Penyambungan ke Lapis 1-2-3 (kalkulasi,
DSM, optimizer) dilakukan pada langkah berikutnya, setelah berkas-berkas itu
ikut diperbarui.
"""

import streamlit as st

from app import validasi_permintaan, opsi_input

st.set_page_config(page_title="EnergiCerdas AI — Uji Input", page_icon="⚡",
                   layout="wide")

OPSI = opsi_input()

st.title("⚡ EnergiCerdas AI")
st.caption("Mode pengujian penerimaan & validasi input")


# ── State daftar peralatan ────────────────────────────────────────────────────
if "peralatan" not in st.session_state:
    st.session_state.peralatan = []


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROFIL RUMAH & METERAN
# ══════════════════════════════════════════════════════════════════════════════
st.header("1. Profil Rumah & Meteran")

c1, c2, c3 = st.columns(3)
with c1:
    luas_m2 = st.number_input("Luas Bangunan (m²)", min_value=1, max_value=100_000,
                              value=45)
with c2:
    penghuni = st.number_input("Jumlah Penghuni", min_value=1, max_value=1_000,
                               value=3)
with c3:
    daya_va = st.selectbox("Daya Tersambung (VA)", OPSI["golongan_daya"], index=1)

jenis = st.radio("Jenis Meteran", ["Pascabayar", "Prabayar"], horizontal=True)
is_prabayar = jenis == "Prabayar"


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA BULAN LALU
# ══════════════════════════════════════════════════════════════════════════════
st.header("2. Data Bulan Lalu")

tagihan_riil = None
sisa_kwh = None
target_hari = 30

if is_prabayar:
    cc1, cc2 = st.columns(2)
    with cc1:
        sisa_kwh = st.number_input("Sisa kWh di Meteran (opsional)",
                                   min_value=0.0, max_value=100_000.0, value=0.0,
                                   help="Kosongkan (biarkan 0) bila tidak diisi.")
    with cc2:
        target_hari = st.number_input("Target Bertahan (hari)", min_value=1,
                                      max_value=366, value=30)
    st.info("Deteksi anomali tidak dijalankan untuk prabayar.")
else:
    tagihan_riil = st.number_input("Tagihan Bulan Lalu (Rp)", min_value=0.0,
                                   max_value=1_000_000_000.0, value=500_000.0,
                                   step=10_000.0,
                                   help="Pembanding untuk deteksi anomali.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PERALATAN
# ══════════════════════════════════════════════════════════════════════════════
st.header("3. Inventaris Peralatan")

with st.form("tambah_alat", clear_on_submit=True):
    f1, f2 = st.columns(2)
    with f1:
        in_nama = st.text_input("Nama Alat", placeholder="cth: AC Kamar")
        in_kategori = st.selectbox("Kategori", OPSI["kategori_peralatan"])
        in_jam = st.number_input("Jam / Hari", min_value=0.1, max_value=24.0,
                                 value=4.0, step=0.5)
    with f2:
        in_teg = st.number_input("Tegangan (V)", min_value=1.0, max_value=1000.0,
                                 value=220.0)
        in_arus = st.number_input("Arus per Unit (A)", min_value=0.001,
                                  max_value=1000.0, value=1.5, format="%.3f")
        in_jml = st.number_input("Jumlah Unit", min_value=1, max_value=10_000,
                                 value=1)

    if st.form_submit_button("Tambahkan", width="stretch"):
        st.session_state.peralatan.append({
            "nama": in_nama, "kategori": in_kategori, "tegangan": in_teg,
            "arus": in_arus, "jam": in_jam, "jumlah": in_jml,
        })

if st.session_state.peralatan:
    st.write(f"**{len(st.session_state.peralatan)} peralatan:**")
    for i, a in enumerate(st.session_state.peralatan):
        col, btn = st.columns([8, 1])
        jml = f" × {a['jumlah']}" if a.get("jumlah", 1) > 1 else ""
        col.write(f"🔌 **{a['nama'] or '(tanpa nama)'}** — {a['kategori']} · "
                  f"{a['tegangan']}V × {a['arus']}A · {a['jam']} j/hari{jml}")
        if btn.button("🗑️", key=f"del{i}"):
            st.session_state.peralatan.pop(i)
            st.rerun()
    if st.button("Hapus Semua"):
        st.session_state.peralatan = []
        st.rerun()
else:
    st.warning("Belum ada peralatan.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDASI
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

if st.button("🔍 Validasi Input", type="primary", width="stretch"):
    permintaan = {
        "luas_m2":     luas_m2,
        "penghuni":    penghuni,
        "daya_va":     daya_va,
        "is_prabayar": is_prabayar,
        "peralatan":   st.session_state.peralatan,
    }
    if is_prabayar:
        if sisa_kwh and sisa_kwh > 0:
            permintaan["sisa_kwh"] = sisa_kwh
        permintaan["target_hari"] = target_hari
    else:
        permintaan["tagihan_riil"] = tagihan_riil

    hasil = validasi_permintaan(permintaan)

    if hasil["ok"]:
        st.success("Semua input valid. Data bersih siap diteruskan ke Lapis 1.")
        st.json(hasil["bersih"])
    else:
        st.error(f"{len(hasil['errors'])} kesalahan ditemukan:")
        for e in hasil["errors"]:
            st.markdown(f"- {e}")