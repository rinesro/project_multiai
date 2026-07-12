"""
core/
=====
Konstanta regulasi dan rumus kalkulasi bersama, dipakai lintas Lapis 1, 2, 3.

Urutan dependensi (acyclic):
    konstanta.py  ->  token.py  ->  kalkulasi.py
"""

from core.konstanta import (
    TARIF_PER_GOLONGAN, TARIF_DEFAULT, GOLONGAN_DAYA,
    PBJT_RUMAH_TANGGA, FAKTOR_EMISI_JAMALI_OM,
    JAM_NYALA_RM, JAM_NYALA_MAKS_TOKEN, AMBANG_ALARM_KWH,
    BATAS_TOLERANSI_ANOMALI, COS_PHI, COS_PHI_DEFAULT,
    KATEGORI_VALID, BATAS_IKE,
)

from core.kalkulasi import (
    get_tarif, hitung_kwh_minimum, hitung_rm_rupiah,
    hitung_watt, hitung_watt_nyata, hitung_kwh_alat,
    hitung_tagihan, hitung_biaya_bulanan, hitung_rentang_biaya,
    hitung_emisi, hitung_ike, hitung_kwh_per_org, deteksi_anomali,
    KETERBATASAN_PREDIKSI,
)

from core.token import (
    hitung_kwh_dari_token, hitung_token_dari_kwh, batas_maks_kwh,
    validasi_pembelian, proyeksi_token, token_untuk_bertahan, jadwal_isi_ulang,
    bulatkan_nominal, kombinasi_denominasi,
)

__all__ = [
    # konstanta
    "TARIF_PER_GOLONGAN", "TARIF_DEFAULT", "GOLONGAN_DAYA",
    "PBJT_RUMAH_TANGGA", "FAKTOR_EMISI_JAMALI_OM",
    "JAM_NYALA_RM", "JAM_NYALA_MAKS_TOKEN", "AMBANG_ALARM_KWH",
    "BATAS_TOLERANSI_ANOMALI", "COS_PHI", "COS_PHI_DEFAULT",
    "KATEGORI_VALID", "BATAS_IKE",
    # kalkulasi
    "get_tarif", "hitung_kwh_minimum", "hitung_rm_rupiah",
    "hitung_watt", "hitung_watt_nyata", "hitung_kwh_alat",
    "hitung_tagihan", "hitung_biaya_bulanan", "hitung_rentang_biaya",
    "hitung_emisi", "hitung_ike", "hitung_kwh_per_org", "deteksi_anomali",
    "KETERBATASAN_PREDIKSI",
    # token
    "hitung_kwh_dari_token", "hitung_token_dari_kwh", "batas_maks_kwh",
    "validasi_pembelian", "proyeksi_token", "token_untuk_bertahan",
    "bulatkan_nominal", "kombinasi_denominasi",
    "jadwal_isi_ulang",
]