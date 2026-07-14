/**
 * lib/preview.ts
 * ===============
 * PENGECUALIAN SENGAJA terhadap prinsip single-source-of-truth proyek
 * ini: dua fungsi di bawah mereplikasi rumus dari core/kalkulasi.py
 * (Python) di TypeScript, KHUSUS untuk preview instan di form —
 * supaya user tidak perlu round-trip ke backend tiap kali mengetik
 * angka. Kalkulasi FINAL yang otoritatif tetap selalu dari backend
 * saat submit form (POST /api/analisis).
 *
 * PENTING: kalau rumus di core/kalkulasi.py berubah, dua fungsi ini
 * HARUS diupdate manual mengikuti. Ini satu-satunya duplikasi rumus
 * yang disengaja di seluruh proyek.
 */

/**
 * Replika core/kalkulasi.py::hitung_kwh_dari_token
 * Dasar pengenaan PBJT prabayar = nominal token itu sendiri.
 * Sumber: Bapenda Provinsi DKI Jakarta (2025); Perda DKI No.1/2024.
 */
export function previewKwhDariToken(
  nominalToken: number,
  tarifKwh: number,
  pbjt: number
): number {
  if (tarifKwh <= 0) return 0;
  const energiRp = nominalToken * (1 - pbjt);
  return Math.round((energiRp / tarifKwh) * 100) / 100;
}

/**
 * Replika core/kalkulasi.py::hitung_hari_berjalan
 * Sengaja TIDAK di-clamp (bisa negatif kalau tanggal di masa depan) —
 * validasi & pesan error tetap tanggung jawab backend saat submit.
 */
export function previewHariBerjalan(tanggalPembelian: string): number {
  const beli = new Date(tanggalPembelian + "T00:00:00");
  const hariIni = new Date();
  hariIni.setHours(0, 0, 0, 0);
  const selisihMs = hariIni.getTime() - beli.getTime();
  return Math.round(selisihMs / (1000 * 60 * 60 * 24));
}
