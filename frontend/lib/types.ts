/**
 * lib/types.ts
 * =============
 * Tipe data ini HARUS tetap sinkron dengan backend/schemas.py (Pydantic).
 * Ini satu-satunya tempat di seluruh frontend yang mendefinisikan bentuk
 * request/response API — komponen lain mengimpor dari sini, tidak
 * mendefinisikan ulang.
 */

export const KATEGORI_ALAT = [
  "Pendingin",
  "Pemanas",
  "Laundry",
  "Memasak",
  "Hiburan/Elektronik",
  "Pencahayaan",
  "Pompa/Motor",
  "Lainnya",
] as const;
export type KategoriAlat = (typeof KATEGORI_ALAT)[number];

export const FOKUS_OPTIMASI = ["Biaya", "Lingkungan"] as const;
export type FokusOptimasi = (typeof FOKUS_OPTIMASI)[number];

export interface PeralatanInput {
  nama: string;
  kategori: KategoriAlat;
  tegangan: number;
  arus: number;
  jam: number;
  jumlah: number;
}

export interface TokenContextInput {
  tanggal_pembelian: string; // ISO date "YYYY-MM-DD"
  sisa_sebelum_beli: number;
  nominal_dibeli: number;
  sisa_saat_ini: number;
}

export interface AnalisisRequest {
  daya_va: number;
  is_prabayar: boolean;
  is_subsidi: boolean;
  luas_rumah: number;
  penghuni: number;
  tagihan_asli?: number | null;
  token_context?: TokenContextInput | null;
  daftar_alat: PeralatanInput[];
  intent_user: FokusOptimasi[];
}

export interface AlatValid {
  nama: string;
  kategori: string;
  tegangan: number;
  arus: number;
  watt: number;
  jam: number;
  jumlah: number;
  kwh_bulan: number;
}

export interface EmisiInfo {
  emisi_kg_bulan: number;
  emisi_kg_tahun: number;
  faktor_emisi: number;
  referensi: string;
}

export interface TokenContextResult {
  tanggal_pembelian: string;
  sisa_sebelum_beli: number;
  nominal_dibeli: number;
  sisa_saat_ini: number;
  kwh_dari_pembelian: number;
  saldo_awal: number;
  hari_berjalan: number;
  token_terpakai_aktual: number;
  estimasi_terpakai_perangkat: number;
}

export type StatusAnomali =
  | "anomali"
  | "normal"
  | "data_belum_cukup"
  | "data_tidak_konsisten"
  | "tanggal_tidak_valid";

export interface HasilDsmItem {
  nama: string;
  kategori: string;
  tegangan: number;
  arus: number;
  watt: number;
  jam: number;
  jumlah: number;
  kwh_bulan: number;
  label_dsm: "Fleksibel" | "Tidak Fleksibel";
  metode: "model" | "fallback";
  valid: boolean;
  pesan: string;
}

export interface LangkahOptimasi {
  nama: string;
  kategori: string;
  jam_awal: number;
  jam_rekomendasi: number;
  kurang_jam: number;
  hemat_kwh: number;
  hemat_rp: number;
  hemat_emisi_kg: number;
}

export interface HasilOptimasi {
  aktif: boolean;
  status: "sudah_efisien" | "efisien" | "cukup_efisien" | "tidak_tercapai";
  zona_awal: string;
  zona_akhir: string;
  ike_awal: number;
  ike_akhir: number;
  target_ike: number;
  total_kwh_akhir: number;
  tagihan_akhir: number;
  emisi_akhir: number;
  hemat_kwh: number;
  hemat_rp: number;
  hemat_emisi_kg: number;
  persen_hemat_rp: number;
  persen_hemat_emisi: number;
  langkah: LangkahOptimasi[];
  pesan: string;
}

export interface AnalisisResponse {
  total_kwh: number;
  biaya_pemakaian: number;
  biaya_pbjt: number;
  biaya_materai: number;
  estimasi_rp: number;
  ike: number;
  label_ike: string;
  kwh_per_org: number;
  emisi_sebelum: EmisiInfo;
  tarif_digunakan: number;
  golongan_daya: string;
  is_prabayar: boolean;
  alat_valid: AlatValid[];
  tagihan_asli: number | null;
  token_context: TokenContextResult | null;
  status_anomali: StatusAnomali;
  selisih_pct: number | null;
  pesan_anomali: string;
  is_anomali: boolean;
  hasil_dsm: HasilDsmItem[];
  hasil_optimasi: HasilOptimasi;
  narasi: string;
}

export interface ReferensiResponse {
  golongan_daya: number[];
  tarif_per_golongan: Record<string, number>;
  tarif_daya_rendah: Record<string, { subsidi: number; non_subsidi: number }>;
  kategori_alat: string[];
  fokus_optimasi: string[];
  pbjt_rumah_tangga: number;
}

export interface ApiErrorBody {
  detail: string | { msg: string; loc: (string | number)[] }[];
}