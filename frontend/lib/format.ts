/**
 * lib/format.ts
 * ==============
 * Utilitas format angka — dipusatkan supaya format Rupiah/kWh
 * konsisten di semua komponen, tidak ditulis ulang di tiap tempat.
 */

export function formatRupiah(nilai: number): string {
  return `Rp ${Math.round(nilai).toLocaleString("id-ID")}`;
}

export function formatKwh(nilai: number, desimal = 2): string {
  return `${nilai.toLocaleString("id-ID", {
    minimumFractionDigits: desimal,
    maximumFractionDigits: desimal,
  })} kWh`;
}

export function formatPersen(nilai: number, desimal = 1): string {
  return `${nilai.toLocaleString("id-ID", {
    minimumFractionDigits: desimal,
    maximumFractionDigits: desimal,
  })}%`;
}

export function formatTanggalIndo(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString("id-ID", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
}
