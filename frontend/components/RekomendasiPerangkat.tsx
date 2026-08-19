"use client";

import { formatKwh, formatPersen, formatRupiah } from "@/lib/format";
import type { HasilDsmItem, HasilOptimasi } from "@/lib/types";

interface PerangkatGanti {
  nama: string;
  tegangan: number;
  arus: number;
  watt: number;
}

interface PerangkatKurangi {
  nama: string;
  jamAwal: number;
  jamRekomendasi: number;
  hematKwh: number;
  hematRp: number;
}

function bucketRekomendasi(hasilDsm: HasilDsmItem[], hasilOpt: HasilOptimasi) {
  const ganti: PerangkatGanti[] = hasilDsm
    .filter((a) => a.label_dsm === "Tidak Fleksibel")
    .map((a) => ({ nama: a.nama, tegangan: a.tegangan, arus: a.arus, watt: a.watt }));

  const kurangi: PerangkatKurangi[] = hasilOpt.aktif
    ? hasilOpt.langkah.map((l) => ({
        nama: l.nama,
        jamAwal: l.jam_awal,
        jamRekomendasi: l.jam_rekomendasi,
        hematKwh: l.hemat_kwh,
        hematRp: l.hemat_rp,
      }))
    : [];

  return { ganti, kurangi };
}

export function RekomendasiPerangkat({
  hasilDsm,
  hasilOpt,
  isPrabayar,
  tampilBiaya,
  tampilLingkungan,
}: {
  hasilDsm: HasilDsmItem[];
  hasilOpt: HasilOptimasi;
  isPrabayar: boolean;
  tampilBiaya: boolean;
  tampilLingkungan: boolean;
}) {
  const { ganti, kurangi } = bucketRekomendasi(hasilDsm, hasilOpt);

  if (ganti.length === 0 && kurangi.length === 0) return null;

  return (
    <div className="space-y-4">
      {kurangi.length > 0 && (
        <div>
          <p className="mb-2 text-sm font-semibold text-teal">
            ⏱️ Bisa kurangi penggunaan perangkat berikut:
          </p>
          <ul className="space-y-1.5">
            {kurangi.map((k, i) => (
              <li
                key={i}
                className="rounded-lg border border-graphite-700 bg-graphite-800/50 px-4 py-2.5 text-sm"
              >
                <span className="font-medium text-cream">{k.nama}</span>{" "}
                <span className="text-cream-dim">
                  — dari {k.jamAwal} jam menjadi {k.jamRekomendasi} jam/hari (hemat{" "}
                  {isPrabayar ? `${formatKwh(k.hematKwh)} token` : `${formatRupiah(k.hematRp)}/bulan`})
                </span>
              </li>
            ))}
          </ul>

          {(tampilBiaya || tampilLingkungan) && (
            <div className="mt-3 rounded-lg border border-teal/30 bg-teal-dim px-4 py-3">
              <p className="text-xs uppercase tracking-wide text-cream-dim">
                Estimasi Total Penghematan
              </p>
              <div className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 font-mono text-sm text-teal">
                {tampilBiaya && (
                  <span>
                    {isPrabayar
                      ? `${formatKwh(hasilOpt.hemat_kwh)} token/bulan`
                      : `${formatRupiah(hasilOpt.hemat_rp)} (-${formatPersen(hasilOpt.persen_hemat_rp)})/bulan`}
                  </span>
                )}
                {tampilLingkungan && (
                  <span>
                    -{hasilOpt.hemat_emisi_kg} kgCO₂ (-{formatPersen(hasilOpt.persen_hemat_emisi)})/bulan
                  </span>
                )}
              </div>
              {tampilLingkungan && (
                <p className="mt-1.5 font-mono text-xs text-cream-dim">
                  Emisi setelah optimasi: {hasilOpt.emisi_akhir} kg/bln
                </p>
              )}
              {tampilBiaya && !isPrabayar && (
                <p className="mt-2 text-[11px] leading-snug text-cream-dim/60">
                  ⚠️ Estimasi — cara PLN menghitung tagihan riil bisa sedikit berbeda dari
                  perhitungan ini.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {ganti.length > 0 && (
        <div>
          <p className="mb-2 text-sm font-semibold text-amber-bright">
            🔄 Ganti penggunaan perangkat berikut dengan yang lebih hemat:
          </p>
          <ul className="space-y-1.5">
            {ganti.map((g, i) => (
              <li
                key={i}
                className="rounded-lg border border-graphite-700 bg-graphite-800/50 px-4 py-2.5 text-sm"
              >
                <span className="font-medium text-cream">{g.nama}</span>{" "}
                <span className="font-mono text-xs text-cream-dim">
                  (tegangan = {g.tegangan}V, arus listrik = {g.arus}A, daya = {g.watt}W)
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
