"use client";

import { formatKwh, formatRupiah } from "@/lib/format";
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

/**
 * Mengelompokkan hasil_dsm + hasil_optimasi jadi dua daftar aksi yang
 * mudah dipahami awam, TANPA menyebut istilah "label DSM" atau
 * "Fleksibel/Tidak Fleksibel" ke user:
 *
 *   - "Tidak Fleksibel" -> tidak bisa dikurangi jam pakainya (alat
 *     yang harus menyala saat dibutuhkan) -> satu-satunya cara hemat
 *     adalah ganti dengan model yang lebih efisien.
 *   - Muncul di hasil_optimasi.langkah -> BENAR-BENAR direkomendasikan
 *     optimizer untuk dikurangi jam pakainya (bukan sekadar berlabel
 *     "Fleksibel" tapi tidak disentuh optimizer).
 *
 * Sengaja dihitung di sini (bukan diminta Gemini menyusunnya) supaya
 * nama alat & angka (volt/ampere/watt) selalu presisi — LLM tidak
 * dilibatkan untuk data teknis, hanya untuk narasi strategi.
 */
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
}: {
  hasilDsm: HasilDsmItem[];
  hasilOpt: HasilOptimasi;
  isPrabayar: boolean;
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
