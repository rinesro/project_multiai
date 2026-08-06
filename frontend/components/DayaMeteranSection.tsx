"use client";

import { Panel, SectionLabel, Field, Select, TogglePill } from "./ui";
import { formatRupiah } from "@/lib/format";
import type { ReferensiResponse } from "@/lib/types";

export function DayaMeteranSection({
  referensi,
  dayaVa,
  onDayaVaChange,
  isPrabayar,
  onIsPrabayarChange,
  isSubsidi,
  onIsSubsidiChange,
}: {
  referensi: ReferensiResponse | null;
  dayaVa: number;
  onDayaVaChange: (v: number) => void;
  isPrabayar: boolean;
  onIsPrabayarChange: (v: boolean) => void;
  isSubsidi: boolean;
  onIsSubsidiChange: (v: boolean) => void;
}) {
  // 450/900 VA: tarif tergantung status subsidi, sumbernya beda dari
  // golongan >=1.300VA yang cuma satu tarif per golongan.
  const tarifRendah = referensi?.tarif_daya_rendah[String(dayaVa)];
  const tarifBiasa = referensi?.tarif_per_golongan[String(dayaVa)];
  const tarif = tarifRendah
    ? isSubsidi
      ? tarifRendah.subsidi
      : tarifRendah.non_subsidi
    : tarifBiasa;
  const pbjt = referensi?.pbjt_rumah_tangga ?? 0.024;
  const tampilkanToggleSubsidi = dayaVa === 450 || dayaVa === 900;

  return (
    <Panel>
      <SectionLabel nomor="01" title="Daya & Meteran" />
      <div className="grid gap-5 sm:grid-cols-2">
        <Field label="Daya Tersambung PLN">
          <Select
            value={dayaVa}
            onChange={(e) => onDayaVaChange(Number(e.target.value))}
          >
            {(referensi?.golongan_daya ?? [450, 900, 1300, 2200, 3500]).map((v) => (
              <option key={v} value={v}>
                {v.toLocaleString("id-ID")} VA
              </option>
            ))}
          </Select>
        </Field>

        <Field label="Jenis Meteran">
          <TogglePill
            value={isPrabayar ? "prabayar" : "pascabayar"}
            onChange={(v) => onIsPrabayarChange(v === "prabayar")}
            options={[
              { value: "prabayar", label: "Prabayar (Token)" },
              { value: "pascabayar", label: "Pascabayar (Tagihan)" },
            ]}
          />
        </Field>
      </div>

      {tampilkanToggleSubsidi && (
        <label className="mt-4 flex items-center gap-2.5 text-sm text-cream-dim">
          <input
            type="checkbox"
            checked={isSubsidi}
            onChange={(e) => onIsSubsidiChange(e.target.checked)}
            className="h-4 w-4 rounded border-graphite-600 bg-graphite-800 accent-amber"
          />
          Pelanggan bersubsidi
          {dayaVa === 900 && (
            <span className="text-xs text-cream-dim/60">
              (Rp605/kWh subsidi vs Rp1.352/kWh RTM/non-subsidi)
            </span>
          )}
        </label>
      )}

      {tarif != null && (
        <div className="mt-5 flex flex-wrap items-baseline gap-x-6 gap-y-1 rounded-lg border border-graphite-700 bg-graphite-800/50 px-4 py-3 font-mono text-sm">
          <span className="text-cream-dim">
            Tarif <span className="text-amber">{formatRupiah(tarif)}/kWh</span>
          </span>
          <span className="text-cream-dim">
            PBJT <span className="text-amber">{(pbjt * 100).toFixed(1)}%</span>
          </span>
          <span className="text-cream-dim">
            Bea Materai{" "}
            <span className="text-amber">
              {isPrabayar ? "tidak berlaku (prabayar)" : ">Rp5 juta"}
            </span>
          </span>
        </div>
      )}
    </Panel>
  );
}
