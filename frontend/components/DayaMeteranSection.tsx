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
}: {
  referensi: ReferensiResponse | null;
  dayaVa: number;
  onDayaVaChange: (v: number) => void;
  isPrabayar: boolean;
  onIsPrabayarChange: (v: boolean) => void;
}) {
  const tarif = referensi?.tarif_per_golongan[String(dayaVa)];
  const pbjt = referensi?.pbjt_rumah_tangga ?? 0.024;

  return (
    <Panel>
      <SectionLabel nomor="01" title="Daya & Meteran" />
      <div className="grid gap-5 sm:grid-cols-2">
        <Field label="Daya Tersambung PLN">
          <Select
            value={dayaVa}
            onChange={(e) => onDayaVaChange(Number(e.target.value))}
          >
            {(referensi?.golongan_daya ?? [900, 1300, 2200, 3500]).map((v) => (
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

      {tarif && (
        <div className="mt-5 flex flex-wrap items-baseline gap-x-6 gap-y-1 rounded-lg border border-graphite-700 bg-graphite-800/50 px-4 py-3 font-mono text-sm">
          <span className="text-cream-dim">
            Tarif <span className="text-amber">{formatRupiah(tarif)}/kWh</span>
          </span>
          <span className="text-cream-dim">
            PBJT <span className="text-amber">{(pbjt * 100).toFixed(1)}%</span>
          </span>
          <span className="text-cream-dim">
            Biaya beban{" "}
            <span className="text-amber">
              {isPrabayar ? "Rp 0 (prabayar)" : "berlaku"}
            </span>
          </span>
        </div>
      )}
    </Panel>
  );
}
