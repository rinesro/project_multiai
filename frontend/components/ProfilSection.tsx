"use client";

import { Panel, SectionLabel, Field, Input } from "./ui";
import { formatKwh } from "@/lib/format";
import { previewHariBerjalan, previewKwhDariToken } from "@/lib/preview";
import type { TokenContextInput } from "@/lib/types";

interface Props {
  luasRumah: number;
  onLuasRumahChange: (v: number) => void;
  penghuni: number;
  onPenghuniChange: (v: number) => void;
  isPrabayar: boolean;
  tagihanAsli: number;
  onTagihanAsliChange: (v: number) => void;
  tokenContext: TokenContextInput;
  onTokenContextChange: (v: TokenContextInput) => void;
  tarifKwh: number;
  pbjt: number;
}

export function ProfilSection({
  luasRumah,
  onLuasRumahChange,
  penghuni,
  onPenghuniChange,
  isPrabayar,
  tagihanAsli,
  onTagihanAsliChange,
  tokenContext,
  onTokenContextChange,
  tarifKwh,
  pbjt,
}: Props) {
  const kwhPreview = previewKwhDariToken(tokenContext.nominal_dibeli, tarifKwh, pbjt);
  const hariPreview = previewHariBerjalan(tokenContext.tanggal_pembelian);

  return (
    <Panel>
      <SectionLabel nomor="02" title="Profil Rumah Tangga & Riwayat Pemakaian" />
      <div className="grid gap-5 sm:grid-cols-2">
        <Field label="Luas Bangunan (m²)">
          <Input
            type="number"
            min={10}
            max={500}
            value={luasRumah}
            onChange={(e) => onLuasRumahChange(Number(e.target.value))}
          />
        </Field>
        <Field label="Jumlah Penghuni">
          <Input
            type="number"
            min={1}
            max={20}
            value={penghuni}
            onChange={(e) => onPenghuniChange(Number(e.target.value))}
          />
        </Field>
      </div>

      <div className="my-6 tear-line" aria-hidden />

      {isPrabayar ? (
        <div>
          <p className="mb-4 text-sm text-cream-dim">
            Data token dipakai untuk deteksi anomali — membandingkan konsumsi
            aktual (selisih saldo token) dengan estimasi dari daftar peralatan.
          </p>
          <div className="grid gap-5 sm:grid-cols-2">
            <Field label="Tanggal Pembelian Token Terakhir">
              <Input
                type="date"
                max={new Date().toISOString().slice(0, 10)}
                value={tokenContext.tanggal_pembelian}
                onChange={(e) =>
                  onTokenContextChange({ ...tokenContext, tanggal_pembelian: e.target.value })
                }
              />
            </Field>
            <Field label="Sisa Token Sebelum Beli (kWh)">
              <Input
                type="number"
                step="0.1"
                min={0}
                value={tokenContext.sisa_sebelum_beli}
                onChange={(e) =>
                  onTokenContextChange({
                    ...tokenContext,
                    sisa_sebelum_beli: Number(e.target.value),
                  })
                }
              />
            </Field>
            <Field
              label="Nominal Token Dibeli (Rp)"
              hint="Sesuai struk bukan termasuk biaya admin bank/e-wallet"
            >
              <Input
                type="number"
                step="5000"
                min={0}
                value={tokenContext.nominal_dibeli}
                onChange={(e) =>
                  onTokenContextChange({
                    ...tokenContext,
                    nominal_dibeli: Number(e.target.value),
                  })
                }
              />
            </Field>
            <Field label="Sisa Token Saat Ini (kWh)">
              <Input
                type="number"
                step="0.1"
                min={0}
                value={tokenContext.sisa_saat_ini}
                onChange={(e) =>
                  onTokenContextChange({ ...tokenContext, sisa_saat_ini: Number(e.target.value) })
                }
              />
            </Field>
          </div>

          <div className="mt-5 grid gap-3 rounded-lg border border-graphite-700 bg-graphite-800/50 px-4 py-3 font-mono text-sm sm:grid-cols-2">
            <span className="text-cream-dim">
              Estimasi kWh dari token{" "}
              <span className="text-amber">{formatKwh(kwhPreview)}</span>
            </span>
            <span className="text-cream-dim">
              Hari sejak pembelian{" "}
              <span className={hariPreview < 0 ? "text-red" : "text-amber"}>
                {hariPreview < 0 ? "tanggal tidak valid" : `${hariPreview} hari`}
              </span>
            </span>
          </div>
        </div>
      ) : (
        <Field label="Tagihan Bulan Lalu (Rp)">
          <Input
            type="number"
            min={50000}
            max={10000000}
            step={10000}
            value={tagihanAsli}
            onChange={(e) => onTagihanAsliChange(Number(e.target.value))}
          />
        </Field>
      )}
    </Panel>
  );
}
