"use client";

import { useState } from "react";
import { Panel, DividerLabel } from "./ui";
import { RekomendasiPerangkat } from "./RekomendasiPerangkat";
import { formatKwh, formatPersen, formatRupiah, formatTanggalIndo } from "@/lib/format";
import type { AnalisisResponse, FokusOptimasi, StatusAnomali } from "@/lib/types";

const STATUS_STYLE: Record<
  StatusAnomali,
  { border: string; bg: string; text: string; label: string }
> = {
  anomali: { border: "border-red/50", bg: "bg-red-dim", text: "text-red", label: "⚠ Anomali Terdeteksi" },
  normal: { border: "border-teal/50", bg: "bg-teal-dim", text: "text-teal", label: "✓ Normal" },
  data_belum_cukup: {
    border: "border-amber-warn/50",
    bg: "bg-amber-warn-dim",
    text: "text-amber-warn",
    label: "ℹ Data Belum Cukup",
  },
  data_tidak_konsisten: {
    border: "border-amber-warn/50",
    bg: "bg-amber-warn-dim",
    text: "text-amber-warn",
    label: "⚠ Data Tidak Konsisten",
  },
  tanggal_tidak_valid: {
    border: "border-amber-warn/50",
    bg: "bg-amber-warn-dim",
    text: "text-amber-warn",
    label: "⚠ Tanggal Tidak Valid",
  },
};

// Status yang memblokir dashboard & rekomendasi sampai user perbaiki
// input dan analisis ulang. Awalnya "data_belum_cukup" dikecualikan
// (dianggap bukan kesalahan input, cuma soal waktu) — tapi setelah
// dicoba langsung, diputuskan tetap diblokir juga supaya konsisten:
// H+1 adalah syarat wajib tanpa kecuali, bukan sekadar rekomendasi.
const STATUS_BLOKIR_DASHBOARD: StatusAnomali[] = [
  "tanggal_tidak_valid",
  "data_tidak_konsisten",
  "data_belum_cukup",
];

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-graphite-700 bg-graphite-800/50 px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-cream-dim">{label}</p>
      <p className="mt-1 font-mono text-lg text-cream">{value}</p>
    </div>
  );
}

function Expandable({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-graphite-700">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium text-cream"
      >
        {title}
        <span className="text-cream-dim">{open ? "−" : "+"}</span>
      </button>
      {open && <div className="border-t border-graphite-700 px-4 py-4">{children}</div>}
    </div>
  );
}

function StrukRow({ label, value, bold = false }: { label: string; value: string; bold?: boolean }) {
  return (
    <div
      className={`flex items-center justify-between font-mono text-sm ${
        bold ? "font-semibold text-amber" : "text-cream-dim"
      }`}
    >
      <span>{label}</span>
      <span className={bold ? "text-amber" : "text-cream"}>{value}</span>
    </div>
  );
}

export function HasilAnalisis({
  hasil,
  intentUser,
}: {
  hasil: AnalisisResponse;
  intentUser: FokusOptimasi[];
}) {
  const s = STATUS_STYLE[hasil.status_anomali];
  const opt = hasil.hasil_optimasi;
  const dashboardDiblokir = STATUS_BLOKIR_DASHBOARD.includes(hasil.status_anomali);

  // Array kosong (user tidak pilih fokus spesifik) dianggap "tampilkan
  // keduanya" — konsisten dengan instruksi_fokus di backend
  // (services/narasi.py) yang juga membolehkan singgung keduanya kalau
  // tidak ada fokus spesifik dipilih.
  const tampilBiaya      = intentUser.length === 0 || intentUser.includes("Biaya");
  const tampilLingkungan = intentUser.length === 0 || intentUser.includes("Lingkungan");

  return (
    <div className="space-y-5">
      {/* Status anomali — selalu tampil, tidak pernah disembunyikan */}
      <div className={`rounded-xl border ${s.border} ${s.bg} px-5 py-4`}>
        <p className={`text-sm font-semibold ${s.text}`}>{s.label}</p>
        <p className="mt-1 text-sm text-cream-dim">{hasil.pesan_anomali}</p>
        {dashboardDiblokir && (
          <p className="mt-3 text-sm font-medium text-cream">
            Perbaiki data di form sebelum ini, lalu klik &quot;Mulai Analisis&quot;
            lagi untuk melihat hasil dan rekomendasi.
          </p>
        )}
      </div>

      {!dashboardDiblokir && (
        <>
      {/* ── KONTEN UTAMA: narasi + rekomendasi aksi ──────────────────────────
          Ini yang dibaca duluan oleh user awam — bahasa strategis dari
          Gemini, langsung disusul daftar aksi konkret (ganti/kurangi)
          yang dihitung deterministik oleh kode, bukan LLM. */}
      <Panel className="border-amber-dim/40">
        <p className="mb-3 text-sm font-semibold text-amber-bright">💡 Rekomendasi EnergiCerdas AI</p>
        <p className="whitespace-pre-line text-sm leading-relaxed text-cream-dim">{hasil.narasi}</p>

        <DividerLabel>
          {hasil.label_ike.includes("Boros")
            ? "Untuk Melakukan Penghematan Bisa Lakukan Hal Ini"
            : "Untuk Lebih Hemat Bisa Lakukan hal ini"}
        </DividerLabel>

        <RekomendasiPerangkat
          hasilDsm={hasil.hasil_dsm}
          hasilOpt={hasil.hasil_optimasi}
          isPrabayar={hasil.is_prabayar}
        />
      </Panel>

      {/* ── DETAIL PENDUKUNG: angka & rincian teknis ────────────────────── */}

      <Panel className="bg-panel-texture text-center">
        <p className="text-xs uppercase tracking-widest text-cream-dim">Total Konsumsi Bulanan</p>
        <p className="digit-glow mt-2 font-mono text-5xl font-semibold text-amber sm:text-6xl">
          {hasil.total_kwh}
          <span className="ml-2 text-2xl text-amber-dim sm:text-3xl">kWh</span>
        </p>
        <p className="mt-2 font-mono text-sm text-cream-dim">
          Tingkat efisiensi listrik: <span className="text-cream">{hasil.label_ike}</span>
        </p>
        <p className="mt-1 text-xs text-cream-dim/60">
          berdasarkan kalibrasi IKE 5-lapis — {hasil.ike.toFixed(2)} kWh/m²/bulan
        </p>
      </Panel>

      {(() => {
        const kartu: React.ReactNode[] = [];
        if (tampilBiaya) {
          kartu.push(
            <MetricCard
              key="biaya"
              label={hasil.is_prabayar ? "Estimasi Nilai Konsumsi" : "Estimasi Tagihan"}
              value={formatRupiah(hasil.estimasi_rp)}
            />
          );
        }
        // "Konsumsi per Penghuni" netral (bukan spesifik biaya/lingkungan) — selalu tampil.
        kartu.push(
          <MetricCard key="konsumsi" label="Konsumsi per Penghuni" value={formatKwh(hasil.kwh_per_org)} />
        );
        if (tampilLingkungan) {
          kartu.push(
            <MetricCard key="emisi" label="Emisi CO₂" value={`${hasil.emisi_sebelum.emisi_kg_bulan} kg/bln`} />
          );
        }
        const gridClass =
          kartu.length === 3 ? "sm:grid-cols-3" : kartu.length === 2 ? "sm:grid-cols-2" : "sm:grid-cols-1";
        return <div className={`grid gap-3 ${gridClass}`}>{kartu}</div>;
      })()}

      {opt.aktif && (() => {
        const kartu: React.ReactNode[] = [];
        if (tampilBiaya) {
          kartu.push(
            hasil.is_prabayar ? (
              <MetricCard key="token-hemat" label="Token Dihemat/Bulan" value={formatKwh(opt.hemat_kwh)} />
            ) : (
              <MetricCard
                key="biaya-hemat"
                label="Hemat Biaya/Bulan"
                value={`${formatRupiah(opt.hemat_rp)} (${formatPersen(opt.persen_hemat_rp)})`}
              />
            )
          );
        }
        if (tampilLingkungan) {
          kartu.push(
            <MetricCard
              key="kurang-emisi"
              label="Kurang Emisi/Bulan"
              value={`${opt.hemat_emisi_kg} kg (-${formatPersen(opt.persen_hemat_emisi)})`}
            />
          );
          kartu.push(
            <MetricCard key="emisi-akhir" label="Emisi Setelah Optimasi" value={`${opt.emisi_akhir} kg/bln`} />
          );
        }
        if (kartu.length === 0) return null;
        const gridClass =
          kartu.length === 3 ? "sm:grid-cols-3" : kartu.length === 2 ? "sm:grid-cols-2" : "sm:grid-cols-1";
        return <div className={`grid gap-3 ${gridClass}`}>{kartu}</div>;
      })()}

      {tampilBiaya && (
      <Expandable title={hasil.is_prabayar ? "📄 Rincian Token (detail teknis)" : "📄 Rincian Tagihan (detail teknis)"}>
        <div className="space-y-2">
          <StrukRow label="Golongan daya" value={hasil.golongan_daya} />
          <StrukRow label="Tarif" value={`${formatRupiah(hasil.tarif_digunakan)}/kWh`} />
          <StrukRow label="Biaya pemakaian" value={formatRupiah(hasil.biaya_pemakaian)} />
          <StrukRow label="PBJT (2,4%)" value={formatRupiah(hasil.biaya_pbjt)} />
          <StrukRow label="Biaya Materai" value={formatRupiah(hasil.biaya_materai)} />
          <div className="my-2 tear-line" aria-hidden />
          <StrukRow label="Estimasi total" value={formatRupiah(hasil.estimasi_rp)} bold />

          {hasil.is_prabayar && hasil.token_context && (
            <>
              <div className="my-3 tear-line" aria-hidden />
              <StrukRow
                label="Tanggal pembelian"
                value={formatTanggalIndo(hasil.token_context.tanggal_pembelian)}
              />
              <StrukRow label="Hari berjalan" value={`${hasil.token_context.hari_berjalan} hari`} />
              <StrukRow label="Saldo awal periode" value={formatKwh(hasil.token_context.saldo_awal)} />
              <StrukRow label="Sisa saat ini" value={formatKwh(hasil.token_context.sisa_saat_ini)} />
              <StrukRow
                label="Token terpakai aktual"
                value={formatKwh(hasil.token_context.token_terpakai_aktual)}
                bold
              />
              <StrukRow
                label="Estimasi dari peralatan"
                value={formatKwh(hasil.token_context.estimasi_terpakai_perangkat)}
              />
            </>
          )}
        </div>
        <p className="mt-4 text-xs leading-relaxed text-cream-dim/70">
          ⚠️{" "}
          {hasil.is_prabayar
            ? "Estimasi nilai konsumsi ini bersifat prediktif — dihitung dari tarif resmi PLN dan spesifikasi/durasi pakai peralatan yang Anda masukkan, bukan dari pembacaan meteran langsung. Nominal aktual bisa berbeda dari saldo token sesungguhnya, dan estimasi ini belum memperhitungkan biaya admin pembelian token (besarannya tergantung channel pembayaran yang Anda gunakan)."
            : "Estimasi tagihan ini bersifat prediktif — dihitung dari tarif resmi PLN dan spesifikasi/durasi pakai peralatan yang Anda masukkan, bukan dari pembacaan meteran langsung. Nominal aktual di rekening/struk bisa berbeda, dan estimasi ini belum memperhitungkan biaya admin (besarannya tergantung channel pembayaran yang Anda gunakan)."}
        </p>
      </Expandable>
      )}
        </>
      )}
    </div>
  );
}

