"use client";

import { useState } from "react";
import { Panel, DividerLabel } from "./ui";
import { RekomendasiPerangkat } from "./RekomendasiPerangkat";
import { formatAngka, formatKg, formatKwh, formatRupiah, formatTanggalIndo, formatWatt } from "@/lib/format";
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

// Zona IKE di mana saran hemat (ganti/kurangi) relevan ditampilkan —
// Cukup Efisien atau lebih boros. Sangat Efisien/Efisien dianggap
// sudah cukup baik, tidak perlu dipaksa saran tambahan.
const ZONA_PERLU_SARAN = ["Cukup Efisien", "Boros", "Sangat Boros"];

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

  // Saran (ganti/kurangi) cuma tampil kalau (a) zona IKE Cukup Efisien
  // atau lebih boros, DAN (b) memang ada sesuatu untuk disarankan —
  // supaya tidak muncul section kosong cuma karena zonanya cocok.
  const gantiList      = hasil.hasil_dsm.filter((a) => a.label_dsm === "Tidak Fleksibel");
  const kurangiList    = opt.aktif ? opt.langkah : [];
  const kelasPerluSaran = ZONA_PERLU_SARAN.includes(hasil.label_ike);
  const adaSaran        = gantiList.length > 0 || kurangiList.length > 0;
  const tampilkanSaran  = kelasPerluSaran && adaSaran;

  const labelEstimasi = hasil.is_prabayar ? "Estimasi Token" : "Estimasi Tagihan";

  return (
    <div className="space-y-5">
      {/* 1. Status anomali — selalu tampil, tidak pernah disembunyikan */}
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
          {/* 2. Estimasi Tagihan/Token — header expander LANGSUNG menunjukkan
              nominalnya (bukan judul generik "detail teknis" yang bikin
              awam bingung). Terbuka -> rincian breakdown, baris terakhir
              cukup "Total" (kata "Estimasi" sudah ada di header). */}
          <Expandable title={`${labelEstimasi}: ${formatRupiah(hasil.estimasi_rp)}`}>
            <div className="space-y-2">
              <StrukRow label="Golongan daya" value={hasil.golongan_daya} />
              <StrukRow label="Tarif" value={`${formatRupiah(hasil.tarif_digunakan)}/kWh`} />
              <StrukRow label="Biaya pemakaian" value={formatRupiah(hasil.biaya_pemakaian)} />
              <StrukRow label="PBJT (2,4%)" value={formatRupiah(hasil.biaya_pbjt)} />
              <StrukRow label="Biaya Materai" value={formatRupiah(hasil.biaya_materai)} />
              <div className="my-2 tear-line" aria-hidden />
              <StrukRow label="Total" value={formatRupiah(hasil.estimasi_rp)} bold />

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
                ? "Estimasi nilai konsumsi ini bersifat prediktif dihitung dari tarif resmi PLN dan spesifikasi/durasi pakai peralatan yang Anda masukkan, bukan dari pembacaan meteran langsung. Nominal aktual bisa berbeda dari saldo token sesungguhnya, dan estimasi ini belum memperhitungkan biaya admin pembelian token (besarannya tergantung channel pembayaran yang Anda gunakan)."
                : "Estimasi tagihan ini bersifat prediktif dihitung dari tarif resmi PLN dan spesifikasi/durasi pakai peralatan yang Anda masukkan, bukan dari pembacaan meteran langsung. Nominal aktual di rekening/struk bisa berbeda, dan estimasi ini belum memperhitungkan biaya admin (besarannya tergantung channel pembayaran yang Anda gunakan)."}
            </p>
          </Expandable>

          {/* 3. Total Konsumsi Bulanan + Tingkat Efisiensi — sama seperti
              sebelumnya, cuma posisinya naik ke urutan ke-3. */}
          <Panel className="bg-panel-texture text-center">
            <p className="text-xs uppercase tracking-widest text-cream-dim">Total Konsumsi Bulanan</p>
            <p className="digit-glow mt-2 font-mono text-5xl font-semibold text-amber sm:text-6xl">
              {formatAngka(hasil.total_kwh, 2)}
              <span className="ml-2 text-2xl text-amber-dim sm:text-3xl">kWh</span>
            </p>
            <p className="mt-2 font-mono text-sm text-cream-dim">
              Tingkat efisiensi listrik: <span className="text-cream">{hasil.label_ike}</span>
            </p>
            <p className="mt-1 text-xs text-cream-dim/60">
              berdasarkan kalibrasi IKE 5 lapis ({hasil.ike.toFixed(2)}) kWh/m²/bulan
            </p>
          </Panel>

          {/* 4. Panel Narasi + saran — narasi menjelaskan angka di atas &
              bawahnya. Saran (ganti/kurangi) cuma muncul kalau zona
              Cukup Efisien+ DAN memang ada sesuatu untuk disarankan. */}
          <Panel className="border-amber-dim/40">
            <p className="mb-3 text-sm font-semibold text-amber-bright">💡 Rekomendasi EnergiCerdas AI</p>
            <p className="whitespace-pre-line text-sm leading-relaxed text-cream-dim">{hasil.narasi}</p>

            {tampilkanSaran && (
              <>
                <DividerLabel>
                  {hasil.label_ike.includes("Boros")
                    ? "Untuk Melakukan Penghematan Bisa Lakukan Hal Ini"
                    : "Untuk Lebih Hemat Bisa Lakukan hal ini"}
                </DividerLabel>
                <RekomendasiPerangkat
                  hasilDsm={hasil.hasil_dsm}
                  hasilOpt={hasil.hasil_optimasi}
                  isPrabayar={hasil.is_prabayar}
                  tampilBiaya={tampilBiaya}
                  tampilLingkungan={tampilLingkungan}
                />
              </>
            )}
          </Panel>

          {/* 5. Kartu metrik utama — sama seperti sebelumnya. Kartu hasil
              greedy_optimizer (Hemat Biaya, Kurang Emisi, Emisi Setelah
              Optimasi) SUDAH DIHAPUS dari sini -- itu duplikasi persis
              dengan kotak "Estimasi Total Penghematan" yang sudah ada
              di dalam RekomendasiPerangkat (poin 4), tepat di bawah
              daftar "Bisa kurangi..." -- sekarang cuma ada SATU tempat
              untuk hasil optimasi, bukan dua. */}
          {(() => {
            const kartu: React.ReactNode[] = [];
            if (tampilBiaya) {
              kartu.push(
                <MetricCard key="biaya" label={labelEstimasi} value={formatRupiah(hasil.estimasi_rp)} />
              );
            }
            kartu.push(
              <MetricCard key="konsumsi" label="Konsumsi per Penghuni" value={formatKwh(hasil.kwh_per_org)} />
            );
            if (tampilLingkungan) {
              kartu.push(
                <MetricCard key="emisi" label="Emisi CO₂" value={`${formatKg(hasil.emisi_sebelum.emisi_kg_bulan)}/bln`} />
              );
            }
            const gridClass =
              kartu.length === 3 ? "sm:grid-cols-3" : kartu.length === 2 ? "sm:grid-cols-2" : "sm:grid-cols-1";
            return <div className={`grid gap-3 ${gridClass}`}>{kartu}</div>;
          })()}

          {/* 6. Klasifikasi Peralatan — judul diganti jadi nama standar
              metodenya (DSM), struktur/isi sama, ditambah penjelasan
              awam di paling bawah (dibatasi tear-line). */}
          <Expandable title="Klasifikasi Perangkat Berdasarkan DSM (Demand Side Management)">
            <ul className="space-y-1.5 text-sm">
              {hasil.hasil_dsm.map((a, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className={a.label_dsm === "Fleksibel" ? "text-teal" : "text-red"}>●</span>
                  <span className="text-cream">{a.nama}</span>
                  <span className="text-cream-dim">— {a.label_dsm}</span>
                </li>
              ))}
            </ul>
            <div className="my-4 tear-line" aria-hidden />
            <div className="space-y-1.5 text-xs leading-relaxed text-cream-dim/70">
              <p>
                <span className="text-teal">● Fleksibel</span>: alat yang durasi/jam pemakaiannya bisa
                dikurangi tanpa mengganggu kebutuhan pokok (mis. AC, mesin cuci), cara hemat: kurangi
                jam pakainya.
              </p>
              <p>
                <span className="text-red">● Tidak Fleksibel</span>: alat yang harus menyala sesuai
                kebutuhan dan sulit dikurangi durasinya (mis. kulkas, router), cara hemat: ganti
                dengan model yang lebih hemat energi, bukan mengurangi jam pakainya.
              </p>
            </div>
          </Expandable>

          {/* Pengingat kapasitas watt vs VA — BUKAN anomali (lihat
              core/kalkulasi.py::cek_kapasitas_watt), jadi SENGAJA tidak
              pakai warna merah/kuning peringatan seperti banner anomali
              di atas. Skenario ekstrem "kalau semua alat nyala bareng",
              yang pada praktiknya jarang benar-benar terjadi — makanya
              nadanya santai, bukan menuduh. Cuma tampil kalau memang
              melebihi, tidak selalu tampil. */}
          {hasil.info_kapasitas_watt.melebihi && (
            <div className="rounded-xl border border-teal/30 bg-teal-dim px-5 py-4">
              <p className="text-sm text-cream">
                <span className="mr-1.5">⚡😅</span>
                Kalau semua alat di atas nyala bareng, totalnya sekitar{" "}
                <span className="font-mono font-semibold text-teal">
                  {formatWatt(hasil.info_kapasitas_watt.total_watt)}
                </span>{" "}
                — di atas kapasitas aman rumah Anda (
                <span className="font-mono">{formatWatt(hasil.info_kapasitas_watt.batas_watt_aman)}</span>
                ). Rumah Anda sering &quot;jepret&quot; nggak?
              </p>
              <p className="mt-1.5 text-xs text-cream-dim/70">
                Ini cuma info santai, bukan alarm — kemungkinan besar Anda memang tidak menyalakan
                semuanya sekaligus.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
