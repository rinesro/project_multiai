"use client";

import { useState } from "react";
import { Panel, SectionLabel, Field, Input, Select, Button, Modal } from "./ui";
import { formatKwh } from "@/lib/format";
import { KATEGORI_ALAT, type PeralatanInput } from "@/lib/types";

function hitungWatt(tegangan: number, arus: number) {
  return Math.round(tegangan * arus * 100) / 100;
}
function hitungKwhBulan(watt: number, jam: number, jumlah: number) {
  return Math.round(((watt * jam * jumlah * 30) / 1000) * 10000) / 10000;
}

/** Batasi durasi nyala ke rentang wajar [0, 24] — sehari cuma 24 jam.
 * Angka di bawah 0 dibulatkan ke 0, di atas 24 dibulatkan ke 24, supaya
 * user tidak perlu menebak kenapa input "ditolak" nilainya cuma
 * disesuaikan ke batas terdekat secara langsung. */
function clampJam(v: number): number {
  if (Number.isNaN(v)) return 0;
  if (v < 0) return 0;
  if (v > 24) return 24;
  return v;
}

const KOSONG: PeralatanInput = {
  nama: "",
  kategori: "Lainnya",
  tegangan: 220,
  arus: 1.5,
  jam: 4,
  jumlah: 1,
};

export function PeralatanSection({
  daftarAlat,
  onChange,
}: {
  daftarAlat: PeralatanInput[];
  onChange: (v: PeralatanInput[]) => void;
}) {
  const [form, setForm] = useState<PeralatanInput>(KOSONG);
  const [modalOpen, setModalOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function tambah() {
    if (!form.nama.trim()) {
      setError("Nama alat wajib diisi.");
      return;
    }
    if (form.tegangan <= 0 || form.arus <= 0) {
      setError("Tegangan dan arus harus lebih dari 0.");
      return;
    }
     
     
     
     
    if (form.jam <= 0) {
      setError("harap cantumkan perangkat yang digunakan (daya nyala>0)");
      return;
    }
    setError(null);
    onChange([...daftarAlat, form]);
    setForm(KOSONG);
     
     
     
    setModalOpen(false);
  }

  function hapus(idx: number) {
    onChange(daftarAlat.filter((_, i) => i !== idx));
  }

  function tutupModal() {
    setModalOpen(false);
    setError(null);
  }

  return (
    <Panel>
      <div className="mb-5 flex items-start justify-between gap-3">
        <SectionLabel
          nomor="04"
          title="Inventarisasi Peralatan Listrik"
          desc="Daya dihitung otomatis."
        />
        <Button type="button" onClick={() => setModalOpen(true)} className="shrink-0">
          + Tambah Alat
        </Button>
      </div>

      {daftarAlat.length === 0 ? (
        <p className="rounded-lg border border-dashed border-graphite-600 px-4 py-6 text-center text-sm text-cream-dim">
          Belum ada peralatan. Klik &quot;+ Tambah Alat&quot; di atas untuk mulai.
        </p>
      ) : (
        <ul className="divide-y divide-graphite-700/60 rounded-lg border border-graphite-700 font-mono text-sm">
          {daftarAlat.map((a, i) => {
            const watt = hitungWatt(a.tegangan, a.arus);
            const kwh = hitungKwhBulan(watt, a.jam, a.jumlah);
            return (
              <li
                key={i}
                className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-3"
              >
                <div className="min-w-0">
                  <p className="truncate font-sans font-medium text-cream">
                    {a.nama}{" "}
                    <span className="font-sans font-normal text-cream-dim">({a.kategori})</span>
                  </p>
                  <p className="text-xs text-cream-dim">
                    {a.tegangan}V × {a.arus}A = {watt.toLocaleString("id-ID")} W/unit
                    {a.jumlah > 1 ? ` × ${a.jumlah}` : ""} · {a.jam} jam/hari
                  </p>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-amber">{formatKwh(kwh)}</span>
                  <button
                    type="button"
                    onClick={() => hapus(i)}
                    aria-label={`Hapus ${a.nama}`}
                    className="text-cream-dim hover:text-red"
                  >
                    ✕
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}

      <Modal open={modalOpen} onClose={tutupModal} title="Tambah Peralatan Baru">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Nama Alat">
            <Input
              placeholder="cth: AC Kamar Tidur"
              value={form.nama}
              onChange={(e) => setForm({ ...form, nama: e.target.value })}
              autoFocus
            />
          </Field>
          <Field label="Kategori">
            <Select
              value={form.kategori}
              onChange={(e) => setForm({ ...form, kategori: e.target.value as PeralatanInput["kategori"] })}
            >
              {KATEGORI_ALAT.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Durasi Nyala (Jam/Hari)" hint="Otomatis dibatasi 0–24 jam — sehari cuma ada 24 jam.">
            <Input
              type="number"
              step="0.5"
              min={0}
              max={24}
              value={form.jam}
              onChange={(e) => setForm({ ...form, jam: clampJam(Number(e.target.value)) })}
            />
          </Field>
          <Field label="Tegangan (V)">
            <Input
              type="number"
              value={form.tegangan}
              onChange={(e) => setForm({ ...form, tegangan: Number(e.target.value) })}
            />
          </Field>
          <Field label="Arus per Unit (A)">
            <Input
              type="number"
              step="0.1"
              min={0.01}
              value={form.arus}
              onChange={(e) => setForm({ ...form, arus: Number(e.target.value) })}
            />
          </Field>
          <Field label="Jumlah Unit">
            <Input
              type="number"
              min={1}
              value={form.jumlah}
              onChange={(e) => setForm({ ...form, jumlah: Number(e.target.value) })}
            />
          </Field>
        </div>

        {error && (
          <p className="mt-4 rounded-lg border border-red/50 bg-red-dim px-3 py-2 text-sm text-red">
            {error}
          </p>
        )}

        <div className="mt-5 flex justify-end gap-2.5">
          <Button type="button" variant="ghost" onClick={tutupModal}>
            Selesai
          </Button>
          <Button type="button" onClick={tambah}>
            + Tambahkan ke Daftar
          </Button>
        </div>
      </Modal>
    </Panel>
  );
}
