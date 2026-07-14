"use client";

import { useState } from "react";
import { Panel, SectionLabel, Field, Input, Select, Button } from "./ui";
import { formatKwh } from "@/lib/format";
import { KATEGORI_ALAT, type PeralatanInput } from "@/lib/types";

function hitungWatt(tegangan: number, arus: number) {
  return Math.round(tegangan * arus * 100) / 100;
}
function hitungKwhBulan(watt: number, jam: number, jumlah: number) {
  return Math.round(((watt * jam * jumlah * 30) / 1000) * 10000) / 10000;
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

  function tambah() {
    if (!form.nama.trim() || form.tegangan <= 0 || form.arus <= 0 || form.jam <= 0) return;
    onChange([...daftarAlat, form]);
    setForm(KOSONG);
  }

  function hapus(idx: number) {
    onChange(daftarAlat.filter((_, i) => i !== idx));
  }

  return (
    <Panel>
      <SectionLabel
        nomor="04"
        title="Inventarisasi Peralatan Listrik"
        desc="Daya dihitung otomatis dari tegangan × arus."
      />

      {daftarAlat.length === 0 ? (
        <p className="rounded-lg border border-dashed border-graphite-600 px-4 py-6 text-center text-sm text-cream-dim">
          Belum ada peralatan. Tambahkan lewat form di bawah.
        </p>
      ) : (
        <ul className="mb-2 divide-y divide-graphite-700/60 rounded-lg border border-graphite-700 font-mono text-sm">
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

      <div className="my-6 tear-line" aria-hidden />

      <p className="mb-4 text-sm font-medium text-cream-dim">Tambah peralatan baru</p>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Nama Alat">
          <Input
            placeholder="cth: AC Kamar Tidur"
            value={form.nama}
            onChange={(e) => setForm({ ...form, nama: e.target.value })}
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
        <Field label="Durasi Nyala (Jam/Hari)">
          <Input
            type="number"
            step="0.5"
            min={0.1}
            max={24}
            value={form.jam}
            onChange={(e) => setForm({ ...form, jam: Number(e.target.value) })}
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
      <Button type="button" variant="secondary" className="mt-4" onClick={tambah}>
        + Tambahkan ke Daftar
      </Button>
    </Panel>
  );
}
