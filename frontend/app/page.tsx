"use client";

import { useEffect, useState } from "react";
import { DayaMeteranSection } from "@/components/DayaMeteranSection";
import { ProfilSection } from "@/components/ProfilSection";
import { OptimasiSection } from "@/components/OptimasiSection";
import { PeralatanSection } from "@/components/PeralatanSection";
import { HasilAnalisis } from "@/components/HasilAnalisis";
import { Button } from "@/components/ui";
import { ApiError, getReferensi, postAnalisis } from "@/lib/api";
import type {
  AnalisisResponse,
  FokusOptimasi,
  PeralatanInput,
  ReferensiResponse,
  TokenContextInput,
} from "@/lib/types";

const TOKEN_KOSONG: TokenContextInput = {
  tanggal_pembelian: new Date().toISOString().slice(0, 10),
  sisa_sebelum_beli: 5,
  nominal_dibeli: 100000,
  sisa_saat_ini: 20,
};

export default function Home() {
  const [referensi, setReferensi] = useState<ReferensiResponse | null>(null);
  const [referensiError, setReferensiError] = useState<string | null>(null);

  const [dayaVa, setDayaVa] = useState(1300);
  const [isPrabayar, setIsPrabayar] = useState(true);
  const [luasRumah, setLuasRumah] = useState(45);
  const [penghuni, setPenghuni] = useState(3);
  const [tagihanAsli, setTagihanAsli] = useState(600000);
  const [tokenContext, setTokenContext] = useState<TokenContextInput>(TOKEN_KOSONG);
  const [intentUser, setIntentUser] = useState<FokusOptimasi[]>(["Biaya"]);
  const [daftarAlat, setDaftarAlat] = useState<PeralatanInput[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasil, setHasil] = useState<AnalisisResponse | null>(null);

  useEffect(() => {
    getReferensi()
      .then(setReferensi)
      .catch((e: unknown) =>
        setReferensiError(
          e instanceof ApiError
            ? e.detail
            : "Tidak bisa terhubung ke server. Periksa apakah backend sedang berjalan."
        )
      );
  }, []);

  const tarifKwh = referensi?.tarif_per_golongan[String(dayaVa)] ?? 0;
  const pbjt = referensi?.pbjt_rumah_tangga ?? 0.024;

  async function submit() {
    if (daftarAlat.length === 0) {
      setError("Tambahkan minimal 1 peralatan sebelum memulai analisis.");
      return;
    }
    setLoading(true);
    setError(null);
    setHasil(null);
    try {
      const result = await postAnalisis({
        daya_va: dayaVa,
        is_prabayar: isPrabayar,
        luas_rumah: luasRumah,
        penghuni,
        tagihan_asli: isPrabayar ? null : tagihanAsli,
        token_context: isPrabayar ? tokenContext : null,
        daftar_alat: daftarAlat,
        intent_user: intentUser,
      });
      setHasil(result);
    } catch (e) {
      setError(
        e instanceof ApiError ? e.detail : "Gagal menghubungi server. Coba lagi beberapa saat."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-4 py-10 sm:py-14">
      <header className="mb-10 text-center">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-amber">⚡ EnergiCerdas AI</p>
        <h1 className="mt-2 text-2xl font-semibold text-cream sm:text-3xl">
          Analisis Konsumsi Listrik Rumah Tangga
        </h1>
        <p className="mx-auto mt-2 max-w-md text-sm text-cream-dim">
          Deteksi anomali pemakaian dan rekomendasi hemat energi untuk rumah
          tangga DKI Jakarta.
        </p>
      </header>

      {referensiError && (
        <div className="mb-6 rounded-lg border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
          {referensiError}
        </div>
      )}

      <div className="space-y-5">
        <DayaMeteranSection
          referensi={referensi}
          dayaVa={dayaVa}
          onDayaVaChange={setDayaVa}
          isPrabayar={isPrabayar}
          onIsPrabayarChange={setIsPrabayar}
        />
        <ProfilSection
          luasRumah={luasRumah}
          onLuasRumahChange={setLuasRumah}
          penghuni={penghuni}
          onPenghuniChange={setPenghuni}
          isPrabayar={isPrabayar}
          tagihanAsli={tagihanAsli}
          onTagihanAsliChange={setTagihanAsli}
          tokenContext={tokenContext}
          onTokenContextChange={setTokenContext}
          tarifKwh={tarifKwh}
          pbjt={pbjt}
        />
        <OptimasiSection intentUser={intentUser} onChange={setIntentUser} />
        <PeralatanSection daftarAlat={daftarAlat} onChange={setDaftarAlat} />

        {error && (
          <div className="rounded-lg border border-red/50 bg-red-dim px-4 py-3 text-sm text-red">
            {error}
          </div>
        )}

        <Button
          onClick={submit}
          disabled={loading || !referensi}
          className="w-full py-3.5 text-base"
        >
          {loading ? "Menganalisis…" : "🚀 Mulai Analisis"}
        </Button>

        {hasil && (
          <div className="pt-4">
            <HasilAnalisis hasil={hasil} />
          </div>
        )}
      </div>
    </main>
  );
}
