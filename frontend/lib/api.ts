/**
 * lib/api.ts
 * ===========
 * Fetch wrapper ke backend FastAPI.
 *
 * NEXT_PUBLIC_API_URL kosong/tidak diset -> request jadi relatif (mis.
 * fetch('/api/referensi')) -> otomatis benar untuk deploy Vercel Services
 * (frontend & backend satu domain yang sama).
 *
 * Untuk development lokal (frontend :3000, backend :8000 terpisah),
 * .env.local WAJIB set NEXT_PUBLIC_API_URL=http://localhost:8000 secara
 * eksplisit — lihat .env.local.example.
 */

import type { AnalisisRequest, AnalisisResponse, ApiErrorBody, ReferensiResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function extractDetail(body: ApiErrorBody | undefined): string {
  if (!body?.detail) return "Terjadi kesalahan tak terduga di server.";
  if (typeof body.detail === "string") return body.detail;
   
  return body.detail[0]?.msg ?? "Data yang dikirim tidak valid.";
}

export async function getReferensi(): Promise<ReferensiResponse> {
  const res = await fetch(`${API_URL}/api/referensi`, { cache: "no-store" });
  if (!res.ok) {
    const body = (await res.json().catch(() => undefined)) as ApiErrorBody | undefined;
    throw new ApiError(res.status, extractDetail(body));
  }
  return res.json();
}

export async function postAnalisis(payload: AnalisisRequest): Promise<AnalisisResponse> {
  const res = await fetch(`${API_URL}/api/analisis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => undefined)) as ApiErrorBody | undefined;
    throw new ApiError(res.status, extractDetail(body));
  }
  return res.json();
}