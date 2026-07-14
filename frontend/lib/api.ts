/**
 * lib/api.ts
 * ===========
 * Fetch wrapper ke backend FastAPI. URL backend diambil dari
 * NEXT_PUBLIC_API_URL (env var) — lihat .env.local.example.
 */

import type { AnalisisRequest, AnalisisResponse, ApiErrorBody, ReferensiResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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
  // Pydantic validation errors (422) berbentuk array — ambil pesan pertama.
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
