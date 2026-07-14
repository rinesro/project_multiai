"use client";

import { ReactNode, SelectHTMLAttributes, InputHTMLAttributes } from "react";

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-graphite-700 bg-graphite-900 p-5 sm:p-7 ${className}`}
    >
      {children}
    </div>
  );
}

/** Label section bergaya panel/skema teknis — "[ 01 ]" bukan angka
 * bulat generik, karena form ini memang punya urutan langkah nyata. */
export function SectionLabel({
  nomor,
  title,
  desc,
}: {
  nomor: string;
  title: string;
  desc?: string;
}) {
  return (
    <div className="mb-5 flex items-start gap-3">
      <span className="font-mono text-sm font-semibold tracking-wider text-amber">
        [{nomor}]
      </span>
      <div>
        <h2 className="text-base font-semibold tracking-wide text-cream sm:text-lg">
          {title}
        </h2>
        {desc && <p className="mt-0.5 text-sm text-cream-dim">{desc}</p>}
      </div>
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-cream-dim">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-cream-dim/70">{hint}</span>}
    </label>
  );
}

const controlClass =
  "w-full rounded-lg border border-graphite-600 bg-graphite-800 px-3 py-2 text-sm text-cream " +
  "placeholder:text-cream-dim/50 outline-none transition-colors focus:border-amber " +
  "disabled:opacity-50";

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={`${controlClass} ${props.className ?? ""}`} />;
}

export function Select(props: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...props} className={`${controlClass} ${props.className ?? ""}`}>
      {props.children}
    </select>
  );
}

export function Button({
  children,
  variant = "primary",
  className = "",
  ...rest
}: {
  children: ReactNode;
  variant?: "primary" | "secondary" | "danger" | "ghost";
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold " +
    "transition-colors disabled:cursor-not-allowed disabled:opacity-40";
  const variants: Record<string, string> = {
    primary: "bg-amber text-graphite-950 hover:bg-amber-bright",
    secondary:
      "border border-graphite-600 bg-graphite-800 text-cream hover:border-amber-dim",
    danger: "border border-red/40 bg-red-dim text-red hover:bg-red/20",
    ghost: "text-cream-dim hover:text-cream",
  };
  return (
    <button {...rest} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}

/** Toggle pill dua opsi — dipakai untuk jenis meteran & fokus optimasi. */
export function TogglePill({
  options,
  value,
  onChange,
}: {
  options: { value: string; label: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-graphite-600 bg-graphite-800 p-1">
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition-colors ${
            value === opt.value
              ? "bg-amber text-graphite-950"
              : "text-cream-dim hover:text-cream"
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function Chip({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`rounded-full border px-4 py-1.5 text-sm font-medium transition-colors ${
        active
          ? "border-amber bg-amber/15 text-amber-bright"
          : "border-graphite-600 text-cream-dim hover:border-graphite-600 hover:text-cream"
      }`}
    >
      {label}
    </button>
  );
}

/** Garis pembatas dengan label di tengah — gaya struk/dokumen cetak,
 * dipakai untuk menandai transisi antar-bagian tanpa terasa asing
 * dari motif tear-line yang sudah ada di seluruh halaman. */
export function DividerLabel({ children }: { children: ReactNode }) {
  return (
    <div className="my-5 flex items-center gap-3" role="separator">
      <div className="h-px flex-1 tear-line" aria-hidden />
      <span className="shrink-0 font-mono text-[11px] uppercase tracking-widest text-cream-dim/70">
        {children}
      </span>
      <div className="h-px flex-1 tear-line" aria-hidden />
    </div>
  );
}
