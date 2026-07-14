import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "EnergiCerdas AI — Analisis Konsumsi Listrik Rumah Tangga",
  description:
    "Analisis konsumsi listrik, deteksi anomali, dan rekomendasi hemat energi untuk rumah tangga DKI Jakarta.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body
        className={`${plexSans.variable} ${plexMono.variable} font-sans antialiased bg-graphite-950 text-cream`}
      >
        {children}
      </body>
    </html>
  );
}
