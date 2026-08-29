# ⚡ EnergiCerdas AI: Platform Rekomendasi Konsumsi Energi Rumah Tangga

**Catatan Perekrut:** Proyek ini adalah Skripsi (Tugas Akhir) untuk gelar Sarjana Komputer, Teknik Informatika, Universitas Gunadarma. Sistem menggabungkan lima komponen AI/ML berbeda dalam satu arsitektur multi-agent untuk mengubah data tagihan listrik rumah tangga menjadi rekomendasi penghematan energi yang actionable, lengkap dengan narasi bahasa natural.

**Live Demo:** [project-multiai.vercel.app](https://project-multiai.vercel.app/)

## 📌 Latar Belakang & Permasalahan

Rumah tangga di DKI Jakarta kesulitan menerjemahkan angka tagihan listrik bulanan menjadi tindakan konkret: alat mana yang boros, apakah pola konsumsinya wajar, dan penghematan mana yang paling realistis untuk dilakukan tanpa mengorbankan kenyamanan. Standar klasifikasi Intensitas Konsumsi Energi (IKE) untuk rumah tangga di Indonesia juga bermasalah — Permen ESDM No.13/2012 yang biasa dijadikan acuan telah dicabut oleh Permen ESDM No.9/2018 tanpa pengganti khusus rumah tangga, sehingga dibutuhkan pendekatan klasifikasi yang tervalidasi dari sumber lain.

## 🧠 Arsitektur Multi-AI

Sistem terdiri dari lima komponen yang bekerja berurutan dalam satu pipeline analisis:

1. **Profil IKE (Fuzzy Argmax)** — mengklasifikasikan intensitas konsumsi energi rumah tangga ke dalam 5 kelas menggunakan fungsi keanggotaan trapesium 1-variabel, dengan ambang batas hasil triangulasi 5 lapis (regulatif, empiris, dan statistik/Jenks natural breaks).
2. **DSM Classifier (LightGBM)** — mengklasifikasikan setiap alat elektronik ke dalam kategori "Fleksibel" atau "Tidak Fleksibel" untuk digeser jadwal pemakaiannya, mencapai **akurasi 97,9%** dan **F1-macro 97,9%** pada data uji, jauh di atas baseline fallback rule-based (61,7%).
3. **Optimizer Penjadwalan** — mencari kombinasi penjadwalan ulang alat-alat fleksibel yang paling optimal untuk menekan biaya, dengan batas maksimum reduksi 50% dari total konsumsi.
4. **Deteksi Anomali** — mendeteksi pola konsumsi yang tidak wajar berbasis aturan (rule-based), termasuk potensi risiko MCB trip dan pelanggaran batas kWh bulanan untuk pelanggan prabayar (aturan "720 jam nyala" PLN).
5. **Generator Narasi (Google Gemini)** — merangkum seluruh hasil analisis menjadi narasi bahasa Indonesia yang mudah dipahami pengguna awam.

## 💰 Perhitungan Tarif & Regulasi

Perhitungan biaya mengikuti struktur tarif PLN (golongan daya 450–6.600 VA) dan Pajak Barang & Jasa Tertentu atas Tenaga Listrik (PBJT) sebesar 2,4% khusus DKI Jakarta sesuai Perda DKI Jakarta No.1/2024, dengan skema perhitungan pajak yang dibedakan antara pelanggan pascabayar dan prabayar.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, LightGBM, scikit-learn
* **Frontend:** Next.js (App Router), React, TypeScript, Tailwind CSS
* **AI Generatif:** Google Gemini API (narasi hasil analisis)
* **Deployment:** Vercel (frontend & backend sebagai layanan terpisah dalam satu proyek), Docker

## 🚀 Cara Menjalankan Proyek

**1. Clone repositori ini (branch `versi5`):**
```bash
git clone -b versi5 https://github.com/rinesro/project_multiai.git
cd project_multiai
```

**2. Menjalankan backend (FastAPI):**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**3. Menjalankan frontend (Next.js):**
```bash
cd frontend
npm install
npm run dev
```

Variabel lingkungan `GEMINI_API_KEY` diperlukan pada backend untuk mengaktifkan generator narasi.

## 📂 Struktur Repositori

* `backend/core/kalkulasi.py` — konstanta regulasi (tarif PLN, PBJT) dan seluruh fungsi perhitungan biaya, kWh, dan emisi
* `backend/action_analist/ike_profiler.py` — klasifikasi fuzzy argmax profil IKE
* `backend/action_analist/anomaly_evaluator.py` — deteksi anomali konsumsi berbasis aturan
* `backend/models/dsm_classifier.py` — pembungkus model LightGBM untuk klasifikasi fleksibilitas alat
* `backend/optimizer/brute_force.py` — optimasi penjadwalan ulang alat fleksibel
* `backend/services/narasi.py` — integrasi Google Gemini untuk narasi hasil analisis
* `backend/data/dsm_metadata.json` — metrik evaluasi model DSM Classifier (akurasi, F1-score)
* `frontend/app/` — antarmuka pengguna (Next.js App Router)
* `test_energicerdas.py` — unit test & system test seluruh modul perhitungan dan klasifikasi

## 💼 Let's Connect!

Dibuat oleh Sandhika Hamzah. Silakan hubungi saya melalui email (sanvinzah@gmail.com) atau LinkedIn untuk diskusi lebih lanjut mengenai peluang kolaborasi data & AI.
