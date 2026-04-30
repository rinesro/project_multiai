# ⚡ EnergiCerdas AI — Setup Guide

## Struktur Folder

```
project/
├── app.py
├── requirements.txt
└── models/
    ├── ai_profile_labels.pkl
    ├── ai_profile_model.pkl
    ├── ai_profile_scaler.pkl
    ├── ai_estimator_features.pkl
    └── ai_estimator_model.pkl
```

## Cara Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ Penting: scikit-learn harus versi **1.6.1** agar kompatibel dengan model yang sudah dilatih.
> Kalau ada conflict: `pip install scikit-learn==1.6.1 --force-reinstall`

### 2. Letakkan model di folder `models/`
Salin semua file `.pkl` ke dalam folder `models/` di direktori yang sama dengan `app.py`.

### 3. Dapatkan Gemini API Key (gratis)
- Buka https://aistudio.google.com/apikey
- Buat API Key baru (gratis, unlimited untuk Gemini Flash)
- Paste ke kolom API Key di aplikasi

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## Cara Kerja Sistem

```
Input Form User
     │
     ▼
[Preprocessing]
 Hitung sub_metering dari wattage × jam pemakaian
     │
     ├──────────────────────────┐
     ▼                          ▼
[AI Profiler]            [AI Estimator]
 K-Means Clustering       Random Forest Regressor
 → Profil konsumsi        → Estimasi kWh
     │                          │
     └──────────┬───────────────┘
                ▼
        [Finishing System]
         Hitung biaya (tarif PLN Q2 2026)
         Hitung emisi CO₂ (ESDM 2020: 0.87 kgCO2/kWh)
                │
                ▼
        [AI Rekomendasi]
         Gemini 2.0 Flash API
         → Rekomendasi natural language
                │
                ▼
        [Tampilkan di Streamlit]
```

## Tarif PLN Q2 2026 (April – Juni 2026)
| Golongan | Tarif |
|---|---|
| Rumah Tangga 900 VA RTM | Rp 1.352,00/kWh |
| Rumah Tangga 1.300–2.200 VA & Bisnis 6.600 VA–200 kVA | Rp 1.444,70/kWh |
| Rumah Tangga ≥ 3.500 VA | Rp 1.699,53/kWh |

## Faktor Emisi
- Grid emission factor: **0,87 kgCO₂/kWh** (Inventarisasi Emisi GRK Sektor Energi, ESDM 2020)