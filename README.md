# Bedah Kasus 1, 2, dan 3 — Visualisasi Data

Nama  : Intan Febriani
NIM   : 241055001
Kelas : F

## Isi folder
| File | Fungsi |
|---|---|
| generate_data.py | Membuat data sintetis 1.000 baris untuk tiap kasus |
| data_kasus1_penjualan.csv | Kasus 1: operasional penjualan |
| data_kasus2_mahasiswa.csv | Kasus 2: profil mahasiswa |
| data_kasus3_cuaca.csv | Kasus 3: pengamatan cuaca |
| analisis.py | Analisis detail ketiga kasus |
| laporan_analisis.txt | Hasil analisis dalam bentuk teks |
| output_analisis/ | Grafik hasil analisis (kasus1, kasus2, kasus3) |
| app.py | Dashboard Streamlit (pilih kasus di sidebar) |

## Cara menjalankan (Windows / PowerShell)
```
python -m pip install -r requirements.txt
python generate_data.py
python analisis.py
python -m streamlit run app.py
```
Dashboard terbuka di http://localhost:8501. Saat pertama kali, Streamlit menanyakan
email: tekan Enter saja untuk melewati.
