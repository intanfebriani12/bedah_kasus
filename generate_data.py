"""
generate_data.py
Membuat 1.000 baris data sintetis untuk tiga bedah kasus:

  Kasus 1 - Dataset Operasional Penjualan  -> data_kasus1_penjualan.csv
  Kasus 2 - Dataset Profil Mahasiswa       -> data_kasus2_mahasiswa.csv
  Kasus 3 - Dataset Pengamatan Cuaca       -> data_kasus3_cuaca.csv

Data dibuat dengan pola yang masuk akal (bukan acak murni) supaya analisisnya
bermakna. Baris contoh asli dari slide selalu dimasukkan sebagai baris pertama.
Seed dikunci (42) sehingga hasil selalu sama setiap dijalankan.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 1000


# =====================================================================
# KASUS 1: OPERASIONAL PENJUALAN
# Produk & Kategori (Nominal), Harga (Kontinu-Rasio),
# Jumlah Terjual (Diskrit-Rasio), Tanggal (Interval)
# =====================================================================
KATALOG = {  # produk: (kategori, harga dasar, rata-rata unit terjual)
    "Laptop": ("Elektronik", 8_500_000, 45), "Smartphone": ("Elektronik", 4_200_000, 70),
    "Tablet": ("Elektronik", 5_300_000, 35), "Headphone": ("Elektronik", 650_000, 95),
    "Smartwatch": ("Elektronik", 1_800_000, 50),
    "Buku": ("Pendidikan", 75_000, 120), "Kamus": ("Pendidikan", 120_000, 60),
    "Alat Tulis Set": ("Pendidikan", 45_000, 150), "Tas Sekolah": ("Pendidikan", 250_000, 80),
    "Kalkulator": ("Pendidikan", 185_000, 70),
    "Kemeja": ("Fashion", 180_000, 110), "Sepatu": ("Fashion", 450_000, 75),
    "Jaket": ("Fashion", 350_000, 60), "Celana Jeans": ("Fashion", 275_000, 90),
    "Rice Cooker": ("Rumah Tangga", 550_000, 55), "Blender": ("Rumah Tangga", 420_000, 50),
    "Setrika": ("Rumah Tangga", 230_000, 65), "Dispenser": ("Rumah Tangga", 780_000, 40),
    "Kopi Bubuk": ("Makanan & Minuman", 55_000, 200), "Teh Celup": ("Makanan & Minuman", 15_000, 260),
    "Biskuit Kaleng": ("Makanan & Minuman", 65_000, 180),
}
BOBOT_KATEGORI = {"Elektronik": .22, "Pendidikan": .22, "Fashion": .20,
                  "Rumah Tangga": .16, "Makanan & Minuman": .20}
# Pengali musiman per bulan (Jan..Des): Lebaran Mar-Apr, ajaran baru Jul, Harbolnas Nov-Des
MUSIM_JUAL = {
    "Elektronik":        [1.0, .9, 1.0, 1.15, 1.0, .95, 1.0, 1.0, .95, 1.0, 1.25, 1.35],
    "Pendidikan":        [1.2, .9, .85, .8, .85, 1.1, 1.5, 1.3, .95, .9, .9, .9],
    "Fashion":           [.95, .9, 1.35, 1.45, 1.0, .95, .9, .9, .95, 1.0, 1.2, 1.3],
    "Rumah Tangga":      [1.0, .95, 1.2, 1.3, 1.0, 1.0, .95, .95, 1.0, 1.0, 1.15, 1.2],
    "Makanan & Minuman": [1.0, 1.0, 1.4, 1.5, 1.0, .95, .95, .95, 1.0, 1.0, 1.1, 1.3],
}


def buat_kasus1(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    produk_list = list(KATALOG)
    per_kat = pd.Series([KATALOG[p][0] for p in produk_list]).value_counts()
    p = np.array([BOBOT_KATEGORI[KATALOG[x][0]] / per_kat[KATALOG[x][0]] for x in produk_list])
    produk = rng.choice(produk_list, size=n - 2, p=p / p.sum())

    hari = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    bobot_bln = np.array([.9, .85, 1.1, 1.2, .9, .95, 1.05, .95, .9, .95, 1.1, 1.3])
    ph = bobot_bln[hari.month - 1]
    tanggal = pd.DatetimeIndex(rng.choice(hari, size=n - 2, p=ph / ph.sum()))

    rows = []
    for prod, tgl in zip(produk, tanggal):
        kat, harga0, unit0 = KATALOG[prod]
        diskon = rng.uniform(.10, .30) if rng.random() < .12 else 0.0
        harga = max(round(harga0 * rng.normal(1, .08) * (1 - diskon) / 1000) * 1000, 1000)
        lam = unit0 * MUSIM_JUAL[kat][tgl.month - 1] * (1 + diskon * 1.8)
        lam *= (1.15 if tgl.dayofweek >= 5 else 1.0) * rng.lognormal(0, .15)
        rows.append((prod, kat, int(harga), int(max(rng.poisson(lam), 1)), tgl, diskon > 0))

    df = pd.DataFrame(rows, columns=["Produk", "Kategori", "Harga (Rp)",
                                     "Jumlah Terjual", "Tanggal", "Promo"])
    asli = pd.DataFrame({"Produk": ["Laptop", "Buku"], "Kategori": ["Elektronik", "Pendidikan"],
                         "Harga (Rp)": [8_500_000, 75_000], "Jumlah Terjual": [45, 120],
                         "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                         "Promo": [False, False]})
    df = pd.concat([asli, df], ignore_index=True)
    # transaksi borongan (outlier wajar)
    idx = rng.choice(df.index[2:], size=8, replace=False)
    df.loc[idx, "Jumlah Terjual"] = (df.loc[idx, "Jumlah Terjual"] * rng.uniform(2.5, 4, 8)).astype(int)
    df = df.sort_values("Tanggal", kind="stable").reset_index(drop=True)
    df.insert(0, "ID Transaksi", [f"TRX-{i:04d}" for i in range(1, len(df) + 1)])
    return df


# =====================================================================
# KASUS 2: PROFIL MAHASISWA
# Jurusan (Nominal), Semester (Ordinal), IPK (Kontinu), Usia (Kontinu)
# =====================================================================
JURUSAN = {  # jurusan: (bobot jumlah mahasiswa, rata-rata IPK, sebaran IPK)
    "Sistem Informasi": (.20, 3.32, .32), "Teknik Informatika": (.22, 3.25, .36),
    "Manajemen": (.20, 3.38, .28), "Akuntansi": (.15, 3.30, .30),
    "Teknik Industri": (.12, 3.15, .38), "Ilmu Komunikasi": (.11, 3.42, .26),
}
# jumlah mahasiswa makin sedikit di semester atas (ada yang lulus/keluar)
BOBOT_SEMESTER = np.array([.17, .16, .15, .14, .12, .11, .09, .06])


def buat_kasus2(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    jur_list = list(JURUSAN)
    pj = np.array([JURUSAN[j][0] for j in jur_list])
    jurusan = rng.choice(jur_list, size=n - 2, p=pj / pj.sum())
    semester = rng.choice(np.arange(1, 9), size=n - 2, p=BOBOT_SEMESTER / BOBOT_SEMESTER.sum())
    jenis = rng.choice(["Reguler", "Karyawan"], size=n - 2, p=[.85, .15])

    rows = []
    for j, s, k in zip(jurusan, semester, jenis):
        _, mu, sd = JURUSAN[j]
        # IPK naik tipis di semester tengah (adaptasi), dibatasi 0-4
        ipk = np.clip(rng.normal(mu + 0.03 * min(s, 6) - 0.09, sd), 1.80, 4.00)
        # usia: masuk kuliah ~18 th, bertambah 0,5 th per semester
        if k == "Reguler":
            usia = 17.6 + s * 0.5 + rng.normal(0, 0.7)
        else:  # kelas karyawan: usia jauh lebih tua -> sumber outlier di box plot
            usia = rng.uniform(23, 38) + s * 0.5
        rows.append((j, int(s), round(float(ipk), 2), int(round(usia)), k))

    df = pd.DataFrame(rows, columns=["Jurusan", "Semester", "IPK", "Usia", "Kelas"])
    asli = pd.DataFrame({"Jurusan": ["Sistem Informasi", "Manajemen"], "Semester": [3, 1],
                         "IPK": [3.85, 3.50], "Usia": [20, 19], "Kelas": ["Reguler", "Reguler"]})
    df = pd.concat([asli, df], ignore_index=True)
    df.insert(0, "ID Mahasiswa", [f"MHS-{i:04d}" for i in range(1, len(df) + 1)])
    return df


# =====================================================================
# KASUS 3: PENGAMATAN CUACA
# Kota (Nominal), Suhu (Interval), Kelembaban (Rasio), Kondisi (Nominal)
# Kolom Tanggal ditambahkan sebagai konteks pengamatan.
# =====================================================================
KOTA = {  # kota: (suhu rata-rata °C, kelembaban rata-rata %)
    "Jakarta": (31.0, 74), "Bandung": (24.0, 82), "Surabaya": (31.5, 70),
    "Medan": (30.0, 78), "Yogyakarta": (29.0, 77), "Makassar": (30.5, 75),
    "Denpasar": (30.0, 76), "Malang": (25.0, 80),
}
# musim hujan Nov-Mar: kelembaban naik, suhu turun sedikit
EFEK_HUJAN = np.array([1, 1, .8, .4, .1, 0, 0, 0, .1, .4, .8, 1])
KONDISI = ["Cerah", "Berawan", "Hujan", "Hujan Petir"]


def buat_kasus3(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    kota = rng.choice(list(KOTA), size=n - 2)
    hari = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    tanggal = pd.DatetimeIndex(rng.choice(hari, size=n - 2))

    rows = []
    for k, t in zip(kota, tanggal):
        suhu0, lembab0 = KOTA[k]
        musim = EFEK_HUJAN[t.month - 1]
        lembab = np.clip(lembab0 + 8 * musim + rng.normal(0, 6), 40, 100)
        # suhu turun saat udara lembab (hubungan negatif suhu-kelembaban)
        suhu = suhu0 - 1.2 * musim - 0.09 * (lembab - lembab0) + rng.normal(0, 1.1)
        # peluang kondisi ditentukan kelembaban (makin lembab, makin mungkin hujan)
        z = (lembab - 78) / 6
        skor = np.array([-1.2 * z + .4, .3, 1.0 * z - .3, 1.4 * z - 1.6])
        p = np.exp(skor) / np.exp(skor).sum()
        kondisi = rng.choice(KONDISI, p=p)
        if kondisi in ("Hujan", "Hujan Petir"):
            suhu -= rng.uniform(.5, 2.0)  # hujan menurunkan suhu
        rows.append((k, round(float(suhu), 1), int(round(lembab)), kondisi, t))

    df = pd.DataFrame(rows, columns=["Kota", "Suhu (°C)", "Kelembaban (%)", "Kondisi", "Tanggal"])
    asli = pd.DataFrame({"Kota": ["Jakarta", "Bandung"], "Suhu (°C)": [32.5, 24.0],
                         "Kelembaban (%)": [75, 85], "Kondisi": ["Cerah", "Hujan"],
                         "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-01"])})
    df = pd.concat([asli, df], ignore_index=True)
    df = df.sort_values("Tanggal", kind="stable").reset_index(drop=True)
    df.insert(0, "ID Pengamatan", [f"OBS-{i:04d}" for i in range(1, len(df) + 1)])
    return df


FILES = {
    "data_kasus1_penjualan.csv": buat_kasus1,
    "data_kasus2_mahasiswa.csv": buat_kasus2,
    "data_kasus3_cuaca.csv": buat_kasus3,
}

if __name__ == "__main__":
    for nama, fungsi in FILES.items():
        data = fungsi()
        data.to_csv(nama, index=False, date_format="%Y-%m-%d")
        print(f"[OK] {nama}: {len(data):,} baris")
        print(data.head(3).to_string(index=False), "\n")
