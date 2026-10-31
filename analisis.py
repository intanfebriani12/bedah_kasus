"""
analisis.py
Analisis data detail untuk tiga bedah kasus.

Jalankan:  python analisis.py
Output  :  laporan_analisis.txt  +  grafik PNG di output_analisis/kasus1, kasus2, kasus3

Warna grafik mengikuti slide: biru = kategorik, oranye = numerik, abu = waktu.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

BIRU, ORANYE, ABU = "#1D8FD1", "#F28C28", "#7A7F87"
BIRU_SKALA = ["#0B4F7A", "#1D8FD1", "#5DB7E8", "#9FD3F2", "#CDE9F9", "#2A7FA8"]
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})

OUT = Path("output_analisis")
laporan: list[str] = []


def tulis(t: str = "") -> None:
    print(t)
    laporan.append(t)


def judul(t: str, c: str = "=") -> None:
    tulis("\n" + c * 76)
    tulis(t)
    tulis(c * 76)


def rupiah(x: float) -> str:
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def fmt_rp(x, _):
    for b, s in ((1e9, "M"), (1e6, "jt"), (1e3, "rb")):
        if abs(x) >= b:
            return f"Rp{x/b:.1f} {s}"
    return f"Rp{x:.0f}"


def simpan(fig, folder: str, nama: str) -> None:
    (OUT / folder).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / folder / nama, bbox_inches="tight")
    plt.close(fig)


def statistik(df: pd.DataFrame, cols: list[str]) -> None:
    d = df[cols].describe().T
    d["median"] = df[cols].median()
    d["skewness"] = df[cols].skew()
    d["kurtosis"] = df[cols].kurt()
    d["CV (%)"] = d["std"] / d["mean"] * 100
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160):
        tulis(d[["count", "mean", "median", "std", "min", "max",
                 "skewness", "kurtosis", "CV (%)"]].to_string())


def kualitas(df: pd.DataFrame, id_col: str) -> None:
    tulis(f"Nilai kosong total : {int(df.isna().sum().sum())}")
    tulis(f"Baris duplikat     : {int(df.drop(columns=id_col).duplicated().sum())}")


def klasifikasi(rows: list[tuple]) -> None:
    t = pd.DataFrame(rows, columns=["Kolom", "Tipe", "Skala", "Visualisasi tepat"])
    tulis(t.to_string(index=False))


def iqr_flag(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([.25, .75])
    return (s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))


# =====================================================================
def kasus1() -> None:
    judul("BEDAH KASUS 1: DATASET OPERASIONAL PENJUALAN")
    df = pd.read_csv("data_kasus1_penjualan.csv", parse_dates=["Tanggal"])
    df["Total (Rp)"] = df["Harga (Rp)"] * df["Jumlah Terjual"]
    df["Bulan"] = df["Tanggal"].dt.to_period("M")
    F = "kasus1"

    judul("1.1 Profil & klasifikasi tipe data", "-")
    tulis(f"{len(df):,} transaksi | {df['Produk'].nunique()} produk | "
          f"{df['Kategori'].nunique()} kategori | {df['Tanggal'].min():%d %b %Y} - "
          f"{df['Tanggal'].max():%d %b %Y}")
    klasifikasi([("Produk", "Kategorik", "Nominal", "Bar / Pie"),
                 ("Kategori", "Kategorik", "Nominal", "Bar / Pie"),
                 ("Harga (Rp)", "Numerik kontinu", "Rasio", "Histogram"),
                 ("Jumlah Terjual", "Numerik diskrit", "Rasio", "Bar / Histogram"),
                 ("Tanggal", "Waktu", "Interval", "Line chart")])
    kualitas(df, "ID Transaksi")

    judul("1.2 Statistik deskriptif", "-")
    statistik(df, ["Harga (Rp)", "Jumlah Terjual", "Total (Rp)"])
    tulis(f"-> Harga sangat menceng kanan: median {rupiah(df['Harga (Rp)'].median())} jauh di bawah "
          f"rata-rata {rupiah(df['Harga (Rp)'].mean())}. Ukuran pusat yang tepat adalah MEDIAN.")

    judul("1.3 Variabel kategorik", "-")
    pk = df.groupby("Kategori").agg(Transaksi=("Produk", "size"), Unit=("Jumlah Terjual", "sum"),
                                    Pendapatan=("Total (Rp)", "sum")).sort_values("Pendapatan", ascending=False)
    pk["% Pendapatan"] = pk["Pendapatan"] / pk["Pendapatan"].sum() * 100
    with pd.option_context("display.float_format", "{:,.1f}".format):
        tulis(pk.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    pk["Transaksi"].sort_values().plot.barh(ax=ax[0], color=BIRU)
    ax[0].set(title="Jumlah transaksi per kategori (bar chart)", ylabel="")
    w, *_ = ax[1].pie(pk["Pendapatan"], colors=BIRU_SKALA, startangle=90,
                      autopct=lambda v: f"{v:.1f}%" if v > 5 else "", pctdistance=.78,
                      wedgeprops={"edgecolor": "white", "linewidth": 2, "width": .45},
                      textprops={"color": "white", "weight": "bold"})
    ax[1].legend(w, pk.index, loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    ax[1].set_title("Kontribusi pendapatan (pie chart)")
    simpan(fig, F, "01_bar_pie_kategori.png")

    judul("1.4 Distribusi numerik & outlier", "-")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(df["Harga (Rp)"], bins=40, ax=ax[0], color=ORANYE)
    ax[0].xaxis.set_major_formatter(mtick.FuncFormatter(fmt_rp))
    ax[0].set_title("Histogram harga (skala asli)")
    sns.histplot(df["Harga (Rp)"], bins=40, log_scale=True, ax=ax[1], color=ORANYE)
    ax[1].xaxis.set_major_formatter(mtick.FuncFormatter(fmt_rp))
    ax[1].set_title("Histogram harga (skala log)")
    simpan(fig, F, "02_histogram_harga.png")
    df["Outlier"] = df.groupby("Produk")["Jumlah Terjual"].transform(iqr_flag)
    tulis(f"Outlier jumlah terjual (IQR per produk): {int(df['Outlier'].sum())} transaksi "
          f"-> pembelian borongan, dipisahkan saat proyeksi penjualan normal.")

    judul("1.5 Pareto pendapatan", "-")
    pp = df.groupby("Produk")["Total (Rp)"].sum().sort_values(ascending=False)
    kum = pp.cumsum() / pp.sum() * 100
    n80 = int((kum < 80).sum() + 1)
    tulis(f"{n80} dari {len(pp)} produk menyumbang ~80% pendapatan: {', '.join(pp.index[:n80])}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(pp.index, pp.values, color=BIRU)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_rp))
    ax.tick_params(axis="x", rotation=60)
    a2 = ax.twinx()
    a2.plot(pp.index, kum.values, color=ORANYE, marker="o", lw=2)
    a2.axhline(80, color=ABU, ls="--")
    a2.set_ylim(0, 105)
    a2.grid(False)
    a2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Diagram Pareto pendapatan per produk")
    simpan(fig, F, "03_pareto.png")

    judul("1.6 Tren waktu", "-")
    bl = df.groupby("Bulan").agg(Pendapatan=("Total (Rp)", "sum"), Unit=("Jumlah Terjual", "sum"))
    tulis(f"Pendapatan tertinggi {bl['Pendapatan'].idxmax()} ({rupiah(bl['Pendapatan'].max())}), "
          f"terendah {bl['Pendapatan'].idxmin()} ({rupiah(bl['Pendapatan'].min())})")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(bl.index.to_timestamp(), bl["Pendapatan"], color=ABU, marker="o", lw=2.5)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_rp))
    ax.set_title("Tren pendapatan bulanan 2024 (line chart)")
    simpan(fig, F, "04_tren_bulanan.png")

    judul("1.7 Hubungan harga - unit & efek promo", "-")
    rho = stats.spearmanr(df["Harga (Rp)"], df["Jumlah Terjual"])
    tulis(f"Spearman rho = {rho.statistic:.3f} (p = {rho.pvalue:.1e}) -> makin mahal, makin "
          f"sedikit unit per transaksi.")
    df["Indeks"] = df["Jumlah Terjual"] / df.groupby("Produk")["Jumlah Terjual"].transform("median")
    pr, nm = df.loc[df["Promo"], "Indeks"], df.loc[~df["Promo"], "Indeks"]
    mw = stats.mannwhitneyu(pr, nm, alternative="greater")
    tulis(f"Promo: indeks unit median {pr.median():.2f} vs normal {nm.median():.2f} "
          f"(Mann-Whitney p = {mw.pvalue:.1e}) -> promo menaikkan penjualan secara signifikan.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="Harga (Rp)", y="Jumlah Terjual", hue="Kategori",
                    alpha=.6, ax=ax, palette="tab10")
    ax.set(xscale="log", yscale="log", title="Harga vs jumlah terjual (log-log)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(fmt_rp))
    simpan(fig, F, "05_scatter_harga_unit.png")

    judul("1.8 Kesimpulan kasus 1", "-")
    tulis(f"1. {pk.index[0]} menyumbang {pk['% Pendapatan'].iloc[0]:.1f}% pendapatan; volume unit "
          f"terbesar dari {pk['Unit'].idxmax()}.")
    tulis(f"2. Prinsip Pareto berlaku: {n80} produk menghasilkan ~80% pendapatan.")
    tulis(f"3. Puncak penjualan {bl['Pendapatan'].idxmax()} (musim belanja akhir tahun).")
    tulis("4. Harga berkorelasi negatif dengan unit terjual; promo efektif mendongkrak volume.")


# =====================================================================
def kasus2() -> None:
    judul("BEDAH KASUS 2: DATASET PROFIL MAHASISWA")
    df = pd.read_csv("data_kasus2_mahasiswa.csv")
    F = "kasus2"

    judul("2.1 Profil & klasifikasi tipe data", "-")
    tulis(f"{len(df):,} mahasiswa | {df['Jurusan'].nunique()} jurusan | semester "
          f"{df['Semester'].min()}-{df['Semester'].max()}")
    klasifikasi([("Jurusan", "Kategorik", "Nominal", "Bar / Pie"),
                 ("Semester", "Kategorik", "Ordinal", "Bar chart berurutan"),
                 ("IPK", "Numerik kontinu", "Interval*", "Histogram / Box plot"),
                 ("Usia", "Numerik kontinu", "Rasio", "Histogram / Box plot")])
    tulis("Catatan: Semester berupa angka, tetapi diperlakukan ORDINAL karena yang penting")
    tulis("urutannya (1 -> 2 -> 3), bukan hasil hitung. Rata-rata semester kurang bermakna;")
    tulis("gunakan median/modus. *IPK sering dianggap interval karena 0 adalah batas skala.")
    kualitas(df, "ID Mahasiswa")
    tulis("-> Duplikat di sini wajar: tanpa nama/NIM, dua mahasiswa berbeda bisa punya jurusan,")
    tulis("   semester, IPK, dan usia yang sama. Data tidak dihapus.")

    judul("2.2 Statistik deskriptif IPK & Usia", "-")
    statistik(df, ["IPK", "Usia"])
    tulis(f"Median semester = {df['Semester'].median():.0f}, modus = {df['Semester'].mode()[0]}")

    judul("2.3 Jurusan (nominal) & Semester (ordinal)", "-")
    fj = df["Jurusan"].value_counts()
    for k, v in fj.items():
        tulis(f"  {k:<20} {v:>4} ({v/len(df):.1%})")
    fs = df["Semester"].value_counts().sort_index()
    tulis("Sebaran semester (berurutan): " + ", ".join(f"S{i}={v}" for i, v in fs.items()))
    tulis("-> Jumlah mahasiswa menurun di semester atas (lulus / cuti / keluar).")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fj.sort_values().plot.barh(ax=ax[0], color=BIRU)
    ax[0].set(title="Jumlah mahasiswa per jurusan (bar chart)", ylabel="")
    ax[1].bar(fs.index.astype(str), fs.values, color=BIRU)
    ax[1].set(title="Mahasiswa per semester (bar chart BERURUTAN)", xlabel="Semester")
    simpan(fig, F, "01_jurusan_semester.png")

    judul("2.4 Distribusi IPK & Usia", "-")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    sns.histplot(df["IPK"], bins=30, kde=True, ax=ax[0, 0], color=ORANYE)
    ax[0, 0].set_title("Histogram IPK")
    sns.boxplot(x=df["IPK"], ax=ax[0, 1], color="#FBD3A8")
    ax[0, 1].set_title("Box plot IPK")
    sns.histplot(df["Usia"], binwidth=1, ax=ax[1, 0], color=ORANYE)
    ax[1, 0].set_title("Histogram usia")
    sns.boxplot(x=df["Usia"], ax=ax[1, 1], color="#FBD3A8")
    ax[1, 1].set_title("Box plot usia (titik = outlier)")
    simpan(fig, F, "02_distribusi_ipk_usia.png")

    sw = stats.shapiro(df["IPK"])
    tulis(f"IPK: skew {df['IPK'].skew():.2f}, uji Shapiro-Wilk p = {sw.pvalue:.1e} -> "
          f"{'tidak normal' if sw.pvalue < .05 else 'mendekati normal'} (sedikit menceng kiri, "
          f"dibatasi maksimum 4,00).")
    out_u = iqr_flag(df["Usia"])
    tulis(f"Usia: {int(out_u.sum())} outlier (IQR), usia {df.loc[out_u, 'Usia'].min()}-"
          f"{df.loc[out_u, 'Usia'].max()} th. Hampir semuanya dari kelas "
          f"{df.loc[out_u, 'Kelas'].mode()[0]} -> outlier ini valid, bukan salah input.")
    out_i = iqr_flag(df["IPK"])
    tulis(f"IPK: {int(out_i.sum())} outlier bawah (IPK <= {df.loc[out_i, 'IPK'].max():.2f}) -> "
          f"kandidat program pendampingan akademik.")
    kat_ipk = pd.cut(df["IPK"], [0, 2.75, 3.0, 3.5, 4.01],
                     labels=["< 2,76", "2,76-3,00 (Memuaskan)", "3,01-3,50 (Sangat memuaskan)",
                             "> 3,50 (Cumlaude)"])
    tulis("Predikat IPK: " + ", ".join(f"{k}: {v}" for k, v in kat_ipk.value_counts().sort_index().items()))

    judul("2.5 IPK per jurusan & per semester", "-")
    gj = df.groupby("Jurusan")["IPK"].agg(["count", "mean", "median", "std"]).sort_values("median", ascending=False)
    with pd.option_context("display.float_format", "{:.2f}".format):
        tulis(gj.to_string())
    kw = stats.kruskal(*[g["IPK"] for _, g in df.groupby("Jurusan")])
    tulis(f"Kruskal-Wallis IPK antar-jurusan: H = {kw.statistic:.2f}, p = {kw.pvalue:.1e} -> "
          f"{'ADA perbedaan signifikan' if kw.pvalue < .05 else 'tidak berbeda signifikan'}.")
    rs = stats.spearmanr(df["Semester"], df["IPK"])
    tulis(f"Spearman semester-IPK: rho = {rs.statistic:.3f} (p = {rs.pvalue:.1e}).")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    urut = gj.index.tolist()
    sns.boxplot(data=df, y="Jurusan", x="IPK", order=urut, ax=ax[0], color="#FBD3A8")
    ax[0].set(title="Sebaran IPK per jurusan", ylabel="")
    sns.pointplot(data=df, x="Semester", y="IPK", ax=ax[1], color=ORANYE, errorbar=("ci", 95))
    ax[1].set_title("Rata-rata IPK per semester (95% CI)")
    simpan(fig, F, "03_ipk_jurusan_semester.png")

    judul("2.6 Hubungan usia - semester", "-")
    rs2 = stats.spearmanr(df["Semester"], df["Usia"])
    reg = df[df["Kelas"] == "Reguler"]
    rs3 = stats.spearmanr(reg["Semester"], reg["Usia"])
    tulis(f"Spearman semester-usia (semua): rho = {rs2.statistic:.3f}")
    tulis(f"Spearman semester-usia (kelas reguler saja): rho = {rs3.statistic:.3f}")
    tulis("-> Hubungan kuat tertutup oleh mahasiswa kelas karyawan. Ini contoh pentingnya")
    tulis("   memisahkan kelompok sebelum menyimpulkan korelasi.")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.stripplot(data=df, x="Semester", y="Usia", hue="Kelas", jitter=.25, alpha=.6,
                  palette={"Reguler": BIRU, "Karyawan": ORANYE}, ax=ax)
    ax.set_title("Usia per semester, dibedakan kelas reguler & karyawan")
    simpan(fig, F, "04_usia_semester.png")

    judul("2.7 Kesimpulan kasus 2", "-")
    tulis(f"1. Jurusan terbanyak {fj.index[0]}; mahasiswa makin sedikit di semester atas.")
    tulis(f"2. Median IPK {df['IPK'].median():.2f}; IPK tertinggi di {gj.index[0]}, terendah di "
          f"{gj.index[-1]} (berbeda signifikan).")
    tulis("3. Outlier usia berasal dari kelas karyawan: valid dan informatif, jangan dihapus.")
    tulis("4. Semester diolah sebagai ordinal: bar chart berurutan, ringkas dengan median/modus.")


# =====================================================================
def kasus3() -> None:
    judul("BEDAH KASUS 3: DATASET PENGAMATAN CUACA")
    df = pd.read_csv("data_kasus3_cuaca.csv", parse_dates=["Tanggal"])
    S, H = "Suhu (°C)", "Kelembaban (%)"
    F = "kasus3"

    judul("3.1 Profil & klasifikasi tipe data", "-")
    tulis(f"{len(df):,} pengamatan | {df['Kota'].nunique()} kota | {df['Kondisi'].nunique()} kondisi")
    klasifikasi([("Kota", "Kategorik", "Nominal", "Bar / Pie"),
                 ("Suhu (°C)", "Numerik", "Interval", "Histogram / Box plot"),
                 ("Kelembaban (%)", "Numerik", "Rasio", "Histogram / Box plot"),
                 ("Kondisi", "Kategorik", "Nominal", "Bar / Pie")])
    kualitas(df, "ID Pengamatan")

    judul("3.2 Bukti interval vs rasio", "-")
    a, b = 20.0, 10.0
    tulis(f"Celsius   : {a:.0f} / {b:.0f} = {a/b:.2f}  (terlihat '2x lebih panas')")
    tulis(f"Fahrenheit: {a*9/5+32:.0f} / {b*9/5+32:.0f} = {(a*9/5+32)/(b*9/5+32):.2f}")
    tulis(f"Kelvin    : {a+273.15:.2f} / {b+273.15:.2f} = {(a+273.15)/(b+273.15):.3f}")
    tulis("-> Rasio berubah tergantung satuan karena 0°C bukan 'tanpa suhu'. Selisih tetap")
    tulis("   bermakna (10°C = 18°F di skala mana pun). Kelembaban 0% benar-benar tidak ada uap")
    tulis("   air, jadi 80% memang dua kali 40%.")

    judul("3.3 Statistik deskriptif", "-")
    statistik(df, [S, H])
    tulis("(CV pada Suhu tidak layak ditafsirkan karena CV butuh nol absolut / skala rasio.)")
    per_kota = df.groupby("Kota").agg(Suhu=(S, "mean"), Kelembaban=(H, "mean"),
                                      Pengamatan=(S, "size")).sort_values("Suhu", ascending=False)
    with pd.option_context("display.float_format", "{:.1f}".format):
        tulis(per_kota.to_string())

    judul("3.4 Kondisi cuaca (nominal)", "-")
    fk = df["Kondisi"].value_counts()
    for k, v in fk.items():
        tulis(f"  {k:<12} {v:>4} ({v/len(df):.1%})")
    ct = pd.crosstab(df["Kota"], df["Kondisi"])
    chi = stats.chi2_contingency(ct)
    v = np.sqrt(chi.statistic / (ct.values.sum() * (min(ct.shape) - 1)))
    tulis(f"Chi-square Kota x Kondisi: chi2 = {chi.statistic:.1f}, p = {chi.pvalue:.1e}, "
          f"Cramer's V = {v:.2f}")
    pct_hujan = (ct[["Hujan", "Hujan Petir"]].sum(axis=1) / ct.sum(axis=1) * 100).sort_values(ascending=False)
    tulis(f"Kota paling sering hujan: {pct_hujan.index[0]} ({pct_hujan.iloc[0]:.0f}%), "
          f"paling jarang: {pct_hujan.index[-1]} ({pct_hujan.iloc[-1]:.0f}%)")

    warna_kondisi = {"Cerah": "#F2B705", "Berawan": "#9AA5B1", "Hujan": "#1D8FD1", "Hujan Petir": "#0B4F7A"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].pie(fk, labels=fk.index, autopct="%1.0f%%", startangle=90,
              colors=[warna_kondisi[k] for k in fk.index], pctdistance=.78,
              wedgeprops={"edgecolor": "white", "linewidth": 2, "width": .45})
    ax[0].set_title("Proporsi kondisi cuaca (pie chart)")
    (ct.div(ct.sum(axis=1), axis=0) * 100)[list(warna_kondisi)].plot.barh(
        stacked=True, ax=ax[1], color=list(warna_kondisi.values()), width=.75)
    ax[1].set(title="Komposisi kondisi per kota (%)", ylabel="", xlabel="%")
    ax[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.1), frameon=False)
    simpan(fig, F, "01_kondisi_kota.png")

    judul("3.5 Distribusi suhu & kelembaban", "-")
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    sns.histplot(df[S], bins=35, kde=True, ax=ax[0, 0], color=ORANYE)
    ax[0, 0].set_title("Histogram suhu (bimodal: dataran rendah vs tinggi)")
    sns.histplot(df[H], bins=30, kde=True, ax=ax[0, 1], color=ORANYE)
    ax[0, 1].set_title("Histogram kelembaban")
    urut = per_kota.index.tolist()
    sns.boxplot(data=df, y="Kota", x=S, order=urut, ax=ax[1, 0], color="#FBD3A8")
    ax[1, 0].set(title="Box plot suhu per kota", ylabel="")
    sns.boxplot(data=df, y="Kota", x=H, order=urut, ax=ax[1, 1], color="#FBD3A8")
    ax[1, 1].set(title="Box plot kelembaban per kota", ylabel="")
    simpan(fig, F, "02_distribusi_suhu_lembab.png")
    tinggi = df[df["Kota"].isin(["Bandung", "Malang"])][S].median()
    rendah = df[~df["Kota"].isin(["Bandung", "Malang"])][S].median()
    tulis(f"Suhu bimodal: Bandung & Malang (dataran tinggi) berpusat di ~{tinggi:.0f}°C, kota lain")
    tulis(f"~{rendah:.0f}°C. Rata-rata gabungan menyesatkan; analisis sebaiknya per kota.")

    judul("3.6 Hubungan suhu - kelembaban - kondisi", "-")
    pr = stats.pearsonr(df[S], df[H])
    tulis(f"Pearson suhu-kelembaban: r = {pr.statistic:.3f} (p = {pr.pvalue:.1e}) -> makin lembab, "
          f"suhu cenderung makin rendah.")
    gk = df.groupby("Kondisi")[[S, H]].median().loc[list(warna_kondisi)]
    with pd.option_context("display.float_format", "{:.1f}".format):
        tulis(gk.to_string())
    kw = stats.kruskal(*[g[H] for _, g in df.groupby("Kondisi")])
    tulis(f"Kruskal-Wallis kelembaban antar-kondisi: p = {kw.pvalue:.1e} -> kelembaban berbeda "
          f"nyata antara cerah dan hujan.")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(data=df, x=H, y=S, hue="Kondisi", palette=warna_kondisi,
                    hue_order=list(warna_kondisi), alpha=.65, ax=ax[0])
    ax[0].set_title(f"Kelembaban vs suhu (r = {pr.statistic:.2f})")
    sns.boxplot(data=df, x="Kondisi", y=H, order=list(warna_kondisi), hue="Kondisi",
                palette=warna_kondisi, legend=False, ax=ax[1])
    ax[1].set_title("Kelembaban menurut kondisi cuaca")
    simpan(fig, F, "03_hubungan_suhu_lembab_kondisi.png")

    judul("3.7 Pola musiman", "-")
    df["Bulan"] = df["Tanggal"].dt.month
    bl = df.groupby("Bulan").agg(Suhu=(S, "mean"), Kelembaban=(H, "mean"),
                                 Hujan=("Kondisi", lambda s: s.isin(["Hujan", "Hujan Petir"]).mean() * 100))
    with pd.option_context("display.float_format", "{:.1f}".format):
        tulis(bl.to_string())
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(bl.index, bl["Kelembaban"], color=ORANYE, marker="o", lw=2.5, label="Kelembaban (%)")
    a2 = ax.twinx()
    a2.plot(bl.index, bl["Suhu"], color=ABU, marker="s", lw=2.5, label="Suhu (°C)")
    a2.grid(False)
    ax.set_xticks(range(1, 13))
    ax.set(title="Pola musiman kelembaban & suhu", xlabel="Bulan", ylabel="Kelembaban (%)")
    a2.set_ylabel("Suhu (°C)")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, .02), frameon=False)
    simpan(fig, F, "04_musiman.png")

    judul("3.8 Kesimpulan kasus 3", "-")
    tulis(f"1. Kota terpanas {per_kota.index[0]}, tersejuk {per_kota.index[-1]}.")
    tulis(f"2. Suhu & kelembaban berkorelasi negatif (r = {pr.statistic:.2f}); hujan terjadi saat "
          f"kelembaban tinggi.")
    tulis("3. Musim hujan (Nov-Mar) menaikkan kelembaban dan frekuensi hujan.")
    tulis("4. Suhu skala interval: hanya selisih yang bermakna, jangan gunakan rasio atau CV.")


if __name__ == "__main__":
    if not Path("data_kasus1_penjualan.csv").exists():
        import generate_data
        for nama, f in generate_data.FILES.items():
            f().to_csv(nama, index=False, date_format="%Y-%m-%d")
    kasus1()
    kasus2()
    kasus3()
    Path("laporan_analisis.txt").write_text("\n".join(laporan), encoding="utf-8")
    print("\n[OK] laporan_analisis.txt dan grafik di folder output_analisis/")
