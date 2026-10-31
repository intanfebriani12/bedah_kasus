"""
app.py — Dashboard Streamlit: Bedah Kasus 1, 2, dan 3
Jalankan:  python -m streamlit run app.py

Warna mengikuti tipe data di slide:
  Biru   = Kategorik (Nominal / Ordinal)
  Oranye = Numerik (Interval / Rasio)
  Abu    = Waktu
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Bedah Kasus Dataset", page_icon="📊", layout="wide")

BIRU, BIRU_TUA, ORANYE, ABU, TINTA = "#1D8FD1", "#0B4F7A", "#F28C28", "#7A7F87", "#1F2A37"
SKALA_BIRU = ["#0B4F7A", "#1D8FD1", "#5DB7E8", "#9FD3F2", "#2A7FA8", "#CDE9F9"]
WARNA_KONDISI = {"Cerah": "#F2B705", "Berawan": "#9AA5B1", "Hujan": "#1D8FD1", "Hujan Petir": "#0B4F7A"}
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
DIR = Path(__file__).parent

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"], .stMarkdown {{ font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 2rem; max-width: 1350px; }}
h1 {{ font-weight: 800; letter-spacing: -0.02em; color: {TINTA}; }}
.tipe {{ border-left: 6px solid; padding: .4rem .9rem; margin: .3rem 0 1rem;
         border-radius: 0 8px 8px 0; font-size: .93rem; line-height: 1.5; }}
.kat {{ border-color: {BIRU}; background: #EAF5FC; }}
.num {{ border-color: {ORANYE}; background: #FEF3E8; }}
.wkt {{ border-color: {ABU}; background: #F1F2F4; }}
.insight {{ background: #fff; border: 1px solid #E3E6EA; border-radius: 10px;
            padding: .75rem 1rem; margin-bottom: .55rem; line-height: 1.5; }}
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #E3E6EA;
            border-radius: 10px; padding: .7rem 1rem; }}
div[data-testid="stMetricValue"] {{ font-weight: 800; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------- helper ---
def rupiah(x: float, singkat: bool = False) -> str:
    if singkat:
        for b, s in ((1e12, "T"), (1e9, "M"), (1e6, "jt")):
            if abs(x) >= b:
                return f"Rp{x/b:,.2f} {s}"
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def angka(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def label(kelas: str, teks: str) -> None:
    st.markdown(f'<div class="tipe {kelas}">{teks}</div>', unsafe_allow_html=True)


def insight(daftar: list[str]) -> None:
    for t in daftar:
        st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)


def rapi(fig: go.Figure, tinggi: int = 380) -> go.Figure:
    fig.update_layout(height=tinggi, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Plus Jakarta Sans, Segoe UI, sans-serif", size=13),
                      title_font=dict(size=16), plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF0F2")
    return fig


def tampil(fig: go.Figure, tinggi: int = 380) -> None:
    st.plotly_chart(rapi(fig, tinggi), width="stretch")


def iqr_flag(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([.25, .75])
    return (s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))


def tabel_klasifikasi(rows: list[tuple]) -> None:
    t = pd.DataFrame(rows, columns=["Kolom", "Tipe data", "Skala", "Alasan", "Visualisasi"])
    warna = {"Kategorik": "#EAF5FC", "Numerik": "#FEF3E8", "Waktu": "#F1F2F4"}
    st.dataframe(t.style.apply(lambda r: [f"background-color: {warna[r['Tipe data'].split()[0]]}"]
                               * len(r), axis=1), hide_index=True, width="stretch")


def statistik(df: pd.DataFrame, cols: list[str], cv: bool = True) -> None:
    num = df[cols]
    t = pd.DataFrame({"Rata-rata": num.mean(), "Median": num.median(), "Std. deviasi": num.std(),
                      "Min": num.min(), "Maks": num.max(), "Skewness": num.skew()})
    if cv:
        t["CV (%)"] = num.std() / num.mean() * 100
    st.dataframe(t.style.format("{:,.2f}"), width="stretch")


def tab_data(df: pd.DataFrame, nama_file: str, id_col: str) -> None:
    q1, q2, q3 = st.columns(3)
    q1.metric("Nilai kosong", int(df.isna().sum().sum()))
    q2.metric("Baris duplikat", int(df.drop(columns=id_col).duplicated().sum()))
    q3.metric("Jumlah baris", angka(len(df)))
    st.dataframe(df, hide_index=True, width="stretch", height=420)
    st.download_button("Unduh data terfilter (CSV)", df.to_csv(index=False).encode("utf-8"),
                       nama_file, "text/csv")


@st.cache_data
def muat(nama: str) -> pd.DataFrame:
    path = DIR / nama
    if not path.exists():  # buat otomatis jika CSV belum ada
        import generate_data
        generate_data.FILES[nama]().to_csv(path, index=False, date_format="%Y-%m-%d")
    df = pd.read_csv(path)
    if "Tanggal" in df:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    return df


def kosong() -> None:
    st.warning("Tidak ada data yang cocok dengan filter. Perluas pilihan filter di sidebar.")
    st.stop()


# =================================================================== UI ===
with st.sidebar:
    st.header("Bedah kasus")
    kasus = st.radio("Pilih dataset", ["1 · Operasional penjualan", "2 · Profil mahasiswa",
                                       "3 · Pengamatan cuaca"], label_visibility="collapsed")
    st.divider()
    st.subheader("Filter")


# ======================================================== KASUS 1 ========
def kasus1() -> None:
    df_all = muat("data_kasus1_penjualan.csv")
    df_all["Total (Rp)"] = df_all["Harga (Rp)"] * df_all["Jumlah Terjual"]
    with st.sidebar:
        kat = st.multiselect("Kategori", sorted(df_all["Kategori"].unique()),
                             default=sorted(df_all["Kategori"].unique()))
        tmin, tmax = df_all["Tanggal"].min().date(), df_all["Tanggal"].max().date()
        rg = st.date_input("Rentang tanggal", (tmin, tmax), min_value=tmin, max_value=tmax)
        promo = st.radio("Status harga", ["Semua", "Promo", "Normal"], horizontal=True)
    t0, t1 = rg if isinstance(rg, tuple) and len(rg) == 2 else (tmin, tmax)
    m = df_all["Kategori"].isin(kat) & df_all["Tanggal"].dt.date.between(t0, t1)
    if promo != "Semua":
        m &= df_all["Promo"] == (promo == "Promo")
    df = df_all[m].copy()

    st.title("Bedah Kasus 1: Dataset Operasional Penjualan")
    st.write(f"{angka(len(df))} dari {angka(len(df_all))} transaksi sintetis, "
             f"{t0:%d/%m/%Y} – {t1:%d/%m/%Y}.")
    if df.empty:
        kosong()

    pend = df["Total (Rp)"].sum()
    c = st.columns(4)
    c[0].metric("Total pendapatan", rupiah(pend, True))
    c[1].metric("Unit terjual", angka(df["Jumlah Terjual"].sum()))
    c[2].metric("Median harga", rupiah(df["Harga (Rp)"].median()))
    c[3].metric("Rata-rata nilai transaksi", rupiah(df["Total (Rp)"].mean(), True))

    t = st.tabs(["Ringkasan", "Kategorik", "Numerik", "Waktu", "Hubungan & promo", "Data"])
    with t[0]:
        tabel_klasifikasi([
            ("Produk", "Kategorik", "Nominal", "Label tanpa urutan", "Bar / Pie"),
            ("Kategori", "Kategorik", "Nominal", "Label tanpa urutan", "Bar / Pie"),
            ("Harga (Rp)", "Numerik kontinu", "Rasio", "Punya nol absolut", "Histogram"),
            ("Jumlah Terjual", "Numerik diskrit", "Rasio", "Hasil hitung barang (bulat)", "Bar / Histogram"),
            ("Tanggal", "Waktu", "Interval", "Selisih bermakna, rasio tidak", "Line chart")])
        pk = df.groupby("Kategori")["Total (Rp)"].sum().sort_values(ascending=False)
        uk = df.groupby("Kategori")["Jumlah Terjual"].sum().sort_values(ascending=False)
        pp = df.groupby("Produk")["Total (Rp)"].sum().sort_values(ascending=False)
        n80 = int((pp.cumsum() / pp.sum() < .8).sum() + 1)
        bl = df.groupby(df["Tanggal"].dt.month)["Total (Rp)"].sum()
        insight([f"<b>{pk.index[0]}</b> menyumbang <b>{pk.iloc[0]/pend:.1%}</b> pendapatan, sedangkan "
                 f"volume unit terbesar dari <b>{uk.index[0]}</b>.",
                 f"<b>{n80} dari {len(pp)} produk</b> menghasilkan ~80% pendapatan (prinsip Pareto).",
                 f"Bulan terbaik <b>{BULAN[bl.idxmax()-1]}</b>, terlemah <b>{BULAN[bl.idxmin()-1]}</b>."])
    with t[1]:
        label("kat", "<b>Kategorik (Nominal)</b> — Produk & Kategori dibandingkan lewat bar dan pie chart.")
        ukuran = st.radio("Ukuran", ["Pendapatan", "Unit terjual", "Jumlah transaksi"], horizontal=True)
        agg = {"Pendapatan": ("Total (Rp)", "sum"), "Unit terjual": ("Jumlah Terjual", "sum"),
               "Jumlah transaksi": ("Produk", "size")}[ukuran]
        a = df.groupby("Kategori").agg(nilai=agg).sort_values("nilai").reset_index()
        k1, k2 = st.columns([3, 2])
        with k1:
            f = px.bar(a, x="nilai", y="Kategori", orientation="h", text_auto=".3s",
                       title=f"{ukuran} per kategori", color_discrete_sequence=[BIRU])
            f.update_layout(xaxis_title=ukuran, yaxis_title="")
            tampil(f)
        with k2:
            f = px.pie(a, names="Kategori", values="nilai", hole=.55, title="Proporsi",
                       color_discrete_sequence=SKALA_BIRU)
            tampil(f)
        pp = df.groupby("Produk")["Total (Rp)"].sum().sort_values(ascending=False)
        kum = pp.cumsum() / pp.sum() * 100
        f = go.Figure()
        f.add_bar(x=pp.index, y=pp.values, name="Pendapatan", marker_color=BIRU)
        f.add_scatter(x=pp.index, y=kum, name="Kumulatif %", yaxis="y2", mode="lines+markers",
                      line=dict(color=ORANYE, width=3))
        f.add_hline(y=80, line_dash="dash", line_color=ABU, yref="y2")
        f.update_layout(title="Diagram Pareto pendapatan per produk",
                        yaxis2=dict(overlaying="y", side="right", range=[0, 105], ticksuffix="%",
                                    showgrid=False))
        tampil(f, 420)
    with t[2]:
        label("num", "<b>Numerik (Rasio)</b> — Harga kontinu, Jumlah Terjual diskrit; keduanya punya nol absolut.")
        statistik(df, ["Harga (Rp)", "Jumlah Terjual", "Total (Rp)"])
        k1, k2 = st.columns(2)
        with k1:
            log = st.toggle("Skala log untuk harga", value=True)
            x = np.log10(df["Harga (Rp)"]) if log else df["Harga (Rp)"]
            f = px.histogram(x=x, nbins=40, title="Histogram harga", color_discrete_sequence=[ORANYE])
            f.update_layout(xaxis_title="log10(Harga)" if log else "Harga (Rp)", yaxis_title="Frekuensi")
            tampil(f)
        with k2:
            f = px.histogram(df, x="Jumlah Terjual", nbins=50, title="Distribusi jumlah terjual",
                             color_discrete_sequence=[ORANYE])
            tampil(f)
        df["Outlier"] = df.groupby("Produk")["Jumlah Terjual"].transform(iqr_flag)
        st.markdown(f"**{int(df['Outlier'].sum())} transaksi outlier** (IQR per produk) — "
                    "kemungkinan pembelian borongan.")
    with t[3]:
        label("wkt", "<b>Waktu (Interval)</b> — tren dibaca dengan line chart.")
        g = st.radio("Agregasi", ["Harian", "Mingguan", "Bulanan"], index=2, horizontal=True)
        ts = df.set_index("Tanggal")["Total (Rp)"].resample({"Harian": "D", "Mingguan": "W-MON",
                                                             "Bulanan": "MS"}[g]).sum().reset_index()
        f = px.line(ts, x="Tanggal", y="Total (Rp)", markers=g != "Harian",
                    title=f"Tren pendapatan {g.lower()}", color_discrete_sequence=[ABU])
        tampil(f, 400)
        tk = df.groupby([df["Tanggal"].dt.to_period("M").dt.to_timestamp(), "Kategori"])["Jumlah Terjual"].sum().reset_index()
        f = px.line(tk, x="Tanggal", y="Jumlah Terjual", color="Kategori", markers=True,
                    title="Unit terjual bulanan per kategori")
        tampil(f, 400)
    with t[4]:
        if len(df) > 10:
            rho = stats.spearmanr(df["Harga (Rp)"], df["Jumlah Terjual"]).statistic
            st.metric("Korelasi Spearman harga–unit", f"{rho:.3f}")
            f = px.scatter(df, x="Harga (Rp)", y="Jumlah Terjual", color="Kategori", log_x=True,
                           log_y=True, opacity=.65, title="Harga vs jumlah terjual (log-log)")
            tampil(f, 450)
        df["Indeks unit"] = df["Jumlah Terjual"] / df.groupby("Produk")["Jumlah Terjual"].transform("median")
        pr, nm = df.loc[df["Promo"], "Indeks unit"], df.loc[~df["Promo"], "Indeks unit"]
        if len(pr) > 3 and len(nm) > 3:
            p = stats.mannwhitneyu(pr, nm, alternative="greater").pvalue
            st.metric("Kenaikan unit saat promo", f"{pr.median()/nm.median()-1:+.0%}",
                      "signifikan (p < 0,05)" if p < .05 else "tidak signifikan")
            st.caption("Indeks unit = jumlah terjual ÷ median produk itu sendiri. Uji: Mann-Whitney U.")
    with t[5]:
        tab_data(df.drop(columns=["Outlier", "Indeks unit"], errors="ignore"),
                 "kasus1_terfilter.csv", "ID Transaksi")


# ======================================================== KASUS 2 ========
def kasus2() -> None:
    df_all = muat("data_kasus2_mahasiswa.csv")
    with st.sidebar:
        jur = st.multiselect("Jurusan", sorted(df_all["Jurusan"].unique()),
                             default=sorted(df_all["Jurusan"].unique()))
        sem = st.slider("Semester", 1, 8, (1, 8))
        kelas = st.radio("Kelas", ["Semua", "Reguler", "Karyawan"], horizontal=True)
    m = df_all["Jurusan"].isin(jur) & df_all["Semester"].between(*sem)
    if kelas != "Semua":
        m &= df_all["Kelas"] == kelas
    df = df_all[m].copy()

    st.title("Bedah Kasus 2: Dataset Profil Mahasiswa")
    st.write(f"{angka(len(df))} dari {angka(len(df_all))} mahasiswa sintetis.")
    if df.empty:
        kosong()

    c = st.columns(4)
    c[0].metric("Jumlah mahasiswa", angka(len(df)))
    c[1].metric("Median IPK", f"{df['IPK'].median():.2f}")
    c[2].metric("IPK > 3,50", f"{(df['IPK'] > 3.5).mean():.1%}")
    c[3].metric("Median usia", f"{df['Usia'].median():.0f} th")

    t = st.tabs(["Ringkasan", "Jurusan & semester", "IPK & usia", "Perbandingan", "Data"])
    with t[0]:
        tabel_klasifikasi([
            ("Jurusan", "Kategorik", "Nominal", "Hanya label pembeda", "Bar / Pie"),
            ("Semester", "Kategorik", "Ordinal", "Ada tingkatan 1 → 2 → 3", "Bar chart berurutan"),
            ("IPK", "Numerik kontinu", "Interval", "Rentang desimal 0–4", "Histogram / Box plot"),
            ("Usia", "Numerik kontinu", "Rasio", "Rentang nilai, nol absolut", "Histogram / Box plot")])
        gj = df.groupby("Jurusan")["IPK"].median().sort_values(ascending=False)
        out = iqr_flag(df["Usia"])
        isi = [f"Jurusan dengan median IPK tertinggi <b>{gj.index[0]}</b> ({gj.iloc[0]:.2f}), "
               f"terendah <b>{gj.index[-1]}</b> ({gj.iloc[-1]:.2f}).",
               f"Semester dibaca sebagai <b>ordinal</b>: median {df['Semester'].median():.0f}, "
               f"modus {df['Semester'].mode()[0]}. Rata-rata semester kurang bermakna."]
        if out.any():
            isi.append(f"<b>{int(out.sum())} outlier usia</b> ({df.loc[out, 'Usia'].min()}–"
                       f"{df.loc[out, 'Usia'].max()} th), mayoritas dari kelas "
                       f"<b>{df.loc[out, 'Kelas'].mode()[0]}</b>. Valid, bukan salah input.")
        if df["Jurusan"].nunique() > 1:
            p = stats.kruskal(*[g["IPK"] for _, g in df.groupby("Jurusan")]).pvalue
            isi.append(f"Uji Kruskal-Wallis IPK antar-jurusan: p = {p:.1e} → "
                       f"{'<b>berbeda signifikan</b>' if p < .05 else 'tidak berbeda signifikan'}.")
        insight(isi)
    with t[1]:
        k1, k2 = st.columns(2)
        with k1:
            label("kat", "<b>Kategorik (Nominal)</b> — Jurusan hanya label pembeda, urutan bebas.")
            fj = df["Jurusan"].value_counts().sort_values().reset_index()
            f = px.bar(fj, x="count", y="Jurusan", orientation="h", text_auto=True,
                       title="Mahasiswa per jurusan", color_discrete_sequence=[BIRU])
            f.update_layout(xaxis_title="Mahasiswa", yaxis_title="")
            tampil(f)
            f = px.pie(fj, names="Jurusan", values="count", hole=.55, title="Proporsi jurusan",
                       color_discrete_sequence=SKALA_BIRU)
            tampil(f)
        with k2:
            label("kat", "<b>Kategorik (Ordinal)</b> — Semester punya urutan, jadi sumbu X wajib berurutan.")
            fs = df["Semester"].value_counts().reindex(range(1, 9), fill_value=0).reset_index()
            fs["Semester"] = "Semester " + fs["Semester"].astype(str)
            f = px.bar(fs, x="Semester", y="count", text_auto=True,
                       title="Mahasiswa per semester (berurutan)", color_discrete_sequence=[BIRU_TUA])
            f.update_layout(yaxis_title="Mahasiswa", xaxis_title="")
            tampil(f)
            st.caption("Bar chart berurutan memperlihatkan jumlah mahasiswa yang menurun di semester "
                       "atas (lulus, cuti, atau keluar). Mengurutkan bar dari terbesar ke terkecil "
                       "di sini justru menghilangkan informasi tingkatan.")
            ct = pd.crosstab(df["Jurusan"], df["Semester"])
            f = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                          title="Jurusan × semester")
            tampil(f, 360)
    with t[2]:
        label("num", "<b>Numerik (Kontinu)</b> — IPK dan Usia berupa rentang. Histogram untuk distribusi, "
              "box plot untuk sebaran & outlier.")
        statistik(df, ["IPK", "Usia"])
        k1, k2 = st.columns(2)
        with k1:
            f = px.histogram(df, x="IPK", nbins=30, title="Histogram IPK", marginal="box",
                             color_discrete_sequence=[ORANYE])
            f.add_vline(x=df["IPK"].median(), line_dash="dash", line_color=TINTA, annotation_text="median")
            tampil(f, 420)
        with k2:
            f = px.histogram(df, x="Usia", title="Histogram usia", marginal="box",
                             color_discrete_sequence=[ORANYE])
            f.update_traces(xbins=dict(size=1), selector=dict(type="histogram"))
            tampil(f, 420)
        pred = pd.cut(df["IPK"], [0, 2.75, 3.0, 3.5, 4.01],
                      labels=["≤ 2,75", "2,76–3,00 Memuaskan", "3,01–3,50 Sangat memuaskan",
                              "> 3,50 Cumlaude"]).value_counts().sort_index().reset_index()
        f = px.bar(pred, x="IPK", y="count", text_auto=True, title="Predikat IPK (dikelompokkan)",
                   color_discrete_sequence=[ORANYE])
        f.update_layout(xaxis_title="", yaxis_title="Mahasiswa")
        tampil(f, 340)
    with t[3]:
        urut = df.groupby("Jurusan")["IPK"].median().sort_values().index.tolist()
        f = px.box(df, x="IPK", y="Jurusan", category_orders={"Jurusan": urut[::-1]},
                   title="Sebaran IPK per jurusan", color_discrete_sequence=[ORANYE])
        tampil(f, 420)
        k1, k2 = st.columns(2)
        with k1:
            g = df.groupby("Semester")["IPK"].agg(["mean", "sem"]).reset_index()
            f = go.Figure(go.Scatter(x=g["Semester"], y=g["mean"], mode="lines+markers",
                                     error_y=dict(array=1.96 * g["sem"].fillna(0)),
                                     line=dict(color=ORANYE, width=3)))
            f.update_layout(title="Rata-rata IPK per semester (95% CI)", xaxis_title="Semester",
                            yaxis_title="IPK", xaxis=dict(dtick=1))
            tampil(f)
        with k2:
            f = px.strip(df, x="Semester", y="Usia", color="Kelas",
                         color_discrete_map={"Reguler": BIRU, "Karyawan": ORANYE},
                         title="Usia per semester menurut kelas")
            tampil(f)
        reg = df[df["Kelas"] == "Reguler"]
        if len(reg) > 10 and len(df) > 10:
            st.markdown(f"Korelasi Spearman semester–usia: **{stats.spearmanr(df['Semester'], df['Usia']).statistic:.2f}** "
                        f"(semua) vs **{stats.spearmanr(reg['Semester'], reg['Usia']).statistic:.2f}** "
                        "(kelas reguler saja). Kelas karyawan menyamarkan hubungan usia–semester.")
    with t[4]:
        tab_data(df, "kasus2_terfilter.csv", "ID Mahasiswa")


# ======================================================== KASUS 3 ========
def kasus3() -> None:
    df_all = muat("data_kasus3_cuaca.csv")
    S, H = "Suhu (°C)", "Kelembaban (%)"
    with st.sidebar:
        kota = st.multiselect("Kota", sorted(df_all["Kota"].unique()),
                              default=sorted(df_all["Kota"].unique()))
        kond = st.multiselect("Kondisi", list(WARNA_KONDISI), default=list(WARNA_KONDISI))
        bln = st.select_slider("Bulan", options=BULAN, value=("Jan", "Des"))
    b0, b1 = BULAN.index(bln[0]) + 1, BULAN.index(bln[1]) + 1
    m = (df_all["Kota"].isin(kota) & df_all["Kondisi"].isin(kond)
         & df_all["Tanggal"].dt.month.between(b0, b1))
    df = df_all[m].copy()

    st.title("Bedah Kasus 3: Dataset Pengamatan Cuaca")
    st.write(f"{angka(len(df))} dari {angka(len(df_all))} pengamatan sintetis, {bln[0]}–{bln[1]} 2024.")
    if df.empty:
        kosong()

    c = st.columns(4)
    c[0].metric("Pengamatan", angka(len(df)))
    c[1].metric("Median suhu", f"{df[S].median():.1f} °C")
    c[2].metric("Median kelembaban", f"{df[H].median():.0f}%")
    c[3].metric("Hari hujan", f"{df['Kondisi'].isin(['Hujan', 'Hujan Petir']).mean():.0%}")

    t = st.tabs(["Ringkasan", "Interval vs rasio", "Kategorik", "Numerik", "Hubungan & musim", "Data"])
    with t[0]:
        tabel_klasifikasi([
            ("Kota", "Kategorik", "Nominal", "Label teks tanpa urutan", "Bar / Pie"),
            ("Suhu (°C)", "Numerik", "Interval", "0°C ≠ tidak ada suhu", "Histogram / Box plot"),
            ("Kelembaban (%)", "Numerik", "Rasio", "0% = benar-benar tidak ada", "Histogram / Box plot"),
            ("Kondisi", "Kategorik", "Nominal", "Label teks tanpa urutan", "Bar / Pie")])
        pk = df.groupby("Kota")[S].mean().sort_values(ascending=False)
        isi = [f"Kota terpanas <b>{pk.index[0]}</b> ({pk.iloc[0]:.1f}°C), tersejuk "
               f"<b>{pk.index[-1]}</b> ({pk.iloc[-1]:.1f}°C)."]
        if len(df) > 10:
            r = stats.pearsonr(df[S], df[H]).statistic
            isi.append(f"Korelasi suhu–kelembaban <b>r = {r:.2f}</b>: makin lembab, suhu cenderung makin rendah.")
        gk = df.groupby("Kondisi")[H].median()
        if {"Cerah", "Hujan"} <= set(gk.index):
            isi.append(f"Median kelembaban saat cerah <b>{gk['Cerah']:.0f}%</b> vs saat hujan "
                       f"<b>{gk['Hujan']:.0f}%</b>.")
        insight(isi)
    with t[1]:
        label("num", "<b>Interval vs rasio</b> — coba ubah suhu dan lihat apakah '2× lebih panas' tetap berlaku.")
        k1, k2 = st.columns(2)
        a = k1.number_input("Suhu A (°C)", value=20.0, step=1.0)
        b = k2.number_input("Suhu B (°C)", value=10.0, step=1.0)
        if b != 0 and (b * 9 / 5 + 32) != 0:
            tbl = pd.DataFrame({
                "Skala": ["Celsius", "Fahrenheit", "Kelvin"],
                "A": [a, a * 9 / 5 + 32, a + 273.15], "B": [b, b * 9 / 5 + 32, b + 273.15]})
            tbl["Selisih A − B"] = tbl["A"] - tbl["B"]
            tbl["Rasio A ÷ B"] = tbl["A"] / tbl["B"]
            st.dataframe(tbl.style.format({"A": "{:.2f}", "B": "{:.2f}", "Selisih A − B": "{:.2f}",
                                           "Rasio A ÷ B": "{:.3f}"}), hide_index=True, width="stretch")
            st.markdown("Rasio berubah di setiap skala karena **0°C bukan berarti tidak ada suhu**. "
                        "Itu ciri skala **interval**: yang bermakna hanya selisih. "
                        "Kelembaban berskala **rasio**: 0% berarti tidak ada uap air, sehingga 80% "
                        "memang dua kali 40%.")
        else:
            st.info("Isi suhu B selain 0 agar rasio bisa dihitung.")
    with t[2]:
        label("kat", "<b>Kategorik (Nominal)</b> — Kota dan Kondisi adalah label teks tanpa urutan.")
        k1, k2 = st.columns([2, 3])
        with k1:
            fk = df["Kondisi"].value_counts().reset_index()
            f = px.pie(fk, names="Kondisi", values="count", hole=.55, title="Proporsi kondisi cuaca",
                       color="Kondisi", color_discrete_map=WARNA_KONDISI)
            tampil(f)
        with k2:
            fkota = df["Kota"].value_counts().sort_values().reset_index()
            f = px.bar(fkota, x="count", y="Kota", orientation="h", text_auto=True,
                       title="Jumlah pengamatan per kota", color_discrete_sequence=[BIRU])
            f.update_layout(xaxis_title="Pengamatan", yaxis_title="")
            tampil(f)
        ct = pd.crosstab(df["Kota"], df["Kondisi"], normalize="index").mul(100).reset_index()
        ct = ct.melt(id_vars="Kota", var_name="Kondisi", value_name="Persen")
        f = px.bar(ct, x="Persen", y="Kota", color="Kondisi", orientation="h",
                   color_discrete_map=WARNA_KONDISI, category_orders={"Kondisi": list(WARNA_KONDISI)},
                   title="Komposisi kondisi cuaca per kota (%)")
        tampil(f, 420)
        tab = pd.crosstab(df["Kota"], df["Kondisi"])
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            chi = stats.chi2_contingency(tab)
            v = np.sqrt(chi.statistic / (tab.values.sum() * (min(tab.shape) - 1)))
            st.markdown(f"Uji chi-square kota × kondisi: p = {chi.pvalue:.1e}, Cramér's V = {v:.2f} → "
                        f"{'kondisi cuaca **bergantung** pada kota' if chi.pvalue < .05 else 'tidak ada hubungan nyata'}.")
    with t[3]:
        label("num", "<b>Numerik</b> — Suhu (interval) dan Kelembaban (rasio).")
        statistik(df, [S, H], cv=False)
        st.caption("CV tidak ditampilkan karena tidak sah untuk suhu (skala interval, tanpa nol absolut).")
        k1, k2 = st.columns(2)
        with k1:
            f = px.histogram(df, x=S, nbins=35, title="Histogram suhu", color_discrete_sequence=[ORANYE])
            tampil(f)
        with k2:
            f = px.histogram(df, x=H, nbins=30, title="Histogram kelembaban", color_discrete_sequence=[ORANYE])
            tampil(f)
        st.caption("Histogram suhu tampak bimodal: Bandung dan Malang (dataran tinggi) membentuk "
                   "kelompok suhu sendiri. Saring kota di sidebar untuk melihatnya.")
        var = st.radio("Box plot per kota", [S, H], horizontal=True)
        urut = df.groupby("Kota")[var].median().sort_values(ascending=False).index.tolist()
        f = px.box(df, x="Kota", y=var, category_orders={"Kota": urut}, points="outliers",
                   title=f"Sebaran {var.lower()} per kota", color_discrete_sequence=[ORANYE])
        tampil(f, 400)
    with t[4]:
        if len(df) > 10:
            r = stats.pearsonr(df[S], df[H]).statistic
            f = px.scatter(df, x=H, y=S, color="Kondisi", color_discrete_map=WARNA_KONDISI,
                           category_orders={"Kondisi": list(WARNA_KONDISI)}, opacity=.65,
                           hover_data=["Kota", "Tanggal"],
                           title=f"Kelembaban vs suhu (r = {r:.2f})")
            tampil(f, 450)
        f = px.box(df, x="Kondisi", y=H, color="Kondisi", color_discrete_map=WARNA_KONDISI,
                   category_orders={"Kondisi": list(WARNA_KONDISI)}, title="Kelembaban menurut kondisi")
        f.update_layout(showlegend=False)
        tampil(f, 360)
        bl = df.groupby(df["Tanggal"].dt.month).agg(Suhu=(S, "mean"), Kelembaban=(H, "mean")).reset_index()
        bl["Bulan"] = bl["Tanggal"].map(lambda x: BULAN[x - 1])
        f = go.Figure()
        f.add_scatter(x=bl["Bulan"], y=bl["Kelembaban"], name="Kelembaban (%)", mode="lines+markers",
                      line=dict(color=ORANYE, width=3))
        f.add_scatter(x=bl["Bulan"], y=bl["Suhu"], name="Suhu (°C)", yaxis="y2", mode="lines+markers",
                      line=dict(color=ABU, width=3))
        f.update_layout(title="Pola musiman suhu & kelembaban", yaxis_title="Kelembaban (%)",
                        yaxis2=dict(overlaying="y", side="right", title="Suhu (°C)", showgrid=False))
        tampil(f, 380)
    with t[5]:
        tab_data(df, "kasus3_terfilter.csv", "ID Pengamatan")


{"1": kasus1, "2": kasus2, "3": kasus3}[kasus[0]]()
st.caption("Data sintetis 1.000 baris per kasus, dibuat dengan generate_data.py (seed 42).")
