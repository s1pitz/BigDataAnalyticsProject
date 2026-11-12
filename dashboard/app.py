import copy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import cube_utils as cu

ROOT = Path(__file__).parent
DATA, DEMO = ROOT / "data", ROOT / "data_demo"
RATING_COLORS = {"1": "#d73027", "2": "#fc8d59", "3": "#fee08b", "4": "#a6d96a", "5": "#1a9850"}
SCEN_COLORS = {"A": "#232F3E", "B": "#FF9900"}
# Konstanta dari 03_helpfulness.ipynb (cell 28): baseline "skor = panjang teks" pada data test.
BASELINE_TEXT_LEN_PRAUC = 0.490

st.set_page_config(page_title="Analisis Ulasan Amazon Video Games", page_icon="🛒", layout="wide")

# ------------------------------------------------------------------ tema oranye Amazon
ORANGE, NAVY = "#FF9900", "#232F3E"
COLORWAY = ["#FF9900", "#232F3E", "#146EB4", "#C45500", "#FEBD69", "#5B6B7F", "#E47911", "#0F8B8D"]
ORANGE_SEQ = ["#FFE0B2", "#FFCC80", "#FFB74D", "#FF9900", "#C45500", "#5C2A00"]

_tpl = copy.deepcopy(pio.templates["plotly_white"])
_tpl.layout.update(
    colorway=COLORWAY,
    font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", size=14, color="#232F3E"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    colorscale=dict(sequential=[[i / (len(ORANGE_SEQ) - 1), c] for i, c in enumerate(ORANGE_SEQ)]),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, bordercolor=ORANGE),
)
for _ax in (_tpl.layout.xaxis, _tpl.layout.yaxis):
    _ax.update(showgrid=True, gridcolor="#E5D2B5", gridwidth=1, showline=True, linecolor="#7B8794",
               zerolinecolor="#C9A56B", ticks="outside", tickcolor="#7B8794")
pio.templates["amazon"] = _tpl
pio.templates.default = "amazon"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, #FFEFD6 0%, #FFDDAA 100%); }
.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 4rem; }

/* judul */
.hero { background: linear-gradient(135deg, #131A22 0%, #232F3E 55%, #37475A 100%);
        border-bottom: 5px solid #FF9900;
        border-radius: 18px; padding: 28px 32px; margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(19, 26, 34, .35); }
.hero .t { color: #fff; font-size: 2rem; font-weight: 800; line-height: 1.25; margin: 0 0 .4rem 0; }
.hero .s { color: #FFE2B8; font-size: .95rem; margin: 0; }
h2 { color: #232F3E !important; font-weight: 700 !important; margin-top: 1.6rem !important;
     padding-bottom: .35rem; border-bottom: 2px solid #F0B866; }
h3 { color: #B45309 !important; font-weight: 600 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: #232F3E !important; opacity: 1 !important; font-size: .95rem; line-height: 1.5; }
hr { border-color: #F0B866 !important; }

/* tab */
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #F0B866; }
button[data-baseweb="tab"] { padding: 10px 18px; border-radius: 10px 10px 0 0; font-weight: 600; color: #5F6B7A; }
button[data-baseweb="tab"][aria-selected="true"] { color: #B45309; background: #FFFFFF; }
[data-baseweb="tab-highlight"] { background-color: #FF9900 !important; height: 3px; }

/* kartu KPI */
[data-testid="stMetric"] { background: #fff; border: 1px solid #F0C27B; border-left: 6px solid #FF9900;
                           border-radius: 14px; padding: 12px 14px; box-shadow: 0 4px 14px rgba(180,83,9,.20); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; color: #37475A !important; opacity: 1 !important; font-weight: 500; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    color: #C45500; font-weight: 700; font-size: clamp(1.25rem, 1.7vw, 1.6rem) !important;
    overflow: visible !important; text-overflow: clip !important; white-space: normal !important;
    line-height: 1.2; }

/* grafik sebagai kartu putih */
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #F0C27B; border-radius: 14px;
                                padding: 8px; box-shadow: 0 4px 14px rgba(180,83,9,.16); }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stExpander"] { background: #fff; border: 1px solid #F0C27B; border-radius: 12px; }

/* filter */
span[data-baseweb="tag"] { background-color: #FF9900 !important; color: #131A22 !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #FF9900 !important; }
[data-testid="stSliderThumbValue"] { color: #B45309 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

DEMO_USED: list[str] = []


# ------------------------------------------------------------------ util pemuatan dan tampilan
def locate(*names):
    for n in names:
        for base, is_demo in ((DATA, False), (DEMO, True)):
            p = base / n
            if p.exists():
                if is_demo and n not in DEMO_USED:
                    DEMO_USED.append(n)
                return p
    return None


@st.cache_data(show_spinner=False)
def _read_table(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def _read_cube(path: str, mtime: float) -> pd.DataFrame:
    return cu.load_cube(path)


def table_file(*names):
    p = locate(*names)
    return None if p is None else _read_table(str(p), p.stat().st_mtime)


def cube_file(name):
    p = locate(name)
    return None if p is None else _read_cube(str(p), p.stat().st_mtime)


def plot(fig, **kw):
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    try:
        st.plotly_chart(fig, use_container_width=True, theme=None, **kw)
    except TypeError:                       # versi Streamlit baru yang menghapus use_container_width
        st.plotly_chart(fig, width="stretch", theme=None, **kw)


def show_table(df, **kw):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, width="stretch", hide_index=True, **kw)


def missing(*names):
    st.error("File belum ada: " + ", ".join(names) + ". Jalankan 05_export_dashboard.ipynb "
             "atau make_demo_data.py, lalu letakkan hasilnya di data/.")


# ------------------------------------------------------------------ data
cube_main = cube_file("cube_main.csv.gz")
cube_text = cube_file("cube_text.csv.gz")
review_len = table_file("review_length.csv")
text_pattern = table_file("text_pattern.csv")
ml_metrics = table_file("ml_metrics.csv")
ml_importance = table_file("ml_importance.csv")
ml_quarterly = table_file("ml_quarterly.csv")
ml_ablation = table_file("ml_ablation.csv")
ml_threshold = table_file("ml_threshold.csv")
g_summary = table_file("graph_summary.csv")
g_degree = table_file("graph_degree.csv")
g_products = table_file("graph_products.csv.gz", "graph_products.csv")
g_reviewers = table_file("graph_reviewers.csv")
g_nodes = table_file("graph_nodes.csv")
g_edges = table_file("graph_edges.csv")
g_comm = table_file("graph_communities.csv")

st.markdown(
    '<div class="hero"><p class="t">Analisis Ulasan Produk Amazon: Kategori Video Games</p>'
    '<p class="s">Dashboard Final Project Big Data Analytics. Sumber: Amazon Review Data 2018 (Ni et al., 2019), '
    'diolah dengan HDFS dan Apache Spark. Semua angka dihitung dari seluruh Clean Layer.</p></div>',
    unsafe_allow_html=True)

if DEMO_USED:
    st.warning("MODE DEMO: file berikut memakai data acak (fiktif) dan tidak boleh dikutip di laporan: "
               + ", ".join(sorted(DEMO_USED)) + ". Letakkan hasil 05_export_dashboard.ipynb di folder data/.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Executive Summary", "2. Perilaku Ulasan", "3. Machine Learning", "4. Graph Analytics"])


def filter_panel(key, cube, with_product):
    """Panel filter di dalam tab. Tab 2 memanggilnya dengan with_product=False (UI produk tidak ada)."""
    with st.container(border=True):
        widths = [3, 2, 3, 2, 2] if with_product else [3, 3, 2, 2]
        cols = st.columns(widths)
        y0, y1 = int(cube["year"].min()), int(cube["year"].max())
        years = cols[0].slider("Tahun", y0, y1, (y0, y1), key=f"{key}_years")
        brands = cols[1].multiselect("Merek", list(cube["brand"].cat.categories), key=f"{key}_brand")
        i = 2
        products = []
        if with_product:
            pool = cube.loc[cube["brand"].isin(brands), "product"] if brands else cube["product"]
            opts = sorted(p for p in pool.unique() if p != cu.OTHER_PRODUCTS)
            products = cols[2].multiselect("Produk", opts, key=f"{key}_product",
                                           help="Hanya produk dengan jumlah ulasan di atas batas ekspor "
                                                "yang dirinci per produk.")
            i = 3
        cats = cols[i].multiselect("Kategori utama", list(cube["main_cat"].cat.categories), key=f"{key}_cat")
        verified = cols[i + 1].radio("Verifikasi", ["Semua", "Verified", "Unverified"], key=f"{key}_ver")
    return {"years": years, "brands": brands, "products": products, "cats": cats, "verified": verified}


# ======================================================================== TAB 1
with tab1:
    if cube_main is None:
        missing("cube_main.csv.gz")
    else:
        f1 = filter_panel("t1", cube_main, with_product=True)
        d = cu.apply_filters(cube_main, f1["years"], f1["brands"], f1["products"], f1["cats"], f1["verified"])
        k = cu.kpis(d)
        if not k:
            st.warning("Tidak ada data untuk kombinasi filter ini. Longgarkan filter.")
        else:
            c = st.columns(6)
            c[0].metric("Total ulasan", cu.fmt_int(k["n"]))
            c[1].metric("Total vote", cu.fmt_int(k["total_vote"]))
            c[2].metric("Rata-rata rating", f"{k['avg_rating']:.2f}")
            c[3].metric("Verified", f"{k['pct_verified'] * 100:.1f}%")
            c[4].metric("Helpful (vote > 0)", f"{k['pct_helpful'] * 100:.1f}%")
            c[5].metric("Panjang rata-rata", f"{k['avg_len']:.0f} kar.")

            st.subheader("Tren ulasan dari tahun ke tahun")
            split = st.checkbox("Pisahkan per merek (6 merek terbanyak pada filter ini). "
                                "Klik nama merek di legenda untuk menampilkan atau menyembunyikannya.",
                                key="t1_split")
            yr = cu.by_year(d)
            if split:
                g = cu.by_year(d, split_brand=True)
                top_b = g.groupby("brand", observed=True)["n"].sum().nlargest(6).index
                g = g[g["brand"].isin(top_b)].copy()
                g["brand"] = g["brand"].astype(str)
                fig = px.line(g, x="year", y="n", color="brand", markers=True,
                              labels={"n": "Jumlah ulasan", "year": "Tahun", "brand": "Merek"})
            else:
                fig = px.bar(yr, x="year", y="n", labels={"n": "Jumlah ulasan", "year": "Tahun"})
                fig.add_scatter(x=yr["year"], y=yr["avg_rating"], name="Rata-rata rating",
                                yaxis="y2", mode="lines+markers")
                fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Rata-rata rating",
                                              range=[1, 5]))
            plot(fig)
            peak = yr.loc[yr["n"].idxmax()]
            st.caption(f"Pada filter ini volume ulasan tertinggi terjadi pada {int(peak['year'])} "
                       f"({cu.fmt_int(peak['n'])} ulasan).")

            p1, p2 = st.columns(2)
            with p1:
                st.subheader("Distribusi rating")
                r = cu.by_rating(d)
                r["rating"] = r["overall"].astype(int).astype(str)
                plot(px.pie(r, names="rating", values="n", color="rating", hole=0.4,
                            color_discrete_map=RATING_COLORS,
                            category_orders={"rating": ["1", "2", "3", "4", "5"]}))
            with p2:
                st.subheader("Verified vs unverified")
                v = cu.by_verified(d)
                plot(px.pie(v, names="status", values="n", color="status", hole=0.4,
                            color_discrete_map={"Verified": "#FF9900", "Unverified": "#8A97A6"}))
                st.caption("Rata-rata rating: " + ", ".join(
                    f"{row.status} {row.avg_rating:.2f}" for row in v.itertuples()))

        # ---- Top / Bottom 10: hanya filter tahun yang berlaku
        st.divider()
        st.subheader("Top / Bottom 10 berdasarkan jumlah ulasan")
        st.caption("Panjang bar = jumlah ulasan. Warna = rata-rata rating (merah 1, hijau 5). "
                   "Bagian ini hanya dipengaruhi filter Tahun.")
        d_year = cu.apply_filters(cube_main, f1["years"])

        def ranking_chart(key, label, exclude, default_min, note):
            a, b = st.columns([1, 2])
            mode = a.radio(f"Urutan {label}", ["Top 10", "Bottom 10"], horizontal=True, key=f"mode_{key}")
            min_rev = b.slider(f"Minimal ulasan per {label}", 1, 2000, default_min, key=f"min_{key}")
            g = cu.top_bottom(d_year, key, mode, min_rev, exclude)
            if g.empty:
                st.info(f"Tidak ada {label} yang memenuhi batas minimal.")
                return
            fig = px.bar(g, x="n", y=key, orientation="h", color="avg_rating", text="n",
                         color_continuous_scale="RdYlGn", range_color=[1, 5],
                         labels={"n": "Jumlah ulasan", key: "", "avg_rating": "Rata-rata rating"})
            fig.update_yaxes(categoryorder="array", categoryarray=g[key].tolist())
            fig.update_layout(height=450)
            plot(fig)
            st.caption(note)

        ranking_chart("brand", "merek", [cu.OTHER_BRANDS, "Unknown"], 100,
                      "Hanya merek teratas dan 'Unknown' yang dirinci saat ekspor; sisanya digabung "
                      "sebagai Other brands dan tidak ditampilkan di sini.")
        ranking_chart("product", "produk", [cu.OTHER_PRODUCTS], 50,
                      "Hanya produk dengan jumlah ulasan di atas batas ekspor yang dirinci. Bottom 10 tanpa "
                      "batas minimal didominasi produk berulasan sangat sedikit, naikkan batas minimalnya.")

# ======================================================================== TAB 2
with tab2:
    if cube_text is None:
        missing("cube_text.csv.gz")
    else:
        f2 = filter_panel("t2", cube_text, with_product=False)
        d2 = cu.apply_filters(cube_text, f2["years"], f2["brands"], (), f2["cats"], f2["verified"])
        if not cu.kpis(d2):
            st.warning("Tidak ada data untuk kombinasi filter ini. Longgarkan filter.")
        else:
            st.subheader("Panjang ulasan dan peluang dianggap helpful")
            lg = cu.by_len_group(d2)
            fig = px.bar(lg, x="len_group", y="helpful_pct", text="helpful_pct", hover_data={"n": True},
                         labels={"len_group": "Panjang ulasan (karakter)", "helpful_pct": "% ulasan helpful",
                                 "n": "Jumlah ulasan"})
            fig.update_traces(texttemplate="%{text:.1f}%")
            plot(fig)
            st.caption("Panjang ulasan adalah fitur terpenting model di Tab 3, dan baseline yang hanya memakai "
                       "panjang teks sudah jauh di atas tebakan acak. Grafik ini memperlihatkan hubungannya.")

            r = cu.by_rating(d2)
            l, rr = st.columns(2)
            with l:
                st.subheader("Rasio helpful per rating")
                plot(px.bar(r, x="overall", y="helpful_pct",
                            labels={"overall": "Rating", "helpful_pct": "% ulasan helpful"}))
            with rr:
                st.subheader("Panjang rata-rata per rating")
                plot(px.bar(r, x="overall", y="avg_len", labels={"overall": "Rating", "avg_len": "Karakter"}))

            st.subheader("Rasio helpful per tahun")
            y = cu.by_year(d2)
            fig = px.line(y, x="year", y="helpful_pct", markers=True,
                          labels={"year": "Tahun", "helpful_pct": "% ulasan helpful"})
            fig.add_vrect(x0=2016.5, x1=max(2016.6, float(y["year"].max()) + 0.5), fillcolor="grey", opacity=0.12,
                          line_width=0)
            plot(fig)
            st.caption("Area abu-abu (setelah 2016): ulasan belum matang karena belum sempat divote, sehingga "
                       "penurunan rasio helpful tidak berarti kualitasnya turun.")

            st.subheader("Verified vs unverified")
            v2 = cu.by_verified(d2)
            show_table(v2[["status", "n", "avg_rating", "helpful_pct", "avg_len"]].rename(columns={
                "status": "Status", "n": "Ulasan", "avg_rating": "Rata-rata rating",
                "helpful_pct": "% helpful", "avg_len": "Panjang rata-rata"}).round(2))
        st.caption("Tab ini tidak memakai filter produk.")

    st.divider()
    st.subheader("Seluruh dataset: distribusi panjang ulasan dan pola teks (tanpa filter)")
    if review_len is not None and text_pattern is not None:
        a, b = st.columns(2)
        with a:
            dl = review_len[~review_len["metrik"].astype(str).isin(["Maksimum", "100"])].copy()
            dl["metrik"] = dl["metrik"].astype(str)

            fig = px.bar(dl, x="metrik", y="panjang_teks", text="panjang_teks",
                        labels={"metrik": "Statistik", "panjang_teks": "Panjang (karakter)"})
            fig.update_xaxes(type="category")   
            plot(fig)
        with b:
            plot(px.bar(text_pattern.sort_values("persen"), x="persen", y="pola", orientation="h",
                        labels={"persen": "% ulasan", "pola": ""}))
            st.caption("Pola yang perlu ditangani saat pembersihan teks (dari EDA Bab 4).")

# ======================================================================== TAB 3
with tab3:
    if ml_metrics is None or ml_importance is None or ml_quarterly is None:
        missing("ml_metrics.csv", "ml_importance.csv", "ml_quarterly.csv")
    else:
        full = ml_metrics[(ml_metrics["engine"] == "Spark MLlib")
                          & (ml_metrics["train_rows"] == ml_metrics["train_rows"].max())]
        r0 = full.iloc[0]
        prevalence = (r0.TP + r0.FN) / (r0.TP + r0.FP + r0.FN + r0.TN)
        gbt = full[full["model"] == "GBT"].set_index("skenario")

        st.subheader("Memprediksi ulasan yang akan dianggap helpful")
        st.write("Skenario **A** memakai fitur eksposur (waktu dan posisi ulasan), jadi lebih mirip memprediksi "
                 "*peluang mendapat vote*. Skenario **B** tanpa eksposur, lebih mendekati *kualitas* ulasan "
                 "sehingga cocok untuk ulasan baru yang belum sempat divote (cold-start).")
        sk_sel = st.multiselect("Skenario", ["A", "B"], default=["A", "B"], key="t3_sk")
        if not sk_sel:
            st.info("Pilih minimal satu skenario.")
        else:
            m1, m2, m3 = st.columns(3)
            for col, s in zip((m1, m2), ("A", "B")):
                if s in sk_sel and s in gbt.index:
                    col.metric(f"PR-AUC GBT, skenario {s}", f"{gbt.loc[s, 'pr_auc']:.3f}")
            m3.metric("PR-AUC tebakan acak", f"{prevalence:.3f}")

            fs = full[full["skenario"].isin(sk_sel)]
            fig = px.bar(fs, x="model", y="pr_auc", color="skenario", barmode="group", text="pr_auc",
                         category_orders={"model": ["LR", "RF", "GBT"]}, color_discrete_map=SCEN_COLORS,
                         labels={"pr_auc": "PR-AUC (data test)", "model": "Model", "skenario": "Skenario"})
            fig.update_traces(texttemplate="%{text:.3f}")
            fig.add_hline(y=prevalence, line_dash="dot", annotation_text=f"Acak ({prevalence:.3f})")
            fig.add_hline(y=BASELINE_TEXT_LEN_PRAUC, line_dash="dash",
                          annotation_text=f"Hanya panjang teks ({BASELINE_TEXT_LEN_PRAUC:.3f})")
            plot(fig)
            with st.expander("Tabel metrik lengkap (Spark MLlib, seluruh data latih)"):
                show_table(full[["skenario", "model", "accuracy", "precision", "recall", "f1", "pr_auc",
                                 "roc_auc", "threshold"]].round(3))
            with st.expander("Spark MLlib vs scikit-learn (data latih 200 ribu baris yang sama)"):
                small = ml_metrics[ml_metrics["train_rows"] == ml_metrics["train_rows"].min()]
                show_table(small[["engine", "skenario", "model", "pr_auc", "fit_seconds"]]
                           .sort_values(["skenario", "model", "engine"]).round(3))

        st.divider()
        l, r = st.columns(2)
        with l:
            st.subheader("Fitur terpenting")
            sk = st.radio("Skenario", ["A", "B"], horizontal=True, key="t3_imp")
            di = ml_importance[ml_importance["skenario"] == sk].nlargest(10, "importance").sort_values("importance")
            plot(px.bar(di, x="importance", y="fitur", orientation="h", labels={"importance": "Importance",
                                                                                 "fitur": ""}))
        with r:
            st.subheader("Kontribusi tiap kelompok fitur")
            if ml_ablation is not None:
                ab = ml_ablation.copy()
                ab["skenario"] = np.where(ab["termasuk_B"], "A dan B", "Hanya A (eksposur)")
                fig = px.bar(ab, x="langkah", y="pr_auc", color="skenario", text="pr_auc",
                             labels={"langkah": "Kelompok fitur ditambahkan", "pr_auc": "PR-AUC (validasi)"})
                fig.update_traces(texttemplate="%{text:.3f}")
                plot(fig)
            st.caption("Eksposur (waktu dan posisi ulasan) menambah PR-AUC paling besar, tetapi itu sinyal "
                       "*kapan* ulasan ditulis, bukan seberapa bagus isinya.")

        st.subheader("Review baru: rasio helpful aktual vs prediksi")
        cols = ["aktual"] + [f"pred_{s}" for s in ("A", "B") if s in sk_sel]
        qd = ml_quarterly.melt(id_vars="kuartal", value_vars=[c for c in cols if c in ml_quarterly.columns],
                               var_name="seri", value_name="persen")
        qd["seri"] = qd["seri"].replace({"aktual": "Aktual (vote > 0)", "pred_A": "Prediksi A (dengan eksposur)",
                                         "pred_B": "Prediksi B (tanpa eksposur)"})
        fig = px.line(qd, x="kuartal", y="persen", color="seri", markers=True,
                      labels={"persen": "% ulasan helpful", "kuartal": "Kuartal"})
        first_imm = ml_quarterly["kuartal"][ml_quarterly["kuartal"] >= "2017Q1"]
        if len(first_imm):
            fig.add_vrect(x0=first_imm.iloc[0], x1=ml_quarterly["kuartal"].iloc[-1], fillcolor="grey",
                          opacity=0.12, line_width=0)
        plot(fig)
        st.caption("Area abu-abu: ulasan 2017 ke atas belum matang. Aktual turun karena belum sempat divote; "
                   "prediksi A ikut turun (ia belajar dari eksposur), sedangkan prediksi B relatif stabil.")

        st.divider()
        st.subheader("Confusion matrix")
        c1, c2 = st.columns(2)
        cm_model = c1.selectbox("Model", ["GBT", "RF", "LR"], key="cm_model")
        cm_sc = c2.selectbox("Skenario", ["A", "B"], key="cm_sc")
        row = full[(full["model"] == cm_model) & (full["skenario"] == cm_sc)]
        if not row.empty:
            row = row.iloc[0]
            z = np.array([[row.TN, row.FP], [row.FN, row.TP]])
            fig = px.imshow(z, text_auto=True, color_continuous_scale=ORANGE_SEQ, aspect="auto",
                            x=["Prediksi: tidak helpful", "Prediksi: helpful"],
                            y=["Aktual: tidak helpful", "Aktual: helpful"])
            fig.update_layout(coloraxis_showscale=False)
            plot(fig)

        st.divider()
        st.subheader("Atur threshold: presisi vs cakupan")
        if ml_threshold is None:
            missing("ml_threshold.csv")
        else:
            st.write("Model memberi skor peluang helpful tiap ulasan. **Threshold** adalah batas skor untuk "
                     "menandai ulasan sebagai helpful. Threshold tinggi: tanda lebih tepat (precision naik) tetapi "
                     "banyak ulasan bagus terlewat (recall turun). Threshold rendah: sebaliknya.")
            a, b = st.columns([1, 3])
            ts = a.radio("Skenario", ["A", "B"], key="thr_sc")
            t = b.slider("Threshold", 0.05, 0.95, 0.50, 0.01, key="thr_t")
            mt = cu.threshold_metrics(ml_threshold, ts, t)
            q = st.columns(4)
            q[0].metric("Precision", f"{mt['precision'] * 100:.1f}%")
            q[1].metric("Recall", f"{mt['recall'] * 100:.1f}%")
            q[2].metric("F1", f"{mt['f1']:.3f}")
            q[3].metric("Ulasan ditandai helpful", f"{mt['flagged_pct'] * 100:.1f}%")
            st.info(f"Dengan threshold {mt['threshold']:.2f}, {mt['flagged_pct'] * 100:.1f}% ulasan ditandai helpful. "
                    f"Dari yang ditandai, {mt['precision'] * 100:.0f}% benar-benar helpful, dan model menangkap "
                    f"{mt['recall'] * 100:.0f}% dari seluruh ulasan helpful. Rata-rata ulasan helpful hanya "
                    f"{mt['prevalence'] * 100:.1f}%, jadi tebakan acak hanya benar {mt['prevalence'] * 100:.1f}% "
                    f"dari yang ditandai.")
            cv = cu.threshold_curve(ml_threshold, ts).melt(id_vars="threshold", value_vars=["precision", "recall", "f1"],
                                                           var_name="metrik", value_name="nilai")
            fig = px.line(cv, x="threshold", y="nilai", color="metrik",
                          labels={"threshold": "Threshold", "nilai": "Nilai", "metrik": ""})
            fig.add_vline(x=mt["threshold"], line_dash="dot")
            if ts in gbt.index:
                fig.add_vline(x=float(gbt.loc[ts, "threshold"]), line_dash="dash", line_color="grey")
                st.caption(f"Garis putus-putus abu-abu = threshold {gbt.loc[ts, 'threshold']:.2f} yang dipilih "
                           f"otomatis dari data validasi (F1 tertinggi).")
            plot(fig)
            cmx = np.array([[mt["TN"], mt["FP"]], [mt["FN"], mt["TP"]]])
            figc = px.imshow(cmx, text_auto=True, color_continuous_scale=ORANGE_SEQ, aspect="auto",
                             x=["Prediksi: tidak helpful", "Prediksi: helpful"],
                             y=["Aktual: tidak helpful", "Aktual: helpful"])
            figc.update_layout(coloraxis_showscale=False, height=320)
            plot(figc)
            st.caption("Threshold dihitung pada data test (GBT, seluruh data latih).")

# ======================================================================== TAB 4
with tab4:
    if any(x is None for x in (g_summary, g_degree, g_products, g_reviewers, g_nodes, g_edges, g_comm)):
        missing("graph_summary.csv", "graph_degree.csv", "graph_products.csv.gz", "graph_reviewers.csv",
                "graph_nodes.csv", "graph_edges.csv", "graph_communities.csv")
    else:
        # ---------------------------------------------------------------- A
        st.header("A. Struktur graph")
        s = g_summary.iloc[0]
        c = st.columns(4)
        c[0].metric("Reviewer", cu.fmt_int(s["n_users"]))
        c[1].metric("Produk", cu.fmt_int(s["n_products"]))
        c[2].metric("Hubungan (edge)", cu.fmt_int(s["n_edges"]))
        c[3].metric("Densitas bipartit", f"{s['density'] * 100:.4f}%")
        gd = g_degree.copy()
        fig = px.scatter(gd, x="degree", y="n_nodes", color="type", log_x=True, log_y=True, opacity=0.6,
                         labels={"degree": "Derajat (jumlah reviewer per produk / produk per reviewer)",
                                 "n_nodes": "Jumlah node", "type": ""})
        plot(fig)
        st.caption("Pola berekor panjang: sebagian besar produk dan reviewer punya hubungan sedikit, sementara "
                   "segelintir node sangat terhubung. Graph sudah disaring: reviewer minimal 2 produk dan produk "
                   "minimal 5 reviewer.")

        # ---------------------------------------------------------------- B
        st.header("B. Produk berpengaruh (PageRank)")
        f = st.columns(4)
        brand_opts = sorted(g_products["brand"].fillna("Unknown").value_counts().head(300).index)
        sel_b = f[0].multiselect("Merek", brand_opts, key="g_brand")
        min_r = f[1].slider("Minimal reviewer", 0, int(g_products["n_reviewers"].max()), 0, step=50, key="g_minr")
        rank_name = f[2].selectbox("Urutkan berdasarkan", ["PageRank", "Jumlah reviewer", "Rata-rata rating",
                                                            "Total vote"], key="g_rank")
        n_show = f[3].slider("Produk di scatter", 20, 500, 200, step=20, key="g_n")
        rank_col = {"PageRank": "pagerank", "Jumlah reviewer": "n_reviewers", "Rata-rata rating": "mean_rating",
                    "Total vote": "votes"}[rank_name]
        pf = g_products[g_products["n_reviewers"] >= min_r]
        if sel_b:
            pf = pf[pf["brand"].isin(sel_b)]
        if pf.empty:
            st.info("Tidak ada produk untuk kombinasi filter ini.")
        else:
            ps = pf.nlargest(n_show, rank_col)
            fig = px.scatter(ps, x="n_reviewers", y="mean_rating", size="pagerank", color="mean_rating",
                             hover_name="title", log_x=True, color_continuous_scale="RdYlGn", range_color=[1, 5],
                             hover_data={"brand": True, "votes": True, "pagerank": ":.1f"},
                             labels={"n_reviewers": "Jumlah reviewer (log)", "mean_rating": "Rata-rata rating"})
            fig.update_traces(marker=dict(sizemin=6, opacity=0.85, line=dict(width=0.8, color="#232F3E")))
            fig.add_hline(y=3, line_dash="dot")
            plot(fig)

            st.subheader("Populer tapi bermasalah")
            q1, q2 = st.columns(2)
            thr_r = q1.slider("Rating rata-rata di bawah", 1.0, 5.0, 3.0, 0.1, key="g_badr")
            thr_n = q2.slider("Jumlah reviewer minimal", 0, int(g_products["n_reviewers"].max()), 500, step=50,
                              key="g_badn")
            bad = pf[(pf["mean_rating"] < thr_r) & (pf["n_reviewers"] >= thr_n)].sort_values("votes", ascending=False)
            if bad.empty:
                st.info("Tidak ada produk yang memenuhi kriteria ini.")
            else:
                st.warning(f"{len(bad)} produk ramai dibahas tetapi dinilai buruk. Urut berdasarkan total vote.")
                show_table(bad[["title", "brand", "n_reviewers", "mean_rating", "votes", "pagerank"]].head(15).round(2))
            st.subheader(f"Top produk menurut {rank_name.lower()}")
            show_table(pf.nlargest(15, rank_col)[["pr_rank", "title", "brand", "n_reviewers", "mean_rating",
                                                  "votes", "pagerank"]].round(2))
        st.caption("PageRank sangat berkorelasi dengan jumlah reviewer (Spearman 0,98 pada hasil notebook 04), "
                   "tetapi tidak mengukur kualitas. Gabungkan dengan rating sebelum dipakai untuk ranking.")

        # ---------------------------------------------------------------- C
        st.header("C. Reviewer berpengaruh (anonim)")
        st.caption("Identitas reviewer tidak ditampilkan; hanya urutan berdasarkan PageRank (Reviewer 1, 2, dan seterusnya).")
        c1, c2 = st.columns(2)
        min_p = c1.slider("Minimal produk diulas", 0, int(g_reviewers["n_products"].max()), 0, key="g_minp")
        flag_p = c2.slider("Batas produk untuk penanda anomali", 20, 500, 100, key="g_flagp")
        rv = g_reviewers[g_reviewers["n_products"] >= min_p]
        if rv.empty:
            st.info("Tidak ada reviewer untuk filter ini.")
        else:
            fig = px.scatter(rv, x="n_products", y="mean_rating", color="votes", size="pagerank", hover_name="reviewer",
                             log_x=True, color_continuous_scale=ORANGE_SEQ,
                             labels={"n_products": "Produk yang diulas (log)", "mean_rating": "Rata-rata rating",
                                     "votes": "Total vote"})
            fig.update_traces(marker=dict(sizemin=6, opacity=0.85, line=dict(width=0.8, color="#232F3E")))
            plot(fig)
            an = rv[(rv["mean_rating"] >= 4.9) & (rv["n_products"] >= flag_p)].sort_values("n_products", ascending=False)
            if len(an):
                st.warning("Pola mencurigakan: rata-rata rating hampir 5,0 pada banyak produk. Perlu diperiksa "
                           "sebagai kemungkinan ulasan tidak organik.")
                show_table(an[["reviewer", "n_products", "mean_rating", "votes"]].round(2))
            else:
                st.caption("Tidak ada reviewer dengan rata-rata rating >= 4,9 pada batas produk ini.")
            show_table(rv.head(100)[["reviewer", "n_products", "votes", "mean_rating", "pagerank"]].round(2))

        # ---------------------------------------------------------------- D
        st.header("D. Jaringan produk (proyeksi produk-produk)")
        st.write("Dua produk terhubung jika diulas oleh reviewer yang sama (Jaccard minimal 0,05). Warna = "
                 "komunitas Louvain, ukuran titik = PageRank. Klik legenda untuk menyembunyikan komunitas.")
        d1, d2, d3 = st.columns(3)
        jmin = float(g_edges["jaccard"].min())
        min_j = d1.slider("Jaccard minimal", jmin, float(g_edges["jaccard"].max()), jmin, 0.01, key="g_jac")
        comm_opts = list(g_comm["label"])
        sel_c = d2.multiselect("Komunitas (kosong = semua)", comm_opts, key="g_comm")
        n_lab = d3.slider("Beri label pada N produk teratas", 0, 40, 12, key="g_lab")
        nd = g_nodes[g_nodes["community_label"].isin(sel_c)] if sel_c else g_nodes
        xs, ys, n_e = cu.network_lines(nd, g_edges, min_j)
        label_set = set(g_nodes.nlargest(n_lab, "pagerank")["asin"]) if n_lab else set()
        palette = COLORWAY + px.colors.qualitative.Dark24
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=0.8, color="rgba(35,47,62,0.35)")))
        pmax = max(float(g_nodes["pagerank"].max()), 1e-9)
        for i, (lab, g) in enumerate(nd.groupby("community_label")):
            hover = [f"{t}<br>{b}<br>rating {r:.2f} | {int(n)} reviewer" for t, b, r, n in
                     zip(g["title"], g["brand"].fillna("Unknown"), g["mean_rating"], g["n_reviewers"])]
            fig.add_trace(go.Scatter(
                x=g["x"], y=g["y"], mode="markers+text" if label_set else "markers", name=str(lab),
                text=[t[:24] if a in label_set else "" for t, a in zip(g["title"], g["asin"])],
                textposition="top center", textfont=dict(size=9), hovertext=hover, hoverinfo="text",
                marker=dict(size=7 + 28 * np.sqrt(g["pagerank"] / pmax), color=palette[i % len(palette)],
                            opacity=0.9, line=dict(width=0.8, color="#232F3E"))))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=720, legend=dict(font=dict(size=9)), margin=dict(l=0, r=0, t=10, b=0))
        plot(fig)
        st.caption(f"{len(nd)} produk dan {n_e} hubungan ditampilkan. Jaringan dibangun dari 300 produk dengan "
                   f"reviewer terbanyak; reviewer yang mengulas lebih dari 100 produk dikeluarkan agar tidak "
                   f"menghubungkan semua produk.")

        # ---------------------------------------------------------------- E
        st.header("E. Komunitas dan produk jembatan")
        cm = g_comm.rename(columns={"label": "Komunitas", "n_products": "Jumlah produk", "avg_rating": "Rata-rata rating",
                                    "top_members": "Anggota utama", "bridge_product": "Produk jembatan"})
        show_table(cm[["Komunitas", "Jumlah produk", "Rata-rata rating", "Anggota utama", "Produk jembatan"]])
        pick = st.selectbox("Lihat anggota komunitas", comm_opts, key="g_pick")
        mem = g_nodes[g_nodes["community_label"] == pick].sort_values("pagerank", ascending=False)
        show_table(mem[["title", "brand", "n_reviewers", "mean_rating", "pagerank", "betweenness"]].round(3))
        st.subheader("Produk jembatan (betweenness tertinggi)")
        show_table(g_nodes.nlargest(10, "betweenness")[["title", "brand", "community_label", "betweenness",
                                                        "mean_rating"]].round(3))
        st.caption("Produk jembatan menghubungkan beberapa komunitas, sehingga cocok untuk promosi lintas ekosistem. "
                   "Produk dalam satu komunitas cocok untuk bundling dan rekomendasi.")

        # ---------------------------------------------------------------- F
        st.header("F. Demo rekomendasi produk")
        st.write("Pilih sebuah produk, lalu lihat produk yang paling sering diulas oleh reviewer yang sama. "
                 "Ini dasar sederhana untuk rekomendasi 'pembeli produk ini juga mengulas...'.")
        by_pr = g_nodes.sort_values("pagerank", ascending=False)
        names = dict(zip(by_pr["asin"], by_pr["title"]))
        asin = st.selectbox("Produk", list(names), format_func=lambda a: names[a], key="g_rec")
        rec = cu.neighbors(g_nodes, g_edges, asin, 10)
        if rec.empty:
            st.info("Produk ini tidak punya tetangga pada ambang Jaccard saat ini.")
        else:
            rec = rec.assign(satu_komunitas=rec["satu_komunitas"].map({True: "Ya", False: "Tidak"}))
            show_table(rec[["title", "brand", "jaccard", "shared", "satu_komunitas", "mean_rating"]].rename(columns={
                "title": "Produk terkait", "brand": "Merek", "jaccard": "Jaccard", "shared": "Reviewer sama",
                "satu_komunitas": "Satu komunitas", "mean_rating": "Rating"}).round(3))
        st.caption("Rekomendasi hanya mencakup 300 produk dengan reviewer terbanyak.")