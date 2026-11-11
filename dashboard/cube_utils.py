"""Logika data murni (pandas) untuk dashboard. Dipisah dari app.py agar bisa diuji tanpa Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd

OTHER_PRODUCTS = "Other products"
OTHER_BRANDS = "Other brands"
LEN_ORDER = ["0", "1-100", "101-250", "251-500", "501-1000", "1000+"]


# ------------------------------------------------------------------ pemuatan
def load_cube(path) -> pd.DataFrame:
    """Baca cube agregat (csv/csv.gz). keep_default_na=False agar merek bernama 'NA' atau 'None' tidak jadi NaN."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"len_group": str})
    df["verified"] = df["verified"].astype(str).str.lower().isin(["true", "1"])
    for c in ("brand", "product", "main_cat", "len_group"):
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def fmt_int(n) -> str:
    """1234567 -> '1.234.567' (format Indonesia)."""
    return f"{int(round(n)):,}".replace(",", ".")


# ------------------------------------------------------------------ filter dan agregasi (Tab 1 dan 2)
def apply_filters(df, years, brands=(), products=(), cats=(), verified="Semua"):
    m = df["year"].between(years[0], years[1])
    if brands:
        m &= df["brand"].isin(list(brands))
    if products and "product" in df.columns:
        m &= df["product"].isin(list(products))
    if cats:
        m &= df["main_cat"].isin(list(cats))
    if verified == "Verified":
        m &= df["verified"]
    elif verified == "Unverified":
        m &= ~df["verified"]
    return df[m]


def kpis(d) -> dict:
    n = d["n_reviews"].sum()
    if n == 0:
        return {}
    out = {
        "n": int(n),
        "avg_rating": float((d["overall"] * d["n_reviews"]).sum() / n),
        "pct_verified": float(d.loc[d["verified"], "n_reviews"].sum() / n),
        "pct_helpful": float(d["n_helpful"].sum() / n),
        "avg_len": float(d["sum_text_len"].sum() / n),
    }
    if "sum_vote" in d.columns:
        out["total_vote"] = int(d["sum_vote"].sum())
    return out


def _with_rs(d):
    return d.assign(rs=d["overall"] * d["n_reviews"])


def by_year(d, split_brand=False) -> pd.DataFrame:
    keys = ["year", "brand"] if split_brand else ["year"]
    g = _with_rs(d).groupby(keys, observed=True).agg(
        n=("n_reviews", "sum"), rs=("rs", "sum"), helpful=("n_helpful", "sum")).reset_index()
    g["avg_rating"] = g["rs"] / g["n"]
    g["helpful_pct"] = g["helpful"] / g["n"] * 100
    return g[g["n"] > 0]


def by_rating(d) -> pd.DataFrame:
    g = d.groupby("overall", observed=True).agg(
        n=("n_reviews", "sum"), helpful=("n_helpful", "sum"), tl=("sum_text_len", "sum")).reset_index()
    g["helpful_pct"] = g["helpful"] / g["n"] * 100
    g["avg_len"] = g["tl"] / g["n"]
    return g


def by_verified(d) -> pd.DataFrame:
    g = _with_rs(d).groupby("verified", observed=True).agg(
        n=("n_reviews", "sum"), rs=("rs", "sum"), helpful=("n_helpful", "sum"),
        tl=("sum_text_len", "sum")).reset_index()
    g["avg_rating"] = g["rs"] / g["n"]
    g["helpful_pct"] = g["helpful"] / g["n"] * 100
    g["avg_len"] = g["tl"] / g["n"]
    g["status"] = g["verified"].map({True: "Verified", False: "Unverified"})
    return g


def by_len_group(d) -> pd.DataFrame:
    g = d.groupby("len_group", observed=True).agg(
        n=("n_reviews", "sum"), helpful=("n_helpful", "sum")).reset_index()
    g["len_group"] = g["len_group"].astype(str)
    g["helpful_pct"] = g["helpful"] / g["n"] * 100
    order = {k: i for i, k in enumerate(LEN_ORDER)}
    return g.sort_values("len_group", key=lambda s: s.map(order)).reset_index(drop=True)


def top_bottom(d, key, mode="Top 10", min_reviews=1, exclude=()) -> pd.DataFrame:
    """Top/Bottom 10 berdasarkan jumlah ulasan; avg_rating dipakai sebagai warna bar."""
    d = d[~d[key].isin(list(exclude))]
    g = _with_rs(d).groupby(key, observed=True).agg(
        n=("n_reviews", "sum"), rs=("rs", "sum")).reset_index()
    g[key] = g[key].astype(str)
    g = g[g["n"] >= max(int(min_reviews), 1)]
    g["avg_rating"] = g["rs"] / g["n"]
    g = g.nlargest(10, "n") if mode == "Top 10" else g.nsmallest(10, "n")
    # bar horizontal: baris pertama berada paling bawah, jadi yang terbesar harus terakhir
    return g.sort_values("n", ascending=(mode == "Top 10")).reset_index(drop=True)


# ------------------------------------------------------------------ Tab 3: threshold
def threshold_metrics(thr: pd.DataFrame, scenario: str, t: float) -> dict:
    """Metrik pada threshold terdekat dari tabel ml_threshold (kolom: skenario, threshold, TP, FP, FN, TN)."""
    d = thr[thr["skenario"] == scenario]
    if d.empty:
        return {}
    row = d.iloc[(d["threshold"] - t).abs().argmin()]
    tp, fp, fn, tn = (int(row[c]) for c in ("TP", "FP", "FN", "TN"))
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(row["threshold"]), "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / total, "flagged_pct": (tp + fp) / total,
            "prevalence": (tp + fn) / total, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def threshold_curve(thr: pd.DataFrame, scenario: str) -> pd.DataFrame:
    d = thr[thr["skenario"] == scenario].copy()
    d["precision"] = np.where(d["TP"] + d["FP"] > 0, d["TP"] / (d["TP"] + d["FP"]).replace(0, np.nan), np.nan)
    d["recall"] = d["TP"] / (d["TP"] + d["FN"])
    d["f1"] = 2 * d["precision"] * d["recall"] / (d["precision"] + d["recall"])
    return d.sort_values("threshold")


# ------------------------------------------------------------------ Tab 4: graph
def network_lines(nodes: pd.DataFrame, edges: pd.DataFrame, min_jaccard: float):
    """Koordinat garis sisi untuk satu trace Plotly (pemisah None antar sisi)."""
    pos = nodes.set_index("asin")[["x", "y"]]
    e = edges[(edges["jaccard"] >= min_jaccard) & edges["p1"].isin(pos.index) & edges["p2"].isin(pos.index)]
    xs, ys = [], []
    for a, b in zip(e["p1"], e["p2"]):
        xs += [pos.at[a, "x"], pos.at[b, "x"], None]
        ys += [pos.at[a, "y"], pos.at[b, "y"], None]
    return xs, ys, len(e)


def neighbors(nodes: pd.DataFrame, edges: pd.DataFrame, asin: str, k: int = 10) -> pd.DataFrame:
    """Produk yang paling sering diulas oleh reviewer yang sama (Jaccard tertinggi)."""
    a = edges[edges["p1"] == asin][["p2", "jaccard", "shared"]].rename(columns={"p2": "asin"})
    b = edges[edges["p2"] == asin][["p1", "jaccard", "shared"]].rename(columns={"p1": "asin"})
    nb = pd.concat([a, b]).merge(nodes, on="asin", how="left")
    me = nodes.loc[nodes["asin"] == asin, "community"]
    nb["satu_komunitas"] = (nb["community"] == me.iloc[0]) if len(me) else False
    return nb.sort_values("jaccard", ascending=False).head(k).reset_index(drop=True)