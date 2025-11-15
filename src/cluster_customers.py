#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
import textwrap
import warnings
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path as MplPath
from numpy.typing import ArrayLike
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# 選用：若環境無 umap/hdbscan，可用 --no-umap 與不加 --run-hdbscan 閃避
try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    import hdbscan  # type: ignore
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Customer clustering for RetailX (K-Means with RFM + preferences, A-version quick enhancements)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 原有參數
    parser.add_argument("--sales", required=True, help="Path to sales CSV")
    parser.add_argument("--customers", required=True, help="Path to customers CSV")
    parser.add_argument("--products", required=True, help="Path to products CSV")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--kmin", type=int, default=4, help="Min K for search")
    parser.add_argument("--kmax", type=int, default=8, help="Max K for search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--winsor", type=float, default=99.5, help="Winsorization top percentile for Monetary (legacy; kept for BC)")
    parser.add_argument("--log-monetary", action="store_true", help="Use log transform for Monetary (mutually exclusive with --winsor in legacy)")
    parser.add_argument("--price-bins", type=int, default=4, help="Number of global price bins (low/med/high/very-high)")
    parser.add_argument("--no-umap", action="store_true", help="Disable UMAP plots even if umap-learn is installed")
    parser.add_argument("--run-hdbscan", action="store_true", help="Run HDBSCAN as a supplemental exploration if available")
    parser.add_argument("--use-minibatch", action="store_true", help="Use MiniBatchKMeans for speed during K search")

    # 新增參數（A 版）
    parser.add_argument("--pca-var", type=float, default=0.0, help="If >0, apply PCA to retain this fraction of explained variance before clustering")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum invoices required to keep a customer for clustering")
    parser.add_argument("--drop-bottom-monetary-quantile", type=float, default=0.0, help="Drop customers with total monetary below this quantile")
    parser.add_argument("--robust-scale", action="store_true", help="Use RobustScaler instead of StandardScaler")
    parser.add_argument("--log-avg-ticket", action="store_true", help="Also apply winsor+log1p to avg_ticket")
    parser.add_argument("--winsor-pct", type=float, default=98.5, help="Winsorization top percentile for heavy-tailed monetary/avg_ticket in A version")
    return parser.parse_args()


def ensure_dirs(base_outdir: str):
    figs = os.path.join(base_outdir, "figures")
    logs = os.path.join(base_outdir, "logs")
    meta = os.path.join(base_outdir, "meta")
    os.makedirs(base_outdir, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(meta, exist_ok=True)
    return figs, logs, meta


def log_issue(log_path: str, message: str):
    with open(log_path, "a", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] {message}\n")


def read_data(sales_path: str, customers_path: str, products_path: str, log_path: str):
    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)
    sales = pd.read_csv(sales_path)

    customers.columns = [c.strip().lower().replace(" ", "_") for c in customers.columns]
    products.columns = [c.strip().lower().replace(" ", "_") for c in products.columns]
    sales.columns = [c.strip().lower().replace(" ", "_") for c in sales.columns]

    sales["invoice_date"] = sales["invoice_date"].astype(str).str.strip()
    sales["invoice_date"] = pd.to_datetime(sales["invoice_date"], format="%d/%m/%Y", errors="coerce")

    invalid_dates = sales[sales["invoice_date"].isna()]
    if len(invalid_dates) > 0:
        log_issue(log_path, f"Invalid invoice_date rows dropped: {len(invalid_dates)} (invalid date format)")
        sales = sales[~sales["invoice_date"].isna()].copy()

    dup_mask = sales.duplicated(subset=["invoice_no"], keep="first")
    if dup_mask.any():
        n_dup = dup_mask.sum()
        log_issue(log_path, f"Duplicate invoice_no dropped: {n_dup}")
        sales = sales[~dup_mask].copy()

    return sales, customers, products


def explode_products(sales: pd.DataFrame, products: pd.DataFrame, log_path: str) -> pd.DataFrame:
    df = sales.copy()
    df["product_id_list"] = df["product_id_list"].astype(str)
    df["product_id_list"] = df["product_id_list"].str.replace('"', "").str.replace("'", "").str.strip()
    df = df.assign(product_id=df["product_id_list"].str.split(",")).explode("product_id")
    df["product_id"] = df["product_id"].str.strip()

    products.columns = [c.strip().lower() for c in products.columns]
    merged = df.merge(products, how="left", left_on="product_id", right_on="product_id")

    invalid = merged[merged["price"].isna()]
    if len(invalid) > 0:
        sample_ids = invalid["product_id"].dropna().unique().tolist()[:10]
        log_issue(log_path, f"Invalid product_id rows dropped: {len(invalid)}. Sample product_ids: {sample_ids}")
        merged = merged[~merged["price"].isna()].copy()

    return merged


def join_customers(trans: pd.DataFrame, customers: pd.DataFrame, log_path: str) -> pd.DataFrame:
    customers.columns = [c.strip().lower() for c in customers.columns]
    merged = trans.merge(customers, how="left", on="customer_id", suffixes=("", "_cust"))
    invalid = merged[merged["gender"].isna() | merged["age"].isna() | merged["payment_method"].isna()]
    if len(invalid) > 0:
        bad_ids = invalid["customer_id"].dropna().unique().tolist()[:10]
        log_issue(log_path, f"Transactions with missing customer profile dropped: {len(invalid)}. Sample customer_ids: {bad_ids}")
        merged = merged[~(merged["gender"].isna() | merged["age"].isna() | merged["payment_method"].isna())].copy()
    return merged


def compute_reference_date(sales: pd.DataFrame) -> pd.Timestamp:
    return sales["invoice_date"].max()


def price_bins_from_products(products: pd.DataFrame, n_bins: int = 4):
    q = np.linspace(0, 1, n_bins + 1)
    quantiles = products["price"].quantile(q).values
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1] + 1e-6
    return quantiles


def assign_price_bucket(price: float, edges: ArrayLike) -> int:
    for i in range(1, len(edges)):
        if price <= edges[i]:
            return i - 1
    return len(edges) - 2


def _winsorize_series(s: pd.Series, upper_pct: float) -> pd.Series:
    if len(s) == 0:
        return s
    upper = np.nanpercentile(s, upper_pct)
    return np.minimum(s, upper)


def _safe_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(scipy_entropy(p, base=np.e))


def build_customer_features(
    trans: pd.DataFrame,
    reference_date: pd.Timestamp,
    price_edges: ArrayLike,
    recent_days: int = 90,
) -> Tuple[pd.DataFrame, Dict]:
    # 發票層級統計
    invoice_stats = trans.groupby(["invoice_no", "customer_id", "invoice_date", "shopping_mall"], as_index=False).agg(
        invoice_amount=("price", "sum"),
        item_count=("product_id", "count"),
    )

    # RFM 基礎
    cust_last_date = invoice_stats.groupby("customer_id")["invoice_date"].max()
    cust_first_date = invoice_stats.groupby("customer_id")["invoice_date"].min()
    recency_days = (reference_date - cust_last_date).dt.days
    tenure_days = (reference_date - cust_first_date).dt.days

    cust_freq = invoice_stats.groupby("customer_id")["invoice_no"].nunique()
    cust_monetary = invoice_stats.groupby("customer_id")["invoice_amount"].sum()

    cust_inv_amount_std = invoice_stats.groupby("customer_id")["invoice_amount"].std().fillna(0.0)
    cust_avg_ticket = invoice_stats.groupby("customer_id")["invoice_amount"].mean()
    cust_avg_items = invoice_stats.groupby("customer_id")["item_count"].mean()
    cust_avg_items_std = invoice_stats.groupby("customer_id")["item_count"].std().fillna(0.0)

    # 購買間隔統計
    inv_dates = invoice_stats.groupby("customer_id")["invoice_date"].apply(lambda s: sorted(s.unique()))
    mean_gap_days = {}
    std_gap_days = {}
    cv_gap = {}
    for cid, dates in inv_dates.items():
        if len(dates) < 2:
            mean_gap_days[cid] = 0.0
            std_gap_days[cid] = 0.0
            cv_gap[cid] = 0.0
            continue
        gaps = np.diff(pd.to_datetime(dates)).astype("timedelta64[D]").astype(float)
        m = float(np.mean(gaps))
        sd = float(np.std(gaps))
        mean_gap_days[cid] = m
        std_gap_days[cid] = sd
        cv_gap[cid] = float(sd / (m + 1e-9))

    mean_gap_days = pd.Series(mean_gap_days)
    std_gap_days = pd.Series(std_gap_days)
    cv_gap = pd.Series(cv_gap)

    # 近 90 天活躍度
    cutoff = reference_date - timedelta(days=recent_days)
    recent_mask = invoice_stats["invoice_date"] >= cutoff
    recent_agg = invoice_stats.assign(recent=recent_mask)
    cust_recent_amount = recent_agg.groupby(["customer_id", "recent"])["invoice_amount"].sum().unstack(fill_value=0)
    cust_recent_freq = recent_agg.groupby(["customer_id", "recent"])["invoice_no"].nunique().unstack(fill_value=0)
    if True not in cust_recent_amount.columns:
        cust_recent_amount[True] = 0.0
    if True not in cust_recent_freq.columns:
        cust_recent_freq[True] = 0
    monetary_recent_ratio = (cust_recent_amount[True] / (cust_recent_amount.sum(axis=1) + 1e-9)).fillna(0.0)
    freq_recent_ratio = (cust_recent_freq[True] / (cust_recent_freq.sum(axis=1) + 1e-9)).fillna(0.0)

    # 商場占比
    mall_counts = invoice_stats.pivot_table(index="customer_id", columns="shopping_mall", values="invoice_no", aggfunc="count", fill_value=0)
    for mall in ["MK", "TKO", "ST", "CYB"]:
        if mall not in mall_counts.columns:
            mall_counts[mall] = 0
    mall_counts = mall_counts[["MK", "TKO", "ST", "CYB"]]
    mall_share = mall_counts.div(mall_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 類目占比（依商品明細）
    cat_counts = trans.pivot_table(index="customer_id", columns="category", values="product_id", aggfunc="count", fill_value=0)
    for cat in ["Electronics", "Clothing", "Groceries", "Books", "Toys"]:
        if cat not in cat_counts.columns:
            cat_counts[cat] = 0
    cat_counts = cat_counts[["Electronics", "Clothing", "Groceries", "Books", "Toys"]]
    cat_share = cat_counts.div(cat_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 偏好簡化
    durable_share = (cat_share.get("Electronics", 0) + cat_share.get("Toys", 0)).astype(float)
    fmcg_share = (cat_share.get("Groceries", 0) + cat_share.get("Books", 0)).astype(float)
    clothing_share = cat_share.get("Clothing", pd.Series(0, index=cat_share.index)).astype(float)
    # top category
    top_category_idx = cat_share.values.argmax(axis=1) if cat_share.shape[1] > 0 else np.zeros(len(cat_share), dtype=int)
    top_category = pd.Series([cat_share.columns[i] if len(cat_share.columns) else "NA" for i in top_category_idx], index=cat_share.index)
    top_category_share = cat_share.max(axis=1) if cat_share.shape[1] > 0 else pd.Series(0.0, index=cat_share.index)
    # 類目多樣性
    category_entropy = cat_share.apply(lambda r: _safe_entropy(r.values), axis=1)

    # 支付占比與多樣性
    pay_counts = trans.pivot_table(index="customer_id", columns="payment_method", values="invoice_no", aggfunc="count", fill_value=0)
    for p in ["Mobile Payment", "Credit Card", "Cash"]:
        if p not in pay_counts.columns:
            pay_counts[p] = 0
    pay_counts = pay_counts[["Mobile Payment", "Credit Card", "Cash"]]
    pay_share = pay_counts.div(pay_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    pay_cash_share = pay_share["Cash"]
    pay_credit_share = pay_share["Credit Card"]
    noncash_share = 1.0 - pay_cash_share
    payment_entropy = pay_share.apply(lambda r: _safe_entropy(r.values), axis=1)

    # 價格分位特徵（以發票中位單品價 → 全局分位）
    trans_price = trans.copy()
    trans_price["price_bucket"] = trans_price["price"].apply(lambda p: assign_price_bucket(p, price_edges))
    # 每張發票的單品價格中位數
    inv_price_median = trans.groupby("invoice_no")["price"].median()
    # 轉為全局分位（0-1）
    global_prices = trans["price"].values
    def to_quantile(v: float) -> float:
        return float((global_prices <= v).mean()) if not np.isnan(v) else 0.0
    inv_price_q = inv_price_median.apply(to_quantile)
    inv_price_q = inv_price_q.to_frame("price_q").join(invoice_stats.set_index("invoice_no")[["customer_id"]], how="left")
    cust_price_q = inv_price_q.groupby("customer_id")["price_q"].agg(
        price_quantile_median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
    )
    cust_price_q["price_quantile_iqr"] = cust_price_q["q3"] - cust_price_q["q1"]
    cust_price_q = cust_price_q[["price_quantile_median", "price_quantile_iqr"]].fillna(0.0)

    # 性別、年齡
    cust_demo = trans.groupby("customer_id", as_index=False).agg(
        age=("age", "first"),
        gender=("gender", "first"),
    ).set_index("customer_id")
    gender_dummies = pd.get_dummies(cust_demo["gender"], prefix="gender")
    age_series = cust_demo["age"].astype(float)

    # 組裝特徵
    feat = pd.DataFrame({
        "recency_days": recency_days,
        "tenure_days": tenure_days,
        "frequency": cust_freq,
        "monetary": cust_monetary,
        "invoice_amount_std": cust_inv_amount_std,
        "avg_ticket": cust_avg_ticket,
        "avg_ticket_std": invoice_stats.groupby("customer_id")["invoice_amount"].std().fillna(0.0),
        "avg_items": cust_avg_items,
        "avg_items_std": cust_avg_items_std,
        "age": age_series,
        "mean_gap_days": mean_gap_days,
        "std_gap_days": std_gap_days,
        "cv_gap": cv_gap,
        "monetary_recent_ratio": monetary_recent_ratio,
        "freq_recent_ratio": freq_recent_ratio,
        "durable_share": durable_share,
        "fmcg_share": fmcg_share,
        "clothing_share": clothing_share,
        "top_category_share": top_category_share,
        "category_entropy": category_entropy,
        "pay_cash_share": pay_cash_share,
        "pay_credit_share": pay_credit_share,
        "noncash_share": noncash_share,
    }).fillna(0.0)

    # 合併商城占比（保留原有）
    feat = feat.join(mall_share, how="left")

    # 加入簡化的 top_category one-hot（前 3 名）
    topcats = top_category.value_counts().index.tolist()[:3]
    topcat_ohe = pd.get_dummies(top_category)
    keep_top = [c for c in topcat_ohe.columns if c in topcats]
    for c in keep_top:
        feat[f"topcat_{c}"] = topcat_ohe[c]
    # 其餘歸為 other
    if len(keep_top) > 0:
        feat["topcat_other"] = 1 - topcat_ohe[keep_top].sum(axis=1)
    else:
        feat["topcat_other"] = 1

    # 加入價格分位特徵
    feat = feat.join(cust_price_q, how="left").fillna(0.0)

    # 保留原本的性別 one-hot
    feat = feat.join(gender_dummies, how="left").fillna(0.0)

    # 原本價格桶（占比）保留但不強制使用（為兼容舊圖表）
    trans_tmp = trans.copy()
    trans_tmp["price_bucket"] = trans_tmp["price"].apply(lambda p: assign_price_bucket(p, price_edges))
    bucket_counts = trans_tmp.pivot_table(index="customer_id", columns="price_bucket", values="product_id", aggfunc="count", fill_value=0)
    n_bins = len(price_edges) - 1
    for b in range(n_bins):
        if b not in bucket_counts.columns:
            bucket_counts[b] = 0
    bucket_counts = bucket_counts[sorted(bucket_counts.columns)]
    bucket_share = bucket_counts.div(bucket_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    price_share_cols = [f"price_bin_{i}" for i in bucket_share.columns]
    bucket_share.columns = price_share_cols
    feat = feat.join(bucket_share, how="left").fillna(0.0)

    feat = feat.fillna(0.0)

    meta = {
        "feature_columns": feat.columns.tolist(),
        "price_edges": [float(x) for x in price_edges],
        "top_categories_used": topcats,
        "recent_days": recent_days,
    }
    return feat, meta


def transform_monetary_and_related(
    feat: pd.DataFrame,
    winsor_pct_legacy: float,
    use_log_monetary: bool,
    winsor_pct: float,
    log_avg_ticket: bool,
) -> Tuple[pd.DataFrame, Dict]:
    f = feat.copy()

    # monetary 變換：A 版建議 winsor 98.5% + log1p（當 --log-monetary 打開）
    if use_log_monetary:
        if winsor_pct is None or math.isnan(winsor_pct):
            winsor_pct = winsor_pct_legacy
        f["monetary"] = _winsorize_series(f["monetary"], winsor_pct)
        f["monetary"] = np.log1p(np.clip(f["monetary"], 0, None))
        m_meta = {"type": "winsor+log1p", "upper_pct": winsor_pct}
    else:
        # 保持舊邏輯：僅 winsor 或 legacy winsor 百分位
        upper = np.nanpercentile(f["monetary"], winsor_pct_legacy)
        f["monetary"] = np.minimum(f["monetary"], upper)
        m_meta = {"type": "winsor", "upper_pct": winsor_pct_legacy, "upper_value": float(upper)}

    # avg_ticket 變換（可選）：winsor + log1p
    if log_avg_ticket and "avg_ticket" in f.columns:
        f["avg_ticket"] = _winsorize_series(f["avg_ticket"], winsor_pct if winsor_pct is not None else 98.5)
        f["avg_ticket"] = np.log1p(np.clip(f["avg_ticket"], 0, None))

    # 對 gap/變異性類特徵做 log1p 平滑
    for col in ["mean_gap_days", "std_gap_days", "cv_gap", "invoice_amount_std", "avg_items_std", "avg_ticket_std"]:
        if col in f.columns:
            f[col] = np.log1p(np.clip(f[col], 0, None))

    transform_meta = {"monetary": m_meta, "log_avg_ticket": bool(log_avg_ticket), "winsor_pct": winsor_pct}
    return f, transform_meta


def standardize_features(feat: pd.DataFrame, robust: bool = False):
    scaler = RobustScaler() if robust else StandardScaler()
    x = scaler.fit_transform(feat.values)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, scaler


def k_search_and_fit(x: np.ndarray, kmin: int, kmax: int, seed: int, use_minibatch: bool, figs_dir: str):
    sse_list = []
    sil_list = []
    k_list = list(range(kmin, kmax + 1))
    best_k = None
    best_sil = -1.0
    best_model = None
    for k in k_list:
        if use_minibatch:
            model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=50, max_iter=600)
        else:
            model = KMeans(n_clusters=k, random_state=seed, n_init=50, max_iter=600)
        labels = model.fit_predict(x)
        sse = model.inertia_
        sse_list.append(sse)
        try:
            sil = silhouette_score(x, labels)
        except Exception:
            sil = np.nan
        sil_list.append(sil)
        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = model

    # 圖
    plt.figure(figsize=(6, 4))
    plt.plot(k_list, sse_list, marker="o")
    plt.title("K selection (SSE - Elbow)")
    plt.xlabel("K")
    plt.ylabel("SSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "k_selection_sse.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(k_list, sil_list, marker="o", color="orange")
    plt.title("K selection (Silhouette)")
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "k_selection_silhouette.png"), dpi=150)
    plt.close()

    return best_k, best_model, {"k_list": k_list, "sse": sse_list, "silhouette": [None if np.isnan(v) else float(v) for v in sil_list]}


def plot_2d_embeddings(x: np.ndarray, labels: np.ndarray, figs_dir: str, seed: int, use_umap: bool):
    pca = PCA(n_components=2, random_state=seed)
    x_pca = pca.fit_transform(x)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette="tab10", s=10, linewidth=0)
    plt.title("Clusters (PCA 2D)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pca_clusters.png"), dpi=150)
    plt.close()

    # UMAP
    if use_umap and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        x_umap = reducer.fit_transform(x)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=x_umap[:, 0], y=x_umap[:, 1], hue=labels, palette="tab10", s=10, linewidth=0)
        plt.title("Clusters (UMAP 2D)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "umap_clusters.png"), dpi=150)
        plt.close()


def plot_cluster_radar(feat: pd.DataFrame, labels: ArrayLike, figs_dir: str):
    candidates = [c for c in ["recency_days", "frequency", "monetary", "avg_ticket", "avg_items",
                              "durable_share", "fmcg_share", "clothing_share",
                              "pay_cash_share", "pay_credit_share", "noncash_share",
                              "monetary_recent_ratio", "freq_recent_ratio"] if c in feat.columns]
    if len(candidates) < 3:
        return
    clusters = pd.Series(labels, index=feat.index, name="cluster")
    df = feat.join(clusters)
    summary = df.groupby("cluster")[candidates].mean()

    norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
    categories = norm.columns.tolist()
    n_var = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_var, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, row in norm.iterrows():
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Cluster Radar (normalized means)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "cluster_radar.png"), dpi=150)
    plt.close()


def run_hdbscan_exploration(x: np.ndarray, figs_dir: str):
    if not HDBSCAN_AVAILABLE:
        return None
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10)
        labels = clusterer.fit_predict(x)
        pca = PCA(n_components=2, random_state=42)
        xp = pca.fit_transform(x)
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=xp[:,0], y=xp[:,1], hue=labels, palette="tab20", s=8, linewidth=0)
        plt.title("HDBSCAN clusters (PCA view)")
        plt.legend(title="label", bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "hdbscan_pca.png"), dpi=150)
        plt.close()
        return {"labels": labels.tolist()}
    except Exception:
        return None


def save_cluster_outputs(customers_index: pd.Index,
                         labels: ArrayLike,
                         feat_original: pd.DataFrame,
                         outdir: str):
    out_customers = pd.DataFrame({
        "customer_id": customers_index,
        "cluster": labels
    }).set_index("customer_id")

    key_cols = [c for c in [
        "recency_days", "tenure_days", "frequency", "monetary", "avg_ticket", "avg_items",
        "durable_share", "fmcg_share", "clothing_share",
        "pay_cash_share", "pay_credit_share", "noncash_share",
        "monetary_recent_ratio", "freq_recent_ratio", "age"
    ] if c in feat_original.columns]
    out_customers = out_customers.join(feat_original[key_cols], how="left")

    out_customers.to_csv(os.path.join(outdir, "customers_clusters.csv"))

    summary = out_customers.groupby("cluster").agg(
        customers=("cluster", "size"),
        recency_days_mean=("recency_days", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        avg_ticket_mean=("avg_ticket", "mean"),
        avg_items_mean=("avg_items", "mean"),
        age_mean=("age", "mean"),
        durable_share_mean=("durable_share", "mean"),
        fmcg_share_mean=("fmcg_share", "mean"),
        clothing_share_mean=("clothing_share", "mean"),
        pay_cash_share_mean=("pay_cash_share", "mean"),
        pay_credit_share_mean=("pay_credit_share", "mean"),
        noncash_share_mean=("noncash_share", "mean"),
        monetary_recent_ratio_mean=("monetary_recent_ratio", "mean"),
        freq_recent_ratio_mean=("freq_recent_ratio", "mean"),
    )
    summary.to_csv(os.path.join(outdir, "clusters_summary.csv"))
    return out_customers, summary


def main():
    args = parse_args()
    np.random.seed(args.seed)

    figs_dir, logs_dir, meta_dir = ensure_dirs(args.outdir)
    log_path = os.path.join(logs_dir, "etl_issues.log")

    # 1) 讀檔與基本清理
    sales, customers, products = read_data(args.sales, args.customers, args.products, log_path)

    # 2) 產品展開並連結價格
    trans = explode_products(sales, products, log_path)

    # 3) 連結顧客檔案
    trans = join_customers(trans, customers, log_path)

    # 4) 參考日
    reference_date = compute_reference_date(sales)

    # 5) 全品項價格分位切割
    price_edges = price_bins_from_products(products, n_bins=args.price_bins)

    # 6) 建立顧客特徵（未標準化）
    feat_raw, feat_meta = build_customer_features(trans, reference_date, price_edges)

    # A 版篩選規則：min-frequency 與底部 monetary 分位剔除
    tmp = pd.DataFrame({
        "frequency": feat_raw["frequency"],
        "monetary": feat_raw["monetary"],
    })
    keep_mask = pd.Series(True, index=feat_raw.index)
    if args.min_frequency > 1:
        keep_mask &= (tmp["frequency"] >= args.min_frequency)
    if args.drop_bottom_monetary_quantile > 0:
        thr = tmp["monetary"].quantile(args.drop_bottom_monetary_quantile)
        keep_mask &= (tmp["monetary"] >= thr)

    filtered_out = feat_raw.loc[~keep_mask].copy()
    if len(filtered_out) > 0:
        filtered_out.to_csv(os.path.join(args.outdir, "filtered_out.csv"))

    feat_raw = feat_raw.loc[keep_mask].copy()

    # 7) 變換：monetary（winsor/log）與 avg_ticket（可選），以及 gap 類平滑
    feat_m, transform_meta = transform_monetary_and_related(
        feat_raw,
        winsor_pct_legacy=args.winsor,
        use_log_monetary=args.log_monetary,
        winsor_pct=args.winsor_pct,
        log_avg_ticket=args.log_avg_ticket,
    )

    # 8) 保存擴充特徵（原尺度 after transform，但尚未標準化）供檢閱
    feat_m.to_csv(os.path.join(args.outdir, "features_extended.csv"))

    # 9) 標準化
    x_full, scaler = standardize_features(feat_m, robust=args.robust_scale)

    # 10) 可選：PCA 前降維用於聚類
    pca_for_clustering: Optional[PCA] = None
    x_for_cluster = x_full
    pca_explained = None
    if args.pca_var and args.pca_var > 0:
        pca_for_clustering = PCA(n_components=args.pca_var, svd_solver="full", random_state=args.seed)
        x_for_cluster = pca_for_clustering.fit_transform(x_full)
        pca_explained = pca_for_clustering.explained_variance_ratio_.tolist()

    # 11) K 搜尋與建模（在 x_for_cluster 空間）
    best_k, best_model, ksearch_info = k_search_and_fit(
        x_for_cluster, args.kmin, args.kmax, args.seed, args.use_minibatch, figs_dir
    )
    if best_k is None or best_model is None:
        print("[ERROR] 無法決定最佳 K，請檢查資料或參數。", file=sys.stderr)
        sys.exit(1)

    labels = best_model.predict(x_for_cluster)

    # 12) 視覺化（PCA/UMAP 基於 x_for_cluster），雷達圖基於 feat_m
    use_umap = (not args.no_umap) and UMAP_AVAILABLE
    if (not args.no_umap) and (not UMAP_AVAILABLE):
        print("[INFO] umap-learn 未安裝或不可用，將僅輸出 PCA 圖。", file=sys.stderr)
    plot_2d_embeddings(x_for_cluster, labels, figs_dir, args.seed, use_umap=use_umap)
    plot_cluster_radar(feat_m, labels, figs_dir)

    # 13) 輸出結果
    out_customers, summary = save_cluster_outputs(feat_m.index, labels, feat_m, args.outdir)

    # 計算 silhouette（在原標準化空間與 PCA 空間）
    sil_full = None
    sil_pca = None
    try:
        sil_full = float(silhouette_score(x_full, labels))
    except Exception:
        sil_full = None
    if pca_for_clustering is not None:
        try:
            sil_pca = float(silhouette_score(x_for_cluster, labels))
        except Exception:
            sil_pca = None

    meta = {
        "reference_date": str(reference_date.date()),
        "price_edges": feat_meta["price_edges"],
        "feature_columns": feat_meta["feature_columns"],
        "selected_feature_columns": feat_m.columns.tolist(),
        "monetary_transform": transform_meta,
        "scaler": "RobustScaler" if args.robust_scale else "StandardScaler",
        "scaler_center": getattr(scaler, "center_", None).tolist() if hasattr(scaler, "center_") else getattr(scaler, "mean_", None).tolist() if hasattr(scaler, "mean_") else None,
        "scaler_scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "k_search": ksearch_info,
        "best_k": int(best_k),
        "random_seed": int(args.seed),
        "umap_enabled": bool(use_umap),
        "pca_var": float(args.pca_var),
        "pca_explained_variance_ratio": pca_explained,
        "silhouette_full": sil_full,
        "silhouette_cluster_space": sil_pca if pca_for_clustering is not None else sil_full,
        "filtering_rules": {
            "min_frequency": int(args.min_frequency),
            "drop_bottom_monetary_quantile": float(args.drop_bottom_monetary_quantile),
            "filtered_out_count": int(len(filtered_out)),
        },
    }
    with open(os.path.join(meta_dir, "features_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 14) HDBSCAN 補充
    if args.run_hdbscan and HDBSCAN_AVAILABLE:
        _ = run_hdbscan_exploration(x_for_cluster, figs_dir)
    elif args.run_hdbscan and (not HDBSCAN_AVAILABLE):
        print("[INFO] hdbscan 未安裝或不可用，略過 HDBSCAN 補充。", file=sys.stderr)

    # 15) 結束提示
    print(textwrap.dedent(f"""
    完成！（A 版快速優化）
    - 最佳 K = {meta["best_k"]}
    - silhouette（原空間）= {meta["silhouette_full"]}
    - silhouette（聚類空間）= {meta["silhouette_cluster_space"]}
    - 篩選剔除人數 = {meta["filtering_rules"]["filtered_out_count"]}
    - 輸出：
      - {args.outdir}/customers_clusters.csv
      - {args.outdir}/clusters_summary.csv
      - {args.outdir}/features_extended.csv
      - {args.outdir}/filtered_out.csv
      - {args.outdir}/figures/
      - {args.outdir}/logs/etl_issues.log
      - {args.outdir}/meta/features_config.json
    """).strip())


if __name__ == "__main__":
    main()