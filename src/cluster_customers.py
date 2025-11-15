#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
import textwrap
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path as MplPath
from numpy.typing import ArrayLike
from scipy.stats import mstats
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
        description="Customer clustering for RetailX (K-Means with RFM + preferences)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sales", required=True, help="Path to sales CSV")
    parser.add_argument("--customers", required=True, help="Path to customers CSV")
    parser.add_argument("--products", required=True, help="Path to products CSV")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--kmin", type=int, default=4, help="Min K for search")
    parser.add_argument("--kmax", type=int, default=8, help="Max K for search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--winsor", type=float, default=99.5, help="Winsorization top percentile for Monetary (mutually exclusive with --log-monetary)")
    parser.add_argument("--log-monetary", action="store_true", help="Use log transform for Monetary (mutually exclusive with --winsor)")
    parser.add_argument("--price-bins", type=int, default=4, help="Number of global price bins (low/med/high/very-high)")
    parser.add_argument("--no-umap", action="store_true", help="Disable UMAP plots even if umap-learn is installed")
    parser.add_argument("--run-hdbscan", action="store_true", help="Run HDBSCAN as a supplemental exploration if available")
    parser.add_argument("--use-minibatch", action="store_true", help="Use MiniBatchKMeans for speed during K search")
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
    # 讀檔
    customers = pd.read_csv(customers_path)
    products = pd.read_csv(products_path)
    sales = pd.read_csv(sales_path)

    # 正規欄位名稱（保險）
    customers.columns = [c.strip().lower().replace(" ", "_") for c in customers.columns]
    products.columns = [c.strip().lower().replace(" ", "_") for c in products.columns]
    sales.columns = [c.strip().lower().replace(" ", "_") for c in sales.columns]

    # 日期解析（DD/MM/YYYY）
    # 有些資料可能有空格，先strip
    sales["invoice_date"] = sales["invoice_date"].astype(str).str.strip()
    sales["invoice_date"] = pd.to_datetime(sales["invoice_date"], format="%d/%m/%Y", errors="coerce")

    # 記錄無效日期
    invalid_dates = sales[sales["invoice_date"].isna()]
    if len(invalid_dates) > 0:
        log_issue(log_path, f"Invalid invoice_date rows dropped: {len(invalid_dates)} (invalid date format)")
        sales = sales[~sales["invoice_date"].isna()].copy()

    # 發票唯一鍵去重（保留首見）
    dup_mask = sales.duplicated(subset=["invoice_no"], keep="first")
    if dup_mask.any():
        n_dup = dup_mask.sum()
        log_issue(log_path, f"Duplicate invoice_no dropped: {n_dup}")
        sales = sales[~dup_mask].copy()

    return sales, customers, products


def explode_products(sales: pd.DataFrame, products: pd.DataFrame, log_path: str) -> pd.DataFrame:
    # 展開 product_id_list -> 每列一個 product_id
    df = sales.copy()
    df["product_id_list"] = df["product_id_list"].astype(str)
    # 去除字串引號與空格
    df["product_id_list"] = df["product_id_list"].str.replace('"', "").str.replace("'", "").str.strip()
    df = df.assign(product_id=df["product_id_list"].str.split(",")).explode("product_id")
    df["product_id"] = df["product_id"].str.strip()

    # 內連到 products 獲取 category, price
    products.columns = [c.strip().lower() for c in products.columns]
    merged = df.merge(products, how="left", left_on="product_id", right_on="product_id")

    # 記錄無效 product_id
    invalid = merged[merged["price"].isna()]
    if len(invalid) > 0:
        # 只記錄數量與前幾個樣本，以免日志過大
        sample_ids = invalid["product_id"].dropna().unique().tolist()[:10]
        log_issue(log_path, f"Invalid product_id rows dropped: {len(invalid)}. Sample product_ids: {sample_ids}")
        merged = merged[~merged["price"].isna()].copy()

    return merged


def join_customers(trans: pd.DataFrame, customers: pd.DataFrame, log_path: str) -> pd.DataFrame:
    customers.columns = [c.strip().lower() for c in customers.columns]
    # 以 customer_id 內連
    merged = trans.merge(customers, how="left", on="customer_id", suffixes=("", "_cust"))
    invalid = merged[merged["gender"].isna() | merged["age"].isna() | merged["payment_method"].isna()]
    if len(invalid) > 0:
        bad_ids = invalid["customer_id"].dropna().unique().tolist()[:10]
        log_issue(log_path, f"Transactions with missing customer profile dropped: {len(invalid)}. Sample customer_ids: {bad_ids}")
        merged = merged[~(merged["gender"].isna() | merged["age"].isna() | merged["payment_method"].isna())].copy()
    return merged


def compute_reference_date(sales: pd.DataFrame) -> pd.Timestamp:
    # 參考日 = 資料中最晚的發票日期
    return sales["invoice_date"].max()


def price_bins_from_products(products: pd.DataFrame, n_bins: int = 4):
    # 使用全品項價格分位數來定義四段價格帶
    q = np.linspace(0, 1, n_bins + 1)
    quantiles = products["price"].quantile(q).values
    # 去重（極端情況：價格重複導致 quantiles 相同，稍微抖動）
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1] + 1e-6
    return quantiles  # 長度 n_bins+1


def assign_price_bucket(price: float, edges: ArrayLike) -> int:
    # 回傳區間索引 0..(len(edges)-2)
    # edges: [q0, q1, q2, q3, q4]
    for i in range(1, len(edges)):
        if price <= edges[i]:
            return i - 1
    return len(edges) - 2


def build_customer_features(trans: pd.DataFrame,
                            reference_date: pd.Timestamp,
                            price_edges: ArrayLike) -> (pd.DataFrame, dict):
    # trans 欄位：invoice_no, customer_id, product_id, invoice_date, shopping_mall, category, price, gender, age, payment_method
    # 先計算發票層級：每張發票的總金額與件數
    invoice_stats = trans.groupby(["invoice_no", "customer_id", "invoice_date", "shopping_mall"], as_index=False).agg(
        invoice_amount=("price", "sum"),
        item_count=("product_id", "count"),
    )

    # 顧客 RFM
    cust_last_date = invoice_stats.groupby("customer_id")["invoice_date"].max()
    recency_days = (reference_date - cust_last_date).dt.days

    cust_freq = invoice_stats.groupby("customer_id")["invoice_no"].nunique()
    cust_monetary = invoice_stats.groupby("customer_id")["invoice_amount"].sum()

    # 金額波動（可選）：單據金額標準差
    cust_inv_amount_std = invoice_stats.groupby("customer_id")["invoice_amount"].std().fillna(0.0)
    # 平均客單、平均件數
    cust_avg_ticket = invoice_stats.groupby("customer_id")["invoice_amount"].mean()
    cust_avg_items = invoice_stats.groupby("customer_id")["item_count"].mean()

    # 商場偏好占比
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

    # 價格帶偏好（依商品價格）
    trans = trans.copy()
    trans["price_bucket"] = trans["price"].apply(lambda p: assign_price_bucket(p, price_edges))
    bucket_counts = trans.pivot_table(index="customer_id", columns="price_bucket", values="product_id", aggfunc="count", fill_value=0)
    # 確保 0..(n_bins-1) 都存在
    n_bins = len(price_edges) - 1
    for b in range(n_bins):
        if b not in bucket_counts.columns:
            bucket_counts[b] = 0
    bucket_counts = bucket_counts[sorted(bucket_counts.columns)]
    bucket_share = bucket_counts.div(bucket_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 支付方式占比（用交易明細推估實際使用比例；若只有客檔方式，仍可作為一種觀察）
    pay_counts = trans.pivot_table(index="customer_id", columns="payment_method", values="invoice_no", aggfunc="count", fill_value=0)
    for p in ["Mobile Payment", "Credit Card", "Cash"]:
        if p not in pay_counts.columns:
            pay_counts[p] = 0
    pay_counts = pay_counts[["Mobile Payment", "Credit Card", "Cash"]]
    pay_share = pay_counts.div(pay_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 人口屬性（客檔）
    cust_demo = trans.groupby("customer_id", as_index=False).agg(
        age=("age", "first"),
        gender=("gender", "first"),
    ).set_index("customer_id")
    # gender one-hot
    gender_dummies = pd.get_dummies(cust_demo["gender"], prefix="gender")
    # 年齡保留連續欄位
    age_series = cust_demo["age"].astype(float)

    # 組裝特徵表
    feat = pd.DataFrame({
        "recency_days": recency_days,
        "frequency": cust_freq,
        "monetary": cust_monetary,
        "invoice_amount_std": cust_inv_amount_std,
        "avg_ticket": cust_avg_ticket,
        "avg_items": cust_avg_items,
        "age": age_series,
    }).fillna(0.0)

    # 合併占比矩陣
    feat = feat.join(mall_share, how="left")
    feat = feat.join(cat_share, how="left")
    # 重命名價格桶欄位
    price_share_cols = [f"price_bin_{i}" for i in bucket_share.columns]
    bucket_share.columns = price_share_cols
    feat = feat.join(bucket_share, how="left")
    # 支付占比
    pay_share.columns = ["pay_mobile", "pay_credit", "pay_cash"]
    feat = feat.join(pay_share, how="left")
    # 性別 one-hot
    feat = feat.join(gender_dummies, how="left")

    feat = feat.fillna(0.0)
    meta = {
        "feature_columns": feat.columns.tolist(),
        "price_edges": [float(x) for x in price_edges],
    }
    return feat, meta


def transform_monetary(feat: pd.DataFrame, winsor_pct: float, use_log: bool):
    f = feat.copy()
    if use_log:
        f["monetary"] = f["monetary"].clip(lower=1e-6)
        f["monetary"] = np.log1p(f["monetary"])
        transform_meta = {"type": "log1p"}
    else:
        # Winsorize 上尾
        upper = np.nanpercentile(f["monetary"], winsor_pct)
        f["monetary"] = np.minimum(f["monetary"], upper)
        transform_meta = {"type": "winsor", "upper_pct": winsor_pct, "upper_value": float(upper)}
    return f, transform_meta


def standardize_features(feat: pd.DataFrame):
    scaler = StandardScaler()
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
            model = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init="auto")
        else:
            model = KMeans(n_clusters=k, random_state=seed, n_init="auto")
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

    # 畫圖
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
    # PCA
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
    # 選擇少量關鍵特徵畫雷達圖，避免圖過亂
    # 可選特徵：["recency_days","frequency","monetary","avg_ticket","avg_items","Electronics","Clothing","Groceries","Books","Toys","pay_mobile","pay_credit","pay_cash"]
    candidates = [c for c in ["recency_days", "frequency", "monetary", "avg_ticket", "avg_items",
                              "Electronics", "Clothing", "Groceries", "Books", "Toys",
                              "pay_mobile", "pay_credit", "pay_cash"] if c in feat.columns]
    if len(candidates) < 3:
        return
    clusters = pd.Series(labels, index=feat.index, name="cluster")
    df = feat.join(clusters)
    summary = df.groupby("cluster")[candidates].mean()

    # 標準化到 0-1 用於雷達圖比較
    norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
    categories = norm.columns.tolist()
    n_var = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_var, endpoint=False).tolist()
    angles += angles[:1]  # 封閉

    plt.figure(figsize=(6, 6))
    for i, row in norm.iterrows():
        values = row.values.tolist()
        values += values[:1]
        ax = plt.subplot(111, polar=True)
    plt.close()  # 避免重複底層繪圖

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    # 繪製每個簇
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
        # 簡單視覺化（PCA）
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
    # 每位顧客的簇與部分特徵
    out_customers = pd.DataFrame({
        "customer_id": customers_index,
        "cluster": labels
    }).set_index("customer_id")

    # 附上部分關鍵特徵以便閱讀（非標準化原尺度）
    key_cols = [c for c in ["recency_days", "frequency", "monetary", "avg_ticket", "avg_items",
                            "Electronics", "Clothing", "Groceries", "Books", "Toys",
                            "pay_mobile", "pay_credit", "pay_cash", "age"] if c in feat_original.columns]
    out_customers = out_customers.join(feat_original[key_cols], how="left")

    out_customers.to_csv(os.path.join(outdir, "customers_clusters.csv"))

    # 簇摘要
    summary = out_customers.groupby("cluster").agg(
        customers=("cluster", "size"),
        recency_days_mean=("recency_days", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
        avg_ticket_mean=("avg_ticket", "mean"),
        avg_items_mean=("avg_items", "mean"),
        age_mean=("age", "mean"),
    )
    # 類目/支付占比均值
    for col in ["Electronics", "Clothing", "Groceries", "Books", "Toys",
                "pay_mobile", "pay_credit", "pay_cash"]:
        if col in out_customers.columns:
            summary[f"{col}_mean"] = out_customers.groupby("cluster")[col].mean()

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

    # 7) Monetary 轉換：winsor 或 log
    if args.log_monetary and args.winsor is not None:
        print("[WARN] --log-monetary 與 --winsor 互斥；已優先採用 --log-monetary。", file=sys.stderr)
    use_log = bool(args.log_monetary)
    winsor_pct = None if use_log else float(args.winsor)
    feat_m, monetary_meta = transform_monetary(feat_raw, winsor_pct if winsor_pct is not None else 99.5, use_log)

    # 8) 標準化
    x, scaler = standardize_features(feat_m)

    # 9) K 搜尋與建模
    best_k, best_model, ksearch_info = k_search_and_fit(
        x, args.kmin, args.kmax, args.seed, args.use_minibatch, figs_dir
    )
    if best_k is None or best_model is None:
        print("[ERROR] 無法決定最佳 K，請檢查資料或參數。", file=sys.stderr)
        sys.exit(1)

    labels = best_model.predict(x)

    # 10) 視覺化（PCA + UMAP(可關閉) + 雷達圖）
    use_umap = (not args.no_umap) and UMAP_AVAILABLE
    if (not args.no_umap) and (not UMAP_AVAILABLE):
        print("[INFO] umap-learn 未安裝或不可用，將僅輸出 PCA 圖。", file=sys.stderr)
    plot_2d_embeddings(x, labels, figs_dir, args.seed, use_umap=use_umap)
    plot_cluster_radar(feat_m, labels, figs_dir)

    # 11) 輸出結果 CSV 與 meta
    out_customers, summary = save_cluster_outputs(feat_m.index, labels, feat_m, args.outdir)

    meta = {
        "reference_date": str(reference_date.date()),
        "price_edges": feat_meta["price_edges"],
        "feature_columns": feat_meta["feature_columns"],
        "monetary_transform": monetary_meta,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "k_search": ksearch_info,
        "best_k": int(best_k),
        "random_seed": int(args.seed),
        "umap_enabled": bool(use_umap),
    }
    with open(os.path.join(meta_dir, "features_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 12) HDBSCAN 補充（可選）
    if args.run_hdbscan and HDBSCAN_AVAILABLE:
        _ = run_hdbscan_exploration(x, figs_dir)
    elif args.run_hdbscan and (not HDBSCAN_AVAILABLE):
        print("[INFO] hdbscan 未安裝或不可用，略過 HDBSCAN 補充。", file=sys.stderr)

    # 13) 結束提示
    print(textwrap.dedent(f"""
    完成！
    - 最佳 K = {meta["best_k"]}
    - 輸出路徑：
      - {args.outdir}/customers_clusters.csv
      - {args.outdir}/clusters_summary.csv
      - {args.outdir}/figures/（K選擇、PCA/UMAP、雷達圖、RFM等）
      - {args.outdir}/logs/etl_issues.log
      - {args.outdir}/meta/features_config.json
    """).strip())


if __name__ == "__main__":
    main()