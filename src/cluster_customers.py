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
from numpy.typing import ArrayLike
from scipy.spatial.distance import pdist
from scipy.stats import entropy as scipy_entropy, pearsonr
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# 選用：若環境無 umap/hdbscan，可用 --no-umap 與不加 --clusterer hdbscan
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
        description="Customer clustering (A+B version: PCA/UMAP embeds, KMeans/GMM/Spectral/HDBSCAN, target-aware weighting)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基礎 I/O
    parser.add_argument("--sales", required=True, help="Path to sales CSV")
    parser.add_argument("--customers", required=True, help="Path to customers CSV")
    parser.add_argument("--products", required=True, help="Path to products CSV")
    parser.add_argument("--outdir", default="output", help="Output directory")

    # 搜尋空間與隨機性
    parser.add_argument("--kmin", type=int, default=4, help="Min K for search")
    parser.add_argument("--kmax", type=int, default=8, help="Max K for search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # A 版保留/兼容參數
    parser.add_argument("--winsor", type=float, default=99.5, help="Legacy winsor for Monetary (kept for BC)")
    parser.add_argument("--log-monetary", action="store_true", help="Use log transform for Monetary")
    parser.add_argument("--price-bins", type=int, default=4, help="Global price bins for legacy price-share")
    parser.add_argument("--no-umap", action="store_true", help="Disable UMAP plots if installed")
    parser.add_argument("--use-minibatch", action="store_true", help="Use MiniBatchKMeans for speed during K search")

    # A 版新增（保留）
    parser.add_argument("--pca-var", type=float, default=0.0, help="If >0 and embed-for-clustering=pca, retain this variance")
    parser.add_argument("--min-frequency", type=int, default=1, help="Minimum invoices required to keep a customer")
    parser.add_argument("--drop-bottom-monetary-quantile", type=float, default=0.0, help="Drop customers below monetary quantile")
    parser.add_argument("--robust-scale", action="store_true", help="Use RobustScaler")
    parser.add_argument("--log-avg-ticket", action="store_true", help="Apply winsor+log1p to avg_ticket")
    parser.add_argument("--winsor-pct", type=float, default=98.5, help="Winsor top percentile for heavy-tailed monetary/avg_ticket")

    # B 版：嵌入空間
    parser.add_argument("--embed-for-clustering", choices=["none", "pca", "umap"], default="none",
                        help="Choose feature space for clustering")
    parser.add_argument("--umap-dim", type=int, default=10, help="UMAP embedding dimension for clustering")
    parser.add_argument("--umap-neighbors", type=int, default=50, help="UMAP neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP metric")
    parser.add_argument("--umap-random-state", type=int, default=42, help="UMAP random_state")

    # B 版：分群器
    parser.add_argument("--clusterer", choices=["kmeans", "gmm", "spectral", "hdbscan"], default="kmeans",
                        help="Clustering algorithm to use")

    # GMM 選項
    parser.add_argument("--gmm-cov", choices=["full", "diag"], default="full", help="GMM covariance_type")
    parser.add_argument("--gmm-ninit", type=int, default=5, help="GMM initializations to try")
    parser.add_argument("--gmm-reg", type=float, default=1e-6, help="GMM reg_covar")
    parser.add_argument("--gmm-select-by", choices=["bic", "silhouette"], default="bic", help="Criterion to pick K for GMM")

    # Spectral 選項
    parser.add_argument("--spectral-kmin", type=int, default=None, help="Min K for Spectral (fallback to --kmin if None)")
    parser.add_argument("--spectral-kmax", type=int, default=None, help="Max K for Spectral (fallback to --kmax if None)")
    parser.add_argument("--spectral-affinity", choices=["rbf"], default="rbf", help="Spectral affinity (rbf)")
    parser.add_argument("--spectral-gamma", type=str, default="auto", help="Gamma value or 'auto' for 1/median_distance")
    parser.add_argument("--spectral-ninit", type=int, default=20, help="n_init for final KMeans inside spectral")
    parser.add_argument("--spectral-norm-laplacian", action="store_true", help="Use normalized Laplacian (default True)",
                        default=True)

    # HDBSCAN 選項
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=30, help="HDBSCAN min_cluster_size")
    parser.add_argument("--hdbscan-min-samples", type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument("--hdbscan-metric", type=str, default="euclidean", help="HDBSCAN metric")
    parser.add_argument("--hdbscan-probability", action="store_true", help="Return soft membership probabilities")
    parser.add_argument("--allow-noise-cluster", action="store_true", help="Keep -1 as noise cluster")

    # 目標導向
    parser.add_argument("--target-recent", choices=["none", "monetary_ratio", "monetary_sum"], default="none",
                        help="Target for target-aware weighting")
    parser.add_argument("--target-weight", type=float, default=0.0, help="Additional weight for top-K features correlated with target")
    parser.add_argument("--target-topk", type=int, default=6, help="Top-K features by |Pearson r| to weight")

    # 穩健性
    parser.add_argument("--stability-seeds", type=int, default=0, help="If >0, run repeated fits with different seeds for stability")
    parser.add_argument("--report-stability", action="store_true", help="Output stability report with NMI/ARI", default=True)
    parser.add_argument("--save-embeddings", action="store_true", help="Save cluster embedding coordinates", default=True)

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
    # 嘗試常見格式，若第一個失敗則 coerce 再 dropna
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
    merged = df.merge(products, how="left", on="product_id")

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
    recent_90d_monetary = cust_recent_amount[True].astype(float)

    # 商場占比
    mall_counts = invoice_stats.pivot_table(index="customer_id", columns="shopping_mall", values="invoice_no", aggfunc="count", fill_value=0)
    for mall in ["MK", "TKO", "ST", "CYB"]:
        if mall not in mall_counts.columns:
            mall_counts[mall] = 0
    mall_counts = mall_counts[["MK", "TKO", "ST", "CYB"]]
    mall_share = mall_counts.div(mall_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # 類目占比
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
    inv_price_median = trans.groupby("invoice_no")["price"].median()
    global_prices = trans["price"].values
    def to_quantile(v: float) -> float:
        return float((global_prices <= v).mean()) if not np.isnan(v) else 0.0
    inv_price_q = inv_price_median.apply(to_quantile)
    inv_price_q = inv_price_q.to_frame("price_q").join(
        invoice_stats.set_index("invoice_no")[["customer_id"]],
        how="left"
    )
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
        "recent_90d_monetary": recent_90d_monetary,
        "durable_share": durable_share,
        "fmcg_share": fmcg_share,
        "clothing_share": clothing_share,
        "top_category_share": top_category_share,
        "category_entropy": category_entropy,
        "pay_cash_share": pay_cash_share,
        "pay_credit_share": pay_credit_share,
        "noncash_share": noncash_share,
    }).fillna(0.0)

    # 合併商城占比
    feat = feat.join(mall_share, how="left")

    # 加入簡化的 top_category one-hot（前 3 名）
    topcats = top_category.value_counts().index.tolist()[:3]
    topcat_ohe = pd.get_dummies(top_category)
    keep_top = [c for c in topcat_ohe.columns if c in topcats]
    for c in keep_top:
        feat[f"topcat_{c}"] = topcat_ohe[c]
    if len(keep_top) > 0:
        feat["topcat_other"] = 1 - topcat_ohe[keep_top].sum(axis=1)
    else:
        feat["topcat_other"] = 1

    # 加入價格分位特徵
    feat = feat.join(cust_price_q, how="left").fillna(0.0)

    # 保留原本的性別 one-hot
    feat = feat.join(gender_dummies, how="left").fillna(0.0)

    # 原本價格桶占比（兼容舊圖表）
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

    # monetary：winsor + log1p（若指定）
    if use_log_monetary:
        if winsor_pct is None or math.isnan(winsor_pct):
            winsor_pct = winsor_pct_legacy
        f["monetary"] = _winsorize_series(f["monetary"], winsor_pct)
        f["monetary"] = np.log1p(np.clip(f["monetary"], 0, None))
        m_meta = {"type": "winsor+log1p", "upper_pct": winsor_pct}
    else:
        upper = np.nanpercentile(f["monetary"], winsor_pct_legacy)
        f["monetary"] = np.minimum(f["monetary"], upper)
        m_meta = {"type": "winsor", "upper_pct": winsor_pct_legacy, "upper_value": float(upper)}

    # avg_ticket
    if log_avg_ticket and "avg_ticket" in f.columns:
        f["avg_ticket"] = _winsorize_series(f["avg_ticket"], winsor_pct if winsor_pct is not None else 98.5)
        f["avg_ticket"] = np.log1p(np.clip(f["avg_ticket"], 0, None))

    # gap/變異類平滑
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


def median_pairwise_distance(x: np.ndarray, sample_cap: int = 5000) -> float:
    n = x.shape[0]
    if n > sample_cap:
        idx = np.random.choice(n, size=sample_cap, replace=False)
        x = x[idx]
    d = pdist(x, metric="euclidean")
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(np.median(d))


def embed_for_clustering(x_scaled: np.ndarray, args) -> Tuple[np.ndarray, Dict]:
    meta = {"method": args.embed_for_clustering}
    if args.embed_for_clustering == "none":
        return x_scaled, meta
    elif args.embed_for_clustering == "pca":
        if not (args.pca_var and args.pca_var > 0):
            # 若未設定 pca_var，保留 0.9 作為合理預設
            pca_var = 0.9
        else:
            pca_var = args.pca_var
        pca = PCA(n_components=pca_var, svd_solver="full", random_state=args.seed)
        x_emb = pca.fit_transform(x_scaled)
        meta.update({
            "pca_var": float(pca_var),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components": int(x_emb.shape[1]),
        })
        return x_emb, meta
    elif args.embed_for_clustering == "umap":
        if not UMAP_AVAILABLE:
            print("[WARN] UMAP not available. Falling back to no embedding.", file=sys.stderr)
            return x_scaled, {"method": "none", "fallback": "umap_unavailable"}
        reducer = umap.UMAP(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            random_state=args.umap_random_state,
        )
        x_emb = reducer.fit_transform(x_scaled)
        meta.update({
            "umap_dim": int(args.umap_dim),
            "umap_neighbors": int(args.umap_neighbors),
            "umap_min_dist": float(args.umap_min_dist),
            "umap_metric": args.umap_metric,
            "n_components": int(x_emb.shape[1]),
        })
        return x_emb, meta
    else:
        return x_scaled, meta


def kmeans_k_search(x: np.ndarray, kmin: int, kmax: int, seed: int, use_minibatch: bool, figs_dir: str):
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
        sse_list.append(model.inertia_)
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


def gmm_k_search(x: np.ndarray, kmin: int, kmax: int, seed: int, cov_type: str, ninit: int, reg: float, select_by: str):
    best_model = None
    best_k = None
    criterion_values = []
    for k in range(kmin, kmax + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type=cov_type, n_init=ninit,
            reg_covar=reg, random_state=seed
        )
        gmm.fit(x)
        labels = gmm.predict(x)
        if select_by == "bic":
            crit = gmm.bic(x)
        else:
            try:
                crit = -silhouette_score(x, labels)  # 負號，越小越好
            except Exception:
                crit = np.inf
        criterion_values.append((k, float(crit), labels, gmm))

    if select_by == "bic":
        best = min(criterion_values, key=lambda t: t[1])
    else:
        best = min(criterion_values, key=lambda t: t[1])
    best_k = int(best[0])
    best_model = best[3]
    labels = best_model.predict(x)
    out = {
        "select_by": select_by,
        "scores": [{"k": int(k), select_by: float(val)} for k, val, _, _ in criterion_values],
    }
    return best_k, best_model, labels, out


def spectral_k_search(x: np.ndarray, kmin: int, kmax: int, seed: int, affinity: str, gamma_param, ninit: int, norm_laplacian: bool):
    # 決定 gamma
    if isinstance(gamma_param, str) and gamma_param == "auto":
        med = median_pairwise_distance(x)
        gamma = 1.0 / max(med, 1e-9)
    else:
        gamma = float(gamma_param)
    k_list = list(range(kmin, kmax + 1))
    best_k = None
    best_sil = -1.0
    best_labels = None
    sil_list = []
    for k in k_list:
        sc = SpectralClustering(
            n_clusters=k,
            affinity=affinity,
            gamma=gamma,
            assign_labels="kmeans",
            random_state=seed,
            n_init=ninit,
        )
        labels = sc.fit_predict(x)
        try:
            sil = silhouette_score(x, labels)
        except Exception:
            sil = np.nan
        sil_list.append(sil)
        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    info = {
        "k_list": k_list,
        "silhouette": [None if np.isnan(v) else float(v) for v in sil_list],
        "gamma": float(gamma),
        "norm_laplacian": bool(norm_laplacian),
    }
    return best_k, best_labels, info


def hdbscan_cluster(x: np.ndarray, min_cluster_size: int, min_samples: int, metric: str, probability: bool):
    if not HDBSCAN_AVAILABLE:
        raise RuntimeError("HDBSCAN not available")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
    labels = clusterer.fit_predict(x)
    probs = getattr(clusterer, "probabilities_", None) if probability else None
    return labels, probs


def plot_2d_embeddings(x: np.ndarray, labels: np.ndarray, figs_dir: str, seed: int, use_umap_plot: bool):
    pca = PCA(n_components=2, random_state=seed)
    x_pca = pca.fit_transform(x)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=labels, palette="tab10", s=10, linewidth=0)
    plt.title("Clusters (PCA 2D of cluster space)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "cluster_space_pca2.png"), dpi=150)
    plt.close()

    if use_umap_plot and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=seed)
        x_umap = reducer.fit_transform(x)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=x_umap[:, 0], y=x_umap[:, 1], hue=labels, palette="tab10", s=10, linewidth=0)
        plt.title("Clusters (UMAP 2D of cluster space)")
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "cluster_space_umap2.png"), dpi=150)
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


def target_aware_weighting(feat_df: pd.DataFrame, args) -> Dict:
    meta = {"enabled": False}
    if args.target_recent == "none" or args.target_weight <= 0 or args.target_topk <= 0:
        return meta

    # 選擇目標
    if args.target_recent == "monetary_ratio":
        if "monetary_recent_ratio" not in feat_df.columns:
            return meta
        y = feat_df["monetary_recent_ratio"].values.astype(float)
        y_name = "monetary_recent_ratio"
    else:
        if "recent_90d_monetary" not in feat_df.columns:
            return meta
        y = feat_df["recent_90d_monetary"].values.astype(float)
        y_name = "recent_90d_monetary"

    # 避免常數 y
    if np.allclose(np.std(y), 0.0):
        return meta

    # 計算與每個數值特徵的皮爾森相關
    numeric_cols = [c for c in feat_df.columns if np.issubdtype(feat_df[c].dtype, np.number)]
    scores = []
    for c in numeric_cols:
        x = feat_df[c].values.astype(float)
        if np.allclose(np.std(x), 0.0):
            continue
        try:
            r, _ = pearsonr(x, y)
        except Exception:
            continue
        if not np.isnan(r):
            scores.append((c, float(abs(r)), float(r)))

    if not scores:
        return meta

    scores.sort(key=lambda t: t[1], reverse=True)
    top = scores[: args.target_topk]
    mult = 1.0 + float(args.target_weight)
    for col, _, _ in top:
        feat_df[col] = feat_df[col] * mult

    meta.update({
        "enabled": True,
        "target": y_name,
        "weight_multiplier": mult,
        "top_features": [{"feature": c, "abs_r": ar, "r": r} for c, ar, r in top],
    })
    return meta


def stability_repeats(x_for_cluster: np.ndarray, feat_df: pd.DataFrame, base_args, cluster_func):
    seeds = [base_args.seed + i + 1 for i in range(base_args.stability_seeds)]
    labels_list = []
    for s in seeds:
        args = argparse.Namespace(**vars(base_args))
        args.seed = s
        labels, _info = cluster_func(x_for_cluster, args)
        labels_list.append(np.asarray(labels))

    # 與主結果對齊後計算 ARI/NMI
    if not labels_list:
        return None
    base_labels, _ = cluster_func(x_for_cluster, base_args)
    report = {"seeds": seeds, "base_seed": base_args.seed, "pairs": []}
    for s, labs in zip(seeds, labels_list):
        ari = float(adjusted_rand_score(base_labels, labs))
        nmi = float(normalized_mutual_info_score(base_labels, labs))
        report["pairs"].append({"seed": int(s), "ARI": ari, "NMI": nmi})
    report["ARI_mean"] = float(np.mean([p["ARI"] for p in report["pairs"]]))
    report["NMI_mean"] = float(np.mean([p["NMI"] for p in report["pairs"]]))
    return report


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

    # 5) 全品項價格分位切割（僅供舊的價格桶佔比）
    price_edges = price_bins_from_products(products, n_bins=args.price_bins)

    # 6) 建立顧客特徵（未標準化）
    feat_raw, feat_meta = build_customer_features(trans, reference_date, price_edges)

    # A 版篩選：min-frequency 與 monetary 下分位
    tmp = pd.DataFrame({"frequency": feat_raw["frequency"], "monetary": feat_raw["monetary"]})
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

    # 7) 變換：monetary/avg_ticket，gap 類平滑
    feat_m, transform_meta = transform_monetary_and_related(
        feat_raw,
        winsor_pct_legacy=args.winsor,
        use_log_monetary=args.log_monetary,
        winsor_pct=args.winsor_pct,
        log_avg_ticket=args.log_avg_ticket,
    )

    # 8) 目標導向加權（在標準化前、嵌入前）
    target_meta = target_aware_weighting(feat_m, args)

    # 9) 保存擴充特徵（原尺度 after transform/weighting，未標準化）
    feat_m.to_csv(os.path.join(args.outdir, "features_extended.csv"))

    # 10) 標準化
    x_scaled, scaler = standardize_features(feat_m, robust=args.robust_scale)

    # 11) 嵌入空間供聚類
    x_for_cluster, embed_meta = embed_for_clustering(x_scaled, args)

    # 12) 聚類（按照 --clusterer）
    def cluster_once(xc: np.ndarray, local_args):
        algo = local_args.clusterer
        if algo == "kmeans":
            best_k, best_model, ksearch_info = kmeans_k_search(
                xc, local_args.kmin, local_args.kmax, local_args.seed, local_args.use_minibatch, figs_dir
            )
            labels = best_model.predict(xc)
            info = {"algo": "kmeans", "best_k": int(best_k), "k_search": ksearch_info}
            return labels, info
        elif algo == "gmm":
            best_k, best_model, labels, gmm_info = gmm_k_search(
                xc, local_args.kmin, local_args.kmax, local_args.seed,
                local_args.gmm_cov, local_args.gmm_ninit, local_args.gmm_reg, local_args.gmm_select_by
            )
            # GMM 組件摘要
            comps = []
            means = best_model.means_
            weights = best_model.weights_
            comps = [{"component": int(i), "weight": float(weights[i])} for i in range(len(weights))]
            gmm_info.update({
                "algo": "gmm",
                "best_k": int(best_k),
                "components": comps,
            })
            return labels, gmm_info
        elif algo == "spectral":
            s_kmin = local_args.spectral_kmin if local_args.spectral_kmin is not None else local_args.kmin
            s_kmax = local_args.spectral_kmax if local_args.spectral_kmax is not None else local_args.kmax
            best_k, labels, sp_info = spectral_k_search(
                xc, s_kmin, s_kmax, local_args.seed, local_args.spectral_affinity,
                local_args.spectral_gamma, local_args.spectral_ninit, local_args.spectral_norm_laplacian
            )
            info = {"algo": "spectral", "best_k": int(best_k), "search": sp_info}
            return labels, info
        elif algo == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                raise RuntimeError("HDBSCAN not available. Install hdbscan or choose another clusterer.")
            labels, probs = hdbscan_cluster(
                xc, local_args.hdbscan_min_cluster_size, local_args.hdbscan_min_samples,
                local_args.hdbscan_metric, local_args.hdbscan_probability
            )
            if not local_args.allow_noise_cluster:
                # 將 -1 噪聲指派到最近的已標記群（最近均值）—簡單後處理
                valid = labels >= 0
                if valid.any() and (~valid).any():
                    centers = []
                    for k in np.unique(labels[valid]):
                        centers.append(xc[labels == k].mean(axis=0))
                    centers = np.vstack(centers)
                    center_ids = np.unique(labels[valid])
                    noise_idx = np.where(~valid)[0]
                    for idx in noise_idx:
                        d = np.linalg.norm(centers - xc[idx], axis=1)
                        labels[idx] = center_ids[np.argmin(d)]
            info = {
                "algo": "hdbscan",
                "min_cluster_size": int(local_args.hdbscan_min_cluster_size),
                "min_samples": int(local_args.hdbscan_min_samples),
                "metric": local_args.hdbscan_metric,
            }
            return labels, info
        else:
            raise ValueError(f"Unknown clusterer: {algo}")

    labels, model_info = cluster_once(x_for_cluster, args)

    # 13) 可視化（在聚類空間上做 2D PCA/UMAP）
    use_umap_plot = (not args.no_umap) and UMAP_AVAILABLE
    if (not args.no_umap) and (not UMAP_AVAILABLE):
        print("[INFO] umap-learn 未安裝或不可用，將僅輸出 PCA 圖。", file=sys.stderr)
    plot_2d_embeddings(x_for_cluster, labels, figs_dir, args.seed, use_umap_plot=use_umap_plot)
    plot_cluster_radar(feat_m, labels, figs_dir)

    # 14) 儲存聚類空間座標
    if args.save_embeddings:
        emb_df = pd.DataFrame(x_for_cluster, index=feat_m.index)
        emb_df.index.name = "customer_id"
        emb_df.to_csv(os.path.join(args.outdir, f"embeddings_{embed_meta['method']}.csv"))

    # 15) 輸出結果表
    out_customers, summary = save_cluster_outputs(feat_m.index, labels, feat_m, args.outdir)

    # 16) silhouette（原標準化空間與聚類空間）
    sil_full = None
    sil_cluster = None
    try:
        sil_full = float(silhouette_score(x_scaled, labels))
    except Exception:
        sil_full = None
    try:
        sil_cluster = float(silhouette_score(x_for_cluster, labels))
    except Exception:
        sil_cluster = None

    # 17) 穩健性評估（可選）
    stability = None
    if args.stability_seeds and args.stability_seeds > 0 and args.report_stability:
        stability = stability_repeats(x_for_cluster, feat_m, args, cluster_once)
        if stability is not None:
            with open(os.path.join(meta_dir, "stability_report.json"), "w", encoding="utf-8") as f:
                json.dump(stability, f, indent=2, ensure_ascii=False)

    # 18) 輸出模型/流程的 meta
    meta = {
        "reference_date": str(reference_date.date()),
        "feature_columns": feat_meta["feature_columns"],
        "selected_feature_columns": feat_m.columns.tolist(),
        "monetary_transform": transform_meta,
        "target_weighting": target_meta,
        "scaler": "RobustScaler" if args.robust_scale else "StandardScaler",
        "kmin": int(args.kmin),
        "kmax": int(args.kmax),
        "random_seed": int(args.seed),
        "embed": embed_meta,
        "clusterer": model_info,
        "silhouette_full": sil_full,
        "silhouette_cluster_space": sil_cluster,
        "filtering_rules": {
            "min_frequency": int(args.min_frequency),
            "drop_bottom_monetary_quantile": float(args.drop_bottom_monetary_quantile),
            "filtered_out_count": int(len(filtered_out)),
        },
    }
    with open(os.path.join(meta_dir, "clusters_model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 額外輸出：若是 GMM，輸出組件摘要
    if model_info.get("algo") == "gmm":
        gmm = None
        # 我們無法直接取到 best_model 物件這裡（在函式內），因此重新 fit 一次以輸出均值/權重（對於固定最佳 K）
        best_k = int(model_info["best_k"])
        gmm = GaussianMixture(
            n_components=best_k, covariance_type=args.gmm_cov, n_init=args.gmm_ninit,
            reg_covar=args.gmm_reg, random_state=args.seed
        ).fit(x_for_cluster)
        comp_df = pd.DataFrame({
            "component": np.arange(best_k),
            "weight": gmm.weights_,
        })
        comp_df.to_csv(os.path.join(args.outdir, "gmm_components_summary.csv"), index=False)

    # 譜式的 affinity/gamma 摘要
    if model_info.get("algo") == "spectral":
        sp = model_info.get("search", {})
        with open(os.path.join(meta_dir, "spectral_affinity_stats.json"), "w", encoding="utf-8") as f:
            json.dump(sp, f, indent=2, ensure_ascii=False)

    # 19) HDBSCAN 補充探索（沿用 A 版，可關閉）
    # 保留：若使用者另外想看 HDBSCAN 在 cluster space 的表現
    # 這裡僅視覺化，不覆蓋主分群
    _ = run_hdbscan_exploration(x_for_cluster, figs_dir)

    # 20) 完成提示
    print(textwrap.dedent(f"""
    完成！（B 版增強）
    - 聚類方法 = {model_info.get('algo')}
    - 嵌入空間 = {embed_meta.get('method')}
    - 最佳 K（若適用） = {model_info.get('best_k', 'N/A')}
    - silhouette（原空間）= {meta["silhouette_full"]}
    - silhouette（聚類空間）= {meta["silhouette_cluster_space"]}
    - 穩健性（若啟用）：{ ('ARI_mean=' + str(stability.get('ARI_mean')) + ', NMI_mean=' + str(stability.get('NMI_mean'))) if stability else '未啟用'}
    - 輸出：
      - {args.outdir}/customers_clusters.csv
      - {args.outdir}/clusters_summary.csv
      - {args.outdir}/features_extended.csv
      - {args.outdir}/embeddings_{embed_meta.get('method')}.csv
      - {args.outdir}/figures/
      - {args.outdir}/logs/etl_issues.log
      - {args.outdir}/meta/clusters_model_meta.json
      - {args.outdir}/meta/stability_report.json（若啟用）
      - {args.outdir}/meta/spectral_affinity_stats.json（若使用 spectral）
      - {args.outdir}/gmm_components_summary.csv（若使用 GMM）
    """).strip())


if __name__ == "__main__":
    main()