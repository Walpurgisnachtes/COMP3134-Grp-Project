# Customer Clustering Pipeline

A reproducible Python pipeline to cluster customers using RFM and preference features, with KMeans (and optional HDBSCAN) plus visualizations. It reads three CSVs (sales, customers, products), performs ETL and feature engineering, standardizes features, searches for the best K, fits KMeans, visualizes clusters (PCA/UMAP), and exports results.

- Main script: src/cluster_customers.py
- Setup scripts: setup_and_run.sh (macOS/Linux), setup_and_run.bat (Windows)
- Dependencies: requirements.txt
- Input data directory: data/

## Contents

- Features
- Project structure
- Data schema
- Quick start
- CLI usage
- Outputs
- Methods and assumptions
- Troubleshooting
- Performance tips
- FAQ
- License

## Features

- ETL with logging for data quality issues
- RFM features: recency, frequency, monetary
- Preference features:
  - Category shares (Electronics, Clothing, Groceries, Books, Toys)
  - Shopping mall shares (MK, TKO, ST, CYB)
  - Global price-band preference shares (quantile-based)
  - Payment method shares (Mobile Payment, Credit Card, Cash)
  - Demographics: age, gender one-hot
  - Basket stats: avg ticket, avg items, invoice amount std
- Robust monetary transform: Winsorization (default) or log1p
- Standardization with scikit-learn StandardScaler
- K search via SSE (elbow) and silhouette; MiniBatchKMeans optional
- 2D embeddings: PCA and optional UMAP
- Optional density clustering exploration with HDBSCAN
- Reproducible outputs and meta/config JSON

## Project structure

- src/cluster_customers.py
- requirements.txt
- setup_and_run.sh
- setup_and_run.bat
- data/
  - sales_27.csv
  - customers_27.csv
  - products_27.csv
- output/ (created on run)
  - figures/
  - logs/
  - meta/

## Data schema

Place these CSVs in data/. Column names are sanitized to lowercase with underscores; still, the following columns are expected.

- sales_27.csv
  - invoice_no: string unique invoice id (duplicates are dropped, first kept)
  - customer_id: string/id
  - product_id_list: comma-separated product_id list per invoice (e.g., "P01,P02,P05")
  - invoice_date: DD/MM/YYYY
  - shopping_mall: categorical among MK, TKO, ST, CYB

- customers_27.csv
  - customer_id
  - gender: e.g., Male/Female/Other (one-hot encoded)
  - age: numeric
  - payment_method: Mobile Payment / Credit Card / Cash (used as a proxy for share; transactions also counted)

- products_27.csv
  - product_id
  - category: Electronics, Clothing, Groceries, Books, Toys
  - price: numeric

Notes:
- Invalid invoice_date rows are dropped and logged.
- Unknown product_id rows after exploding product_id_list are dropped and logged.
- Transactions with missing core customer profile fields (gender, age, payment_method) are dropped and logged.

## Quick start

macOS/Linux:
1) Make script executable
- chmod +x setup_and_run.sh
2) Run with defaults
- ./setup_and_run.sh
3) Optional flags, e.g.
- ./setup_and_run.sh --no-umap
- ./setup_and_run.sh --run-hdbscan

Windows:
1) Double-click or execute in cmd
- setup_and_run.bat
2) Optional flags
- setup_and_run.bat --no-umap
- setup_and_run.bat --run-hdbscan

The scripts will:
- Create a .venv virtual environment
- Install dependencies
- Validate input files
- Run the pipeline with default parameters

## CLI usage

You can also run the Python script directly.

- python src/cluster_customers.py --help

Key arguments:
- --sales PATH: sales CSV path (required)
- --customers PATH: customers CSV path (required)
- --products PATH: products CSV path (required)
- --outdir DIR: output directory (default: output)
- --kmin INT: min K for search (default: 4)
- --kmax INT: max K for search (default: 8)
- --seed INT: random seed (default: 42)
- --winsor FLOAT: top percentile for monetary winsorization (default: 99.5)
- --log-monetary: use log1p for monetary instead of winsor (mutually exclusive)
- --price-bins INT: number of global price buckets (default: 4)
- --no-umap: disable UMAP plots, even if installed
- --run-hdbscan: run HDBSCAN cluster exploration (if installed)
- --use-minibatch: use MiniBatchKMeans during K search for speed

Examples:
- python src/cluster_customers.py --sales data/sales_27.csv --customers data/customers_27.csv --products data/products_27.csv
- python src/cluster_customers.py --sales data/sales_27.csv --customers data/customers_27.csv --products data/products_27.csv --kmin 3 --kmax 10 --log-monetary --no-umap

## Outputs

Created under outdir (default: output):

- customers_clusters.csv
  - Columns: customer_id, cluster, plus selected original-scale features (recency_days, frequency, monetary, avg_ticket, avg_items, category shares, payment shares, age)
- clusters_summary.csv
  - Per-cluster counts and means of key features and shares
- figures/
  - k_selection_sse.png: SSE vs K
  - k_selection_silhouette.png: silhouette vs K
  - pca_clusters.png: 2D PCA scatter colored by cluster
  - umap_clusters.png: 2D UMAP scatter (if enabled)
  - cluster_radar.png: normalized mean feature radar
  - hdbscan_pca.png: HDBSCAN PCA plot (if run)
- logs/etl_issues.log
  - Dropped/invalid rows summary
- meta/features_config.json
  - Reference date, price quantile edges, feature columns, monetary transform details, scaler params, K search metrics, best_k, seed, umap_enabled

Console output shows the detected best K and key file paths.

## Methods and assumptions

- Reference date is the max invoice_date in sales
- Price buckets are global quantiles computed from products.price with deduped edges
- Monetary transform
  - Winsorization caps extreme values at the specified percentile to reduce outlier influence
  - Alternatively log1p stabilizes variance; choose via --log-monetary
- Standardization using StandardScaler across all features
- K search uses inertia (SSE) and silhouette; best_k is selected by max silhouette among tested Ks
- PCA/UMAP are for visualization only; clustering operates in standardized feature space
- HDBSCAN is optional exploratory clustering; results are not exported as main labels

## Troubleshooting

- Missing files: Ensure data/*.csv exist with the exact filenames used in scripts or pass custom paths via CLI.
- Date parsing errors: invoice_date must be DD/MM/YYYY; invalid rows are dropped and logged.
- Empty or tiny dataset: If best_k cannot be determined, verify sufficient customers and transactions after cleaning.
- UMAP/HDBSCAN not installed: The pipeline will continue without them. Install via requirements or use --no-umap.
- Memory/performance:
  - Use --use-minibatch for faster K search on large datasets
  - Reduce --kmax or feature set if needed
  - Ensure you’re running in a 64-bit Python with enough RAM

## Performance tips

- Start with --kmin 3 --kmax 10 and --use-minibatch for speed
- For very large data, consider:
  - Increasing MiniBatchKMeans batch_size
  - Sampling customers for initial K search, then refit on full data
- Disable UMAP (--no-umap) to save time

## FAQ

- Can I customize features?
  - Yes. Edit build_customer_features in src/cluster_customers.py. We can add seasonality (month/weekday shares), weekend share, time-since-first-purchase, or product affinities.
- My categories/malls differ.
  - The code ensures expected columns; unseen values become zeroed columns. Update the fixed lists in build_customer_features if your taxonomy differs.
- How are gender and payment handled?
  - gender is one-hot; payment_method shares are inferred from transactions. If only a single default per customer exists, it still contributes as a share.

## License

Provide your project’s license here (e.g., MIT). If unsure, add a LICENSE file at the repo root.

## Acknowledgements

Built with pandas, scikit-learn, umap-learn, hdbscan, matplotlib, seaborn, and scipy.

Need help tailoring features or plots to your data? Open an issue or ask for adjustments.