# Customer Clustering Pipeline

A reproducible Python pipeline to cluster customers using RFM and preference features, with KMeans (and optional HDBSCAN) plus visualizations. It reads three CSVs (sales, customers, products), performs ETL and feature engineering, standardizes features, searches for the best K, fits KMeans, visualizes clusters (PCA/UMAP), and exports results.

- Main script: src/cluster_customers.py
- Setup scripts: setup_and_run.sh (macOS/Linux), setup_and_run.bat (Windows)
- Dependencies: requirements.txt
- Input data directory: data/

## Contents

- Features
- Project structure
- Requirements
- Quick start
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
- setup_and_run.bat
- data/
  - sales_27.csv
  - customers_27.csv
  - products_27.csv
- output/ (created on run)
  - figures/
  - logs/
  - meta/

## Requirements
- Python 3.12.6

## Quick start

Find an algorithm you like in `bat_examples.md` and run in PowerShell, for example:
`./setup_and_run.bat --embed-for-clustering umap --clusterer kmeans --kmin 3 --kmax 8 --target-recent monetary_ratio --target-weight 0.3 --target-topk 6 --log-monetary`

The scripts will:
- Create a .venv virtual environment
- Install dependencies
- Validate input files
- Run the pipeline

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

- Start with --kmin 3 --kmax 8
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

## Acknowledgements

Built with pandas, scikit-learn, umap-learn, hdbscan, matplotlib, seaborn, and scipy.

Need help tailoring features or plots to your data? Open an issue or ask for adjustments.