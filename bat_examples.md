- UMAP 10 維 + KMeans

`./setup_and_run.bat --embed-for-clustering umap --umap-dim 10 --umap-neighbors 50 --umap-min-dist 0.1 --umap-metric cosine --kmin 3 --kmax 8 --use-minibatch --log-monetary --log-avg-ticket --min-frequency 2 --drop-bottom-monetary-quantile 0.1`

- 譜式分群（RBF，自動 gamma）

`./setup_and_run.bat --embed-for-clustering umap --umap-dim 10 --clusterer spectral --spectral-kmin 3 --spectral-kmax 8 --spectral-affinity rbf --spectral-gamma auto --log-monetary --min-frequency 2 --drop-bottom-monetary-quantile 0.1`

- GMM 搜尋（BIC 選 K）

`./setup_and_run.bat --embed-for-clustering pca --pca-var 0.9 --clusterer gmm --gmm-cov full --gmm-ninit 5 --gmm-select-by bic --kmin 3 --kmax 8 --log-monetary`

- HDBSCAN（非球狀群；允許噪聲）

`./setup_and_run.bat --embed-for-clustering umap --umap-dim 10 --clusterer hdbscan --hdbscan-min-cluster-size 40 --hdbscan-min-samples 10 --allow-noise-cluster --log-monetary`

- 目標導向加權 [正在使用中]

`./setup_and_run.bat --embed-for-clustering umap --clusterer kmeans --kmin 3 --kmax 8 --target-recent monetary_ratio --target-weight 0.3 --target-topk 6 --log-monetary`