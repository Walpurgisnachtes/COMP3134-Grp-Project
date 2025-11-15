#!/usr/bin/env bash
set -euo pipefail

# 基本檔案檢查
if [[ ! -f "requirements.txt" ]]; then
  echo "[ERROR] requirements.txt not found."
  exit 1
fi
if [[ ! -f "src/cluster_customers.py" ]]; then
  echo "[ERROR] src/cluster_customers.py not found."
  exit 1
fi
if [[ ! -f "data/sales_27.csv" ]]; then
  echo "[ERROR] data/sales_27.csv not found."
  exit 1
fi
if [[ ! -f "data/customers_27.csv" ]]; then
  echo "[ERROR] data/customers_27.csv not found."
  exit 1
fi
if [[ ! -f "data/products_27.csv" ]]; then
  echo "[ERROR] data/products_27.csv not found."
  exit 1
fi

# 選擇 python 執行器
PYTHON=python3
if ! command -v $PYTHON >/dev/null 2>&1; then
  PYTHON=python
fi
if ! command -v $PYTHON >/dev/null 2>&1; then
  echo "[ERROR] Python is not installed or not in PATH."
  exit 1
fi

# 建立 venv
if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment (.venv) ..."
  $PYTHON -m venv .venv
fi

# 啟動 venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 升級 pip 並安裝依賴
python -m pip install -U pip
pip install -r requirements.txt

# 預設參數（可由使用者於命令尾端追加，例如 --no-umap）
SALES="data/sales_27.csv"
CUSTOMERS="data/customers_27.csv"
PRODUCTS="data/products_27.csv"
OUTDIR="output"
KMIN=4
KMAX=8
PRICEBINS=4
WINSOR=99.5
SEED=42

EXTRA_ARGS=("$@")

# 執行主程式
python src/cluster_customers.py \
  --sales "$SALES" \
  --customers "$CUSTOMERS" \
  --products "$PRODUCTS" \
  --outdir "$OUTDIR" \
  --kmin $KMIN \
  --kmax $KMAX \
  --price-bins $PRICEBINS \
  --winsor $WINSOR \
  --seed $SEED \
  "${EXTRA_ARGS[@]}"