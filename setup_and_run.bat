@echo off
setlocal enabledelayedexpansion


REM 基本檔案檢查
if not exist requirements.txt (
  echo [ERROR] requirements.txt not found.
  exit /b 1
)
if not exist src\cluster_customers.py (
  echo [ERROR] src\cluster_customers.py not found.
  exit /b 1
)
if not exist data\sales_27.csv (
  echo [ERROR] data\sales_27.csv not found.
  exit /b 1
)
if not exist data\customers_27.csv (
  echo [ERROR] data\customers_27.csv not found.
  exit /b 1
)
if not exist data\products_27.csv (
  echo [ERROR] data\products_27.csv not found.
  exit /b 1
)

REM 建立 venv
if not exist .venv (
  echo "Creating virtual environment (.venv) ..."
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Ensure Python is installed and on PATH.
    exit /b 1
  )
)

REM 啟動 venv
call .venv\Scripts\activate

REM 升級 pip 並安裝依賴
python -m pip install -U pip
pip install -r requirements.txt

REM 預設參數（可自行修改）
set SALES=data\sales_27.csv
set CUSTOMERS=data\customers_27.csv
set PRODUCTS=data\products_27.csv
set OUTDIR=output
set KMIN=4
set KMAX=8
set PRICEBINS=4
set WINSOR=99.5
set SEED=42

REM 傳遞可變參數（允許使用者在命令尾端追加，例如 --no-umap）
set EXTRA=%*

REM 執行主程式
python src\cluster_customers.py --sales %SALES% --customers %CUSTOMERS% --products %PRODUCTS% --outdir %OUTDIR% --kmin %KMIN% --kmax %KMAX% --price-bins %PRICEBINS% --winsor %WINSOR% --seed %SEED% %EXTRA%

endlocal