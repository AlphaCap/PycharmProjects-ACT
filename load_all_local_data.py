import pandas as pd
import os

# ---------- LOAD ALL STOCKS FROM data/daily/ ----------

stock_dir = "data/daily"
stock_data = {}

for filename in os.listdir(stock_dir):
    if filename.endswith(".csv"):
        symbol = filename.replace(".csv", "")
        file_path = os.path.join(stock_dir, filename)
        try:
            df = pd.read_csv(file_path)
            stock_data[symbol] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

print(f"Loaded {len(stock_data)} stocks from {stock_dir}")

# ---------- LOAD ALL SECTOR ETFS FROM data/etf_historical/ ----------

etf_dir = "data/etf_historical"
etf_data = {}

for filename in os.listdir(etf_dir):
    if filename.endswith("_historical.csv"):
        symbol = filename.replace("_historical.csv", "")
        file_path = os.path.join(etf_dir, filename)
        try:
            df = pd.read_csv(file_path)
            etf_data[symbol] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

print(f"Loaded {len(etf_data)} ETFs from {etf_dir}")

# ---------- STOCK_DATA and ETF_DATA now hold ALL your saved data ----------

# Example: print all loaded stock symbols
print("Stock symbols:", list(stock_data.keys()))
print("ETF symbols:", list(etf_data.keys()))