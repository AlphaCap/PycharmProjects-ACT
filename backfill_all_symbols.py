import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from data_update import get_sp500_symbols, save_price_data

def get_polygon_daily_data(symbol, days):
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY environment variable is not set.")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            if not df.empty:
                df["Date"] = pd.to_datetime(df["t"], unit="ms")
                df.rename(columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume"
                }, inplace=True)
                return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return pd.DataFrame()

# --- CONFIGURATION ---
HISTORY_DAYS = 1000
SLEEP_SECONDS = 12

if not os.getenv("POLYGON_API_KEY"):
    raise RuntimeError(
        "POLYGON_API_KEY environment variable is not set.\n"
        "Set it in your terminal before running this script."
    )

symbols = get_sp500_symbols()
print(f"Starting sequential backfill for {len(symbols)} symbols, {HISTORY_DAYS} days each.")
print(f"Respecting Polygon free tier rate limit: 1 request every {SLEEP_SECONDS} seconds.")

for idx, symbol in enumerate(symbols):
    print(f"({idx+1}/{len(symbols)}) Downloading {symbol} ...")
    try:
        df = get_polygon_daily_data(symbol, HISTORY_DAYS)
        if df is not None and not df.empty:
            save_price_data(symbol, df)
            print(f"Downloaded and saved {len(df)} rows for {symbol}.")
        else:
            print(f"Warning: No data returned for {symbol}.")
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")
    time.sleep(SLEEP_SECONDS)

print("Backfill complete.")
