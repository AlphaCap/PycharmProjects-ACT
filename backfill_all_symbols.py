from data_update import get_sp500_symbols, get_polygon_daily_data, save_price_data
import time
import os

# --- POLYGON API KEY (HARDCODED FOR PRIVATE REPO) ---
POLYGON_API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"  # <-- Your current Polygon API key
os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY

# --- CONFIGURATION ---
HISTORY_DAYS = 200           # Number of days of history to fetch (adjust as needed)
SLEEP_SECONDS = 12            # Polygon free API: 5 requests per min = 12 sec per call

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
