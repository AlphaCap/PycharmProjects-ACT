import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from data_manager import get_sp500_symbols, save_price_data, load_price_data
from nGS_Strategy import NGSStrategy  # ADD THIS IMPORT

# --- CONFIGURATION ---
HISTORY_DAYS = 2
SLEEP_SECONDS = 12

# Set your Polygon API key here if not set in environment
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    POLYGON_API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"
    os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY
    print("Warning: No POLYGON_API_KEY found in environment. Using hardcoded key.")

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

# ADD THIS NEW FUNCTION
def process_with_indicators(symbol, new_df):
    """
    Process the downloaded data to include indicators.
    """
    try:
        # Load existing data if any
        existing_df = load_price_data(symbol)
        
        if not existing_df.empty:
            # Merge with existing data
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            new_df['Date'] = pd.to_datetime(new_df['Date'])
            
            # Remove overlapping dates
            new_dates = set(new_df['Date'].dt.date)
            existing_df = existing_df[~existing_df['Date'].dt.date.isin(new_dates)]
            
            # Combine
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        else:
            combined_df = new_df.copy()
        
        # Calculate indicators
        strategy = NGSStrategy()
        df_with_indicators = strategy.calculate_indicators(combined_df)
        
        if df_with_indicators is not None and not df_with_indicators.empty:
            return df_with_indicators
        else:
            return combined_df
            
    except Exception as e:
        print(f"Error calculating indicators for {symbol}: {e}")
        return new_df

if not os.getenv("POLYGON_API_KEY"):
    raise RuntimeError(
        "POLYGON_API_KEY environment variable is not set.\n"
        "Set it in your terminal before running this script."
    )

symbols = ['AAPL', 'MSFT', 'GOOGL']  # Simple test list
print(f"Starting sequential backfill for {len(symbols)} symbols, {HISTORY_DAYS} days each.")
print(f"Respecting Polygon free tier rate limit: 1 request every {SLEEP_SECONDS} seconds.")

# FOR TESTING: Process only first 3 symbols
test_symbols = symbols[:3]  # REMOVE THIS LINE TO PROCESS ALL SYMBOLS

for idx, symbol in enumerate(test_symbols):
    print(f"({idx+1}/{len(test_symbols)}) Downloading {symbol} ...")
    try:
        # Download raw data
        df = get_polygon_daily_data(symbol, HISTORY_DAYS)
        if df is not None and not df.empty:
            print(f"  Downloaded {len(df)} rows")
            
            # Process with indicators - THIS IS THE KEY CHANGE
            df_with_indicators = process_with_indicators(symbol, df)
            
            # Save the complete data
            save_price_data(symbol, df_with_indicators)
            
            # Verify indicators were added
            indicator_cols = ['BBAvg', 'UpperBB', 'LowerBB', 'ATR']
            has_indicators = sum(1 for col in indicator_cols if col in df_with_indicators.columns and not df_with_indicators[col].isna().all())
            
            print(f"  ✅ Saved {len(df_with_indicators)} rows with {has_indicators}/4 key indicators")
        else:
            print(f"  ❌ No data returned for {symbol}.")
    except Exception as e:
        print(f"  ❌ Error downloading {symbol}: {e}")
    time.sleep(SLEEP_SECONDS)

print("Backfill complete with indicators!")
