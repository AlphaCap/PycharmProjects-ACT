import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from data_manager import get_sp500_symbols, save_price_data, load_price_data
from nGS_Strategy import NGSStrategy

# --- CONFIGURATION ---
HISTORY_DAYS = 180
SLEEP_SECONDS = 12

# Set your Polygon API key
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    POLYGON_API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"
    os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY

def get_polygon_daily_data(symbol, days):
    """Download daily data from Polygon API."""
    api_key = os.getenv('POLYGON_API_KEY')
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {
        "adjusted": "true",
        "sort": "asc", 
        "limit": 5000,
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df["Date"] = pd.to_datetime(df["t"], unit="ms")
                df.rename(columns={
                    "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
                }, inplace=True)
                return df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return pd.DataFrame()
    except Exception as e:
        print(f"    Error downloading {symbol}: {e}")
        return pd.DataFrame()

def process_symbol_with_indicators(symbol, days):
    """Process a symbol with the same successful approach as the test."""
    try:
        # 1. Download new data
        new_df = get_polygon_daily_data(symbol, days)
        if new_df.empty:
            print(f"    ❌ No new data for {symbol}")
            return False
        
        # 2. Load and merge with existing data  
        try:
            existing_df = load_price_data(symbol)
            if not existing_df.empty:
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                new_dates = set(new_df['Date'].dt.date)
                existing_df = existing_df[~existing_df['Date'].dt.date.isin(new_dates)]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.sort_values('Date').reset_index(drop=True)
                print(f"    Merged {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total")
            else:
                combined_df = new_df.copy()
                print(f"    No existing data, using {len(combined_df)} new rows")
        except Exception as e:
            print(f"    Warning loading existing data: {e}")
            combined_df = new_df.copy()
        
        # 3. Calculate indicators (same as successful test)
        strategy = NGSStrategy()
        df_with_indicators = strategy.calculate_indicators(combined_df)
        
        if df_with_indicators is not None and not df_with_indicators.empty:
            # 4. Save complete data
            save_price_data(symbol, df_with_indicators)
            
            # 5. Verify indicators are present - CHECK ALL INDICATORS
            all_indicator_cols = [
                'BBAvg', 'BBSDev', 'UpperBB', 'LowerBB', 
                'High_Low', 'High_Close', 'Low_Close', 'TR', 'ATR', 'ATRma',
                'LongPSAR', 'ShortPSAR', 'PSAR_EP', 'PSAR_AF', 'PSAR_IsLong', 
                'oLRSlope', 'oLRAngle', 'oLRIntercept', 'TSF', 
                'oLRSlope2', 'oLRAngle2', 'oLRIntercept2', 'TSF5', 
                'Value1', 'ROC', 'LRV', 'LinReg', 'oLRValue', 'oLRValue2', 
                'SwingLow', 'SwingHigh'
            ]
            present_count = sum(1 for col in all_indicator_cols 
                              if col in df_with_indicators.columns and not df_with_indicators[col].isna().all())
            
            print(f"    ✅ Saved {len(df_with_indicators)} rows with {present_count}/31 indicators")
            return True
        else:
            print(f"    ❌ Failed to calculate indicators for {symbol}")
            return False
            
    except Exception as e:
        print(f"    ❌ Error processing {symbol}: {e}")
        return False

def main():
    """Apply the working fix to all symbols."""
    print("=== APPLYING INDICATOR FIX TO ALL SYMBOLS ===")
    print("Using the same successful approach that worked for AAPL\n")
    
    # Get all symbols
    symbols = get_sp500_symbols()
    
    if not symbols:
        print("❌ No symbols found!")
        return
    
    print(f"Found {len(symbols)} symbols to process")
    
    # For safety, let's start with first 20 symbols
    # Remove this line to process all 500 symbols
    batch_symbols = symbols
    
    print(f"Processing batch of {len(batch_symbols)} symbols...")
    print(f"Rate limit: 1 request every {SLEEP_SECONDS} seconds\n")
    
    success_count = 0
    error_count = 0
    already_complete = 0
    
    for idx, symbol in enumerate(batch_symbols):
        print(f"({idx+1}/{len(batch_symbols)}) Processing {symbol}...")
        
        # Quick check: does this symbol already have indicators?
        try:
            existing = load_price_data(symbol)
            if not existing.empty and 'BBAvg' in existing.columns and 'ATR' in existing.columns and 'LRV' in existing.columns:
                # Check if indicators actually have values (not all NaN)
                key_indicators = ['BBAvg', 'ATR', 'LRV', 'TSF', 'oLRValue']
                has_values = all(col in existing.columns and not existing[col].isna().all() for col in key_indicators)
                if has_values:
                    print(f"    ✅ Already has complete indicators, skipping")
                    already_complete += 1
                    continue
        except:
            pass
        
        # Process the symbol
        success = process_symbol_with_indicators(symbol, HISTORY_DAYS)
        
        if success:
            success_count += 1
        else:
            error_count += 1
        
        # Rate limiting
        if idx < len(batch_symbols) - 1:
            print(f"    Waiting {SLEEP_SECONDS} seconds...")
            time.sleep(SLEEP_SECONDS)
        
        print()  # Empty line for readability
    
    # Summary
    print("=== BATCH COMPLETE ===")
    print(f"✅ Successfully processed: {success_count}")
    print(f"✅ Already had indicators: {already_complete}")
    print(f"❌ Errors: {error_count}")
    print(f"📊 Total processed: {len(batch_symbols)}")
    
    success_rate = (success_count + already_complete) / len(batch_symbols) * 100
    print(f"🎯 Success rate: {success_rate:.1f}%")
    
    if success_count > 0:
        print(f"\n🎉 SUCCESS! {success_count} more symbols now have complete OHLC + indicator data")
        print("Your 6-month rolling database is being populated with indicators!")
        
        if len(batch_symbols) < len(symbols):
            print(f"\n🔄 To process all {len(symbols)} symbols:")
            print("   Change 'batch_symbols = symbols[:20]' to 'batch_symbols = symbols'")
            print("   This will take about 2 hours with rate limiting")

if __name__ == "__main__":
    main()