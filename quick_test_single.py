import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from data_manager import save_price_data, load_price_data
from nGS_Strategy import NGSStrategy

def test_single_symbol_with_indicators():
    """Test downloading and processing one symbol with indicators."""
    
    symbol = "AAPL"
    print(f"=== TESTING {symbol} WITH INDICATORS ===")
    
    # API setup
    api_key = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"
    
    # Download raw data
    print("1. Downloading raw data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Get a week of data
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": api_key}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            new_df = pd.DataFrame(data["results"])
            new_df["Date"] = pd.to_datetime(new_df["t"], unit="ms")
            new_df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
            new_df = new_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            print(f"   ✅ Downloaded {len(new_df)} rows")
        else:
            print("   ❌ No data in API response")
            return
    else:
        print(f"   ❌ API error: {response.status_code}")
        return
    
    # Load existing data and merge
    print("2. Loading existing data...")
    try:
        existing_df = load_price_data(symbol)
        if not existing_df.empty:
            print(f"   Found {len(existing_df)} existing rows")
            
            # Merge
            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
            new_dates = set(new_df['Date'].dt.date)
            existing_df = existing_df[~existing_df['Date'].dt.date.isin(new_dates)]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            print(f"   Combined to {len(combined_df)} total rows")
        else:
            combined_df = new_df.copy()
            print(f"   No existing data, using {len(combined_df)} new rows")
    except Exception as e:
        print(f"   Warning: Error loading existing data: {e}")
        combined_df = new_df.copy()
    
    # Calculate indicators
    print("3. Calculating indicators...")
    try:
        strategy = NGSStrategy()
        df_with_indicators = strategy.calculate_indicators(combined_df)
        
        if df_with_indicators is not None and not df_with_indicators.empty:
            print(f"   ✅ Calculated indicators successfully")
            
            # Check which indicators we got
            indicator_cols = ['BBAvg', 'UpperBB', 'LowerBB', 'ATR', 'LongPSAR', 'ShortPSAR']
            present_indicators = []
            for col in indicator_cols:
                if col in df_with_indicators.columns and not df_with_indicators[col].isna().all():
                    present_indicators.append(col)
            
            print(f"   Present indicators: {present_indicators}")
            
            # Save the data
            print("4. Saving data...")
            save_price_data(symbol, df_with_indicators)
            
            # Verify what was saved
            print("5. Verifying saved data...")
            saved_df = load_price_data(symbol)
            if not saved_df.empty:
                saved_indicators = []
                for col in indicator_cols:
                    if col in saved_df.columns and not saved_df[col].isna().all():
                        saved_indicators.append(col)
                
                print(f"   ✅ Verified: {len(saved_df)} rows with indicators: {saved_indicators}")
                
                # Show sample of latest data
                latest = saved_df.tail(1).iloc[0]
                print(f"\n📊 Latest data for {symbol}:")
                print(f"   Date: {latest['Date']}")
                print(f"   Close: ${latest['Close']:.2f}")
                if 'BBAvg' in saved_indicators:
                    print(f"   BB Mid: ${latest['BBAvg']:.2f}")
                if 'UpperBB' in saved_indicators:
                    print(f"   BB Upper: ${latest['UpperBB']:.2f}")
                if 'LowerBB' in saved_indicators:
                    print(f"   BB Lower: ${latest['LowerBB']:.2f}")
                if 'ATR' in saved_indicators:
                    print(f"   ATR: {latest['ATR']:.2f}")
                
                print(f"\n🎉 SUCCESS! {symbol} now has complete data with indicators")
                return True
            else:
                print("   ❌ No data found after saving")
                return False
        else:
            print("   ❌ Failed to calculate indicators")
            return False
            
    except Exception as e:
        print(f"   ❌ Error calculating indicators: {e}")
        return False

if __name__ == "__main__":
    success = test_single_symbol_with_indicators()
    
    if success:
        print("\n✅ Test successful! You can now apply this to all symbols.")
        print("The approach works - your indicators are being calculated and saved.")
    else:
        print("\n❌ Test failed. Check the error messages above.")