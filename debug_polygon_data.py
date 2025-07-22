"""
Debug Polygon API Data Retrieval
Test what data is actually available from Polygon API
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from polygon_config import POLYGON_API_KEY

def test_polygon_date_ranges():
    """Test what date ranges Polygon API actually returns"""
    
    # Test XLK (Technology ETF) with different date ranges
    symbol = "XLK"
    
    # Test different start dates
    test_dates = [
        ("2020-01-01", "2025-07-21", "5+ years"),
        ("2021-01-01", "2025-07-21", "4+ years"), 
        ("2022-01-01", "2025-07-21", "3+ years"),
        ("2023-01-01", "2025-07-21", "2+ years"),
    ]
    
    print(f"🔍 Testing Polygon API data availability for {symbol}")
    print("=" * 60)
    
    for start_date, end_date, description in test_dates:
        print(f"\n📅 Testing {description} ({start_date} to {end_date}):")
        
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            params = {
                'apikey': POLYGON_API_KEY,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    results = data['results']
                    
                    # Convert timestamps to dates
                    first_date = pd.to_datetime(results[0]['t'], unit='ms').strftime('%Y-%m-%d')
                    last_date = pd.to_datetime(results[-1]['t'], unit='ms').strftime('%Y-%m-%d')
                    
                    print(f"   ✅ {len(results)} records available")
                    print(f"   📊 Actual range: {first_date} to {last_date}")
                    
                    # Calculate actual years
                    start_dt = pd.to_datetime(first_date)
                    end_dt = pd.to_datetime(last_date)
                    years = (end_dt - start_dt).days / 365.25
                    print(f"   📈 Actual years: {years:.1f}")
                    
                else:
                    print(f"   ❌ No data returned")
                    if 'message' in data:
                        print(f"   💬 Message: {data['message']}")
                        
            elif response.status_code == 429:
                print(f"   ⏳ Rate limited")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                if response.text:
                    print(f"   💬 Response: {response.text[:200]}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_account_limits():
    """Test if there are account-specific limitations"""
    print(f"\n🔑 Testing Polygon API Account Info:")
    
    try:
        # Try to get account status (if endpoint exists)
        url = "https://api.polygon.io/v3/reference/tickers"
        params = {
            'apikey': POLYGON_API_KEY,
            'limit': 1  # Just get 1 result to test
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            print("   ✅ API key is working")
            data = response.json()
            if 'status' in data:
                print(f"   📊 API Status: {data['status']}")
        else:
            print(f"   ⚠️  API Response: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error testing account: {e}")

def suggest_alternatives():
    """Suggest alternative approaches"""
    print(f"\n💡 ALTERNATIVES IF POLYGON IS LIMITED:")
    print("=" * 50)
    print("1. 📊 Yahoo Finance (free, unlimited history):")
    print("   pip install yfinance")
    print("   import yfinance as yf")
    print("   data = yf.download('XLK', start='2020-01-01')")
    print("")
    print("2. 📈 Alpha Vantage (free tier, good history):")
    print("   Different API with potentially better free limits")
    print("")
    print("3. 🔄 Use 2 years for now:")
    print("   2 years might be sufficient for initial optimization")
    print("   Can always get more data later")

if __name__ == "__main__":
    print("🧪 Polygon API Data Availability Test")
    print("=" * 50)
    
    test_polygon_date_ranges()
    test_account_limits()
    suggest_alternatives()
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("- If Polygon limits free accounts to ~2 years, that's normal")
    print("- 2 years is still usable for optimization (shorter walk-forward periods)")
    print("- Consider Yahoo Finance for longer history if needed")
