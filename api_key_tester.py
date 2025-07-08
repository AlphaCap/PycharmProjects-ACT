import requests
import json
from datetime import datetime, timedelta

# Test your Polygon API key and connection
API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"

def test_api_key_status():
    """Test if the API key is valid and check account status."""
    print("=== POLYGON API KEY TEST ===")
    
    # Test 1: Basic API connectivity
    print("\n1. Testing API connectivity...")
    try:
        url = f"https://api.polygon.io/v1/meta/symbols/AAPL/company"
        params = {"apiKey": API_KEY}
        
        response = requests.get(url, params=params, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API connection successful")
        elif response.status_code == 401:
            print("   ❌ API key invalid or expired")
            return False
        elif response.status_code == 403:
            print("   ❌ API key forbidden (check permissions)")
            return False
        else:
            print(f"   ⚠️  Unexpected status: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False
    
    # Test 2: Check daily bars endpoint specifically
    print("\n2. Testing daily bars endpoint...")
    
    # Use a date range that should definitely have data (last Friday)
    today = datetime.now()
    # Go back to find last Friday
    days_back = 1
    while days_back <= 7:
        test_date = today - timedelta(days=days_back)
        if test_date.weekday() == 4:  # Friday
            break
        days_back += 1
    
    if days_back > 7:
        # Fallback to a known date with data
        test_date = datetime(2024, 12, 31)  # Known trading day
    
    date_str = test_date.strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{date_str}/{date_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": API_KEY
    }
    
    print(f"   Testing date: {date_str} ({test_date.strftime('%A')})")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   API Status: {data.get('status', 'Unknown')}")
            print(f"   Results Count: {data.get('resultsCount', 0)}")
            print(f"   Query Count: {data.get('queryCount', 0)}")
            
            if data.get('resultsCount', 0) > 0:
                print("   ✅ Daily bars endpoint working")
                print(f"   Sample result: {data.get('results', [{}])[0]}")
                return True
            else:
                print("   ⚠️  No data returned (might be holiday/weekend)")
                print(f"   Full response: {json.dumps(data, indent=2)}")
        elif response.status_code == 429:
            print("   ⚠️  Rate limited")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Request error: {e}")
    
    return False

def test_recent_trading_days():
    """Test multiple recent trading days to find one with data."""
    print("\n3. Testing recent trading days...")
    
    today = datetime.now()
    
    # Test last 10 days to find trading days
    for days_back in range(1, 11):
        test_date = today - timedelta(days=days_back)
        
        # Skip weekends
        if test_date.weekday() >= 5:
            continue
            
        date_str = test_date.strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{date_str}/{date_str}"
        params = {"adjusted": "true", "apiKey": API_KEY}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('resultsCount', 0) > 0:
                    print(f"   ✅ Found data for {date_str} ({test_date.strftime('%A')})")
                    print(f"      Results: {data.get('resultsCount')}")
                    return date_str
                else:
                    print(f"   ❌ No data for {date_str} ({test_date.strftime('%A')})")
            else:
                print(f"   ❌ Error {response.status_code} for {date_str}")
                
        except Exception as e:
            print(f"   ❌ Error for {date_str}: {e}")
    
    print("   ❌ No recent trading days found with data")
    return None

if __name__ == "__main__":
    print(f"Testing API Key: {API_KEY[:10]}...{API_KEY[-5:]}")
    
    # Run tests
    basic_test = test_api_key_status()
    recent_data = test_recent_trading_days()
    
    print(f"\n=== SUMMARY ===")
    if basic_test and recent_data:
        print("✅ API key is working correctly")
        print(f"✅ Found recent data for: {recent_data}")
        print("\nYour API should be working. The issue might be:")
        print("1. Date range in your script")
        print("2. Market holidays")
        print("3. Symbol-specific issues")
    elif basic_test:
        print("⚠️  API key works but no recent data found")
        print("This might indicate market holidays or API issues")
    else:
        print("❌ API key test failed")
        print("Check your API key at: https://polygon.io/dashboard")