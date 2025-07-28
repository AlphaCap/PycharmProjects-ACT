import pandas as pd
from polygon import RESTClient
from polygon_config import POLYGON_API_KEY
from datetime import datetime, timedelta
import os
import json

def load_polygon_data(symbols):
    data = {}
    cache_dir = 'data'
    last_fetch_file = os.path.join(cache_dir, 'last_fetch_dates.json')
    
    # Load last fetch dates
    last_fetch = {'daily': {'full_history_date': None, 'last_update': None}}
    if os.path.exists(last_fetch_file):
        try:
            with open(last_fetch_file, 'r') as f:
                last_fetch = json.load(f)
        except Exception as e:
            print(f"Error reading {last_fetch_file}: {e}")

    today = datetime.now().strftime('%Y-%m-%d')
    last_update = last_fetch.get('daily', {}).get('last_update')
    fetch_start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

    try:
        client = RESTClient(api_key=POLYGON_API_KEY)
        for symbol in symbols:
            cache_file = os.path.join(cache_dir, f'{symbol}_daily.csv')
            df = pd.DataFrame()

            # Check if cached data is valid
            if last_update == today and os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file)
                    df['date'] = pd.to_datetime(df['date'])
                    print(f"Loaded cached data for {symbol} from {cache_file}")
                except Exception as e:
                    print(f"Error loading cached data for {symbol}: {e}")

            # Fetch new data if cache is invalid or outdated
            if df.empty or last_update != today:
                try:
                    aggs = client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=fetch_start_date,
                        to=today,
                        limit=5000
                    )
                    if not aggs:
                        print(f"No data returned for {symbol}")
                        data[symbol] = pd.DataFrame()
                        continue

                    df = pd.DataFrame([{
                        'date': pd.to_datetime(agg.timestamp, unit='ms'),
                        'open': agg.open,
                        'high': agg.high,
                        'low': agg.low,
                        'close': agg.close,
                        'volume': agg.volume
                    } for agg in aggs])
                    df = df.sort_values('date')

                    # Save to cache
                    os.makedirs(cache_dir, exist_ok=True)
                    df.to_csv(cache_file, index=False)
                    print(f"Saved data for {symbol} to {cache_file}")
                except Exception as e:
                    print(f"Failed to fetch data for {symbol}: {e}")
                    df = pd.DataFrame()

            data[symbol] = df

        # Update last_fetch_dates.json
        last_fetch['daily']['full_history_date'] = fetch_start_date
        last_fetch['daily']['last_update'] = today
        try:
            with open(last_fetch_file, 'w') as f:
                json.dump(last_fetch, f, indent=2)
            print(f"Updated {last_fetch_file}")
        except Exception as e:
            print(f"Error updating {last_fetch_file}: {e}")

        if not any(df.shape[0] > 0 for df in data.values()):
            print("Warning: No valid data loaded for any symbols")
    except Exception as e:
        print(f"Error initializing Polygon client: {e}")
        data = {symbol: pd.DataFrame() for symbol in symbols}

    print(f"load_polygon_data returning: {type(data)}, keys: {list(data.keys())}")
    return data
