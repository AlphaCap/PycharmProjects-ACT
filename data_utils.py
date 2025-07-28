import pandas as pd
from polygon import RESTClient
from polygon_config import POLYGON_API_KEY
from datetime import datetime, timedelta

def load_polygon_data(symbols):
    data = {}
    try:
        client = RESTClient(api_key=POLYGON_API_KEY)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Match 180-day retention

        for symbol in symbols:
            try:
                # Fetch daily aggregates (OHLCV)
                aggs = client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.strftime("%Y-%m-%d"),
                    to=end_date.strftime("%Y-%m-%d"),
                    limit=5000
                )
                if not aggs:
                    print(f"No data returned for {symbol}")
                    data[symbol] = pd.DataFrame()
                    continue

                # Convert to DataFrame
                df = pd.DataFrame([{
                    'date': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                } for agg in aggs])
                df = df.sort_values('date')
                data[symbol] = df
            except Exception as e:
                print(f"Failed to load data for {symbol}: {e}")
                data[symbol] = pd.DataFrame()

        if not any(df.shape[0] > 0 for df in data.values()):
            print("Warning: No valid data loaded for any symbols")
    except Exception as e:
        print(f"Error initializing Polygon client: {e}")
        data = {symbol: pd.DataFrame() for symbol in symbols}

    print(f"load_polygon_data returning: {type(data)}, keys: {list(data.keys())}")
    return data
