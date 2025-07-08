import pandas as pd
from polygon import RESTClient
from data_manager import save_price_data
from datetime import date, timedelta, datetime

API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"
symbol = "AAPL"
# Remove: csv_path = f"data/{symbol}.csv"

# Set date range (last 365 days)
to_date = date.today()
from_date = to_date - timedelta(days=365)
from_str = from_date.strftime("%Y-%m-%d")
to_str = to_date.strftime("%Y-%m-%d")

client = RESTClient(API_KEY)

bars = []
for bar in client.list_aggs(
    symbol,
    1,
    "day",
    from_=from_str,
    to=to_str,
    sort="desc",
    limit=200,
    adjusted=True,
):
    bars.append({
        "Date": datetime.fromtimestamp(bar.timestamp / 1000).date().isoformat(),
        "Open": bar.open,
        "High": bar.high,
        "Low": bar.low,
        "Close": bar.close,
        "Volume": bar.volume
    })

df = pd.DataFrame(bars)
df = df.sort_values("Date")

indicator_cols = [
    "BBAvg", "BBSDev", "UpperBB", "LowerBB",
    "High_Low", "High_Close", "Low_Close", "TR", "ATR", "ATRma",
    "LongPSAR", "ShortPSAR", "PSAR_EP", "PSAR_AF", "PSAR_IsLong",
    "oLRSlope", "oLRAngle", "oLRIntercept", "TSF",
    "oLRSlope2", "oLRAngle2", "oLRIntercept2", "TSF5",
    "Value1", "ROC", "LRV", "LinReg",
    "oLRValue", "oLRValue2", "SwingLow", "SwingHigh"
]
for col in indicator_cols:
    if col not in df.columns:
        df[col] = None

save_price_data(symbol, df)
print(f"Saved {len(df)} rows to data/daily/{symbol}.csv")
print("AAPL fetched dates:", df["Date"].tail(10).tolist())
print("Min date:", df["Date"].min(), "Max date:", df["Date"].max())
print("Number of rows:", len(df))