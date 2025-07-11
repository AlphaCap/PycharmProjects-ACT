import pandas as pd
import os

# Daily data with price + indicator columns
DAILY_COLUMNS = [
    "Date", "Open", "High", "Low", "Close", "Volume",
    "BBAvg", "BBSDev", "UpperBB", "LowerBB",
    "High_Low", "High_Close", "Low_Close", "TR", "ATR", "ATRma",
    "LongPSAR", "ShortPSAR", "PSAR_EP", "PSAR_AF", "PSAR_IsLong",
    "oLRSlope", "oLRAngle", "oLRIntercept", "TSF",
    "oLRSlope2", "oLRAngle2", "oLRIntercept2", "TSF5",
    "Value1", "ROC", "LRV", "LinReg",
    "oLRValue", "oLRValue2", "SwingLow", "SwingHigh"
]

os.makedirs("data/daily", exist_ok=True)
aapl_file = "data/daily/AAPL.csv"
if not os.path.exists(aapl_file):
    pd.DataFrame(columns=DAILY_COLUMNS).to_csv(aapl_file, index=False)
    print(f"Created {aapl_file}")
else:
    print(f"{aapl_file} already exists.")

# Trade history file columns
TRADE_COLUMNS = [
    "symbol", "type", "entry_date", "exit_date",
    "entry_price", "exit_price", "shares", "profit", "exit_reason"
]
os.makedirs("data/trades", exist_ok=True)
trade_file = "data/trades/trade_history.csv"
if not os.path.exists(trade_file):
    pd.DataFrame(columns=TRADE_COLUMNS).to_csv(trade_file, index=False)
    print(f"Created {trade_file}")
else:
    print(f"{trade_file} already exists.")