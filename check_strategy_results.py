import os
import pandas as pd
from datetime import datetime

print("=== CHECKING STRATEGY RESULTS ===")

# Check what files were actually created/updated by the strategy runs
print("1. Checking for trade/position files created by strategy...")

files_to_check = [
    'data/trades/trade_history.csv',
    'data/positions.csv', 
    'data/current_positions.csv',
    'data/me_ratio_history.csv'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        stat = os.stat(file_path)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size
        print(f"✓ {file_path}")
        print(f"  Modified: {mod_time}")
        print(f"  Size: {size} bytes")
        
        # Try to read and show sample content
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"  Rows: {len(df)}")
                if not df.empty:
                    print(f"  Columns: {list(df.columns)}")
                    print(f"  Sample: {df.head(2).to_string()}")
                else:
                    print("  File is empty")
        except Exception as e:
            print(f"  Error reading: {e}")
        print()
    else:
        print(f"✗ Missing: {file_path}")

# Check if there are any other result files
print("2. Looking for any other CSV files in data directory...")
for root, dirs, files in os.walk('data'):
    for file in files:
        if file.endswith('.csv') and 'trade' in file.lower():
            full_path = os.path.join(root, file)
            print(f"Found trade-related file: {full_path}")

# Check the sp500 symbols file
print("3. Checking symbol source file...")
sp500_file = 'data/sp500_symbols.txt'
if os.path.exists(sp500_file):
    with open(sp500_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    print(f"✓ {sp500_file}: {len(symbols)} symbols")
    print(f"  First 10: {symbols[:10]}")
    print(f"  Last 10: {symbols[-10:]}")
else:
    print(f"✗ Missing: {sp500_file}")

print("\n=== ANALYSIS COMPLETE ===")
