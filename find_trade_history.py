import os
import pandas as pd

print("=== SEARCHING FOR TRADE HISTORY FILES ===\n")

# Search for CSV files
csv_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            csv_files.append(full_path)

print(f"Found {len(csv_files)} CSV files:\n")

# Look specifically for trade-related files
trade_files = []
for file in csv_files:
    if 'trade' in file.lower() or 'history' in file.lower():
        trade_files.append(file)
        print(f"📊 {file}")
        
        # Try to read and show info
        try:
            df = pd.read_csv(file)
            if 'profit' in df.columns:
                print(f"   ✓ Has {len(df)} rows, Total profit: ${df['profit'].sum():,.2f}")
            else:
                print(f"   ✓ Has {len(df)} rows, Columns: {list(df.columns)[:5]}...")
        except Exception as e:
            print(f"   ✗ Error reading: {e}")
        print()

# Also check specific directories
print("\n=== CHECKING SPECIFIC DIRECTORIES ===")
dirs_to_check = ['data', 'data/trades', 'results', 'reports', 'gSTDayTrader_results']

for dir_path in dirs_to_check:
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            print(f"\n{dir_path}/:")
            for f in csv_files[:5]:  # Show first 5
                print(f"  - {f}")
                
# Look for the file with 848 trades
print("\n=== LOOKING FOR FILE WITH ~848 TRADES ===")
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if len(df) == 848 or (840 < len(df) < 856):  # Close to 848
            print(f"🎯 FOUND IT: {file}")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            if 'profit' in df.columns:
                print(f"   Total profit: ${df['profit'].sum():,.2f}")
    except:
        pass