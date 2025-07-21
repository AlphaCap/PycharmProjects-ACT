from nGS_Revised_Strategy import NGSStrategy
from data_manager import get_positions
import os
from datetime import datetime

print("=== DEBUGGING POSITION SYSTEM ===")
print(f"Current time: {datetime.now()}")
print("")

# Check strategy positions
print("1. Loading strategy...")
strategy = NGSStrategy()
print(f"   Strategy initialized")
print(f"   Total positions loaded: {len(strategy.positions)}")
print(f"   Retention cutoff date: {strategy.cutoff_date}")
print(f"   Cash available: ${strategy.cash:,.2f}")
print("")

if strategy.positions:
    print("2. Strategy positions found:")
    for symbol, pos in strategy.positions.items():
        print(f"   {symbol}: {pos}")
else:
    print("2. No positions in strategy object")
print("")

# Check data_manager positions
print("3. Checking data_manager positions...")
try:
    positions_from_dm = get_positions()
    print(f"   Positions from data_manager: {len(positions_from_dm)}")
    if positions_from_dm:
        print("   Sample positions:")
        for i, pos in enumerate(positions_from_dm[:5]):
            print(f"     {i+1}. {pos}")
    else:
        print("   No positions from data_manager")
except Exception as e:
    print(f"   Error getting positions from data_manager: {e}")
print("")

# Check if position files exist
print("4. Checking position files...")
position_files = [
    'data/positions.csv',
    'data/trades/positions.csv', 
    'data/current_positions.csv'
]

for file_path in position_files:
    if os.path.exists(file_path):
        print(f"   ✓ Found: {file_path}")
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"     Lines in file: {len(lines)}")
                if len(lines) > 1:
                    print(f"     Header: {lines[0]}")
                    if len(lines) > 2:
                        print(f"     First data row: {lines[1]}")
        except Exception as e:
            print(f"     Error reading {file_path}: {e}")
    else:
        print(f"   ✗ Missing: {file_path}")
print("")

# Check recent trading activity
print("5. Checking recent trading activity...")
try:
    print(f"   Total trades in strategy: {len(strategy.trades)}")
    if strategy.trades:
        print("   Recent trades:")
        for trade in strategy.trades[-5:]:
            print(f"     {trade['symbol']} {trade['type']} on {trade['exit_date']}: ${trade['profit']:+.2f}")
    else:
        print("   No trades found in strategy")
except Exception as e:
    print(f"   Error checking trades: {e}")
print("")

# Check data files timestamps
print("6. Checking data file timestamps...")
try:
    data_dir = 'data/daily'
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"   CSV files in data/daily: {len(csv_files)}")
        
        # Check timestamps of first few files
        for file in csv_files[:5]:
            file_path = os.path.join(data_dir, file)
            timestamp = os.path.getmtime(file_path)
            mod_time = datetime.fromtimestamp(timestamp)
            print(f"     {file}: modified {mod_time}")
    else:
        print("   data/daily directory not found")
except Exception as e:
    print(f"   Error checking file timestamps: {e}")

print("\n=== DEBUG COMPLETE ===")
