import os
import pandas as pd

print("=== DEBUGGING DATA LOADING ===\n")

# 1. Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}\n")

# 2. Check if trade_history.csv exists
if os.path.exists("trade_history.csv"):
    print("✓ trade_history.csv EXISTS")
    
    # 3. Try to read it
    try:
        df = pd.read_csv("trade_history.csv")
        print(f"✓ Successfully loaded {len(df)} trades")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Total profit: ${df['profit'].sum():,.2f}")
        
        # 4. Show sample data
        print("\nFirst 3 trades:")
        print(df.head(3))
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")
else:
    print("✗ trade_history.csv NOT FOUND")

# 5. Test data_manager
print("\n=== TESTING DATA_MANAGER ===")
try:
    from data_manager import get_trades_history, get_portfolio_metrics
    
    trades = get_trades_history()
    print(f"\nget_trades_history() returned {len(trades)} trades")
    
    metrics = get_portfolio_metrics(100000)
    print(f"\nPortfolio metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
        
except Exception as e:
    print(f"\nError in data_manager: {e}")
    import traceback
    traceback.print_exc()

# 6. Check for positions.csv
print("\n=== CHECKING OTHER FILES ===")
print(f"positions.csv exists: {os.path.exists('positions.csv')}")
<<<<<<< HEAD
print(f"signals.csv exists: {os.path.exists('recent_signals.csv')}")
=======
print(f"signals.csv exists: {os.path.exists('recent_signals.csv')}")
>>>>>>> f612c139bbee93d0e08532f249d5d28e26216b45
