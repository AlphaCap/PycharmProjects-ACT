import pandas as pd
import os

print("=== CHECKING EXISTING INDICATOR FILES ===\n")

# Check if AAPL file exists
files_to_check = ['AAPL', 'MSFT', 'GOOGL']

for symbol in files_to_check:
    filepath = f'data/daily/{symbol}.csv'
    
    if os.path.exists(filepath):
        print(f"✅ Found {symbol}.csv")
        
        # Load and check indicators
        df = pd.read_csv(filepath)
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check for key indicators
        key_indicators = ['BBAvg', 'UpperBB', 'LowerBB', 'ATR', 'LongPSAR', 'ShortPSAR', 'oLRSlope', 'LinReg', 'SwingLow', 'SwingHigh']
        present = [col for col in key_indicators if col in df.columns]
        
        print(f"   Key indicators present: {len(present)}/10")
        print(f"   Present: {present}")
        
        if len(present) >= 8:
            print(f"   ✅ {symbol} has FULL INDICATORS")
        elif len(present) >= 4:
            print(f"   ⚠️  {symbol} has PARTIAL indicators")
        else:
            print(f"   ❌ {symbol} has FEW indicators")
        
        print()
    else:
        print(f"❌ {symbol}.csv not found at {filepath}")
        print()

print("If files show PARTIAL or FEW indicators, we need to re-run backfill with all 37 indicators.")