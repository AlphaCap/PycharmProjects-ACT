# test_trade_loading.py - Diagnose trade history loading issues
import os
import pandas as pd
from data_manager import get_trades_history, TRADES_HISTORY_FILE

def diagnose_trade_loading() -> None:
    """
    Diagnose issues with loading trade history data.

    This function checks file existence, size, direct readability, and function
    behavior to troubleshoot loading problems.

    Raises:
        OSError: If file operations fail.
        pd.errors.ParserError: If CSV parsing fails.
    """
    print("=== TRADE HISTORY LOADING DIAGNOSTIC ===")
    print()
    
    # Check if file exists
    print("1. Checking trade history file:", TRADES_HISTORY_FILE)
    if os.path.exists(TRADES_HISTORY_FILE):
        print("   ✓ File exists")
        
        # Check file size
        file_size: int = os.path.getsize(TRADES_HISTORY_FILE)
        print("   ✓ File size:", file_size, "bytes")
        
        # Try reading directly
        try:
            df_direct: pd.DataFrame = pd.read_csv(TRADES_HISTORY_FILE)
            print("   ✓ Direct read successful:", len(df_direct), "rows")
            print("   ✓ Columns:", list(df_direct.columns))
            
            if not df_direct.empty:
                print("   ✓ Sample data:")
                print(df_direct.head().to_string())
            else:
                print("   ✗ File is empty")
                
        except Exception as e:
            print("   ✗ Error reading file directly:", e)
    else:
        print("   ✗ File does not exist")
        print("   Looking for:", os.path.abspath(TRADES_HISTORY_FILE))
    
    # Test data_manager function
    print("\n2. Testing get_trades_history() function:")
    try:
        df_function: pd.DataFrame = get_trades_history()
        print("   ✓ Function call successful:", len(df_function), "rows")
        
        if not df_function.empty:
            print("   ✓ Function returned data:")
            print(df_function.head().to_string())
        else:
            print("   ✗ Function returned empty DataFrame")
            
    except Exception as e:
        print("   ✗ Error calling function:", e)
    
    # Check current directory
    print("\n3. Current working directory:", os.getcwd())
    
    # Look for any CSV files in data directory
    print("\n4. Files in data directory:")
    if os.path.exists("data"):
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith('.csv'):
                    full_path: str = os.path.join(root, file)
                    size: int = os.path.getsize(full_path)
                    print("  ", full_path, "(", size, "bytes)", sep="")
    else:
        print("   No data directory found")
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    diagnose_trade_loading()