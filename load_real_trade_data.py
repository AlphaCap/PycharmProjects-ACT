# load_real_trade_data.py - Replace test data with real historical trades
import pandas as pd
import os
import shutil
from datetime import datetime

def backup_test_data():
    """Backup the current test data before replacing"""
    test_file = "data/trades/trade_history.csv"
    backup_file = f"data/trades/trade_history_test_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    if os.path.exists(test_file):
        shutil.copy2(test_file, backup_file)
        print(f"✓ Backed up test data to: {backup_file}")
        return backup_file
    return None

def validate_trade_data(df):
    """Validate that the trade data has the required format"""
    print("=== VALIDATING TRADE DATA ===")
    
    required_columns = ['symbol', 'type', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'shares', 'profit']
    
    # Check columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"✗ Missing required columns: {missing_columns}")
        return False
    
    print(f"✓ All required columns present: {required_columns}")
    
    # Check data types and content
    issues = []
    
    # Check for empty data
    if df.empty:
        issues.append("DataFrame is empty")
    else:
        print(f"✓ Data contains {len(df)} trades")
    
    # Check date formats
    try:
        pd.to_datetime(df['entry_date'])
        pd.to_datetime(df['exit_date'])
        print("✓ Dates can be parsed")
    except Exception as e:
        issues.append(f"Date parsing error: {e}")
    
    # Check numeric columns
    for col in ['entry_price', 'exit_price', 'shares', 'profit']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                pd.to_numeric(df[col])
                print(f"✓ {col} can be converted to numeric")
            except:
                issues.append(f"{col} contains non-numeric values")
        else:
            print(f"✓ {col} is numeric")
    
    # Check for reasonable values
    if (df['entry_price'] <= 0).any():
        issues.append("Some entry prices are <= 0")
    if (df['exit_price'] <= 0).any():
        issues.append("Some exit prices are <= 0")
    if (df['shares'] == 0).any():
        issues.append("Some trades have 0 shares")
    
    # Check trade types
    valid_types = ['long', 'short']
    invalid_types = df[~df['type'].str.lower().isin(valid_types)]['type'].unique()
    if len(invalid_types) > 0:
        print(f"⚠ Warning: Unusual trade types found: {invalid_types}")
        print("  Expected: 'long' or 'short'")
    
    if issues:
        print("✗ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All validation checks passed!")
        return True

def show_data_summary(df):
    """Show summary of the trade data"""
    print("\n=== TRADE DATA SUMMARY ===")
    
    print(f"Total Trades: {len(df)}")
    print(f"Date Range: {df['entry_date'].min()} to {df['exit_date'].max()}")
    print(f"Symbols: {df['symbol'].nunique()} unique ({', '.join(sorted(df['symbol'].unique())[:10])}{'...' if df['symbol'].nunique() > 10 else ''})")
    
    # Trade types
    type_counts = df['type'].value_counts()
    print(f"Trade Types: {dict(type_counts)}")
    
    # Performance summary
    total_profit = df['profit'].sum()
    winning_trades = len(df[df['profit'] > 0])
    win_rate = winning_trades / len(df) * 100
    
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Win Rate: {win_rate:.1f}% ({winning_trades}/{len(df)})")
    print(f"Avg Profit per Trade: ${df['profit'].mean():.2f}")
    
    # Show sample trades
    print(f"\nSample Trades (first 5):")
    sample_cols = ['symbol', 'type', 'entry_date', 'exit_date', 'profit']
    print(df[sample_cols].head().to_string(index=False))

def load_real_data_interactive():
    """Interactive function to help user load their real trade data"""
    print("=== REAL TRADE DATA LOADER ===\n")
    
    print("This script will help you replace the test data with your real historical trades.")
    print("I'll guide you through the process step by step.\n")
    
    # Method selection
    print("How do you want to provide your real trade data?")
    print("1. I have a CSV file ready in the correct format")
    print("2. I have a CSV file but it needs column mapping")
    print("3. I have an Excel file")
    print("4. I want to manually enter a few trades for testing")
    print("5. I need help exporting from my trading platform")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        load_csv_direct()
    elif choice == "2":
        load_csv_with_mapping()
    elif choice == "3":
        load_excel_file()
    elif choice == "4":
        create_manual_trades()
    elif choice == "5":
        show_export_help()
    else:
        print("Invalid choice. Please run the script again.")

def load_csv_direct():
    """Load CSV file that's already in the correct format"""
    print("\n=== LOADING CSV FILE (DIRECT) ===")
    
    file_path = input("Enter the full path to your CSV file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return
    
    try:
        # Read the file
        df = pd.read_csv(file_path)
        print(f"✓ Successfully read {len(df)} rows from {file_path}")
        
        # Show current format
        print(f"\nFile columns: {list(df.columns)}")
        print("Sample data:")
        print(df.head())
        
        # Validate
        if validate_trade_data(df):
            show_data_summary(df)
            
            # Confirm replacement
            confirm = input(f"\nReplace test data with this real data? (y/n): ").strip().lower()
            if confirm == 'y':
                backup_test_data()
                df.to_csv("data/trades/trade_history.csv", index=False)
                print("✓ Real trade data loaded successfully!")
                print("\nNext steps:")
                print("1. Run: python calculate_daily_me_ratio_fixed.py")
                print("2. Run: streamlit run app.py")
            else:
                print("Operation cancelled.")
        
    except Exception as e:
        print(f"✗ Error reading file: {e}")

def load_csv_with_mapping():
    """Load CSV file and map columns to required format"""
    print("\n=== LOADING CSV WITH COLUMN MAPPING ===")
    
    file_path = input("Enter the full path to your CSV file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Successfully read {len(df)} rows")
        print(f"Available columns: {list(df.columns)}")
        print("\nSample data:")
        print(df.head())
        
        # Column mapping
        print(f"\nNow I'll help you map your columns to the required format:")
        required_columns = ['symbol', 'type', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'shares', 'profit']
        
        column_mapping = {}
        for req_col in required_columns:
            print(f"\nWhich column contains '{req_col}'?")
            print(f"Available: {list(df.columns)}")
            user_col = input(f"Column for '{req_col}': ").strip()
            
            if user_col in df.columns:
                column_mapping[req_col] = user_col
            else:
                print(f"✗ Column '{user_col}' not found!")
                return
        
        # Create mapped DataFrame
        mapped_df = pd.DataFrame()
        for req_col, user_col in column_mapping.items():
            mapped_df[req_col] = df[user_col]
        
        print("\nMapped data:")
        print(mapped_df.head())
        
        if validate_trade_data(mapped_df):
            show_data_summary(mapped_df)
            
            confirm = input(f"\nSave this mapped data? (y/n): ").strip().lower()
            if confirm == 'y':
                backup_test_data()
                mapped_df.to_csv("data/trades/trade_history.csv", index=False)
                print("✓ Mapped trade data saved successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def create_manual_trades():
    """Allow user to manually enter a few real trades"""
    print("\n=== MANUAL TRADE ENTRY ===")
    print("Enter your real trades one by one. Type 'done' when finished.")
    
    trades = []
    
    while True:
        print(f"\n--- Trade #{len(trades) + 1} ---")
        symbol = input("Symbol (e.g., AAPL): ").strip().upper()
        if symbol.lower() == 'done':
            break
            
        trade_type = input("Type (long/short): ").strip().lower()
        entry_date = input("Entry date (YYYY-MM-DD): ").strip()
        exit_date = input("Exit date (YYYY-MM-DD): ").strip()
        entry_price = float(input("Entry price: $"))
        exit_price = float(input("Exit price: $"))
        shares = int(input("Shares: "))
        
        # Calculate profit
        if trade_type == 'long':
            profit = (exit_price - entry_price) * shares
        else:
            profit = (entry_price - exit_price) * shares
        
        print(f"Calculated profit: ${profit:.2f}")
        
        trades.append({
            'symbol': symbol,
            'type': trade_type,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'profit': profit
        })
        
        continue_entry = input("Add another trade? (y/n): ").strip().lower()
        if continue_entry != 'y':
            break
    
    if trades:
        df = pd.DataFrame(trades)
        print(f"\n{len(trades)} trades entered:")
        print(df)
        
        if validate_trade_data(df):
            backup_test_data()
            df.to_csv("data/trades/trade_history.csv", index=False)
            print("✓ Manual trades saved successfully!")

def show_export_help():
    """Show help for exporting from trading platforms"""
    print("\n=== TRADING PLATFORM EXPORT HELP ===")
    print("Here's how to export trade history from common platforms:")
    
    print("\n1. TD Ameritrade / Charles Schwab:")
    print("   - Log into thinkorswim or web platform")
    print("   - Go to Account Info > Order Status")
    print("   - Filter for 'Filled' orders")
    print("   - Export to CSV")
    
    print("\n2. Interactive Brokers:")
    print("   - Go to Account Management")
    print("   - Reports > Activity Statements")
    print("   - Select date range and download CSV")
    
    print("\n3. E*TRADE:")
    print("   - Go to Accounts > Portfolio")
    print("   - Click 'Order Status & History'")
    print("   - Export transactions")
    
    print("\n4. Robinhood:")
    print("   - Go to Account > Statements & History")
    print("   - Download monthly statements")
    print("   - Extract trade data from PDFs/CSVs")
    
    print("\nAfter exporting, run this script again and choose option 2 for column mapping.")

if __name__ == "__main__":
    load_real_data_interactive()