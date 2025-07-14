def check_data_manager_config():
    """Check what data_manager.py is expecting"""
    print(f"\n" + "-" * 40)
    print("DATA MANAGER CONFIGURATION CHECK")
    print("-" * 40)
    
    try:
        # Try to import data_manager to see its configuration
        import data_manager as dm
        
        # Check the SP500_SYMBOLS_FILE path
        if hasattr(dm, 'SP500_SYMBOLS_FILE'):
            expected_path = dm.SP500_SYMBOLS_FILE
            print(f"data_manager expects: {expected_path}")
            print(f"Full path: {os.path.abspath(expected_path)}")
            print(f"File exists: {os.path.exists(expected_path)}")
            
            if os.path.exists(expected_path):
                print("✓ data_manager can find its S&P 500 file")
            else:
                print("❌ data_manager CANNOT find its S&P 500 file")
        
        # Test the actual function
        try:
            symbols = dm.get_sp500_symbols()
            print(f"dm.get_sp500_symbols() returns: {len(symbols)} symbols")
            if symbols:
                print(f"Sample symbols: {symbols[:5]}")
                print("✓ data_manager can load S&P 500 symbols successfully")
            else:
                print("❌ data_manager returns empty symbol list")
        except Exception as e:
            print(f"❌ Error calling dm.get_sp500_symbols(): {e}")
            
    except ImportError as e:
        print(f"❌ Cannot import data_manager: {e}")
    except Exception as e:
        print(f"❌ Error checking data_manager: {e}")#!/usr/bin/env python3
"""
M/E Ratio Diagnostic Tool
Run from command prompt to check M/E ratio indicator data
"""

import os
import pandas as pd
import sys
from datetime import datetime, timedelta

def check_me_ratio_data():
    """Diagnostic tool to check M/E ratio indicator in daily data files"""
    print("=" * 60)
    print("M/E RATIO DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check if data directory exists
    daily_dir = os.path.join("data", "daily")
    if not os.path.exists(daily_dir):
        print(f"❌ ERROR: Daily data directory not found: {daily_dir}")
        return
    
    print(f"✓ Daily data directory found: {daily_dir}")
    
    # Get list of symbol files
    symbol_files = [f for f in os.listdir(daily_dir) if f.endswith('.csv')]
    print(f"✓ Found {len(symbol_files)} symbol data files")
    
    if len(symbol_files) == 0:
        print("❌ ERROR: No symbol data files found")
        return
    
    # Sample some files to check
    sample_size = min(10, len(symbol_files))
    sample_files = symbol_files[:sample_size]
    print(f"\n📊 Checking sample of {sample_size} files:")
    
    me_ratio_found = 0
    valid_me_values = 0
    total_files_checked = 0
    
    for filename in sample_files:
        symbol = filename.replace('.csv', '')
        filepath = os.path.join(daily_dir, filename)
        
        try:
            # Load the file
            df = pd.read_csv(filepath)
            total_files_checked += 1
            
            print(f"\n📈 {symbol}:")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check if ME_Ratio column exists
            if 'ME_Ratio' in df.columns:
                me_ratio_found += 1
                print(f"   ✓ ME_Ratio column found")
                
                # Check for valid values
                valid_values = df['ME_Ratio'][df['ME_Ratio'].notna() & (df['ME_Ratio'] > 0)]
                if len(valid_values) > 0:
                    valid_me_values += 1
                    print(f"   ✓ Valid ME_Ratio values: {len(valid_values)}")
                    print(f"   📊 Sample values: {valid_values.head().tolist()}")
                    print(f"   📊 Min: {valid_values.min():.2f}, Max: {valid_values.max():.2f}, Avg: {valid_values.mean():.2f}")
                    
                    # Show recent dates with M/E values
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        recent_me = df[df['ME_Ratio'].notna() & (df['ME_Ratio'] > 0)].tail(3)
                        if not recent_me.empty:
                            print(f"   📅 Recent M/E data:")
                            for _, row in recent_me.iterrows():
                                print(f"      {row['Date'].strftime('%Y-%m-%d')}: {row['ME_Ratio']:.2f}%")
                else:
                    print(f"   ❌ No valid ME_Ratio values found")
            else:
                print(f"   ❌ ME_Ratio column missing")
                
        except Exception as e:
            print(f"   ❌ Error reading {filename}: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Files checked: {total_files_checked}")
    print(f"Files with ME_Ratio column: {me_ratio_found}")
    print(f"Files with valid ME_Ratio values: {valid_me_values}")
    
    if me_ratio_found == 0:
        print("\n❌ ISSUE: No files contain ME_Ratio column")
        print("   The M/E ratio indicator is not being calculated or saved")
        print("   Check if the strategy is running and calculating indicators properly")
    elif valid_me_values == 0:
        print("\n⚠️  WARNING: ME_Ratio column exists but no valid values found")
        print("   The M/E ratio calculation might have an issue")
    else:
        print(f"\n✓ SUCCESS: M/E ratio data found in {valid_me_values} files")
        print("   M/E ratio indicator appears to be working correctly")
    
    # Run additional diagnostics
    find_sp500_file()
    check_data_manager_config()
    
def find_sp500_file():
    """Find the S&P 500 symbols file in various locations"""
    print(f"\n" + "-" * 40)
    print("FINDING S&P 500 SYMBOLS FILE")
    print("-" * 40)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check multiple possible locations and file types
    possible_files = [
        "sp500_symbols.txt",
        "sp500_symbols.csv", 
        os.path.join("data", "sp500_symbols.txt"),
        os.path.join("data", "sp500_symbols.csv"),
        os.path.join("..", "sp500_symbols.txt"),
        os.path.join("..", "sp500_symbols.csv"),
        os.path.join(".", "sp500_symbols.txt"),
        os.path.join(".", "sp500_symbols.csv")
    ]
    
    found_files = []
    
    for file_path in possible_files:
        abs_path = os.path.abspath(file_path)
        if os.path.exists(file_path):
            found_files.append((file_path, abs_path))
            print(f"✓ FOUND: {file_path}")
            print(f"   Full path: {abs_path}")
            
            # Try to read and count symbols
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r') as f:
                        symbols = [line.strip() for line in f if line.strip()]
                elif file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    if 'Symbol' in df.columns:
                        symbols = df['Symbol'].tolist()
                    else:
                        symbols = df.iloc[:, 0].tolist()  # First column
                
                print(f"   Contains: {len(symbols)} symbols")
                print(f"   Sample: {symbols[:5]}")
                
            except Exception as e:
                print(f"   Error reading: {e}")
    
    if not found_files:
        print("❌ No S&P 500 symbols file found in any location")
        
        # List all files in current directory for debugging
        print(f"\n📁 Files in current directory:")
        try:
            files = os.listdir(current_dir)
            txt_csv_files = [f for f in files if f.endswith(('.txt', '.csv'))]
            if txt_csv_files:
                for f in txt_csv_files[:10]:  # Show first 10
                    print(f"   {f}")
            else:
                print("   No .txt or .csv files found")
        except Exception as e:
            print(f"   Error listing files: {e}")
    
    return found_files

def check_specific_symbol(symbol):
    """Check a specific symbol's M/E ratio data"""
    print(f"\n" + "=" * 60)
    print(f"DETAILED CHECK FOR {symbol.upper()}")
    print("=" * 60)
    
    filename = os.path.join("data", "daily", f"{symbol.upper()}.csv")
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    try:
        df = pd.read_csv(filename)
        print(f"✓ File loaded: {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)}")
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"✓ Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        if 'ME_Ratio' in df.columns:
            print(f"\n📊 ME_Ratio Analysis:")
            me_values = df['ME_Ratio']
            total_values = len(me_values)
            null_values = me_values.isna().sum()
            zero_values = (me_values == 0).sum()
            valid_values = me_values[me_values.notna() & (me_values > 0)]
            
            print(f"   Total rows: {total_values}")
            print(f"   Null values: {null_values}")
            print(f"   Zero values: {zero_values}")
            print(f"   Valid values (> 0): {len(valid_values)}")
            
            if len(valid_values) > 0:
                print(f"   Min: {valid_values.min():.2f}")
                print(f"   Max: {valid_values.max():.2f}")
                print(f"   Average: {valid_values.mean():.2f}")
                
                print(f"\n📅 Recent M/E Ratio values:")
                recent_data = df[df['ME_Ratio'].notna() & (df['ME_Ratio'] > 0)].tail(10)
                for _, row in recent_data.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d') if 'Date' in df.columns else 'N/A'
                    print(f"   {date_str}: {row['ME_Ratio']:.2f}%")
            else:
                print("   ❌ No valid M/E ratio values found")
        else:
            print("❌ ME_Ratio column not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Check specific symbol
        symbol = sys.argv[1]
        check_specific_symbol(symbol)
    else:
        # General diagnostic
        check_me_ratio_data()
    
    print(f"\n" + "=" * 60)
    print("USAGE:")
    print("  python me_ratio_diagnostic.py           # General check")
    print("  python me_ratio_diagnostic.py AAPL      # Check specific symbol")
    print("=" * 60)
