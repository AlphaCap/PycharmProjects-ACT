#!/usr/bin/env python3
"""
Trade Data Diagnostic Tool
Check what's actually in the trade history file
"""

import os
import pandas as pd
from datetime import datetime

def diagnose_trade_data():
    """Examine the trade history file in detail"""
    print("=" * 60)
    print("TRADE DATA DIAGNOSTIC")
    print("=" * 60)
    
    trades_file = "data/trades/trade_history.csv"
    
    # Check if file exists
    if not os.path.exists(trades_file):
        print(f"❌ Trade history file not found: {trades_file}")
        
        # Check for alternative locations
        alt_files = [
            "trade_history.csv",
            "data/trade_history.csv",
            "trades.csv"
        ]
        
        for alt_file in alt_files:
            if os.path.exists(alt_file):
                print(f"✓ Found alternative file: {alt_file}")
                trades_file = alt_file
                break
        else:
            print("❌ No trade history file found anywhere")
            return
    
    print(f"✓ Reading trade data from: {trades_file}")
    
    try:
        # Load the data
        df = pd.read_csv(trades_file)
        
        print(f"📊 BASIC INFO:")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   File size: {os.path.getsize(trades_file)} bytes")
        
        if df.empty:
            print("❌ File is empty!")
            return
        
        # Show first few rows
        print(f"\n📋 FIRST 5 ROWS:")
        print(df.head().to_string())
        
        # Check data types and ranges
        print(f"\n🔍 DATA ANALYSIS:")
        
        # Date analysis
        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
            print(f"   Entry dates: {df['entry_date'].min()} to {df['entry_date'].max()}")
            
            # Check for future dates
            today = datetime.now()
            future_entries = df[df['entry_date'] > today]
            if not future_entries.empty:
                print(f"   ⚠️  WARNING: {len(future_entries)} trades have FUTURE entry dates!")
        
        if 'exit_date' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
            print(f"   Exit dates: {df['exit_date'].min()} to {df['exit_date'].max()}")
            
            # Check for future dates
            future_exits = df[df['exit_date'] > today]
            if not future_exits.empty:
                print(f"   ⚠️  WARNING: {len(future_exits)} trades have FUTURE exit dates!")
        
        # Symbol analysis
        if 'symbol' in df.columns:
            unique_symbols = df['symbol'].nunique()
            print(f"   Unique symbols: {unique_symbols}")
            if unique_symbols > 100:
                print(f"   ⚠️  WARNING: {unique_symbols} symbols seems very high!")
            
            # Show symbol counts
            symbol_counts = df['symbol'].value_counts().head(10)
            print(f"   Top symbols:")
            for symbol, count in symbol_counts.items():
                print(f"      {symbol}: {count} trades")
        
        # Profit analysis
        if 'profit' in df.columns:
            total_profit = df['profit'].sum()
            avg_profit = df['profit'].mean()
            print(f"   Total profit: ${total_profit:,.2f}")
            print(f"   Average profit: ${avg_profit:.2f}")
            
            # Check for unrealistic profits
            huge_profits = df[df['profit'].abs() > 10000]
            if not huge_profits.empty:
                print(f"   ⚠️  WARNING: {len(huge_profits)} trades with profit > $10,000")
        
        # Shares analysis
        if 'shares' in df.columns:
            avg_shares = df['shares'].mean()
            max_shares = df['shares'].max()
            print(f"   Average shares: {avg_shares:.0f}")
            print(f"   Maximum shares: {max_shares:,.0f}")
            
            if max_shares > 10000:
                print(f"   ⚠️  WARNING: Maximum shares {max_shares:,} seems very high!")
        
        # Trade type analysis
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            print(f"   Trade types:")
            for trade_type, count in type_counts.items():
                print(f"      {trade_type}: {count} trades")
        
        # Check for synthetic/test data patterns
        print(f"\n🔎 SYNTHETIC DATA CHECKS:")
        
        # Check for round numbers (common in synthetic data)
        if 'profit' in df.columns:
            round_profits = df[df['profit'] % 100 == 0]
            if len(round_profits) / len(df) > 0.5:
                print(f"   ⚠️  WARNING: {len(round_profits)}/{len(df)} profits are round numbers (synthetic?)")
        
        # Check for regular patterns in dates
        if 'entry_date' in df.columns and 'exit_date' in df.columns:
            df['hold_days'] = (df['exit_date'] - df['entry_date']).dt.days
            if df['hold_days'].std() < 1:  # Very consistent hold times
                print(f"   ⚠️  WARNING: Very consistent hold times (synthetic?)")
        
        # Check for identical values
        if 'entry_price' in df.columns:
            unique_prices = df['entry_price'].nunique()
            if unique_prices < len(df) * 0.5:
                print(f"   ⚠️  WARNING: Too many identical entry prices (synthetic?)")
        
        # Recent trades check
        print(f"\n📅 RECENT TRADES (last 10):")
        recent = df.tail(10)[['symbol', 'entry_date', 'exit_date', 'profit']].copy()
        if 'entry_date' in recent.columns:
            recent['entry_date'] = recent['entry_date'].dt.strftime('%Y-%m-%d')
        if 'exit_date' in recent.columns:
            recent['exit_date'] = recent['exit_date'].dt.strftime('%Y-%m-%d')
        print(recent.to_string(index=False))
        
        # Check overlap analysis (how many trades open simultaneously)
        print(f"\n🔄 POSITION OVERLAP ANALYSIS:")
        analyze_position_overlap(df)
        
    except Exception as e:
        print(f"❌ Error reading trade data: {e}")

def analyze_position_overlap(df):
    """Analyze how many positions were open simultaneously"""
    
    if 'entry_date' not in df.columns or 'exit_date' not in df.columns:
        print("   Cannot analyze - missing date columns")
        return
    
    try:
        # Convert dates
        df = df.copy()
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        
        # Find date range
        min_date = df['entry_date'].min()
        max_date = df['exit_date'].max()
        
        # Sample some dates to check overlap
        sample_dates = pd.date_range(start=min_date, end=max_date, periods=10)
        
        max_overlap = 0
        max_date = None
        
        for check_date in sample_dates:
            # Find trades open on this date
            open_trades = df[
                (df['entry_date'] <= check_date) & 
                (df['exit_date'] > check_date)
            ]
            
            overlap_count = len(open_trades)
            if overlap_count > max_overlap:
                max_overlap = overlap_count
                max_date = check_date
            
            print(f"   {check_date.strftime('%Y-%m-%d')}: {overlap_count} open positions")
        
        print(f"   📊 Maximum simultaneous positions: {max_overlap} on {max_date.strftime('%Y-%m-%d') if max_date else 'N/A'}")
        
        if max_overlap > 50:
            print(f"   ⚠️  WARNING: {max_overlap} simultaneous positions is extremely high!")
            print(f"   This suggests synthetic/test data or calculation error")
        
    except Exception as e:
        print(f"   Error in overlap analysis: {e}")

def check_file_source():
    """Check if this might be synthetic data"""
    print(f"\n🔍 FILE SOURCE CHECK:")
    
    trades_file = "data/trades/trade_history.csv"
    if os.path.exists(trades_file):
        # Check file creation time
        create_time = datetime.fromtimestamp(os.path.getctime(trades_file))
        modify_time = datetime.fromtimestamp(os.path.getmtime(trades_file))
        
        print(f"   File created: {create_time}")
        print(f"   Last modified: {modify_time}")
        
        # Check if created very recently (might be synthetic)
        if (datetime.now() - create_time).days < 7:
            print(f"   ⚠️  WARNING: File is very recent - might be synthetic test data")
    
    # Look for strategy files that might generate synthetic data
    strategy_files = [
        "nGS_Revised_Strategy.py",
        "strategy.py",
        "backtest.py",
        "synthetic_data.py"
    ]
    
    for file in strategy_files:
        if os.path.exists(file):
            print(f"   Found strategy file: {file}")

if __name__ == "__main__":
    diagnose_trade_data()
    check_file_source()
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print("If you see:")
    print("- Future dates")
    print("- 100+ simultaneous positions") 
    print("- 500%+ M/E ratios")
    print("- Very regular patterns")
    print("")
    print("This is likely SYNTHETIC/TEST data, not real trades!")
    print("Check if you're running backtest data instead of live trades.")
    print(f"{'='*60}")