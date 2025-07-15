#!/usr/bin/env python3
"""
List Daily M/E Ratios - Command Prompt Tool
Shows historical daily M/E ratios calculated from trade data.
"""

import os
import pandas as pd
import sys
from datetime import datetime, timedelta
from typing import List, Dict

def load_trade_history() -> pd.DataFrame:
    """
    Load trade history data.

    Returns:
        pd.DataFrame: DataFrame containing trade history.

    Raises:
        FileNotFoundError: If the trade history file is missing.
        pd.errors.EmptyDataError: If the file is empty.
    """
    trades_file: str = "data/trades/trade_history.csv"
    
    if not os.path.exists(trades_file):
        print(f"❌ Trade history file not found: {trades_file}")
        return pd.DataFrame()
    
    try:
        df: pd.DataFrame = pd.read_csv(trades_file)
        print(f"✓ Loaded {len(df)} trades from {trades_file}")
        return df
    except Exception as e:
        print(f"❌ Error loading trade history: {e}")
        return pd.DataFrame()

def calculate_daily_me_ratios(trades_df: pd.DataFrame, initial_value: float = 100000, 
                            days_back: int = 30) -> List[Dict]:
    """
    Calculate daily M/E ratios from trade history.

    Args:
        trades_df (pd.DataFrame): DataFrame containing trade history.
        initial_value (float, optional): Initial portfolio value. Defaults to 100000.
        days_back (int, optional): Number of days to look back. Defaults to 30.

    Returns:
        List[Dict]: List of daily ratio data.

    Raises:
        ValueError: If trade data is empty or malformed.
    """
    if trades_df.empty:
        print("❌ No trade data available")
        return []
    
    # Convert dates
    trades_df = trades_df.copy()
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Get date range (last N days)
    end_date: datetime = datetime.now()
    start_date: datetime = end_date - timedelta(days=days_back)
    
    # Filter trades to relevant period
    relevant_trades: pd.DataFrame = trades_df[
        (trades_df['entry_date'] <= end_date) & 
        (trades_df['exit_date'] >= start_date)
    ]
    
    if relevant_trades.empty:
        print(f"❌ No trades found in last {days_back} days")
        return []
    
    print(f"📊 Calculating M/E ratios for last {days_back} days...")
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create daily date range (trading days only)
    date_range: pd.DatetimeIndex = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_ratios: List[Dict] = []
    
    for current_date in date_range:
        # Skip weekends
        if current_date.weekday() >= 5:
            continue
        
        # Calculate portfolio equity on this date
        closed_trades: pd.DataFrame = trades_df[trades_df['exit_date'] <= current_date]
        realized_pnl: float = closed_trades['profit'].sum() if not closed_trades.empty else 0
        portfolio_equity: float = initial_value + realized_pnl
        
        # Find open positions on this date
        open_trades: pd.DataFrame = trades_df[
            (trades_df['entry_date'] <= current_date) & 
            (trades_df['exit_date'] > current_date)
        ]
        
        if not open_trades.empty:
            # Calculate position values
            long_positions: pd.DataFrame = open_trades[open_trades['type'] == 'long']
            short_positions: pd.DataFrame = open_trades[open_trades['type'] == 'short']
            
            long_value: float = (
                (long_positions['entry_price'] * long_positions['shares']).sum()
                if not long_positions.empty else 0
            )
            short_value: float = (
                (short_positions['entry_price'] * short_positions['shares']).sum()
                if not short_positions.empty else 0
            )
            
            total_position_value: float = long_value + short_value
            
            # Calculate M/E ratio
            if portfolio_equity > 0:
                me_ratio: float = (total_position_value / portfolio_equity) * 100
                me_ratio = max(me_ratio, 0.0)
            else:
                me_ratio = 0.0
            
            position_count: int = len(open_trades)
        else:
            me_ratio = 0.0
            total_position_value = 0
            position_count = 0
        
        daily_ratios.append({
            'date': current_date,
            'me_ratio': me_ratio,
            'portfolio_equity': portfolio_equity,
            'position_value': total_position_value,
            'position_count': position_count
        })
    
    return daily_ratios

def display_me_ratios(daily_ratios: List[Dict], show_details: bool = False) -> None:
    """
    Display M/E ratios in formatted table.

    Args:
        daily_ratios (List[Dict]): List of daily ratio data.
        show_details (bool, optional): Show detailed metrics. Defaults to False.
    """
    if not daily_ratios:
        print("❌ No M/E ratio data to display")
        return
    
    print(f"\n{'='*80}")
    print("DAILY M/E RATIOS")
    print(f"{'='*80}")
    
    # Header
    if show_details:
        print(f"{'Date':>12} {'M/E Ratio':>10} {'Portfolio':>12} {'Positions':>12} {'Count':>6}")
        print(f"{'':>12} {'(%)':>10} {'Equity ($)':>12} {'Value ($)':>12} {'':>6}")
    else:
        print(f"{'Date':>12} {'M/E Ratio':>10} {'Positions':>6}")
        print(f"{'':>12} {'(%)':>10} {'Count':>6}")
    
    print("-" * 80)
    
    # Data rows
    for ratio_data in daily_ratios:
        date_str: str = ratio_data['date'].strftime('%Y-%m-%d')
        me_ratio: float = ratio_data['me_ratio']
        
        if show_details:
            portfolio: float = ratio_data['portfolio_equity']
            position_val: float = ratio_data['position_value']
            count: int = ratio_data['position_count']
            print(f"{date_str:>12} {me_ratio:>9.1f}% {portfolio:>11,.0f} {position_val:>11,.0f} {count:>6}")
        else:
            count = ratio_data['position_count']
            print(f"{date_str:>12} {me_ratio:>9.1f}% {count:>6}")
    
    # Summary statistics
    print("-" * 80)
    me_values: List[float] = [r['me_ratio'] for r in daily_ratios if r['me_ratio'] > 0]
    
    if me_values:
        print(f"{'SUMMARY':>12}")
        print(f"{'Average:':>12} {sum(me_values)/len(me_values):>9.1f}%")
        print(f"{'Maximum:':>12} {max(me_values):>9.1f}%")
        print(f"{'Minimum:':>12} {min(me_values):>9.1f}%")
        print(f"{'Days with positions:':>20} {len(me_values)}/{len(daily_ratios)}")
    else:
        print(f"{'No positions during period':>30}")
    
    print(f"{'='*80}")

def main() -> None:
    """
    Main function to run M/E ratio listing tool.

    Raises:
        ValueError: If command-line arguments are invalid.
    """
    print("Daily M/E Ratio Calculator")
    print("=" * 40)
    
    # Parse command line arguments
    days_back: int = 30  # Default
    show_details: bool = False
    
    if len(sys.argv) > 1:
        try:
            days_back = int(sys.argv[1])
        except ValueError:
            if sys.argv[1] == '--details':
                show_details = True
            else:
                print(f"Invalid argument: {sys.argv[1]}")
                print("Usage: python list_me_ratios.py [days_back] [--details]")
                return
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '--details':
            show_details = True
    
    # Load and process data
    trades_df: pd.DataFrame = load_trade_history()
    
    if trades_df.empty:
        return
    
    # Calculate M/E ratios
    daily_ratios: List[Dict] = calculate_daily_me_ratios(trades_df, days_back=days_back)
    
    # Display results
    display_me_ratios(daily_ratios, show_details=show_details)

def show_current_me_ratio() -> None:
    """
    Show current M/E ratio from positions file.
    """
    print(f"\n{'-'*40}")
    print("CURRENT M/E RATIO")
    print(f"{'-'*40}")
    
    positions_file: str = "positions.csv"
    
    if not os.path.exists(positions_file):
        print("❌ No current positions file found")
        return
    
    try:
        positions_df: pd.DataFrame = pd.read_csv(positions_file)
        
        if positions_df.empty:
            print("No current positions")
            return
        
        # Calculate current M/E ratio
        long_positions: pd.DataFrame = positions_df[positions_df['shares'] > 0]
        short_positions: pd.DataFrame = positions_df[positions_df['shares'] < 0]
        
        long_value: float = (
            (long_positions['current_price'] * long_positions['shares']).sum()
            if not long_positions.empty else 0
        )
        short_value: float = (
            (short_positions['current_price'] * short_positions['shares'].abs()).sum()
            if not short_positions.empty else 0
        )
        
        total_position_value: float = long_value + short_value
        
        # Estimate current portfolio value (this would need to be calculated properly)
        current_portfolio_value: float = 100000  # Placeholder
        
        if current_portfolio_value > 0:
            current_me_ratio: float = (total_position_value / current_portfolio_value) * 100
        else:
            current_me_ratio = 0.0
        
        print(f"Current M/E Ratio: {current_me_ratio:.1f}%")
        print(f"Open Positions: {len(positions_df)}")
        print(f"Total Position Value: ${total_position_value:,.0f}")
        
    except Exception as e:
        print(f"❌ Error reading positions: {e}")

if __name__ == "__main__":
    main()
    show_current_me_ratio()
    
    print(f"\n{'='*40}")
    print("USAGE:")
    print("  python list_me_ratios.py                    # Last 30 days")
    print("  python list_me_ratios.py 60                 # Last 60 days") 
    print(f"{'='*40}")