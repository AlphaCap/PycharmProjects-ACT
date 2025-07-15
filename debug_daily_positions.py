# debug_daily_positions.py - Show actual daily positions and M/E calculations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_me_calculation():
    """Show daily open positions and M/E calculations"""
    
    # Load trades
    trades_df = pd.read_csv('data/trades/trade_history.csv')
    print(f"Loaded {len(trades_df)} trades from trade_history.csv")
    print(f"Total profit: ${trades_df['profit'].sum():,.2f}\n")
    
    # Convert dates
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Initial value
    initial_value = 100000
    
    # Sample some specific dates to debug
    # Let's check a few dates spread across the trading period
    start_date = trades_df['entry_date'].min()
    end_date = trades_df['exit_date'].max()
    
    print(f"Trading period: {start_date.date()} to {end_date.date()}\n")
    
    # Check 10 sample dates
    sample_dates = pd.date_range(start=start_date, end=end_date, periods=10)
    
    print("=== SAMPLE DAILY POSITIONS ===\n")
    
    me_ratios = []
    
    for date in sample_dates:
        # Find open positions on this date
        open_positions = trades_df[
            (trades_df['entry_date'] <= date) & 
            (trades_df['exit_date'] >= date)
        ]
        
        if not open_positions.empty:
            # Calculate values
            long_positions = open_positions[open_positions['shares'] > 0]
            short_positions = open_positions[open_positions['shares'] < 0]
            
            # Position values
            long_value = (long_positions['entry_price'] * long_positions['shares']).sum() if not long_positions.empty else 0
            short_value = (short_positions['entry_price'] * short_positions['shares'].abs()).sum() if not short_positions.empty else 0
            total_position_value = long_value + short_value
            
            # Portfolio equity
            closed_trades = trades_df[trades_df['exit_date'] < date]
            cumulative_profit = closed_trades['profit'].sum() if not closed_trades.empty else 0
            portfolio_equity = initial_value + cumulative_profit
            
            # M/E ratio
            me_ratio = (total_position_value / portfolio_equity * 100) if portfolio_equity > 0 else 0
            me_ratios.append(me_ratio)
            
            print(f"Date: {date.date()}")
            print(f"  Open positions: {len(open_positions)}")
            print(f"  - Long: {len(long_positions)} positions, value: ${long_value:,.0f}")
            print(f"  - Short: {len(short_positions)} positions, value: ${short_value:,.0f}")
            print(f"  Total position value: ${total_position_value:,.0f}")
            print(f"  Portfolio equity: ${portfolio_equity:,.0f}")
            print(f"  M/E Ratio: {me_ratio:.1f}%")
            print()
            
            # Show actual positions for first date
            if date == sample_dates[0]:
                print("  Detailed positions:")
                for _, pos in open_positions.head(5).iterrows():
                    value = pos['entry_price'] * abs(pos['shares'])
                    print(f"    {pos['symbol']}: {pos['shares']} shares @ ${pos['entry_price']:.2f} = ${value:,.0f}")
                if len(open_positions) > 5:
                    print(f"    ... and {len(open_positions) - 5} more positions")
                print()
    
    if me_ratios:
        print(f"\n=== SUMMARY ===")
        print(f"Average M/E Ratio: {np.mean(me_ratios):.1f}%")
        print(f"Max M/E Ratio: {max(me_ratios):.1f}%")
        print(f"Min M/E Ratio: {min(me_ratios):.1f}%")
    
    # Check for potential issues
    print(f"\n=== CHECKING FOR ISSUES ===")
    
    # Check share counts
    print(f"Average shares per trade: {trades_df['shares'].abs().mean():.0f}")
    print(f"Max shares in a trade: {trades_df['shares'].abs().max()}")
    
    # Check if there are huge positions
    large_positions = trades_df[trades_df['shares'].abs() > 1000]
    if not large_positions.empty:
        print(f"\nFound {len(large_positions)} trades with >1000 shares:")
        for _, trade in large_positions.head(3).iterrows():
            value = trade['entry_price'] * abs(trade['shares'])
            print(f"  {trade['symbol']}: {trade['shares']} shares @ ${trade['entry_price']:.2f} = ${value:,.0f}")
    
    # Calculate what a typical position value is
    trades_df['position_value'] = trades_df['entry_price'] * trades_df['shares'].abs()
    avg_position_value = trades_df['position_value'].mean()
    print(f"\nAverage position value: ${avg_position_value:,.0f}")
    
    # Check for overlapping trades (many positions open at once)
    max_concurrent = 0
    for date in pd.date_range(start=start_date, end=end_date, freq='D'):
        concurrent = len(trades_df[
            (trades_df['entry_date'] <= date) & 
            (trades_df['exit_date'] >= date)
        ])
        max_concurrent = max(max_concurrent, concurrent)
    
    print(f"Maximum concurrent positions: {max_concurrent}")

if __name__ == "__main__":
    debug_me_calculation()