# calculate_historical_me.py - Run this to calculate proper M/E ratios
import pandas as pd
import numpy as np
from datetime import datetime
import json

def calculate_daily_me_ratios():
    """Calculate daily M/E ratios from trade history"""
    
    # Load trades
    trades_df = pd.read_csv('data/trades/trade_history.csv')
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Initial portfolio value
    initial_value = 100000
    
    # Get date range
    start_date = trades_df['entry_date'].min()
    end_date = trades_df['exit_date'].max()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"Analyzing M/E ratios from {start_date.date()} to {end_date.date()}")
    print(f"Total trading days: {len(date_range)}")
    
    # Calculate daily metrics
    daily_data = []
    
    for current_date in date_range:
        # Find all positions open on this date
        open_positions = trades_df[
            (trades_df['entry_date'] <= current_date) & 
            (trades_df['exit_date'] >= current_date)
        ]
        
        # Calculate closed P&L up to this date
        closed_trades = trades_df[trades_df['exit_date'] < current_date]
        cumulative_profit = closed_trades['profit'].sum() if not closed_trades.empty else 0
        
        # Portfolio equity
        portfolio_equity = initial_value + cumulative_profit
        
        if not open_positions.empty:
            # Calculate position values
            long_positions = open_positions[open_positions['shares'] > 0]
            short_positions = open_positions[open_positions['shares'] < 0]
            
            # Long value
            long_value = (long_positions['entry_price'] * long_positions['shares']).sum() if not long_positions.empty else 0
            
            # Short value (using absolute shares)
            short_value = (short_positions['entry_price'] * short_positions['shares'].abs()).sum() if not short_positions.empty else 0
            
            # Total position value
            total_position_value = long_value + short_value
            
            # M/E ratio
            me_ratio = (total_position_value / portfolio_equity) * 100
            
            daily_data.append({
                'date': current_date,
                'open_positions': len(open_positions),
                'long_positions': len(long_positions),
                'short_positions': len(short_positions),
                'portfolio_equity': portfolio_equity,
                'total_position_value': total_position_value,
                'long_value': long_value,
                'short_value': short_value,
                'me_ratio': me_ratio
            })
    
    # Create DataFrame
    daily_df = pd.DataFrame(daily_data)
    
    if not daily_df.empty:
        # Calculate statistics
        avg_me_ratio = daily_df['me_ratio'].mean()
        max_me_ratio = daily_df['me_ratio'].max()
        min_me_ratio = daily_df['me_ratio'].min()
        
        print(f"\n=== M/E Ratio Statistics ===")
        print(f"Average M/E Ratio: {avg_me_ratio:.1f}%")
        print(f"Maximum M/E Ratio: {max_me_ratio:.1f}%")
        print(f"Minimum M/E Ratio: {min_me_ratio:.1f}%")
        print(f"Average Open Positions: {daily_df['open_positions'].mean():.1f}")
        
        # Save results
        daily_df.to_csv('historical_me_ratios.csv', index=False)
        
        # Save summary for data_manager
        summary = {
            'average_me_ratio': avg_me_ratio,
            'max_me_ratio': max_me_ratio,
            'min_me_ratio': min_me_ratio,
            'calculation_date': datetime.now().isoformat(),
            'total_days_analyzed': len(daily_df)
        }
        
        with open('me_ratio_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved detailed M/E data to 'historical_me_ratios.csv'")
        print(f"✅ Saved summary to 'me_ratio_summary.json'")
        
        # Show sample of high M/E days
        high_me_days = daily_df[daily_df['me_ratio'] > 150].sort_values('me_ratio', ascending=False).head(5)
        if not high_me_days.empty:
            print(f"\n=== Days with Highest M/E Ratios ===")
            for _, row in high_me_days.iterrows():
                print(f"{row['date'].date()}: {row['me_ratio']:.1f}% ({row['open_positions']} positions)")
    
    return daily_df

if __name__ == "__main__":
    calculate_daily_me_ratios()
