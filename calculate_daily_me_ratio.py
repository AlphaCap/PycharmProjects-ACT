# calculate_daily_me_ratio.py - Calculate and store daily M/E ratios
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def calculate_daily_me_ratios(initial_value=100000):
    """
    Calculate daily M/E ratios from trade history and save as indicator data
    """
    print("=== CALCULATING DAILY M/E RATIOS ===\n")
    
    # Load trades
    trades_df = pd.read_csv('data/trades/trade_history.csv')
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Get date range
    start_date = trades_df['entry_date'].min()
    end_date = trades_df['exit_date'].max()
    
    print(f"Analyzing period: {start_date.date()} to {end_date.date()}")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize tracking
    daily_me_data = []
    position_tracker = {}  # symbol -> {shares, entry_price, type}
    
    # Process each day
    for current_date in date_range:
        # Add new positions that start today
        new_entries = trades_df[trades_df['entry_date'].dt.date == current_date.date()]
        for _, trade in new_entries.iterrows():
            symbol = trade['symbol']
            if symbol not in position_tracker:
                position_tracker[symbol] = {
                    'shares': 0,
                    'entry_prices': [],
                    'total_cost': 0
                }
            
            # Add to position
            position_tracker[symbol]['shares'] += trade['shares']
            position_tracker[symbol]['entry_prices'].append(trade['entry_price'])
            position_tracker[symbol]['total_cost'] += abs(trade['shares']) * trade['entry_price']
        
        # Remove positions that exit today
        exits = trades_df[trades_df['exit_date'].dt.date == current_date.date()]
        for _, trade in exits.iterrows():
            symbol = trade['symbol']
            if symbol in position_tracker:
                position_tracker[symbol]['shares'] -= trade['shares']
                # Remove if position is closed
                if abs(position_tracker[symbol]['shares']) < 0.01:
                    del position_tracker[symbol]
        
        # Calculate current metrics
        total_long_value = 0
        total_short_value = 0
        long_positions = 0
        short_positions = 0
        
        for symbol, pos_data in position_tracker.items():
            if pos_data['shares'] > 0:
                # Long position
                avg_price = pos_data['total_cost'] / abs(pos_data['shares']) if pos_data['shares'] != 0 else 0
                position_value = abs(pos_data['shares']) * avg_price
                total_long_value += position_value
                long_positions += 1
            elif pos_data['shares'] < 0:
                # Short position
                avg_price = pos_data['total_cost'] / abs(pos_data['shares']) if pos_data['shares'] != 0 else 0
                position_value = abs(pos_data['shares']) * avg_price
                total_short_value += position_value
                short_positions += 1
        
        # Calculate portfolio equity (initial + realized profits to date)
        closed_to_date = trades_df[trades_df['exit_date'] < current_date]
        cumulative_profit = closed_to_date['profit'].sum() if not closed_to_date.empty else 0
        portfolio_equity = initial_value + cumulative_profit
        
        # Calculate M/E ratio
        total_position_value = total_long_value + total_short_value
        me_ratio = (total_position_value / portfolio_equity * 100) if portfolio_equity > 0 else 0
        
        # Store daily data
        daily_me_data.append({
            'Date': current_date,
            'Portfolio_Equity': portfolio_equity,
            'Long_Value': total_long_value,
            'Short_Value': total_short_value,
            'Total_Position_Value': total_position_value,
            'ME_Ratio': me_ratio,
            'Long_Positions': long_positions,
            'Short_Positions': short_positions,
            'Total_Positions': long_positions + short_positions,
            'Cumulative_Profit': cumulative_profit
        })
    
    # Create DataFrame
    me_df = pd.DataFrame(daily_me_data)
    
    # Save to CSV
    output_file = 'data/me_ratio_history.csv'
    os.makedirs('data', exist_ok=True)
    me_df.to_csv(output_file, index=False)
    print(f"\nSaved M/E ratio history to {output_file}")
    
    # Calculate statistics
    print("\n=== M/E RATIO STATISTICS ===")
    print(f"Average M/E Ratio: {me_df['ME_Ratio'].mean():.1f}%")
    print(f"Maximum M/E Ratio: {me_df['ME_Ratio'].max():.1f}%")
    print(f"Minimum M/E Ratio: {me_df['ME_Ratio'].min():.1f}%")
    print(f"Standard Deviation: {me_df['ME_Ratio'].std():.1f}%")
    
    # Position statistics
    print(f"\n=== POSITION STATISTICS ===")
    print(f"Average Total Positions: {me_df['Total_Positions'].mean():.1f}")
    print(f"Maximum Positions: {me_df['Total_Positions'].max()}")
    print(f"Average Long Positions: {me_df['Long_Positions'].mean():.1f}")
    print(f"Average Short Positions: {me_df['Short_Positions'].mean():.1f}")
    
    # Show sample data
    print("\n=== SAMPLE DATA (First 10 days) ===")
    print(me_df[['Date', 'ME_Ratio', 'Long_Positions', 'Short_Positions', 'Portfolio_Equity']].head(10))
    
    # Save summary for quick access
    summary = {
        'average_me_ratio': float(me_df['ME_Ratio'].mean()),
        'max_me_ratio': float(me_df['ME_Ratio'].max()),
        'min_me_ratio': float(me_df['ME_Ratio'].min()),
        'std_me_ratio': float(me_df['ME_Ratio'].std()),
        'average_positions': float(me_df['Total_Positions'].mean()),
        'calculation_date': datetime.now().isoformat(),
        'total_days': len(me_df)
    }
    
    import json
    with open('me_ratio_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to me_ratio_summary.json")
    
    # Create a chart
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # M/E Ratio over time
        ax1.plot(me_df['Date'], me_df['ME_Ratio'], label='M/E Ratio', color='blue', linewidth=2)
        ax1.axhline(y=me_df['ME_Ratio'].mean(), color='red', linestyle='--', 
                    label=f'Average ({me_df["ME_Ratio"].mean():.1f}%)')
        ax1.set_ylabel('M/E Ratio (%)')
        ax1.set_title('Historical Margin-to-Equity Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Position counts
        ax2.plot(me_df['Date'], me_df['Long_Positions'], label='Long Positions', color='green')
        ax2.plot(me_df['Date'], me_df['Short_Positions'], label='Short Positions', color='red')
        ax2.plot(me_df['Date'], me_df['Total_Positions'], label='Total Positions', 
                 color='black', linestyle='--')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Positions')
        ax2.set_title('Position Count Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('me_ratio_history.png', dpi=150)
        print("\nSaved chart to me_ratio_history.png")
        plt.close()
        
    except ImportError:
        print("\nMatplotlib not available - skipping chart generation")
    
    return me_df

if __name__ == "__main__":
    me_df = calculate_daily_me_ratios()
