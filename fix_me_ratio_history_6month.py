# fix_me_ratio_history_6month.py
"""
Script to calculate M/E ratio history for the last 6 months only (rolling window).
This respects the strategy's 6-month data retention policy.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from data_manager import get_trades_history, format_dollars, ensure_dir
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_daily_me_ratios_6month(initial_portfolio_value: float = 100000):
    """
    Calculate daily M/E ratios from trade history for the last 6 months only.
    
    Args:
        initial_portfolio_value: Starting portfolio value
    """
    try:
        # Load trade history
        trades_df = get_trades_history()
        
        if trades_df.empty:
            logger.error("No trade history available for M/E calculation")
            return False
        
        logger.info(f"Original trade history: {len(trades_df)} trades")
        
        # FILTER TO LAST 6 MONTHS ONLY
        six_months_ago = datetime.now() - timedelta(days=180)
        
        # Convert dates
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Filter to last 6 months (entry date or exit date within 6 months)
        trades_df = trades_df[
            (trades_df['entry_date'] >= six_months_ago) | 
            (trades_df['exit_date'] >= six_months_ago)
        ]
        
        if trades_df.empty:
            logger.error("No trades in the last 6 months")
            return False
        
        logger.info(f"Filtered to last 6 months: {len(trades_df)} trades")
        
        # Get date range (last 6 months only)
        start_date = max(trades_df['entry_date'].min(), six_months_ago)
        end_date = trades_df['exit_date'].max()
        
        logger.info(f"M/E calculation date range: {start_date.date()} to {end_date.date()}")
        logger.info(f"Days to process: {(end_date - start_date).days}")
        
        # Create daily date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Calculate daily metrics
        daily_data = []
        
        for current_date in date_range:
            # Find positions open on this date
            open_positions = trades_df[
                (trades_df['entry_date'] <= current_date) & 
                (trades_df['exit_date'] > current_date)
            ]
            
            # Calculate portfolio equity up to this date
            # For 6-month rolling: use trades that closed before this date AND within our window
            closed_trades = trades_df[
                (trades_df['exit_date'] <= current_date) &
                (trades_df['exit_date'] >= six_months_ago)
            ]
            cumulative_profit = closed_trades['profit'].sum() if not closed_trades.empty else 0
            portfolio_equity = initial_portfolio_value + cumulative_profit
            
            # Calculate total position value
            total_position_value = 0
            position_count = 0
            
            if not open_positions.empty:
                for _, position in open_positions.iterrows():
                    # Use entry price as the position value
                    position_value = abs(position['shares']) * position['entry_price']
                    total_position_value += position_value
                    position_count += 1
            
            # Calculate M/E ratio
            me_ratio = (total_position_value / portfolio_equity * 100) if portfolio_equity > 0 else 0
            
            # Store daily data
            daily_data.append({
                'Date': current_date,
                'Portfolio_Equity': portfolio_equity,
                'Total_Position_Value': total_position_value,
                'ME_Ratio': me_ratio,
                'Open_Positions': position_count,
                'Cumulative_Profit': cumulative_profit
            })
        
        # Create DataFrame
        me_history_df = pd.DataFrame(daily_data)
        
        # Save to file
        output_dir = 'data'
        ensure_dir(os.path.join(output_dir, 'temp.csv'))
        output_file = os.path.join(output_dir, 'me_ratio_history.csv')
        
        me_history_df.to_csv(output_file, index=False)
        
        # Calculate summary statistics
        avg_me = me_history_df['ME_Ratio'].mean()
        max_me = me_history_df['ME_Ratio'].max()
        min_me = me_history_df['ME_Ratio'].min()
        
        # Save summary
        summary = {
            'calculation_date': datetime.now().isoformat(),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_days': len(me_history_df),
            'trades_used': len(trades_df),
            'average_me_ratio': avg_me,
            'max_me_ratio': max_me,
            'min_me_ratio': min_me,
            'initial_portfolio_value': initial_portfolio_value,
            'rolling_window_days': 180
        }
        
        import json
        summary_file = 'me_ratio_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Success! M/E ratio history calculated and saved to {output_file}")
        print(f"Summary saved to {summary_file}")
        print(f"Statistics (6-month rolling window):")
        print(f"   - Total days: {len(me_history_df)}")
        print(f"   - Trades used: {len(trades_df)}")
        print(f"   - Average M/E: {avg_me:.1f}%")
        print(f"   - Max M/E: {max_me:.1f}%")
        print(f"   - Min M/E: {min_me:.1f}%")
        
        # Show sample data
        print(f"\nSample M/E ratio data (last 10 days):")
        sample_df = me_history_df.tail(10)[['Date', 'ME_Ratio', 'Open_Positions', 'Portfolio_Equity']]
        sample_df['Date'] = sample_df['Date'].dt.strftime('%Y-%m-%d')
        sample_df['Portfolio_Equity'] = sample_df['Portfolio_Equity'].apply(lambda x: f"${x:,.0f}")
        sample_df['ME_Ratio'] = sample_df['ME_Ratio'].apply(lambda x: f"{x:.1f}%")
        print(sample_df.to_string(index=False))
        
        # Warning about data retention
        if len(get_trades_history()) > 200:  # More than ~6.5 months
            print(f"\nWARNING: Found {len(get_trades_history())} trades in history file.")
            print(f"Your 6-month rolling data cleanup may not be working properly.")
            print(f"Consider running data cleanup to remove trades older than 6 months.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error calculating daily M/E ratios: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_me_calculation():
    """Verify the M/E calculation results"""
    try:
        # Check if files exist
        me_file = 'data/me_ratio_history.csv'
        summary_file = 'me_ratio_summary.json'
        
        if not os.path.exists(me_file):
            print(f"Error: M/E history file not found: {me_file}")
            return False
        
        if not os.path.exists(summary_file):
            print(f"Error: Summary file not found: {summary_file}")
            return False
        
        # Load and verify data
        me_df = pd.read_csv(me_file, parse_dates=['Date'])
        
        import json
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("M/E ratio calculation verification:")
        print(f"   - History file: {len(me_df)} days of data")
        print(f"   - Date range: {summary['start_date'][:10]} to {summary['end_date'][:10]}")
        print(f"   - Trades used: {summary['trades_used']}")
        print(f"   - Average M/E: {summary['average_me_ratio']:.1f}%")
        print(f"   - Rolling window: {summary['rolling_window_days']} days")
        print(f"   - Calculation date: {summary['calculation_date'][:10]}")
        
        return True
        
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("M/E Ratio History Calculator (6-Month Rolling Window)")
    print("=" * 60)
    
    # Calculate M/E ratios for last 6 months only
    success = calculate_daily_me_ratios_6month()
    
    if success:
        print("\n" + "=" * 60)
        verify_me_calculation()
        print("\nHistorical M/E ratio calculation complete!")
        print("The historical page should now show correct 6-month M/E ratios.")
    else:
        print("\nM/E ratio calculation failed!")
        print("Check the error messages above.")
