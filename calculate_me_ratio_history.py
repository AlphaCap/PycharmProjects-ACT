# calculate_me_ratio_history.py
"""
Script to calculate proper daily M/E ratio history for the historical page.
This will generate the data needed for accurate historical M/E ratio display.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from data_manager import get_trades_history, format_dollars, ensure_dir
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_daily_me_ratios(initial_portfolio_value: float = 100000):
    """
    Calculate daily M/E ratios from trade history and save to file.
    
    Args:
        initial_portfolio_value: Starting portfolio value
    """
    try:
        # Load trade history
        trades_df = get_trades_history()
        
        if trades_df.empty:
            logger.error("No trade history available for M/E calculation")
            return False
        
        logger.info(f"Calculating daily M/E ratios from {len(trades_df)} trades")
        
        # Convert dates
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Get date range
        start_date = trades_df['entry_date'].min()
        end_date = trades_df['exit_date'].max()
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
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
            closed_trades = trades_df[trades_df['exit_date'] <= current_date]
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
            'average_me_ratio': avg_me,
            'max_me_ratio': max_me,
            'min_me_ratio': min_me,
            'initial_portfolio_value': initial_portfolio_value
        }
        
        import json
        summary_file = 'me_ratio_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ M/E ratio history calculated and saved to {output_file}")
        logger.info(f"✅ Summary saved to {summary_file}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   - Total days: {len(me_history_df)}")
        logger.info(f"   - Average M/E: {avg_me:.1f}%")
        logger.info(f"   - Max M/E: {max_me:.1f}%")
        logger.info(f"   - Min M/E: {min_me:.1f}%")
        
        # Show sample data
        print("\n📋 Sample M/E ratio data (last 10 days):")
        sample_df = me_history_df.tail(10)[['Date', 'ME_Ratio', 'Open_Positions', 'Portfolio_Equity']]
        sample_df['Date'] = sample_df['Date'].dt.strftime('%Y-%m-%d')
        sample_df['Portfolio_Equity'] = sample_df['Portfolio_Equity'].apply(lambda x: f"${x:,.0f}")
        sample_df['ME_Ratio'] = sample_df['ME_Ratio'].apply(lambda x: f"{x:.1f}%")
        print(sample_df.to_string(index=False))
        
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
            print(f"❌ M/E history file not found: {me_file}")
            return False
        
        if not os.path.exists(summary_file):
            print(f"❌ Summary file not found: {summary_file}")
            return False
        
        # Load and verify data
        me_df = pd.read_csv(me_file, parse_dates=['Date'])
        
        import json
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("✅ M/E ratio calculation verification:")
        print(f"   - History file: {len(me_df)} days of data")
        print(f"   - Date range: {summary['start_date'][:10]} to {summary['end_date'][:10]}")
        print(f"   - Average M/E: {summary['average_me_ratio']:.1f}%")
        print(f"   - Calculation date: {summary['calculation_date'][:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("M/E Ratio History Calculator")
    print("=" * 40)
    
    # Calculate M/E ratios
    success = calculate_daily_me_ratios()
    
    if success:
        print("\n" + "=" * 40)
        verify_me_calculation()
        print("\n✅ Historical M/E ratio calculation complete!")
        print("   The historical page should now show correct M/E ratios.")
    else:
        print("\n❌ M/E ratio calculation failed!")
        print("   Check the error messages above and ensure trade history exists.")
