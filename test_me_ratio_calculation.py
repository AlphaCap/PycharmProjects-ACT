# test_me_ratio_calculation.py - Test M/E ratio calculation with sample data
import pandas as pd
import os
from datetime import datetime
from typing import Optional, List

def create_sample_trade_data() -> pd.DataFrame:
    """
    Create sample trade data for testing M/E ratio calculation.

    Returns:
        pd.DataFrame: DataFrame containing sample trade data.

    Raises:
        OSError: If directory creation or file writing fails.
    """
    # Create sample trades
    sample_trades: List[dict] = [
        {
            'symbol': 'AAPL',
            'type': 'long',
            'entry_date': '2024-01-15',
            'exit_date': '2024-01-20',
            'entry_price': 150.00,
            'exit_price': 155.00,
            'shares': 100,
            'profit': 500.00
        },
        {
            'symbol': 'MSFT',
            'type': 'short',
            'entry_date': '2024-01-16',
            'exit_date': '2024-01-22',
            'entry_price': 300.00,
            'exit_price': 295.00,
            'shares': 50,
            'profit': 250.00
        },
        {
            'symbol': 'GOOGL',
            'type': 'long',
            'entry_date': '2024-01-18',
            'exit_date': '2024-01-25',
            'entry_price': 120.00,
            'exit_price': 125.00,
            'shares': 80,
            'profit': 400.00
        },
        {
            'symbol': 'TSLA',
            'type': 'long',
            'entry_date': '2024-01-20',
            'exit_date': '2024-01-28',
            'entry_price': 200.00,
            'exit_price': 190.00,
            'shares': 60,
            'profit': -600.00
        }
    ]
    
    # Create DataFrame
    df: pd.DataFrame = pd.DataFrame(sample_trades)
    
    # Ensure data directory exists
    os.makedirs('data/trades', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/trades/trade_history.csv', index=False)
    print("Sample trade data created in data/trades/trade_history.csv")
    print("\nSample trades:")
    print(df)
    
    return df

def manual_me_calculation_check() -> None:
    """
    Manually verify M/E calculation for specific dates.
    """
    print("\n" + "="*60)
    print("MANUAL M/E RATIO VERIFICATION")
    print("="*60)
    
    # Manually check January 18, 2024
    print("\nJanuary 18, 2024 Analysis:")
    print("- AAPL: 100 shares × $150 = $15,000 (long position)")
    print("- MSFT: 50 shares × $300 = $15,000 (short position)")
    print("- GOOGL: 80 shares × $120 = $9,600 (entered today)")
    print("- Total Position Value = $15,000 + $15,000 + $9,600 = $39,600")
    print("- Portfolio Equity = $100,000 (no exits yet)")
    print("- M/E Ratio = ($39,600 / $100,000) × 100 = 39.6%")
    
    print("\nJanuary 22, 2024 Analysis (after MSFT exit):")
    print("- AAPL: Still open, 100 shares × $150 = $15,000")
    print("- MSFT: Closed with $250 profit")
    print("- GOOGL: Still open, 80 shares × $120 = $9,600")
    print("- Total Position Value = $15,000 + $9,600 = $24,600")
    print("- Portfolio Equity = $100,000 + $250 = $100,250")
    print("- M/E Ratio = ($24,600 / $100,250) × 100 = 24.54%")

def run_me_calculation_test() -> Optional[pd.DataFrame]:
    """
    Run the M/E calculation and show results.

    Returns:
        Optional[pd.DataFrame]: DataFrame of M/E ratios if successful, None otherwise.

    Raises:
        ImportError: If the calculation module is not found.
        Exception: For other runtime errors.
    """
    print("\n" + "="*60)
    print("RUNNING M/E RATIO CALCULATION")
    print("="*60)
    
    # Import the calculation function
    try:
        from calculate_daily_me_ratio_fixed import calculate_daily_me_ratios
        
        # Run the calculation
        me_df: pd.DataFrame = calculate_daily_me_ratios(initial_value=100000)
        
        print("\n" + "="*60)
        print("DETAILED DAILY BREAKDOWN")
        print("="*60)
        
        # Show detailed breakdown for verification
        key_dates: List[str] = ['2024-01-18', '2024-01-22', '2024-01-25']
        
        for date in key_dates:
            if date in me_df['Date'].dt.strftime('%Y-%m-%d').values:
                row = me_df[me_df['Date'].dt.strftime('%Y-%m-%d') == date].iloc[0]
                print(f"\n{date}:")
                print(f"  Long Value: ${row['Long_Value']:,.2f}")
                print(f"  Short Value: ${row['Short_Value']:,.2f}")
                print(f"  Total Position Value: ${row['Total_Position_Value']:,.2f}")
                print(f"  Portfolio Equity: ${row['Portfolio_Equity']:,.2f}")
                print(f"  M/E Ratio: {row['ME_Ratio']:.2f}%")
                print(f"  Long Positions: {row['Long_Positions']}")
                print(f"  Short Positions: {row['Short_Positions']}")
                print(f"  Cumulative Profit: ${row['Cumulative_Profit']:,.2f}")
        
        return me_df
        
    except ImportError:
        print("Could not import calculate_daily_me_ratio module")
        print("Make sure the file is in the same directory")
        return None
    except Exception as e:
        print(f"Error running M/E calculation: {e}")
        return None

def validate_me_calculation(me_df: Optional[pd.DataFrame]) -> None:
    """
    Validate the M/E calculation results.

    Args:
        me_df (Optional[pd.DataFrame]): DataFrame of M/E ratios, or None if unavailable.
    """
    if me_df is None:
        print("No data to validate")
        return
        
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check 1: M/E ratio should never exceed reasonable limits
    max_me: float = me_df['ME_Ratio'].max()
    print("✓ Max M/E Ratio:", f"{max_me:.2f}% (should be reasonable)")
    
    # Check 2: Portfolio equity should generally increase with profits
    final_equity: float = me_df['Portfolio_Equity'].iloc[-1]
    initial_equity: float = me_df['Portfolio_Equity'].iloc[0]
    total_profit: float = me_df['Cumulative_Profit'].iloc[-1]
    
    print("✓ Initial Equity:", f"${initial_equity:,.2f}")
    print("✓ Final Equity:", f"${final_equity:,.2f}")
    print("✓ Total Profit:", f"${total_profit:,.2f}")
    print("✓ Equity Check:", f"{initial_equity + total_profit} = {final_equity} ✓" if abs(initial_equity + total_profit - final_equity) < 0.01 else "✗ Equity mismatch!")
    
    # Check 3: M/E ratio calculation spot check
    sample_row = me_df.iloc[10] if len(me_df) > 10 else me_df.iloc[-1]
    calculated_me: float = (sample_row['Total_Position_Value'] / sample_row['Portfolio_Equity']) * 100
    stored_me: float = sample_row['ME_Ratio']
    
    print("✓ Spot Check M/E Calculation:")
    print("  Calculated:", f"{calculated_me:.2f}%")
    print("  Stored:", f"{stored_me:.2f}%")
    print("  Match:", "✓" if abs(calculated_me - stored_me) < 0.01 else "✗")

if __name__ == "__main__":
    print("M/E Ratio Calculation Test")
    print("=" * 50)
    
    # Step 1: Create sample data
    create_sample_trade_data()
    
    # Step 2: Show manual calculation
    manual_me_calculation_check()
    
    # Step 3: Run the calculation
    me_df: Optional[pd.DataFrame] = run_me_calculation_test()
    
    # Step 4: Validate results
    validate_me_calculation(me_df)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("Files created:")
    print("- data/trades/trade_history.csv (sample trade data)")
    print("- data/me_ratio_history.csv (daily M/E ratios)")
    print("- me_ratio_summary.json (summary statistics)")
    print("- me_ratio_history.png (chart, if matplotlib available)")
    print("python calculate_daily_me_ratio_fixed.py")
