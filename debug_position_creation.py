from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
import pandas as pd
from datetime import datetime

print("=== DEBUGGING POSITION CREATION ===")

# Load one symbol that we know has signals
symbols = ['AAPL']  # We know this has a signal on 2025-04-09
data = load_polygon_data(symbols)

strategy = NGSStrategy()
print(f"Initial cash: ${strategy.cash:,.2f}")
print(f"Retention cutoff: {strategy.cutoff_date}")

for symbol, df in data.items():
    print(f"\n--- Processing {symbol} in detail ---")
    
    # Process through the full strategy pipeline
    df_indicators = strategy.calculate_indicators(df)
    df_signals = strategy.generate_signals(df_indicators)
    
    # Find the signal we know exists
    signal_rows = df_signals[df_signals['Signal'] != 0]
    print(f"Found {len(signal_rows)} signal rows")
    
    for idx, row in signal_rows.iterrows():
        signal_date = row['Date']
        signal_type = row['SignalType']
        signal_value = row['Signal']
        shares = row['Shares']
        close_price = row['Close']
        
        print(f"\nSignal found:")
        print(f"  Date: {signal_date}")
        print(f"  Type: {signal_type}")
        print(f"  Direction: {'Long' if signal_value > 0 else 'Short'}")
        print(f"  Shares: {shares}")
        print(f"  Price: ${close_price:.2f}")
        print(f"  Cost: ${shares * close_price:,.2f}")
        
        # Check if this date is within retention period
        if isinstance(signal_date, str):
            signal_datetime = datetime.strptime(signal_date[:10], '%Y-%m-%d')
        else:
            signal_datetime = signal_date.to_pydatetime() if hasattr(signal_date, 'to_pydatetime') else signal_date
            
        is_in_retention = signal_datetime >= strategy.cutoff_date.replace(tzinfo=None)
        print(f"  Within retention period: {is_in_retention}")
        print(f"  Signal date: {signal_datetime}")
        print(f"  Cutoff date: {strategy.cutoff_date}")
        
        # Check cash availability
        trade_cost = abs(shares * close_price)
        can_afford = trade_cost <= strategy.cash
        print(f"  Can afford trade: {can_afford} (${trade_cost:,.2f} <= ${strategy.cash:,.2f})")
        
        # Check what happens when we try to process this entry
        print(f"  Attempting to create position...")

    # Now run the full position management
    print(f"\nRunning full position management for {symbol}...")
    print(f"Positions before: {len(strategy.positions)}")
    
    result_df = strategy.manage_positions(df_signals, symbol)
    
    print(f"Positions after: {len(strategy.positions)}")
    print(f"Current positions: {strategy.positions}")
    print(f"Cash remaining: ${strategy.cash:,.2f}")
    print(f"Trades generated: {len(strategy.trades)}")
    
    if strategy.trades:
        print("Trades:")
        for trade in strategy.trades:
            print(f"  {trade}")

print(f"\n=== POSITION CREATION DEBUG COMPLETE ===")