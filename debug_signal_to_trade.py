from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
import pandas as pd
from datetime import datetime

print("=== DEBUGGING SIGNAL TO TRADE CONVERSION ===")

# Test with a broader sample
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'JPM', 'JNJ', 'PG']
print(f"Testing {len(symbols)} major stocks...")

data = load_polygon_data(symbols)
strategy = NGSStrategy()

print(f"Strategy settings:")
print(f"  Cash available: ${strategy.cash:,.2f}")
print(f"  Position size: ${strategy.inputs['PositionSize']}")
print(f"  Price range: ${strategy.inputs['MinPrice']}-${strategy.inputs['MaxPrice']}")
print(f"  Retention cutoff: {strategy.cutoff_date}")

total_signals = 0
total_trades = 0
signal_details = []

for symbol, df in data.items():
    print(f"\n--- Processing {symbol} ---")
    
    # Step 1: Calculate indicators
    df_indicators = strategy.calculate_indicators(df)
    if df_indicators is None:
        print(f"  ❌ Failed indicators")
        continue
    
    # Step 2: Generate signals
    df_signals = strategy.generate_signals(df_indicators)
    signals = df_signals[df_signals['Signal'] != 0]
    signal_count = len(signals)
    total_signals += signal_count
    
    print(f"  Signals generated: {signal_count}")
    
    if signal_count > 0:
        for _, signal in signals.iterrows():
            signal_date = signal['Date']
            signal_type = signal['SignalType']
            signal_direction = 'Long' if signal['Signal'] > 0 else 'Short'
            shares = signal['Shares']
            price = signal['Close']
            cost = abs(shares * price)
            
            print(f"    {signal_date}: {signal_type} ({signal_direction}) - {shares} shares @ ${price:.2f} = ${cost:,.2f}")
            
            # Check if this would pass position creation filters
            signal_datetime = pd.to_datetime(signal_date).to_pydatetime()
            within_retention = signal_datetime >= strategy.cutoff_date.replace(tzinfo=None)
            can_afford = cost <= strategy.cash
            
            print(f"      Within retention: {within_retention}")
            print(f"      Can afford: {can_afford}")
            
            signal_details.append({
                'symbol': symbol,
                'date': signal_date,
                'type': signal_type,
                'direction': signal_direction,
                'cost': cost,
                'within_retention': within_retention,
                'can_afford': can_afford
            })
    
    # Step 3: Run position management
    trades_before = len(strategy.trades)
    strategy.manage_positions(df_signals, symbol)
    trades_after = len(strategy.trades)
    new_trades = trades_after - trades_before
    
    print(f"  Trades created: {new_trades}")
    total_trades += new_trades
    
    if new_trades > 0:
        print("  New trades:")
        for trade in strategy.trades[-new_trades:]:
            print(f"    {trade}")

print(f"\n=== SUMMARY ===")
print(f"Total signals generated: {total_signals}")
print(f"Total trades executed: {total_trades}")
print(f"Signal-to-trade conversion rate: {(total_trades/total_signals*100) if total_signals > 0 else 0:.1f}%")

if total_signals > 0 and total_trades == 0:
    print("\n❌ SIGNALS FOUND BUT NO TRADES EXECUTED!")
    print("Checking what's blocking signal execution...")
    
    blocked_reasons = {'retention': 0, 'cash': 0, 'other': 0}
    for signal in signal_details:
        if not signal['within_retention']:
            blocked_reasons['retention'] += 1
        elif not signal['can_afford']:
            blocked_reasons['cash'] += 1
        else:
            blocked_reasons['other'] += 1
    
    print(f"Signals blocked by retention filter: {blocked_reasons['retention']}")
    print(f"Signals blocked by cash constraint: {blocked_reasons['cash']}")
    print(f"Signals blocked by other reasons: {blocked_reasons['other']}")

elif total_signals == 0:
    print("\n✅ NO SIGNALS GENERATED - This might be normal market conditions")
    print("The strategy patterns may not be occurring in current market environment")

print(f"\n=== DEBUG COMPLETE ===")
