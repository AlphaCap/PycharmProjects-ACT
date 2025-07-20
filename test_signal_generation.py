from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
import pandas as pd

print("=== TESTING SIGNAL GENERATION ===")

# Test with just a few symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
print(f"Testing with symbols: {symbols}")

# Load data
data = load_polygon_data(symbols)
print(f"Loaded data for {len(data)} symbols")

# Initialize strategy
strategy = NGSStrategy()
print(f"Strategy initialized")
print(f"Retention cutoff: {strategy.cutoff_date}")
print(f"Price range filter: ${strategy.inputs['MinPrice']} - ${strategy.inputs['MaxPrice']}")

# Process each symbol and check for signals
for symbol, df in data.items():
    print(f"\n--- Processing {symbol} ---")
    print(f"Raw data: {len(df)} rows")
    
    # Calculate indicators
    df_indicators = strategy.calculate_indicators(df)
    if df_indicators is None:
        print(f"❌ Failed to calculate indicators for {symbol}")
        continue
    
    print(f"With indicators: {len(df_indicators)} rows")
    
    # Generate signals
    df_signals = strategy.generate_signals(df_indicators)
    
    # Check for any signals
    long_signals = df_signals[df_signals['Signal'] == 1]
    short_signals = df_signals[df_signals['Signal'] == -1]
    
    print(f"Long signals: {len(long_signals)}")
    print(f"Short signals: {len(short_signals)}")
    
    if len(long_signals) > 0:
        print("Long signal dates:", long_signals['Date'].dt.strftime('%Y-%m-%d').tolist())
        print("Long signal types:", long_signals['SignalType'].tolist())
    
    if len(short_signals) > 0:
        print("Short signal dates:", short_signals['Date'].dt.strftime('%Y-%m-%d').tolist())
        print("Short signal types:", short_signals['SignalType'].tolist())
    
    # Check recent price ranges to see if they're in valid range
    recent_prices = df_indicators['Close'].tail(10)
    in_range = recent_prices[(recent_prices >= strategy.inputs['MinPrice']) & 
                            (recent_prices <= strategy.inputs['MaxPrice'])]
    print(f"Recent prices in range ${strategy.inputs['MinPrice']}-${strategy.inputs['MaxPrice']}: {len(in_range)}/10")
    print(f"Recent price range: ${recent_prices.min():.2f} - ${recent_prices.max():.2f}")

print(f"\n=== TESTING COMPLETE ===")
