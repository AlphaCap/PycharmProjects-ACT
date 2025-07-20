from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
import pandas as pd
import json
from datetime import datetime

def debug_signal_indicators():
    """
    Debug which specific indicators triggered each entry signal.
    Records detailed indicator values for analysis.
    """
    print("=== SIGNAL INDICATOR DEBUGGER ===")
    
    # Load broader symbol set for more signals
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'JPM', 'JNJ', 'PG', 
              'WMT', 'V', 'PFE', 'KO', 'DIS', 'BA', 'CAT', 'HD', 'MCD', 'UNH']
    
    print(f"Loading data for {len(symbols)} symbols...")
    data = load_polygon_data(symbols)
    strategy = NGSStrategy()
    
    all_signal_details = []
    
    for symbol, df in data.items():
        print(f"Processing {symbol}...")
        
        # Calculate indicators and generate signals
        df_indicators = strategy.calculate_indicators(df)
        if df_indicators is None:
            continue
            
        df_signals = strategy.generate_signals(df_indicators)
        signals = df_signals[df_signals['Signal'] != 0].copy()
        
        if len(signals) == 0:
            continue
            
        print(f"  Found {len(signals)} signals")
        
        # For each signal, capture all indicator values
        for idx, signal_row in signals.iterrows():
            signal_date = signal_row['Date']
            signal_type = signal_row['SignalType'] 
            signal_direction = 'Long' if signal_row['Signal'] > 0 else 'Short'
            
            # Get the previous day's data for comparison (since signals look at i-1)
            if idx > 0:
                prev_row = df_signals.iloc[idx-1]
            else:
                prev_row = signal_row  # fallback
            
            # Capture all relevant indicator values on signal date
            indicator_snapshot = {
                'symbol': symbol,
                'date': str(signal_date)[:10],
                'signal_type': signal_type,
                'direction': signal_direction,
                'shares': int(signal_row['Shares']),
                
                # Price data
                'open': round(signal_row['Open'], 2),
                'high': round(signal_row['High'], 2), 
                'low': round(signal_row['Low'], 2),
                'close': round(signal_row['Close'], 2),
                'prev_open': round(prev_row['Open'], 2),
                'prev_high': round(prev_row['High'], 2),
                'prev_low': round(prev_row['Low'], 2), 
                'prev_close': round(prev_row['Close'], 2),
                
                # Bollinger Bands
                'bb_upper': round(signal_row['UpperBB'], 2) if pd.notna(signal_row['UpperBB']) else None,
                'bb_lower': round(signal_row['LowerBB'], 2) if pd.notna(signal_row['LowerBB']) else None,
                'bb_avg': round(signal_row['BBAvg'], 2) if pd.notna(signal_row['BBAvg']) else None,
                
                # ATR indicators
                'atr': round(signal_row['ATR'], 2) if pd.notna(signal_row['ATR']) else None,
                'atr_ma': round(signal_row['ATRma'], 2) if pd.notna(signal_row['ATRma']) else None,
                
                # Linear Regression indicators (key for signal logic)
                'lr_value': round(signal_row['oLRValue'], 4) if pd.notna(signal_row['oLRValue']) else None,
                'lr_value2': round(signal_row['oLRValue2'], 4) if pd.notna(signal_row['oLRValue2']) else None,
                'lr_slope': round(signal_row['oLRSlope'], 4) if pd.notna(signal_row['oLRSlope']) else None,
                'lr_angle': round(signal_row['oLRAngle'], 2) if pd.notna(signal_row['oLRAngle']) else None,
                
                # Swing levels
                'swing_high': round(signal_row['SwingHigh'], 2) if pd.notna(signal_row['SwingHigh']) else None,
                'swing_low': round(signal_row['SwingLow'], 2) if pd.notna(signal_row['SwingLow']) else None,
                
                # PSAR
                'psar_long': round(signal_row['LongPSAR'], 2) if pd.notna(signal_row['LongPSAR']) else None,
                'psar_short': round(signal_row['ShortPSAR'], 2) if pd.notna(signal_row['ShortPSAR']) else None,
                'psar_is_long': int(signal_row['PSAR_IsLong']) if pd.notna(signal_row['PSAR_IsLong']) else None,
            }
            
            # Add specific pattern checks based on signal type
            if 'Engf L' in signal_type:  # Engulfing Long pattern
                pattern_checks = {
                    'open_below_prev_close': signal_row['Open'] < prev_row['Close'],
                    'close_above_prev_open': signal_row['Close'] > prev_row['Open'],
                    'current_green': signal_row['Close'] > signal_row['Open'],
                    'prev_red': prev_row['Close'] < prev_row['Open'],
                    'prev_close_near_low': abs(prev_row['Close'] - prev_row['Low']) / prev_row['Close'] < 0.05,
                    'gap_within_2atr': (signal_row['High'] - prev_row['Close']) <= (signal_row['ATR'] * 2) if pd.notna(signal_row['ATR']) else None,
                    'prev_body_size': prev_row['Open'] - prev_row['Close'] > 0.05,
                    'lr_value_trend': (signal_row['oLRValue'] >= signal_row['oLRValue2']) if (pd.notna(signal_row['oLRValue']) and pd.notna(signal_row['oLRValue2'])) else None,
                    'low_near_bb_lower': (signal_row['Low'] <= signal_row['LowerBB'] * 1.02) if pd.notna(signal_row['LowerBB']) else None,
                    'close_below_bb_upper': (signal_row['Close'] <= signal_row['UpperBB'] * 0.95) if pd.notna(signal_row['UpperBB']) else None,
                }
                indicator_snapshot.update(pattern_checks)
                
            elif 'Engf S' in signal_type:  # Engulfing Short pattern  
                pattern_checks = {
                    'open_above_prev_close': signal_row['Open'] > prev_row['Close'],
                    'close_below_prev_open': signal_row['Close'] < prev_row['Open'],
                    'current_red': signal_row['Close'] < signal_row['Open'],
                    'prev_green': prev_row['Close'] > prev_row['Open'],
                    'prev_close_near_high': (prev_row['Close'] - prev_row['Low']) <= (signal_row['ATR'] * 2) if pd.notna(signal_row['ATR']) else None,
                    'prev_body_size': prev_row['Close'] - prev_row['Open'] > 0.05,
                    'lr_value_trend': (signal_row['oLRValue'] <= signal_row['oLRValue2']) if (pd.notna(signal_row['oLRValue']) and pd.notna(signal_row['oLRValue2'])) else None,
                    'high_near_bb_upper': (signal_row['High'] >= signal_row['UpperBB'] * 0.98) if pd.notna(signal_row['UpperBB']) else None,
                    'close_above_bb_lower': (signal_row['Close'] >= signal_row['LowerBB'] * 1.05) if pd.notna(signal_row['LowerBB']) else None,
                }
                indicator_snapshot.update(pattern_checks)
            
            all_signal_details.append(indicator_snapshot)
    
    # Sort by date (most recent first)
    all_signal_details.sort(key=lambda x: x['date'], reverse=True)
    
    print(f"\n=== FOUND {len(all_signal_details)} TOTAL SIGNALS ===")
    
    # Show last 10 signals in detail
    print(f"\n=== LAST 10 SIGNALS (Most Recent First) ===")
    for i, signal in enumerate(all_signal_details[:10]):
        print(f"\n--- Signal #{i+1}: {signal['symbol']} {signal['date']} ---")
        print(f"Type: {signal['signal_type']} ({signal['direction']})")
        print(f"Price: ${signal['close']} (Open: ${signal['open']}, Prev Close: ${signal['prev_close']})")
        print(f"Shares: {signal['shares']}")
        
        # Show Bollinger Band position
        if signal['bb_upper'] and signal['bb_lower']:
            bb_position = "middle"
            if signal['close'] <= signal['bb_lower'] * 1.02:
                bb_position = "near lower"
            elif signal['close'] >= signal['bb_upper'] * 0.98:
                bb_position = "near upper"
            print(f"BB Position: {bb_position} (Upper: ${signal['bb_upper']}, Lower: ${signal['bb_lower']})")
        
        # Show LR trend
        if signal['lr_value'] and signal['lr_value2']:
            lr_trend = "up" if signal['lr_value'] >= signal['lr_value2'] else "down"
            print(f"LR Trend: {lr_trend} (Value: {signal['lr_value']:.4f}, Value2: {signal['lr_value2']:.4f})")
        
        # Show pattern-specific checks
        pattern_keys = [k for k in signal.keys() if k not in ['symbol', 'date', 'signal_type', 'direction', 'shares', 
                       'open', 'high', 'low', 'close', 'prev_open', 'prev_high', 'prev_low', 'prev_close',
                       'bb_upper', 'bb_lower', 'bb_avg', 'atr', 'atr_ma', 'lr_value', 'lr_value2', 
                       'lr_slope', 'lr_angle', 'swing_high', 'swing_low', 'psar_long', 'psar_short', 'psar_is_long']]
        
        if pattern_keys:
            print("Pattern Checks:")
            for key in pattern_keys:
                if signal[key] is not None:
                    status = "✓" if signal[key] else "✗"
                    print(f"  {status} {key}: {signal[key]}")
    
    # Save detailed results to file
    with open('signal_analysis.json', 'w') as f:
        json.dump(all_signal_details, f, indent=2, default=str)
    
    print(f"\n✓ Saved detailed analysis to signal_analysis.json")
    print(f"✓ Total signals analyzed: {len(all_signal_details)}")
    
    # Summary statistics
    if all_signal_details:
        long_signals = len([s for s in all_signal_details if s['direction'] == 'Long'])
        short_signals = len([s for s in all_signal_details if s['direction'] == 'Short'])
        
        print(f"\nSignal Summary:")
        print(f"  Long signals: {long_signals}")
        print(f"  Short signals: {short_signals}")
        print(f"  Date range: {all_signal_details[-1]['date']} to {all_signal_details[0]['date']}")
        
        # Most common signal types
        signal_types = {}
        for signal in all_signal_details:
            sig_type = signal['signal_type']
            signal_types[sig_type] = signal_types.get(sig_type, 0) + 1
        
        print(f"  Most common patterns:")
        for sig_type, count in sorted(signal_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {sig_type}: {count}")

if __name__ == "__main__":
    debug_signal_indicators()
