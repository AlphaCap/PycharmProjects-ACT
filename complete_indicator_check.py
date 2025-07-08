import pandas as pd
import os
from data_manager import load_price_data, get_sp500_symbols

def check_complete_indicators():
    """
    Check for ALL indicators that nGS_Strategy calculates.
    """
    print("=== COMPLETE INDICATOR CHECK ===\n")
    print("Checking for ALL 31+ indicators from nGS_Strategy...\n")
    
    # ALL indicators from nGS_Strategy.py
    all_indicators = {
        'Bollinger Bands': ['BBAvg', 'BBSDev', 'UpperBB', 'LowerBB'],
        'True Range': ['High_Low', 'High_Close', 'Low_Close', 'TR', 'ATR', 'ATRma'],
        'Parabolic SAR': ['LongPSAR', 'ShortPSAR', 'PSAR_EP', 'PSAR_AF', 'PSAR_IsLong'],
        'Linear Regression 1': ['oLRSlope', 'oLRAngle', 'oLRIntercept', 'TSF'],
        'Linear Regression 2': ['oLRSlope2', 'oLRAngle2', 'oLRIntercept2', 'TSF5'],
        'Additional': ['Value1', 'ROC', 'LRV', 'LinReg', 'oLRValue', 'oLRValue2'],
        'Swing Points': ['SwingLow', 'SwingHigh']
    }
    
    # Flatten all indicators
    all_indicator_list = []
    for group in all_indicators.values():
        all_indicator_list.extend(group)
    
    print(f"Total indicators to check: {len(all_indicator_list)}\n")
    
    # Get symbols to test
    symbols = get_sp500_symbols()
    test_symbols = symbols[:5]  # Test first 5
    
    results = {
        'complete': [],
        'partial': [],
        'missing': [],
        'no_data': []
    }
    
    for symbol in test_symbols:
        print(f"Checking {symbol}...")
        
        try:
            df = load_price_data(symbol)
            
            if df.empty:
                print(f"  ❌ No data found")
                results['no_data'].append(symbol)
                continue
            
            # Check each indicator group
            group_results = {}
            total_present = 0
            
            for group_name, indicators in all_indicators.items():
                present_in_group = 0
                for indicator in indicators:
                    if indicator in df.columns and not df[indicator].isna().all():
                        present_in_group += 1
                        total_present += 1
                
                group_results[group_name] = f"{present_in_group}/{len(indicators)}"
                
            print(f"  📊 Total indicators: {total_present}/{len(all_indicator_list)}")
            
            # Show breakdown by group
            for group_name, result in group_results.items():
                status = "✅" if result.split('/')[0] == result.split('/')[1] else "⚠️"
                print(f"     {status} {group_name}: {result}")
            
            # Categorize overall result
            if total_present >= len(all_indicator_list) * 0.9:  # 90%+ complete
                results['complete'].append(symbol)
                print(f"  ✅ COMPLETE: {symbol} has most indicators")
            elif total_present >= len(all_indicator_list) * 0.5:  # 50%+ partial
                results['partial'].append(symbol)
                print(f"  ⚠️  PARTIAL: {symbol} has some indicators")
            else:
                results['missing'].append(symbol)
                print(f"  ❌ MISSING: {symbol} lacks most indicators")
            
            # Show sample of latest data with key indicators
            if total_present > 0:
                latest = df.tail(1).iloc[0]
                print(f"     Latest date: {latest['Date']}")
                print(f"     Close: ${latest['Close']:.2f}")
                
                # Show key indicators if present
                key_indicators = ['BBAvg', 'ATR', 'LRV', 'TSF', 'oLRValue']
                for indicator in key_indicators:
                    if indicator in df.columns and not pd.isna(latest[indicator]):
                        print(f"     {indicator}: {latest[indicator]:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['no_data'].append(symbol)
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"✅ Complete indicators (90%+): {len(results['complete'])} symbols")
    print(f"⚠️  Partial indicators (50-90%): {len(results['partial'])} symbols") 
    print(f"❌ Missing indicators (<50%): {len(results['missing'])} symbols")
    print(f"🚫 No data: {len(results['no_data'])} symbols")
    
    if results['complete']:
        print(f"\nSymbols with complete indicators: {results['complete']}")
    
    if results['partial']:
        print(f"\nSymbols needing indicator completion: {results['partial']}")
    
    if results['missing']:
        print(f"\nSymbols missing most indicators: {results['missing']}")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    if len(results['complete']) == len(test_symbols):
        print("🎉 All test symbols have complete indicators!")
        print("Your nGS strategy should work perfectly with this data.")
    else:
        print("🔧 Run the updated apply_to_all_symbols.py to add ALL indicators")
        print("   This will ensure your 6-month rolling database has all 31+ indicators")
    
    return results

def show_detailed_breakdown(symbol):
    """Show detailed breakdown of all indicators for one symbol."""
    print(f"=== DETAILED BREAKDOWN: {symbol} ===\n")
    
    try:
        df = load_price_data(symbol)
        if df.empty:
            print(f"No data found for {symbol}")
            return
        
        print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total rows: {len(df)}\n")
        
        # Group indicators and check each one
        all_indicators = {
            'Basic OHLCV': ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
            'Bollinger Bands': ['BBAvg', 'BBSDev', 'UpperBB', 'LowerBB'],
            'True Range': ['High_Low', 'High_Close', 'Low_Close', 'TR', 'ATR', 'ATRma'],
            'Parabolic SAR': ['LongPSAR', 'ShortPSAR', 'PSAR_EP', 'PSAR_AF', 'PSAR_IsLong'],
            'Linear Regression 3-period': ['oLRSlope', 'oLRAngle', 'oLRIntercept', 'TSF'],
            'Linear Regression 5-period': ['oLRSlope2', 'oLRAngle2', 'oLRIntercept2', 'TSF5'],
            'Rate of Change': ['Value1', 'ROC', 'LRV', 'LinReg', 'oLRValue', 'oLRValue2'],
            'Swing Points': ['SwingLow', 'SwingHigh']
        }
        
        for group_name, indicators in all_indicators.items():
            print(f"{group_name}:")
            for indicator in indicators:
                if indicator in df.columns:
                    non_null = df[indicator].notna().sum()
                    if non_null > 0:
                        latest_val = df[indicator].iloc[-1]
                        if pd.notna(latest_val):
                            print(f"  ✅ {indicator}: {non_null}/{len(df)} values, latest: {latest_val}")
                        else:
                            print(f"  ⚠️  {indicator}: {non_null}/{len(df)} values, latest: NaN")
                    else:
                        print(f"  ❌ {indicator}: All NaN")
                else:
                    print(f"  ❌ {indicator}: Missing column")
            print()
        
    except Exception as e:
        print(f"Error checking {symbol}: {e}")

if __name__ == "__main__":
    # Run complete check
    results = check_complete_indicators()
    
    # Show detailed breakdown for AAPL
    print("\n" + "="*60)
    show_detailed_breakdown("AAPL")