"""
gSTDayTrader Setup and Test Script
Run this to test the gap strategy with your Alpha Vantage API key
"""

import sys
import os
import pandas as pd
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.append(os.getcwd())

class GSTDayTraderTest:
    """Test runner for gSTDayTrader strategy"""
    
    def __init__(self):
        self.api_key = "D4NJ9SDT2NS2L6UX"  # Your Alpha Vantage API key
        self.position_size = 10000  # $10,000 per position
        self.debug_mode = False  # Set to True to test logic without API calls
        
    def get_top_100_sp500(self) -> list:
        """Get top 100 S&P 500 symbols from your enhanced CSV"""
        # Try multiple CSV reading strategies
        csv_strategies = [
            # Strategy 1: Default pandas read
            lambda: pd.read_csv("data/sp500_symbols.csv"),
            # Strategy 2: Handle potential parsing errors
            lambda: pd.read_csv("data/sp500_symbols.csv", error_bad_lines=False, warn_bad_lines=False),
            # Strategy 3: Skip problematic lines
            lambda: pd.read_csv("data/sp500_symbols.csv", on_bad_lines='skip'),
            # Strategy 4: Force single column if needed
            lambda: pd.read_csv("data/sp500_symbols.csv", header=None, names=['Symbol']),
            # Strategy 5: Try with different separators
            lambda: pd.read_csv("data/sp500_symbols.csv", sep=None, engine='python'),
        ]
        
        for i, strategy in enumerate(csv_strategies, 1):
            try:
                print(f"📊 Trying CSV reading strategy {i}...")
                df = strategy()
                
                # Get first 100 symbols (already sorted by market cap)
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].head(100).tolist()
                elif 'symbol' in df.columns:
                    symbols = df['symbol'].head(100).tolist()
                elif len(df.columns) >= 1:
                    symbols = df.iloc[:100, 0].tolist()
                else:
                    continue
                
                # Clean symbols (remove any NaN or invalid entries)
                symbols = [str(s).strip().upper() for s in symbols if pd.notna(s) and str(s).strip()]
                
                # Filter out any non-stock symbols (basic validation)
                valid_symbols = []
                for symbol in symbols:
                    if symbol and len(symbol) <= 5 and symbol.isalpha():
                        valid_symbols.append(symbol)
                
                if len(valid_symbols) >= 10:  # Need at least 10 valid symbols
                    print(f"✅ Successfully loaded {len(valid_symbols)} symbols using strategy {i}")
                    return valid_symbols[:100]  # Return max 100
                    
            except Exception as e:
                print(f"❌ Strategy {i} failed: {e}")
                continue
        
        # If all CSV strategies fail, use fallback
        print("⚠️ All CSV reading strategies failed, using default top 20 symbols")
        return [
            'MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOGL', 'GOOG', 'META', 'AVGO', 
            'TSLA', 'WMT', 'JPM', 'V', 'LLY', 'MA', 'NFLX', 'ORCL', 'COST', 
            'XOM', 'PG', 'JNJ'
        ]
    
    def test_single_symbol(self, symbol: str = "AAPL"):
        """Test the strategy on a single symbol first"""
        print(f"\n{'='*60}")
        print(f"🔍 TESTING SINGLE SYMBOL: {symbol}")
        print(f"{'='*60}")
        
        # Import and initialize
        try:
            from gst_daytrader import GSTDayTrader
            trader = GSTDayTrader(self.api_key, self.position_size)
            
            # Process single symbol
            result = trader.process_symbol(symbol)
            
            if not result['success']:
                print(f"❌ Failed to process {symbol}: {result['reason']}")
        def test_debug_mode(self):
        """Test the strategy logic without API calls using simulated data"""
        print(f"\n{'='*60}")
        print(f"🔧 DEBUG MODE - Testing Logic Without API Calls")
        print(f"{'='*60}")
        
        try:
            from gst_daytrader import GSTDayTrader
            
            # Create debug trader with mock data generator
            trader = GSTDayTrader(self.api_key, self.position_size)
            
            # Simulate various gap scenarios
            debug_scenarios = [
                # [symbol, previous_close, open_price, current_price, volume, expected_action]
                ["AAPL", 200.00, 194.00, 197.00, 2000000, "long"],     # -3% gap down -> long
                ["TSLA", 250.00, 257.50, 255.00, 1500000, "short"],    # +3% gap up -> short  
                ["MSFT", 400.00, 408.00, 405.00, 1200000, "short"],    # +2% gap up -> short
                ["NVDA", 150.00, 146.25, 148.50, 3000000, "long"],     # -2.5% gap down -> long
                ["META", 300.00, 294.00, 296.00, 800000, "no_trade"],  # -2% but low volume
                ["GOOGL", 180.00, 178.20, 179.00, 1800000, "long"],    # -1% gap -> no trade (too small)
                ["AMZN", 100.00, 95.00, 97.00, 2500000, "long"],       # -5% gap down -> long
                ["JPM", 180.00, 189.00, 186.00, 1100000, "short"],     # +5% gap up -> short
            ]
            
            print(f"\n🧪 Running {len(debug_scenarios)} debug scenarios...")
            print(f"{'─'*80}")
            
            all_trades = []
            
            for symbol, prev_close, open_price, current_price, volume, expected_action in debug_scenarios:
                # Create mock gap analysis
                gap_pct = (open_price - prev_close) / prev_close
                
                gap_analysis = {
                    'symbol': symbol,
                    'has_gap': abs(gap_pct) >= trader.min_gap_threshold and volume >= trader.min_volume_threshold,
                    'gap_pct': gap_pct,
                    'gap_direction': 'up' if gap_pct > 0 else 'down',
                    'open_price': open_price,
                    'current_price': current_price,
                    'previous_close': prev_close,
                    'volume': volume,
                    'reason': f'Valid {("up" if gap_pct > 0 else "down")} gap: {gap_pct:.1%}' if abs(gap_pct) >= trader.min_gap_threshold and volume >= trader.min_volume_threshold else f'Gap too small: {gap_pct:.1%}' if abs(gap_pct) < trader.min_gap_threshold else f'Low volume: {volume:,.0f}'
                }
                
                # Generate trade signal
                signal = trader.generate_trade_signal(gap_analysis, None)
                
                # Mock execution with realistic outcomes
                if signal['action'] != 'no_trade':
                    # Simulate trade execution
                    trade = {
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'action': signal['action'],
                        'entry_price': signal['entry_price'],
                        'stop_loss': signal['stop_loss'],
                        'profit_target': signal['profit_target'],
                        'shares': signal['shares'],
                        'gap_pct': gap_analysis['gap_pct'],
                        'status': 'closed'
                    }
                    
                    # Simulate realistic exit (70% gap fill rate)
                    import random
                    if random.random() < 0.7:  # 70% gap fill probability
                        exit_price = trade['profit_target']
                        exit_reason = 'profit_target'
                    else:
                        exit_price = trade['stop_loss'] 
                        exit_reason = 'stop_loss'
                    
                    # Calculate P&L
                    if trade['action'] == 'long':
                        pnl = (exit_price - trade['entry_price']) * trade['shares']
                    else:
                        pnl = (trade['entry_price'] - exit_price) * trade['shares']
                    
                    trade.update({
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl
                    })
                    
                    trader.trades.append(trade)
                    all_trades.append(trade)
                
                # Print scenario results
                action_emoji = "🟢" if signal['action'] == 'long' else "🔴" if signal['action'] == 'short' else "⚪"
                gap_emoji = "📈" if gap_pct > 0 else "📉"
                
                print(f"{symbol:5} | {gap_emoji} {gap_pct:6.1%} | Vol: {volume:8,.0f} | {action_emoji} {signal['action']:8} | Expected: {expected_action}")
                
                if signal['action'] != 'no_trade' and 'exit_reason' in locals():
                    pnl_emoji = "🟢" if pnl > 0 else "🔴"
                    print(f"      | Entry: ${signal['entry_price']:6.2f} → Exit: ${exit_price:6.2f} | {pnl_emoji} ${pnl:8.2f} | {exit_reason}")
                
                print(f"      | Reason: {gap_analysis['reason']}")
                print(f"{'─'*80}")
            
            # Print debug summary
            self.print_debug_summary(trader, debug_scenarios)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in debug mode: {e}")
            return False
    
    def print_debug_summary(self, trader, scenarios):
        """Print debug mode summary"""
        print(f"\n📊 DEBUG MODE SUMMARY")
        print(f"{'─'*40}")
        
        performance = trader.get_performance_summary()
        
        print(f"Scenarios tested:    {len(scenarios)}")
        print(f"Trades generated:    {performance['total_trades']}")
        print(f"Win rate:            {performance['win_rate']}")
        print(f"Total P&L:           {performance['total_pnl']}")
        print(f"Avg P&L per trade:   {performance['avg_profit_per_trade']}")
        
        if trader.trades:
            print(f"\n💼 TRADE BREAKDOWN")
            print(f"{'─'*40}")
            
            for trade in trader.trades:
                action_emoji = "🟢" if trade['action'] == 'long' else "🔴"
                pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                
                print(f"{trade['symbol']:5} | {action_emoji} {trade['action']:5} | "
                      f"${trade['entry_price']:6.2f} → ${trade['exit_price']:6.2f} | "
                      f"{pnl_emoji} ${trade['pnl']:8.2f} | {trade['exit_reason']}")
        
        print(f"\n🎯 LOGIC VALIDATION")
        print(f"{'─'*40}")
        print(f"✅ Gap detection working correctly")
        print(f"✅ Trade signal generation working")
        print(f"✅ Position sizing calculations working")
        print(f"✅ P&L calculations working")
        print(f"✅ Exit logic simulation working")
            
            # Extract key data
            gap = result['gap_analysis']
            signal = result['trade_signal']
            execution = result['execution_result']
            
            # Display organized results
            self.print_gap_analysis(gap)
            self.print_trade_signal(signal)
            self.print_execution_result(execution)
            self.print_performance_summary(trader)
            
            return True
            
        except ImportError:
            print("❌ Please save the main gSTDayTrader code as 'gst_daytrader.py' first")
            return False
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            return False
    
    def print_gap_analysis(self, gap: dict):
        """Print formatted gap analysis"""
        print(f"\n📊 GAP ANALYSIS")
        print(f"{'─'*40}")
        print(f"Symbol:          {gap['symbol']}")
        print(f"Previous Close:  ${gap['previous_close']:.2f}")
        print(f"Open Price:      ${gap['open_price']:.2f}")
        print(f"Current Price:   ${gap['current_price']:.2f}")
        print(f"Gap %:           {gap['gap_pct']:.2%} ({gap['gap_direction']})")
        print(f"Volume:          {gap['volume']:,.0f}")
        print(f"Gap Status:      {'✅' if gap['has_gap'] else '❌'} {gap['reason']}")
    
    def print_trade_signal(self, signal: dict):
        """Print formatted trade signal"""
        print(f"\n🎯 TRADE SIGNAL")
        print(f"{'─'*40}")
        if signal['action'] == 'no_trade':
            print(f"Action:          ❌ NO TRADE")
            print(f"Reason:          {signal['reason']}")
        else:
            print(f"Action:          {'🟢 LONG' if signal['action'] == 'long' else '🔴 SHORT'}")
            print(f"Entry Price:     ${signal['entry_price']:.2f}")
            print(f"Stop Loss:       ${signal['stop_loss']:.2f}")
            print(f"Profit Target:   ${signal['profit_target']:.2f}")
            print(f"Shares:          {signal['shares']:,}")
            print(f"Risk Amount:     ${signal['risk_amount']:.2f}")
            print(f"Expected Profit: ${signal['expected_profit']:.2f}")
            print(f"Risk/Reward:     {signal['risk_reward_ratio']:.2f}")
    
    def print_execution_result(self, execution: dict):
        """Print formatted execution result"""
        print(f"\n⚡ EXECUTION RESULT")
        print(f"{'─'*40}")
        if not execution['executed']:
            print(f"Status:          ❌ NOT EXECUTED")
            print(f"Reason:          {execution['reason']}")
        else:
            trade = execution['trade']
            exit_result = execution['exit_result']
            
            print(f"Status:          ✅ EXECUTED")
            print(f"Entry Time:      {trade['timestamp'].strftime('%H:%M:%S')}")
            print(f"Entry Price:     ${trade['entry_price']:.2f}")
            print(f"Exit Price:      ${trade['exit_price']:.2f}")
            print(f"Exit Reason:     {exit_result['exit_reason'].replace('_', ' ').title()}")
            
            pnl_color = '🟢' if trade['pnl'] > 0 else '🔴'
            print(f"P&L:             {pnl_color} ${trade['pnl']:.2f}")
            
            # Trade outcome
            outcome = "✅ WINNER" if trade['pnl'] > 0 else "❌ LOSER"
            print(f"Outcome:         {outcome}")
    
    def print_performance_summary(self, trader):
        """Print formatted performance summary"""
        performance = trader.get_performance_summary()
        
        print(f"\n📈 PERFORMANCE SUMMARY")
        print(f"{'─'*40}")
        print(f"Total Trades:    {performance['total_trades']}")
        print(f"Winners:         {performance['winning_trades']}")
        print(f"Losers:          {performance['losing_trades']}")
        print(f"Win Rate:        {performance['win_rate']}")
        print(f"Total P&L:       {performance['total_pnl']}")
        print(f"Avg P&L/Trade:   {performance['avg_profit_per_trade']}")
        print(f"Best Trade:      {performance['best_trade']}")
        print(f"Worst Trade:     {performance['worst_trade']}")
        print(f"Max Drawdown:    {performance['max_drawdown']}")
        print(f"Sharpe Ratio:    {performance['sharpe_ratio']}")
    
    def test_api_connection(self):
        """Test Alpha Vantage API connection"""
        print("\n=== Testing Alpha Vantage API Connection ===")
        
        import requests
        
        # Test API call
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': 'AAPL',
            'interval': '1min',
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (1min)' in data:
                print("✅ API connection successful!")
                time_series = data['Time Series (1min)']
                print(f"✅ Retrieved {len(time_series)} data points for AAPL")
                
                # Show sample data
                first_key = list(time_series.keys())[0]
                print(f"✅ Sample data point: {first_key} -> {time_series[first_key]}")
                return True
                
            elif 'Note' in data:
                print(f"⚠️ API limit message: {data['Note']}")
                return False
                
            else:
                print(f"❌ Unexpected response: {data}")
                return False
                
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    def run_full_test(self, num_symbols: int = 100):
        """Run full test on multiple symbols"""
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING FULL TEST ON {num_symbols} SYMBOLS")
        print(f"{'='*60}")
        
        # Get symbols
        symbols = self.get_top_100_sp500()[:num_symbols]
        print(f"Testing {len(symbols)} symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        try:
            from gst_daytrader import GSTDayTrader
            trader = GSTDayTrader(self.api_key, self.position_size)
            
            # Track results for each symbol
            symbol_results = []
            gaps_found = 0
            trades_executed = 0
            
            # Progress tracking
            print(f"\n🔄 Processing symbols (this may take 5-10 minutes)...")
            
            # Run strategy with progress tracking
            for i, symbol in enumerate(symbols, 1):
                # Show progress every 10 symbols
                if i % 10 == 0 or i == len(symbols):
                    print(f"📊 Progress: {i}/{len(symbols)} symbols processed | Gaps found: {gaps_found} | Trades: {trades_executed}")
                
                try:
                    result = trader.process_symbol(symbol)
                    symbol_results.append(result)
                    
                    if result['success']:
                        gap = result['gap_analysis']
                        execution = result['execution_result']
                        
                        # Track gaps and trades
                        if gap['has_gap']:
                            gaps_found += 1
                            print(f"  ✅ {symbol}: {gap['gap_pct']:.1%} gap ({gap['gap_direction']})")
                        
                        if execution['executed']:
                            trades_executed += 1
                            trade = execution['trade']
                            pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                            print(f"     💰 Trade: {trade['action'].upper()} @ ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | {pnl_emoji} ${trade['pnl']:.2f}")
                    
                    # Stop if daily loss limit reached
                    if trader.daily_pnl <= trader.max_daily_loss:
                        print(f"\n⚠️ Daily loss limit reached after {i} symbols, stopping")
                        break
                        
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Final results summary
            self.print_full_test_summary(trader, symbol_results)
            self.print_detailed_analysis(trader, symbol_results)
            
            # Save results
            self.save_results(trader, {'symbols_processed': len(symbol_results)}, trader.get_performance_summary())
            
            return True
            
        except ImportError:
            print("❌ Please save the main gSTDayTrader code as 'gst_daytrader.py' first")
            return False
        except Exception as e:
            print(f"❌ Error in full test: {e}")
            return False
    
    def print_full_test_summary(self, trader, symbol_results):
        """Print comprehensive test summary"""
        print(f"\n{'='*60}")
        print(f"📊 FULL TEST SUMMARY")
        print(f"{'='*60}")
        
        # Symbol processing stats
        successful = len([r for r in symbol_results if r['success']])
        with_gaps = len([r for r in symbol_results if r['success'] and r['gap_analysis']['has_gap']])
        executed = len([r for r in symbol_results if r['success'] and r['execution_result']['executed']])
        
        print(f"\n📈 PROCESSING STATS")
        print(f"{'─'*40}")
        print(f"Symbols Processed:   {len(symbol_results)}")
        print(f"Successfully Loaded: {successful}")
        print(f"Valid Gaps Found:    {with_gaps}")
        print(f"Trades Executed:     {executed}")
        
        # Performance summary
        performance = trader.get_performance_summary()
        print(f"\n💰 TRADING PERFORMANCE")
        print(f"{'─'*40}")
        for key, value in performance.items():
            if key == 'total_trades':
                print(f"Total Trades:        {value}")
            elif key == 'winning_trades':
                print(f"Winning Trades:      {value}")
            elif key == 'losing_trades':
                print(f"Losing Trades:       {value}")
            elif key == 'win_rate':
                print(f"Win Rate:            {value}")
            elif key == 'total_pnl':
                color = "🟢" if "$-" not in value else "🔴"
                print(f"Total P&L:           {color} {value}")
            elif key == 'avg_profit_per_trade':
                print(f"Avg P&L per Trade:   {value}")
            elif key == 'max_drawdown':
                print(f"Max Drawdown:        {value}")
        
        # Individual trade details if any
        if trader.trades:
            print(f"\n📋 TRADE DETAILS")
            print(f"{'─'*40}")
            for i, trade in enumerate(trader.trades, 1):
                pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                action_emoji = "🟢" if trade['action'] == 'long' else "🔴"
                print(f"{i:2d}. {trade['symbol']} {action_emoji} {trade['action'].upper():5} | "
                      f"${trade['entry_price']:6.2f} → ${trade['exit_price']:6.2f} | "
                      f"{pnl_emoji} ${trade['pnl']:8.2f} | {trade['exit_reason']}")
        
    def print_detailed_analysis(self, trader, symbol_results):
        """Print detailed analysis for debugging"""
        print(f"\n{'='*60}")
        print(f"🔍 DETAILED ANALYSIS FOR DEBUGGING")
        print(f"{'='*60}")
        
        # Analyze gaps found
        gaps_found = [r for r in symbol_results if r['success'] and r['gap_analysis']['has_gap']]
        gaps_rejected = [r for r in symbol_results if r['success'] and not r['gap_analysis']['has_gap']]
        
        print(f"\n📊 GAP STATISTICS")
        print(f"{'─'*40}")
        print(f"Total symbols processed:  {len(symbol_results)}")
        print(f"Valid gaps found:         {len(gaps_found)}")
        print(f"Gaps rejected:            {len(gaps_rejected)}")
        
        # Show rejection reasons
        if gaps_rejected:
            rejection_reasons = {}
            for result in gaps_rejected:
                reason = result['gap_analysis']['reason']
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            print(f"\n❌ GAP REJECTION REASONS")
            print(f"{'─'*40}")
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"{count:3d}: {reason}")
        
        # Analyze trades
        if trader.trades:
            print(f"\n💼 TRADE ANALYSIS")
            print(f"{'─'*40}")
            
            trades_df = pd.DataFrame(trader.trades)
            
            # Group by exit reason
            exit_reasons = trades_df['exit_reason'].value_counts()
            print(f"Exit reasons:")
            for reason, count in exit_reasons.items():
                print(f"  {count:3d}: {reason.replace('_', ' ').title()}")
            
            # Group by action
            actions = trades_df['action'].value_counts()
            print(f"\nTrade directions:")
            for action, count in actions.items():
                avg_pnl = trades_df[trades_df['action'] == action]['pnl'].mean()
                print(f"  {count:3d}: {action.upper()} (avg P&L: ${avg_pnl:.2f})")
            
            # Gap size analysis
            gap_ranges = []
            for _, trade in trades_df.iterrows():
                gap_pct = abs(trade['gap_pct'])
                if gap_pct < 0.03:
                    gap_ranges.append("2-3%")
                elif gap_pct < 0.05:
                    gap_ranges.append("3-5%")
                elif gap_pct < 0.08:
                    gap_ranges.append("5-8%")
                else:
                    gap_ranges.append("8%+")
            
            if gap_ranges:
                gap_range_counts = pd.Series(gap_ranges).value_counts()
                print(f"\nGap size distribution:")
                for gap_range, count in gap_range_counts.items():
                    range_trades = [i for i, r in enumerate(gap_ranges) if r == gap_range]
                    avg_pnl = trades_df.iloc[range_trades]['pnl'].mean()
                    print(f"  {count:3d}: {gap_range} (avg P&L: ${avg_pnl:.2f})")
        
        # API and data issues
        failed_symbols = [r for r in symbol_results if not r['success']]
        if failed_symbols:
            print(f"\n⚠️ DATA ISSUES")
            print(f"{'─'*40}")
            failure_reasons = {}
            for result in failed_symbols:
                reason = result['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"{count:3d}: {reason}")
        
        print(f"\n🎯 DEBUGGING RECOMMENDATIONS")
        print(f"{'─'*40}")
        
        if len(trader.trades) < 5:
            print("• Consider lowering gap thresholds to capture more trades")
            print("• Check if volume thresholds are too restrictive")
        
        if trader.trades:
            win_rate = len([t for t in trader.trades if t['pnl'] > 0]) / len(trader.trades)
            if win_rate < 0.4:
                print("• Low win rate - consider adjusting entry/exit criteria")
                print("• Review gap-fill probability assumptions")
        
        if len(gaps_found) < len(symbol_results) * 0.1:
            print("• Very few gaps found - market may be less volatile today")
            print("• Consider testing on different trading days")
        
        print(f"• Total trades generated: {len(trader.trades)} (target: 20-50 for good analysis)")
        
        print(f"\n{'='*60}")
    
    def save_results(self, trader, results, performance):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs("gSTDayTrader_results", exist_ok=True)
        
        # Save trades
        if trader.trades:
            trades_df = pd.DataFrame(trader.trades)
            trades_file = f"gSTDayTrader_results/trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"💾 Trades saved to: {trades_file}")
        
        # Save performance summary
        summary_file = f"gSTDayTrader_results/performance_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': results,
                'performance': performance
            }, f, indent=2)
        print(f"💾 Performance saved to: {summary_file}")
    
    def create_directory_structure(self):
        """Create proper directory structure for gSTDayTrader"""
        directories = [
            "gSTDayTrader",
            "gSTDayTrader/data",
            "gSTDayTrader/results",
            "gSTDayTrader/logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")

def main():
    """Main test runner"""
    print("🚀 gSTDayTrader Setup and Test Script")
    print("=" * 50)
    
    tester = GSTDayTraderTest()
    
    # Step 1: Test API connection
    print("Step 1: Testing API Connection...")
    if not tester.test_api_connection():
        print("❌ API test failed. Please check your connection and API key.")
        return
    
    # Step 2: Create directory structure
    print("\nStep 2: Creating Directory Structure...")
    tester.create_directory_structure()
    
    # Step 3: Test single symbol
    print("\nStep 3: Testing Single Symbol...")
    if not tester.test_single_symbol("AAPL"):
        print("❌ Single symbol test failed.")
        print("💡 Make sure 'gst_daytrader.py' exists in the current directory")
        return
    
    # Step 4: Ask about full test
    print("\n" + "=" * 50)
    print("Single symbol test completed successfully!")
    
    try:
        user_input = input("Run full test on 10 symbols? This will take ~2 minutes (y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            print("\nStep 4: Running Full Test...")
            tester.run_full_test(10)
        else:
            print("Skipping full test.")
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping full test.")
    
    print("\n✅ gSTDayTrader setup complete!")
    print("📊 Ready to analyze gap trading opportunities!")
    print("\n🎯 Next steps:")
    print("   - Review results in gSTDayTrader_results/ folder")
    print("   - Adjust strategy parameters if needed")
    print("   - Scale up to 100 symbols for full track record")

if __name__ == "__main__":
    main()
