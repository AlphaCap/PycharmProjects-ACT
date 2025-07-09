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
        
    def get_top_100_sp500(self) -> list:
        """Get top 100 S&P 500 symbols from your enhanced CSV"""
        try:
            # Load your enhanced S&P 500 symbols
            df = pd.read_csv("data/sp500_symbols.csv")
            
            # Get first 100 symbols (already sorted by market cap)
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].head(100).tolist()
            elif 'symbol' in df.columns:
                symbols = df['symbol'].head(100).tolist()
            else:
                symbols = df.iloc[:100, 0].tolist()
            
            print(f"✅ Loaded {len(symbols)} symbols from sp500_symbols.csv")
            return symbols
            
        except FileNotFoundError:
            print("⚠️ sp500_symbols.csv not found, using default top 20")
            # Fallback to top 20 liquid stocks
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
                return False
            
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
    
    def run_full_test(self, num_symbols: int = 10):
        """Run full test on multiple symbols"""
        print(f"\n{'='*60}")
        print(f"🚀 RUNNING FULL TEST ON {num_symbols} SYMBOLS")
        print(f"{'='*60}")
        
        # Get symbols
        symbols = self.get_top_100_sp500()[:num_symbols]
        print(f"Testing symbols: {', '.join(symbols)}")
        
        try:
            from gst_daytrader import GSTDayTrader
            trader = GSTDayTrader(self.api_key, self.position_size)
            
            # Track results for each symbol
            symbol_results = []
            
            # Run strategy with progress tracking
            for i, symbol in enumerate(symbols, 1):
                print(f"\n{'─'*60}")
                print(f"📊 Processing {symbol} ({i}/{num_symbols})")
                print(f"{'─'*60}")
                
                result = trader.process_symbol(symbol)
                symbol_results.append(result)
                
                if result['success']:
                    gap = result['gap_analysis']
                    execution = result['execution_result']
                    
                    # Quick summary for each symbol
                    gap_status = "✅" if gap['has_gap'] else "❌"
                    trade_status = "✅ EXECUTED" if execution['executed'] else "❌ NO TRADE"
                    
                    print(f"Gap: {gap_status} {gap['gap_pct']:.1%} | Trade: {trade_status}")
                    
                    if execution['executed']:
                        trade = execution['trade']
                        pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
                        print(f"P&L: {pnl_emoji} ${trade['pnl']:.2f}")
                else:
                    print(f"❌ Failed: {result['reason']}")
            
            # Final results summary
            self.print_full_test_summary(trader, symbol_results)
            
            # Save results
            self.save_results(trader, {'symbols_processed': len(symbols)}, trader.get_performance_summary())
            
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
        
        print(f"\n{'='*60}")
        print(f"✅ Test completed successfully!")
        print(f"{'='*60}")
    
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
