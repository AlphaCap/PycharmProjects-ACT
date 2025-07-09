"""
Updated gSTDayTrader Test Script with Enhanced Fetching
Tests the gap strategy with improved data fetching for 100+ symbols
"""

import sys
import os
import pandas as pd
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.append(os.getcwd())

class GSTDayTraderTest:
    """Enhanced test runner for gSTDayTrader strategy"""
    
    def __init__(self):
        self.api_key = "D4NJ9SDT2NS2L6UX"  # Your Alpha Vantage API key
        self.max_risk_per_trade = 50  # Reduced to $50 for safety
        
    def test_enhanced_fetcher(self):
        """Test the enhanced data fetcher"""
        print(f"\n{'='*60}")
        print(f"🔍 TESTING ENHANCED DATA FETCHER")
        print(f"{'='*60}")
        
        try:
            from gst_enhanced_fetcher import GSTEnhancedFetcher
            
            fetcher = GSTEnhancedFetcher(self.api_key)
            
            # Test with small batch first
            test_symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL']
            
            print(f"📊 Testing batch fetch with {len(test_symbols)} symbols...")
            results = fetcher.batch_get_symbols_data(test_symbols)
            
            print(f"\n📈 FETCH RESULTS")
            print(f"{'─'*40}")
            
            successful = 0
            for symbol, data in results.items():
                status = "✅" if data['has_data'] else "❌"
                print(f"{symbol:5} | {status} | ", end="")
                
                if data['has_data']:
                    successful += 1
                    prev_close = data['previous_close']
                    intraday_points = len(data['intraday_data']) if data['intraday_data'] is not None else 0
                    print(f"Prev: ${prev_close:7.2f} | Points: {intraday_points:4}")
                else:
                    print("No data available")
            
            print(f"{'─'*40}")
            print(f"Success rate: {successful}/{len(test_symbols)} ({successful/len(test_symbols)*100:.1f}%)")
            print(f"API requests used: {fetcher.requests_made}/{fetcher.max_requests}")
            
            return successful > 0
            
        except ImportError:
            print("❌ Please create 'gst_enhanced_fetcher.py' first")
            return False
        except Exception as e:
            print(f"❌ Error testing enhanced fetcher: {e}")
            return False
    
    def test_fixed_trader_single(self, symbol: str = "AAPL"):
        """Test the fixed trader on a single symbol"""
        print(f"\n{'='*60}")
        print(f"🎯 TESTING FIXED TRADER: {symbol}")
        print(f"{'='*60}")
        
        try:
            from gst_daytrader import GSTDayTrader
            
            # Use reduced risk for testing
            trader = GSTDayTrader(self.api_key, self.max_risk_per_trade)
            
            print(f"💰 Max risk per trade: ${trader.max_risk_per_trade}")
            print(f"💰 Max position value: ${trader.max_position_value}")
            print(f"📊 Gap thresholds: {trader.min_gap_threshold:.1%} - {trader.max_gap_threshold:.1%}")
            print(f"🛡️  Stop loss: {trader.stop_loss_pct:.1%}")
            print(f"🎯 Profit target: {trader.profit_target_pct:.1%}")
            
            # Process single symbol
            result = trader.process_symbol(symbol)
            
            if not result['success']:
                print(f"❌ Failed to process {symbol}: {result['reason']}")
                return False
            
            # Display results
            gap = result['gap_analysis']
            signal = result['trade_signal']
            execution = result['execution_result']
            
            self.print_analysis_results(gap, signal, execution)
            self.print_trader_performance(trader)
            
            return True
            
        except ImportError:
            print("❌ Please save the fixed gSTDayTrader code as 'gst_daytrader.py'")
            return False
        except Exception as e:
            print(f"❌ Error testing fixed trader: {e}")
            return False
    
    def test_batch_processing(self, max_symbols: int = 10):
        """Test batch processing with multiple symbols"""
        print(f"\n{'='*60}")
        print(f"🚀 TESTING BATCH PROCESSING ({max_symbols} symbols)")
        print(f"{'='*60}")
        
        try:
            from gst_daytrader import GSTDayTrader
            from gst_enhanced_fetcher import GSTEnhancedFetcher
            
            # Initialize components
            trader = GSTDayTrader(self.api_key, self.max_risk_per_trade)
            fetcher = GSTEnhancedFetcher(self.api_key)
            
            # Get symbols
            all_symbols = fetcher.get_top_100_symbols()
            test_symbols = all_symbols[:max_symbols]
            
            print(f"📊 Processing {len(test_symbols)} symbols...")
            print(f"🔄 Symbols: {', '.join(test_symbols)}")
            
            # Run strategy
            results = trader.run_strategy(test_symbols)
            
            # Print batch results
            print(f"\n📈 BATCH RESULTS")
            print(f"{'─'*40}")
            print(f"Symbols processed:   {results['symbols_processed']}")
            print(f"Symbols with data:   {results['symbols_with_data']}")
            print(f"Gaps found:          {results['gaps_found']}")
            print(f"Total trades:        {results['total_trades']}")
            print(f"Successful trades:   {results['successful_trades']}")
            print(f"Failed trades:       {results['failed_trades']}")
            print(f"API errors:          {results['api_errors']}")
            print(f"Duration:            {results['duration']}")
            
            # Performance summary
            performance = trader.get_performance_summary()
            print(f"\n💼 PERFORMANCE")
            print(f"{'─'*40}")
            for key, value in performance.items():
                if 'trades' in key.lower():
                    print(f"{key.replace('_', ' ').title():20}: {value}")
            
            print(f"Win rate:            {performance['win_rate']}")
            print(f"Total P&L:           {performance['total_pnl']}")
            print(f"Avg P&L per trade:   {performance['avg_profit_per_trade']}")
            print(f"Max drawdown:        {performance['max_drawdown']}")
            
            # Save results
            trader.save_trades_to_csv()
            trader.save_performance_report()
            
            return results['total_trades'] > 0
            
        except Exception as e:
            print(f"❌ Error in batch processing: {e}")
            return False
    
    def print_analysis_results(self, gap, signal, execution):
        """Print formatted analysis results"""
        print(f"\n📊 GAP ANALYSIS")
        print(f"{'─'*40}")
        print(f"Previous Close:  ${gap['previous_close']:.2f}")
        print(f"Open Price:      ${gap['open_price']:.2f}")
        print(f"Current Price:   ${gap['current_price']:.2f}")
        print(f"Gap %:           {gap['gap_pct']:.2%} ({gap['gap_direction']})")
        print(f"Volume:          {gap['volume']:,.0f}")
        print(f"Gap Status:      {'✅' if gap['has_gap'] else '❌'} {gap['reason']}")
        
        print(f"\n🎯 TRADE SIGNAL")
        print(f"{'─'*40}")
        if signal['action'] == 'no_trade':
            print(f"Action:          ❌ NO TRADE")
            print(f"Reason:          {signal['reason']}")
        else:
            action_emoji = '🟢' if signal['action'] == 'long' else '🔴'
            print(f"Action:          {action_emoji} {signal['action'].upper()}")
            print(f"Entry Price:     ${signal['entry_price']:.2f}")
            print(f"Stop Loss:       ${signal['stop_loss']:.2f}")
            print(f"Profit Target:   ${signal['profit_target']:.2f}")
            print(f"Shares:          {signal['shares']:,}")
            print(f"Position Value:  ${signal['position_value']:,.2f}")
            print(f"Risk Amount:     ${signal['risk_amount']:.2f}")
            print(f"Expected Profit: ${signal['expected_profit']:.2f}")
            print(f"Risk/Reward:     {signal['risk_reward_ratio']:.2f}")
        
        print(f"\n⚡ EXECUTION")
        print(f"{'─'*40}")
        if not execution['executed']:
            print(f"Status:          ❌ NOT EXECUTED")
            print(f"Reason:          {execution['reason']}")
        else:
            trade = execution['trade']
            exit_result = execution['exit_result']
            
            pnl_emoji = '🟢' if trade['pnl'] > 0 else '🔴'
            outcome = "✅ WINNER" if trade['pnl'] > 0 else "❌ LOSER"
            
            print(f"Status:          ✅ EXECUTED")
            print(f"Exit Price:      ${trade['exit_price']:.2f}")
            print(f"Exit Reason:     {exit_result['exit_reason'].replace('_', ' ').title()}")
            print(f"P&L:             {pnl_emoji} ${trade['pnl']:.2f}")
            print(f"Outcome:         {outcome}")
    
    def print_trader_performance(self, trader):
        """Print trader performance summary"""
        performance = trader.get_performance_summary()
        
        print(f"\n📈 TRADER PERFORMANCE")
        print(f"{'─'*40}")
        print(f"Total Trades:    {performance['total_trades']}")
        print(f"Win Rate:        {performance['win_rate']}")
        print(f"Total P&L:       {performance['total_pnl']}")
        print(f"Best Trade:      {performance['best_trade']}")
        print(f"Worst Trade:     {performance['worst_trade']}")
        print(f"Max Drawdown:    {performance['max_drawdown']}")
    
    def create_directory_structure(self):
        """Create directory structure for gSTDayTrader"""
        directories = [
            "data",
            "data/cache", 
            "results",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")

def main():
    """Main test runner for gSTDayTrader"""
    print("🚀 gSTDayTrader Enhanced Test System")
    print("=" * 60)
    
    tester = GSTDayTraderTest()
    
    # Step 1: Create directories
    print("Step 1: Creating Directory Structure...")
    tester.create_directory_structure()
    
    # Step 2: Test enhanced fetcher
    print("\nStep 2: Testing Enhanced Data Fetcher...")
    fetcher_works = tester.test_enhanced_fetcher()
    
    if not fetcher_works:
        print("\n❌ Enhanced fetcher test failed")
        print("📝 Make sure you created 'gst_enhanced_fetcher.py'")
        return
    
    # Step
