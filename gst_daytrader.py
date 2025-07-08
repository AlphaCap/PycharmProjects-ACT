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

# Import the gSTDayTrader (save the previous artifact as gst_daytrader.py)
# from gst_daytrader import GSTDayTrader

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
        print(f"\n=== Testing Single Symbol: {symbol} ===")
        
        # Import and initialize (you'll need to save the main artifact as gst_daytrader.py)
        try:
            from gst_daytrader import GSTDayTrader
            trader = GSTDayTrader(self.api_key, self.position_size)
            
            # Process single symbol
            result = trader.process_symbol(symbol)
            
            print(f"Result: {result}")
            
            # Show performance
            performance = trader.get_performance_summary()
            print(f"\nPerformance: {performance}")
            
            # Show trades if any
            if trader.trades:
                trades_df = pd.DataFrame(trader.trades)
                print(f"\nTrades:")
                print(trades_df.to_string())
            
            return True
            
        except ImportError:
            print("❌ Please save the main gSTDayTrader code as 'gst_daytrader.py' first")
            return False
        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            return False
    
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
        print(f"\n=== Running Full Test on {num_symbols} Symbols ===")
        
        # Get symbols
        symbols = self.get_top_100_sp500()[:num_symbols]
        print(f"Testing symbols: {symbols}")
        
        try:
            from gst_daytrader import GSTDayTrader
            trader = GSTDayTrader(self.api_key, self.position_size)
            
            # Run strategy
            results = trader.run_strategy(symbols)
            
            print(f"\n=== Results ===")
            print(f"Symbols processed: {results['symbols_processed']}")
            print(f"Symbols with data: {results['symbols_with_data']}")
            print(f"Total trades: {results['total_trades']}")
            
            # Performance summary
            performance = trader.get_performance_summary()
            print(f"\n=== Performance Summary ===")
            for key, value in performance.items():
                print(f"{key}: {value}")
            
            # Save results
            self.save_results(trader, results, performance)
            
            return True
            
        except ImportError:
            print("❌ Please save the main gSTDayTrader code as 'gst_daytrader.py' first")
            return False
        except Exception as e:
            print(f"❌ Error in full test: {e}")
            return False
    
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
    if not tester.test_api_connection():
        print("❌ API test failed. Please check your connection and API key.")
        return
    
    # Step 2: Create directory structure
    tester.create_directory_structure()
    
    # Step 3: Test single symbol
    print("\n" + "=" * 50)
    if not tester.test_single_symbol("AAPL"):
        print("❌ Single symbol test failed.")
        print("💡 Next steps:")
        print("   1. Save the main gSTDayTrader code as 'gst_daytrader.py'")
        print("   2. Run this test script again")
        return
    
    # Step 4: Run full test
    print("\n" + "=" * 50)
    user_input = input("Run full test on 10 symbols? (y/n): ").lower()
    if user_input == 'y':
        tester.run_full_test(10)
    
    print("\n✅ gSTDayTrader setup complete!")
    print("📊 Ready to analyze gap trading opportunities!")

if __name__ == "__main__":
    main()
