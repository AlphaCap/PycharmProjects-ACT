# main.py
import argparse
import logging
import os
from datetime import datetime
import json

# Import our modules
from data_manager import initialize as init_data_manager
from nGS_Strategy import NGSStrategy
import daily_update
import reporting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ngs_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_system():
    """Initialize the system and create necessary directories."""
    logger.info("Setting up system directories and files")
    
    # Create directories
    for directory in ["data", "logs", "reports", "config"]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize data manager
    init_data_manager()
    
    # Create default config if it doesn't exist
    if not os.path.exists("config/system_config.json"):
        config = {
            "account_size": 100000,
            "position_size": 5000,
            "max_positions": 20,
            "min_price": 10,
            "max_price": 500,
            "update_time": "16:30",  # Market close
            "reporting_time": "17:30"
        }
        with open("config/system_config.json", "w") as f:
            json.dump(config, f, indent=4)
    
    # Update SP500 list
    daily_update.update_sp500_list()
    
    logger.info("System setup complete")
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="nGS Trading System for SP500")
    parser.add_argument("--setup", action="store_true", help="Set up the system")
    parser.add_argument("--update", action="store_true", help="Run daily data update")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--days", type=int, default=30, help="Days to include in report")
    parser.add_argument("--symbol", type=str, help="Process a specific symbol")
    return parser.parse_args()

def process_single_symbol(symbol):
    """Process a single symbol for testing."""
    from data_manager import load_combined_data
    
    logger.info(f"Processing single symbol: {symbol}")
    
    # Try to load existing data
    df = load_combined_data(symbol)
    
    # If no data, download it
    if df is None or df.empty:
        df = daily_update.download_symbol_data(symbol)
        if df.empty:
            logger.error(f"Failed to download data for {symbol}")
            return False
    
    # Process with strategy
    strategy = NGSStrategy()
    result = strategy.process_symbol(symbol, df)
    
    if result is not None and not result.empty:
        logger.info(f"Successfully processed {symbol}, result has {len(result)} rows")
        
        # Show the last few bars with signals
        signal_rows = result[(result['Signal'] != 0) | (result['ExitSignal'] != 0)]
        if not signal_rows.empty:
            print("\nSignal rows:")
            print(signal_rows[['Date', 'Close', 'Signal', 'SignalType', 'ExitSignal', 'ExitType']].tail(5))
        else:
            print("\nNo signals generated")
            
        return True
    else:
        logger.error(f"Failed to process {symbol}")
        return False

def main():
    """Main entry point for the system."""
    args = parse_arguments()
    
    if args.setup:
        setup_system()
    
    elif args.update:
        daily_update.main()
    
    elif args.report:
        reporting.generate_performance_report(days=args.days)
        reporting.current_positions_report()
    
    elif args.symbol:
        process_single_symbol(args.symbol)
    
    else:
        # Print help
        print("nGS Trading System for SP500")
        print("=" * 40)
        print("Usage options:")
        print("  --setup    : Set up the system directories and files")
        print("  --update   : Run daily data update and strategy")
        print("  --report   : Generate performance report")
        print("  --symbol X : Process a specific symbol X")
        print("\nFor more details, use --help")

if __name__ == "__main__":
    print("=" * 50)
    print(f"nGS Trading System for SP500 - {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 50)
    main()
