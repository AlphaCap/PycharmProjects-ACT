
# main.py - Section 1: Imports and Setup
import argparse
import logging
import os
import json
import sys
from datetime import datetime, timedelta
import time

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

# Constants
VERSION = "1.0.0"
CONFIG_FILE = "config/system_config.json"

def setup_system():
    """Initialize the system and create necessary directories."""
    logger.info("Setting up system directories and files")
    
    # Create directories
    for directory in ["data", "logs", "reports", "config"]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize data manager
    init_data_manager()
    
    # Create default config if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        config = {
            "account_size": 100000,
            "position_size": 5000,
            "max_positions": 20,
            "min_price": 10,
            "max_price": 500,
            "update_time": "16:30",  # Market close
            "reporting_time": "17:30",
            "history_days": 200,
            "max_workers": 8,
            "batch_size": 100
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        
        # Write config file
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Created default configuration file: {CONFIG_FILE}")
    
    # Update SP500 list
    symbols = daily_update.update_sp500_list()
    logger.info(f"Updated SP500 list with {len(symbols)} symbols")
    
    print(f"System setup complete. Created necessary directories and configuration files.")
    return True
# main.py - Section 2: Command Processing Functions
def process_single_symbol(symbol):
    """Process a single symbol for testing."""
    from data_manager import load_combined_data
    
    logger.info(f"Processing single symbol: {symbol}")
    print(f"Processing {symbol}...")
    
    # Try to load existing data
    df = load_combined_data(symbol)
    
    # If no data, download it
    if df is None or df.empty:
        print(f"No existing data found for {symbol}, downloading...")
        df = daily_update.download_single_symbol(symbol)
        if df is None or df.empty:
            logger.error(f"Failed to download data for {symbol}")
            print(f"Error: Failed to download data for {symbol}")
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
            print(signal_rows[['Date', 'Close', 'Signal', 'SignalType', 'ExitSignal', 'ExitType']].tail(5).to_string())
        else:
            print("\nNo signals generated in the recent bars")
        
        # Show current position if exists
        if symbol in strategy.positions and strategy.positions[symbol]['shares'] != 0:
            pos = strategy.positions[symbol]
            print("\nCurrent position:")
            print(f"Shares: {pos['shares']}")
            print(f"Entry Price: ${pos['entry_price']:.2f}")
            print(f"Entry Date: {pos['entry_date']}")
            print(f"Days Held: {pos['bars_since_entry']}")
            
            # Calculate current profit
            last_price = result['Close'].iloc[-1]
            profit = (last_price - pos['entry_price']) * pos['shares'] if pos['shares'] > 0 else (pos['entry_price'] - last_price) * abs(pos['shares'])
            print(f"Current Profit: ${profit:.2f}")
        else:
            print("\nNo active position for this symbol")
            
        return True
    else:
        logger.error(f"Failed to process {symbol}")
        print(f"Error: Failed to process {symbol}")
        return False

def run_scheduled_tasks():
    """Run scheduled tasks based on configuration."""
    logger.info("Starting scheduled task runner")
    print("Starting scheduled task runner...")
    
    # Load configuration
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = {}
    else:
        logger.warning(f"Config file not found: {CONFIG_FILE}")
        config = {}
    
    # Get scheduled times
    update_time = config.get('update_time', '16:30')
    reporting_time = config.get('reporting_time', '17:30')
    
    print(f"Scheduled update time: {update_time}")
    print(f"Scheduled reporting time: {reporting_time}")
    print("Press Ctrl+C to stop...\n")
    
    try:
        while True:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            
            # Check for data update time
            if current_time == update_time:
                logger.info("Running scheduled data update")
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled data update...")
                daily_update.main()
                time.sleep(60)  # Sleep to avoid multiple runs
            
            # Check for reporting time
            elif current_time == reporting_time:
                logger.info("Running scheduled reporting")
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled reporting...")
                reporting.main()
                time.sleep(60)  # Sleep to avoid multiple runs
            
            # Sleep for a while before checking again
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("Scheduled task runner stopped by user")
        print("\nScheduled task runner stopped")

def display_system_status():
    """Display system status and information."""
    from data_manager import get_positions, get_metadata, get_symbols_count
    
    print("\n" + "="*60)
    print(f"nGS TRADING SYSTEM STATUS - v{VERSION}")
    print("="*60)
    
    # System info
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"System Directory: {os.getcwd()}")
    
    # Data statistics
    symbol_count = get_symbols_count()
    positions = get_positions()
    metadata = get_metadata()
    
    print("\nData Statistics:")
    print(f"Total Symbols: {symbol_count}")
    print(f"Active Positions: {len(positions)}")
    
    # Show last update time
    last_update = metadata.get("daily_update.last_run", "Never")
    print(f"Last Data Update: {last_update}")
    
    # Show performance stats if available
    win_rate = metadata.get("performance.win_rate", "Unknown")
    total_profit_30d = metadata.get("performance.total_profit_30d", "Unknown")
    
    print("\nPerformance Summary:")
    print(f"Win Rate: {win_rate}%")
    print(f"30-Day Profit: ${total_profit_30d}")
    
    # Configuration
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            print("\nConfiguration:")
            print(f"Account Size: ${config.get('account_size', 'Not set')}")
            print(f"Position Size: ${config.get('position_size', 'Not set')}")
            print(f"Update Time: {config.get('update_time', 'Not set')}")
        except:
            print("\nConfiguration: Error loading config file")
    else:
        print("\nConfiguration: No config file found")
    
    print("="*60)
# main.py - Section 3: CLI and Main Function
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f"nGS Trading System for SP500 v{VERSION}",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("--setup", action="store_true", 
                       help="Set up the system directories and files")
    
    parser.add_argument("--update", action="store_true", 
                       help="Run daily data update")
    
    parser.add_argument("--report", action="store_true", 
                       help="Generate performance report")
    
    parser.add_argument("--days", type=int, default=30, 
                       help="Days to include in report (default: 30)")
    
    parser.add_argument("--symbol", type=str, 
                       help="Process a single symbol for testing")
    
    parser.add_argument("--status", action="store_true", 
                       help="Display system status")
    
    parser.add_argument("--scheduler", action="store_true", 
                       help="Run scheduled tasks (press Ctrl+C to stop)")
    
    parser.add_argument("--version", action="store_true", 
                       help="Show version information")
    
    return parser.parse_args()

def main():
    """Main entry point for the system."""
    args = parse_arguments()
    
    # Check for API key
    if not os.getenv('POLYGON_API_KEY') and (args.update or args.symbol or args.scheduler):
        print("ERROR: Polygon API key not set. Please set the POLYGON_API_KEY environment variable.")
        print("Example: export POLYGON_API_KEY='your_api_key_here'")
        return 1
    
    # Process commands
    if args.setup:
        setup_system()
    
    elif args.update:
        daily_update.main()
    
    elif args.report:
        reporting.main(days=args.days)
    
    elif args.symbol:
        process_single_symbol(args.symbol)
    
    elif args.status:
        display_system_status()
    
    elif args.scheduler:
        run_scheduled_tasks()
    
    elif args.version:
        print(f"nGS Trading System v{VERSION}")
        print("Neural Grid Strategy for SP500")
    
    else:
        # Print welcome message
        print("\n" + "="*60)
        print(f"nGS TRADING SYSTEM FOR SP500 - v{VERSION}")
        print("=" * 60)
        print("Usage options:")
        print("  --setup     : Set up the system directories and files")
        print("  --update    : Run daily data update and strategy")
        print("  --report    : Generate performance report")
        print("  --symbol X  : Process a specific symbol X")
        print("  --status    : Display system status")
        print("  --scheduler : Run scheduled tasks")
        print("  --version   : Show version information")
        print("\nFor more details, use --help")
    
    return 0

if __name__ == "__main__":
    print(f"nGS Trading System v{VERSION} - {datetime.now().strftime('%Y-%m-%d')}")
    sys.exit(main())
