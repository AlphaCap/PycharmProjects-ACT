# main.py - Complete revised version with simplified import approach
import argparse
import logging
import os
import json
import sys
from datetime import datetime, timedelta
import time

# Import our modules that work with direct imports
from data_manager import initialize as init_data_manager
from data_manager import get_positions, get_metadata, get_symbols_count
from data_manager import load_combined_data, get_sp500_symbols
from nGS_Strategy import NGSStrategy

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

# Helper functions to execute code from other files
def execute_daily_update_main():
    """Execute the main function from daily_update.py"""
    logger.info("Executing daily update")
    exec(open('daily_update.py').read())
    
def execute_daily_update_update_sp500_list():
    """Execute the update_sp500_list function from daily_update.py"""
    # Create a namespace to capture the return value
    namespace = {}
    exec("""
from datetime import datetime, timedelta
import pandas as pd
import requests
import logging
import os

# Simple logger setup if it doesn't exist
if 'logger' not in locals():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def update_sp500_list():
    try:
        # Fallback to Wikipedia
        logger.info("Getting SP500 list from Wikipedia")
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = tables[0]
        symbols = sp500['Symbol'].tolist()
        
        # Clean symbols
        symbols = [s.replace('.', '-') for s in symbols]
        
        # Save to CSV
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"symbol": symbols}).to_csv("data/sp500_symbols.csv", index=False)
        logger.info(f"Updated SP500 list with {len(symbols)} symbols from Wikipedia")
        return symbols
    except Exception as e:
        logger.error(f"Error updating SP500 list: {e}")
        # If update fails, try to load existing list
        if os.path.exists("data/sp500_symbols.csv"):
            symbols = pd.read_csv("data/sp500_symbols.csv")["symbol"].tolist()
            logger.info(f"Loaded {len(symbols)} symbols from existing SP500 list")
            return symbols
        return []

symbols = update_sp500_list()
""", namespace)
    return namespace.get('symbols', [])

def execute_daily_update_download_single_symbol(symbol):
    """Execute the download_single_symbol function for a given symbol"""
    namespace = {'target_symbol': symbol}
    exec("""
import pandas as pd
import requests
import logging
import os
import time
from datetime import datetime, timedelta

# Simple logger setup if it doesn't exist
if 'logger' not in locals():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Get Polygon API key
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

def get_polygon_daily_data(symbol, days=200):
    """Download historical daily data for a symbol using Polygon API."""
    if not POLYGON_API_KEY:
        logger.error("Cannot download data: No Polygon API key provided")
        return pd.DataFrame()
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Polygon API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # API endpoint for aggregated daily bars
        endpoint = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date_str}/{end_date_str}"
        
        # Make API request with retry logic
        attempts = 0
        retry_attempts = 3
        rate_limit_pause = 12
        while attempts < retry_attempts:
            try:
                response = requests.get(
                    endpoint,
                    params={'apiKey': POLYGON_API_KEY, 'adjusted': 'true'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if results exist
                    if 'results' in data and data['results']:
                        # Convert to DataFrame
                        df = pd.DataFrame(data['results'])
                        
                        # Rename columns to match our expected format
                        column_map = {
                            't': 'timestamp',  # Unix timestamp in milliseconds
                            'o': 'Open',
                            'h': 'High',
                            'l': 'Low',
                            'c': 'Close',
                            'v': 'Volume'
                        }
                        df = df.rename(columns=column_map)
                        
                        # Convert timestamp to datetime
                        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
                        
                        # Select and order columns
                        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        
                        # Sort by date
                        df = df.sort_values('Date')
                        
                        logger.info(f"Downloaded {len(df)} bars for {symbol}")
                        return df
                    else:
                        logger.warning(f"No data returned for {symbol}")
                        return pd.DataFrame()
                
                elif response.status_code == 429:
                    # Rate limited - pause and retry
                    logger.warning(f"Rate limited on {symbol}, pausing for {rate_limit_pause} seconds")
                    attempts += 1
                    time.sleep(rate_limit_pause)
                    continue
                    
                else:
                    logger.error(f"Error {response.status_code} for {symbol}: {response.text}")
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Request error for {symbol}: {e}")
                attempts += 1
                
        logger.error(f"Failed to download data for {symbol} after {retry_attempts} attempts")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()

def save_price_data(symbol, df):
    """Save price data to disk"""
    if df is None or df.empty:
        return False
        
    # Create directory if it doesn't exist
    os.makedirs("data/daily", exist_ok=True)
    
    # Save the file
    file_path = f"data/daily/{symbol}.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Saved data for {symbol} to {file_path}")
    return True

# Download data for the target symbol
symbol = target_symbol
df = get_polygon_daily_data(symbol, 200)
if not df.empty:
    save_price_data(symbol, df)
    result_df = df
else:
    result_df = None
""", namespace)
    return namespace.get('result_df')

def execute_reporting_main(days=30, export_html=True):
    """Execute the main function from reporting.py with arguments"""
    namespace = {'args_days': days, 'args_export_html': export_html}
    exec("""
# Execute the entire reporting.py file with our parameters
days = args_days
export_html = args_export_html

# The rest of reporting.py will be loaded and executed with these variables in scope
""" + open('reporting.py').read(), namespace)

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
    symbols = execute_daily_update_update_sp500_list()
    logger.info(f"Updated SP500 list with {len(symbols)} symbols")
    
    print(f"System setup complete. Created necessary directories and configuration files.")
    return True

def process_single_symbol(symbol):
    """Process a single symbol for testing."""
    logger.info(f"Processing single symbol: {symbol}")
    print(f"Processing {symbol}...")
    
    # Try to load existing data
    df = load_combined_data(symbol)
    
    # If no data, download it
    if df is None or df.empty:
        print(f"No existing data found for {symbol}, downloading...")
        df = execute_daily_update_download_single_symbol(symbol)
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
                execute_daily_update_main()
                time.sleep(60)  # Sleep to avoid multiple runs
            
            # Check for reporting time
            elif current_time == reporting_time:
                logger.info("Running scheduled reporting")
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled reporting...")
                execute_reporting_main()
                time.sleep(60)  # Sleep to avoid multiple runs
            
            # Sleep for a while before checking again
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("Scheduled task runner stopped by user")
        print("\nScheduled task runner stopped")

def display_system_status():
    """Display system status and information."""
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
        execute_daily_update_main()
    
    elif args.report:
        execute_reporting_main(days=args.days)
    
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
