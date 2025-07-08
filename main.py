from nGS_Strategy import NGSStrategy
from data_manager import load_price_data, update_metadata
from data_manager import get_positions, init_metadata as get_metadata
from data_manager import initialize as init_data_manager
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
import json
import os
import logging
import argparse
from utils.polygon_api import PolygonClient
import sys
sys.path.append('C:/ACT/Python NEW 2025')

# Import from data_manager

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


def update_sp500_list():
    """Update the S&P 500 symbols list using real S&P 500 companies."""
    try:
        # Try to load from existing CSV file first
        if os.path.exists("data/sp500_symbols.csv"):
            try:
                df = pd.read_csv("data/sp500_symbols.csv")
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].dropna().tolist()
                elif 'symbol' in df.columns:
                    symbols = df['symbol'].dropna().tolist()
                else:
                    symbols = df.iloc[:, 0].dropna().tolist()
                
                symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
                
                if symbols and len(symbols) > 100:
                    logger.info(f"Loaded {len(symbols)} symbols from existing SP500 CSV file")
                    return symbols
            except Exception as e:
                logger.warning(f"Could not load S&P 500 symbols from CSV file: {e}")
        
        # Real S&P 500 symbols as of July 2025 (503 symbols)
        logger.info("Using hardcoded real S&P 500 symbols list (503 symbols)")
        symbols = [
            "MSFT", "NVDA", "AAPL", "AMZN", "GOOG", "GOOGL", "META", "AVGO", "BRK.B", "TSLA",
            "WMT", "JPM", "V", "LLY", "MA", "NFLX", "ORCL", "COST", "XOM", "PG",
            "JNJ", "HD", "BAC", "ABBV", "KO", "PLTR", "PM", "TMUS", "UNH", "GE",
            "CRM", "CSCO", "WFC", "CVX", "IBM", "ABT", "LIN", "MCD", "INTU", "NOW",
            "AXP", "MS", "DIS", "T", "ISRG", "ACN", "MRK", "AMD", "RTX", "VZ",
            "BKNG", "GS", "UBER", "PEP", "ADBE", "TXN", "BX", "CAT", "PGR", "QCOM",
            "SCHW", "SPGI", "BA", "AMGN", "BLK", "TMO", "BSX", "NEE", "HON", "SYK",
            "C", "TJX", "DE", "DHR", "GILD", "AMAT", "UNP", "PANW", "PFE", "ADP",
            "GEV", "ETN", "CMCSA", "LOW", "ANET", "MU", "CB", "CRWD", "VRTX", "MMC",
            "APH", "LMT", "MDT", "LRCX", "ADI", "COP", "KKR", "KLAC", "PLD", "ICE",
            "SBUX", "WELL", "MO", "AMT", "CME", "BMY", "SO", "TT", "WM", "CEG",
            "NKE", "DASH", "HCA", "FI", "CTAS", "DUK", "EQIX", "SHW", "MCK", "ELV",
            "MCO", "INTC", "ABNB", "PH", "MDLZ", "AJG", "CI", "UPS", "TDG", "CDNS",
            "CVS", "FTNT", "AON", "RSG", "ORLY", "MMM", "DELL", "APO", "COF", "ZTS",
            "ECL", "SNPS", "RCL", "GD", "WMB", "CL", "MAR", "ITW", "PYPL", "HWM",
            "CMG", "PNC", "NOC", "MSI", "USB", "EMR", "JCI", "WDAY", "BK", "COIN",
            "ADSK", "KMI", "APD", "EOG", "AZO", "TRV", "MNST", "AXON", "ROP", "CHTR",
            "SPG", "DLR", "CARR", "CSX", "HLT", "FCX", "VST", "NEM", "PAYX", "NSC",
            "AFL", "COR", "ALL", "AEP", "MET", "PWR", "PSA", "TFC", "FDX", "GWW",
            "NXPI", "REGN", "OKE", "O", "AIG", "SRE", "BDX", "AMP", "MPC", "NDAQ",
            "PCAR", "CTVA", "TEL", "CPRT", "FAST", "D", "ROST", "PSX", "SLB", "URI",
            "LHX", "GM", "EW", "CMI", "VRSK", "KDP", "KMB", "TGT", "KR", "MSCI",
            "GLW", "FICO", "CCI", "EXC", "FIS", "TTWO", "IDXX", "HES", "OXY", "KVUE",
            "AME", "FANG", "F", "YUM", "VLO", "PEG", "GRMN", "CTSH", "XEL", "OTIS",
            "CBRE", "BKR", "EA", "PRU", "DHI", "CAH", "RMD", "HIG", "ED", "ROK",
            "TRGP", "EBAY", "SYY", "ACGL", "ETR", "WAB", "MCHP", "VMC", "PCG", "DXCM",
            "ODFL", "EQT", "WEC", "IR", "LYV", "EFX", "DAL", "VICI", "MLM", "EXR",
            "CSGP", "MPWR", "A", "TKO", "GEHC", "HSY", "IT", "CCL", "LULU", "BRO",
            "KHC", "XYL", "WTW", "NRG", "STZ", "IRM", "GIS", "ANSS", "RJF", "MTB",
            "VTR", "AVB", "BR", "LEN", "DD", "K", "LVS", "WRB", "STT", "NUE",
            "ROL", "EXE", "KEYS", "HUM", "DTE", "UAL", "CNC", "AWK", "TSCO", "STX",
            "EQR", "VRSN", "IQV", "FITB", "GDDY", "AEE", "TPL", "PPG", "DRI", "PPL",
            "IP", "DG", "VLTO", "TYL", "FTV", "SMCI", "EL", "MTD", "DOV", "FOXA",
            "CHD", "WBD", "SBAC", "ATO", "ES", "CNP", "STE", "CPAY", "HPE", "HPQ",
            "HBAN", "CINF", "CDW", "FE", "TDY", "FOX", "CBOE", "ADM", "SW", "SYF",
            "EXPE", "PODD", "NTAP", "LH", "NVR", "HUBB", "NTRS", "ON", "CMS", "ULTA",
            "WAT", "AMCR", "TROW", "DVN", "EIX", "PTC", "INVH", "DOW", "PHM", "MKC",
            "STLD", "RF", "DLTR", "TSN", "IFF", "LII", "CTRA", "BIIB", "DGX", "ERIE",
            "WSM", "WY", "WDC", "LUV", "LDOS", "JBL", "GPN", "L", "ESS", "NI",
            "ZBH", "GEN", "LYB", "MAA", "CFG", "KEY", "FSLR", "HAL", "PKG", "GPC",
            "PFG", "TRMB", "FFIV", "HRL", "SNA", "RL", "NWS", "FDS", "TPR", "PNR",
            "DECK", "WST", "MOH", "DPZ", "CLX", "NWSA", "LNT", "BAX", "BBY", "EXPD",
            "J", "ZBRA", "EVRG", "CF", "BALL", "PAYC", "EG", "UDR", "APTV", "COO",
            "HOLX", "KIM", "AVY", "OMC", "JBHT", "IEX", "TER", "TXT", "MAS", "INCY",
            "BF.B", "JKHY", "REG", "BXP", "ALGN", "SOLV", "CPT", "BLDR", "DOC", "UHS",
            "ARE", "NDSN", "JNPR", "ALLE", "SJM", "BEN", "CHRW", "AKAM", "POOL", "HST",
            "MOS", "RVTY", "SWKS", "CAG", "PNW", "MRNA", "TAP", "DVA", "AIZ", "CPB",
            "SWK", "VTRS", "EPAM", "LKQ", "GL", "BG", "KMX", "WBA", "DAY", "HAS",
            "AOS", "EMN", "HII", "NCLH", "MGM", "WYNN", "HSIC", "IPG", "FRT", "MKTX",
            "PARA", "LW", "MTCH", "AES", "TECH", "GNRC", "CRL", "ALB", "APA", "IVZ",
            "MHK", "ENPH", "CZR"
        ]
             
    except Exception as e:
        logger.error(f"Error updating stock list: {e}")
        # Fallback to minimal list if everything fails
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        pd.DataFrame({"symbol": symbols}).to_csv(
            "data/sp500_symbols.csv", index=False)
        logger.info(
            f"Updated stock list with {
                len(symbols)} symbols from Polygon API")
        return symbols

    except Exception as e:
        logger.error(f"Error updating stock list: {e}")
        # If update fails, try to load existing list
        if os.path.exists("data/sp500_symbols.csv"):
            symbols = pd.read_csv("data/sp500_symbols.csv")["symbol"].tolist()
            logger.info(
                f"Loaded {
                    len(symbols)} symbols from existing SP500 list")
            return symbols
        return []


def download_single_symbol(symbol, days=200):
    """Download data for a single symbol using Polygon API."""
    from utils.config import POLYGON_API_KEY  # Import from config file

    if not POLYGON_API_KEY:
        logger.error("Cannot download data: No Polygon API key provided")
        return None

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates for Polygon API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # API endpoint for aggregated daily bars
        endpoint = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date_str}/{end_date_str}"

        # Make API request
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
                    't': 'timestamp',
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

                # Save the data
                os.makedirs("data/daily", exist_ok=True)
                df.to_csv(f"data/daily/{symbol}.csv", index=False)

                logger.info(
                    f"Downloaded and saved {
                        len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return None

        else:
            logger.error(
                f"Error {
                    response.status_code} for {symbol}: {
                    response.text}")
            return None

    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return None


def run_daily_update():
    """Run the daily update process with API throttling."""
    logger.info("Starting daily update process")
    print("Starting daily update process...")

    try:
        # Update the SP500 list
        symbols = update_sp500_list()

        update_symbols = symbols
        print(f"Processing {len(update_symbols)} symbols...")

        # Process each symbol with throttling
        for i, symbol in enumerate(update_symbols):
            print(
                f"Downloading data for {symbol}... ({i + 1}/{len(update_symbols)})")
            df = download_single_symbol(symbol)

            if df is not None and not df.empty:
                # Process with strategy
                strategy = NGSStrategy()
                strategy.process_symbol(symbol, df)

            # Apply throttling - respect Polygon API rate limits
            # For free tier: 5 calls per minute = 12 seconds between calls
            if i < len(update_symbols) - \
                    1:  # Don't sleep after the last symbol
                print(f"Throttling API requests... waiting 12 seconds")
                time.sleep(12)  # Wait 12 seconds between API calls

        # Update metadata with last run time
        update_metadata("daily_update.last_run",
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        print("Daily update process complete.")
        return True

    except Exception as e:
        logger.error(f"Error in daily update process: {e}")
        print(f"Error in daily update process: {e}")
        return False


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
    symbols = update_sp500_list()
    logger.info(f"Updated SP500 list with {len(symbols)} symbols")

    print(f"System setup complete. Created necessary directories and configuration files.")
    return True


def process_single_symbol(symbol):
    """Process a single symbol for testing."""
    logger.info(f"Processing single symbol: {symbol}")
    print(f"Processing {symbol}...")

    # Try to load existing data
    df = load_price_data(symbol)

    # If no data, download it
    if df is None or df.empty:
        print(f"No existing data found for {symbol}, downloading...")
        df = download_single_symbol(symbol)
        if df is None or df.empty:
            logger.error(f"Failed to download data for {symbol}")
            print(f"Error: Failed to download data for {symbol}")
            return False

    # Process with strategy
    strategy = NGSStrategy()
    result = strategy.process_symbol(symbol, df)

    if result is not None and not result.empty:
        logger.info(
            f"Successfully processed {symbol}, result has {
                len(result)} rows")

        # Show the last few bars with signals
        signal_rows = result[(result['Signal'] != 0) |
                             (result['ExitSignal'] != 0)]
        if not signal_rows.empty:
            print("\nSignal rows:")
            print(signal_rows[['Date',
                               'Close',
                               'Signal',
                               'SignalType',
                               'ExitSignal',
                               'ExitType']].tail(5).to_string())
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
            profit = (last_price - pos['entry_price']) * pos['shares'] if pos['shares'] > 0 else (
                pos['entry_price'] - last_price) * abs(pos['shares'])
            print(f"Current Profit: ${profit:.2f}")
        else:
            print("\nNo active position for this symbol")

        return True
    else:
        logger.error(f"Failed to process {symbol}")
        print(f"Error: Failed to process {symbol}")
        return False


def run_reporting(days=30):
    """Run the basic reporting functionality."""
    logger.info(f"Running simplified reporting for last {days} days")
    print(f"Running simplified reporting for last {days} days...")

    try:
        from data_manager import get_trades_history

        # Get trade history
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%d")
        trades_df = get_trades_history()

        if trades_df.empty:
            print(f"No trades found in the last {days} days")
            return True

        # Calculate basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = trades_df['profit'].sum()

        # Print summary
        print("\n" + "=" * 50)
        print(f"PERFORMANCE SUMMARY - LAST {days} DAYS")
        print("=" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Profit: ${total_profit:.2f}")

        # Current positions
        positions = get_positions()
        if positions:
            print("\n" + "=" * 50)
            print("CURRENT POSITIONS")
            print("=" * 50)
            for symbol, pos in positions.items():
                print(
                    f"{symbol}: {
                        pos['shares']} shares, Entry: ${
                        pos['entry_price']:.2f}")

        return True

    except Exception as e:
        logger.error(f"Error in reporting: {e}")
        print(f"Error generating report: {e}")
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
                print(
                    f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled data update...")
                run_daily_update()
                time.sleep(60)  # Sleep to avoid multiple runs

            # Check for reporting time
            elif current_time == reporting_time:
                logger.info("Running scheduled reporting")
                print(
                    f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled reporting...")
                run_reporting()
                time.sleep(60)  # Sleep to avoid multiple runs

            # Sleep for a while before checking again
            time.sleep(30)

    except KeyboardInterrupt:
        logger.info("Scheduled task runner stopped by user")
        print("\nScheduled task runner stopped")


def display_system_status():
    """Display system status and information."""
    print("\n" + "=" * 60)
    print(f"nGS TRADING SYSTEM STATUS - v{VERSION}")
    print("=" * 60)

    # System info
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"System Directory: {os.getcwd()}")

    # Data statistics
    symbol_count = 0
    if os.path.exists("data/sp500_symbols.csv"):
        try:
            symbols_df = pd.read_csv("data/sp500_symbols.csv")
            symbol_count = len(symbols_df)
        except Exception as e:
            logger.error(f"Error counting symbols: {e}")
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
    print(f"Win Rate: {win_rate}")
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
        except BaseException:
            print("\nConfiguration: Error loading config file")
    else:
        print("\nConfiguration: No config file found")

    print("=" * 60)


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
    polygon_api_key = os.getenv('POLYGON_API_KEY')

    # Use hardcoded key if environment variable not found
    if not polygon_api_key:
        polygon_api_key = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"  # Hardcoded API key
        # Set it for downstream modules that might use the environment variable
        os.environ['POLYGON_API_KEY'] = polygon_api_key
        print("Using hardcoded Polygon API key.")

    # Final check (should not trigger now that we have a fallback)
    if not polygon_api_key and (args.update or args.symbol or args.scheduler):
        print("ERROR: Polygon API key not set. Please set the POLYGON_API_KEY environment variable.")
        print("Example: export POLYGON_API_KEY='your_api_key_here'")
        return 1

    # Process commands
    if args.setup:
        setup_system()

    elif args.update:
        run_daily_update()

    elif args.report:
        run_reporting(days=args.days)

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
        print("\n" + "=" * 60)
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
    
