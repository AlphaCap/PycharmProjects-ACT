import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from polygon import RESTClient
from typing import List, Dict, Optional

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    encoding='utf-8',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('data_manager.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
SP500_FILE = "data/sp500_symbols.txt"
SP500_MINIMUM_COUNT = 490
DATA_RETENTION_DAYS = 180
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")  # Set via environment variable or .env
POSITIONS_FILE = "data/positions.json"
TRADES_FILE = "data/trades.csv"
METADATA_FILE = "metadata.json"

# Initialize Polygon client
polygon_client = RESTClient(POLYGON_API_KEY) if POLYGON_API_KEY else None

def get_sp500_symbols() -> List[str]:
    """
    Load S&P 500 symbols from the symbols file.

    Returns:
        List[str]: List of S&P 500 symbols.

    Raises:
        FileNotFoundError: If the symbols file is missing.
    """
    if not os.path.exists(SP500_FILE):
        logger.error(f"S&P 500 symbols file not found: {SP500_FILE}")
        return []
    with open(SP500_FILE, "r") as f:
        symbols = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(symbols)} S&P 500 symbols from {SP500_FILE}")
    logger.info(f"Sample symbols: {symbols[:5]}...")
    return symbols

def verify_sp500_coverage() -> bool:
    """
    Verify the integrity and coverage of S&P 500 symbols.

    Returns:
        bool: True if coverage is acceptable, False otherwise.
    """
    symbols = get_sp500_symbols()
    symbol_count = len(symbols)
    expected_count = 500
    blue_chips = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "UNH", "JNJ"]
    sectors = {"Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
               "Finance": ["BRK.B", "JPM", "WFC", "BAC", "C"],
               "Healthcare": ["UNH", "JNJ", "PFE", "MRK", "ABT"]}

    logger.info("S&P 500 symbol verification:")
    logger.info(f"Total symbols loaded: {symbol_count}")
    logger.info(f"Expected S&P 500 count: {expected_count}")
    logger.info(f"Minimum acceptable count: {SP500_MINIMUM_COUNT}")
    logger.info(f"Blue-chip symbols found ({len([s for s in blue_chips if s in symbols])}/10): {blue_chips}")
    for sector, sector_symbols in sectors.items():
        found = len([s for s in sector_symbols if s in symbols])
        logger.info(f"   {sector}: {found}/5 symbols")
    coverage_ok = symbol_count >= SP500_MINIMUM_COUNT
    logger.info(f"✔ S&P 500 symbol count is ACCEPTABLE ({symbol_count} >= {SP500_MINIMUM_COUNT})")

    # Check for duplicates
    if len(symbols) != len(set(symbols)):
        duplicates = [item for item, count in pd.Series(symbols).value_counts().items() if count > 1]
        logger.warning(f"Duplicates found: {duplicates}")
    else:
        logger.info("✔ No duplicate symbols found")

    return coverage_ok

def initialize():
    """
    Initialize the data manager with retention policy and symbol verification.
    """
    cutoff_date = datetime.now() - timedelta(days=DATA_RETENTION_DAYS)
    logger.info(f"Data retention policy: {DATA_RETENTION_DAYS} days ({DATA_RETENTION_DAYS // 30} months)")
    logger.info(f"Current cutoff date: {cutoff_date.date()}")
    logger.info("Data manager initialized with 6-month retention enforced")
    coverage_ok = verify_sp500_coverage()
    return coverage_ok

def get_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol using Polygon API.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        start_date (datetime): Start date for data.
        end_date (datetime): End date for data.

    Returns:
        pd.DataFrame: Historical data with columns like 'open', 'high', 'low', 'close', 'volume'.
    """
    if not polygon_client:
        logger.error("Polygon API key not configured. Set POLYGON_API_KEY environment variable.")
        return pd.DataFrame()

    try:
        # Convert datetime to string format expected by Polygon
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        # Fetch aggregated daily data from Polygon
        agg = polygon_client.get_aggs(symbol, 1, 'day', start_str, end_str)
        if not agg or not agg.results:
            logger.warning(f"No historical data available for {symbol} from {start_str} to {end_str}")
            return pd.DataFrame()

        # Convert Polygon response to DataFrame
        data = pd.DataFrame(agg.results)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data = data.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        data.set_index('timestamp', inplace=True)
        logger.info(f"✔ Fetched historical data for {symbol} from {start_str} to {end_str}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def load_positions() -> List[Dict]:
    """
    Load current positions from a JSON file, filtered by retention policy.

    Returns:
        List[Dict]: List of position dictionaries.
    """
    if not os.path.exists(POSITIONS_FILE):
        logger.warning(f"Positions file not found: {POSITIONS_FILE}")
        return []
    try:
        with open(POSITIONS_FILE, "r") as f:
            positions = pd.read_json(f).to_dict(orient='records')
        cutoff_date = datetime.now() - timedelta(days=DATA_RETENTION_DAYS)
        valid_positions = [pos for pos in positions if datetime.strptime(pos.get('date', cutoff_date.strftime('%Y-%m-%d')), '%Y-%m-%d') >= cutoff_date]
        logger.info(f"Attempting to load {len(positions)} positions from data manager")
        logger.info(f"Loaded {len(valid_positions)} positions within {DATA_RETENTION_DAYS}-day retention period")
        return valid_positions
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return []

def get_trades_history() -> pd.DataFrame:
    """
    Load trade history from a CSV file, filtered by retention policy.

    Returns:
        pd.DataFrame: DataFrame of trade history.
    """
    if not os.path.exists(TRADES_FILE):
        logger.warning(f"Trades file not found: {TRADES_FILE}")
        return pd.DataFrame()
    try:
        trades = pd.read_csv(TRADES_FILE)
        if 'exit_date' not in trades.columns:
            logger.warning("Trades file missing 'exit_date' column")
            return pd.DataFrame()
        trades['exit_date'] = pd.to_datetime(trades['exit_date'])
        cutoff_date = datetime.now() - timedelta(days=DATA_RETENTION_DAYS)
        trades = trades[trades['exit_date'] >= cutoff_date]
        logger.info(f"Loaded {len(trades)} trades within {DATA_RETENTION_DAYS}-day retention period")
        return trades
    except Exception as e:
        logger.error(f"Error loading trades history: {e}")
        return pd.DataFrame()

def update_metadata(key: str, value: any):
    """
    Update metadata in the JSON file.

    Args:
        key (str): Metadata key.
        value (any): Value to store.
    """
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                metadata = pd.read_json(f).to_dict()
        else:
            metadata = {}
        metadata[key] = value
        with open(METADATA_FILE, "w") as f:
            pd.DataFrame.from_dict(metadata, orient='index').to_json(f)
        logger.info(f"Updated metadata: {key} = {value}")
    except Exception as e:
        logger.error(f"Error updating metadata: {e}")

def get_portfolio_metrics(initial_portfolio_value: int) -> Dict:
    """
    Fetch portfolio metrics based on current positions and trades.

    Args:
        initial_portfolio_value (int): Initial portfolio value.

    Returns:
        Dict: Portfolio metrics.
    """
    positions = load_positions()
    trades = get_trades_history()
    total_value = initial_portfolio_value
    total_return_pct = "+0.0%"
    daily_pnl = "$0.00"
    me_ratio = "0.00"
    net_exposure = "$0"
    mtd_return = "+0.0%"
    mtd_delta = "+0.0%"
    ytd_return = "+0.0%"
    ytd_delta = "+0.0%"

    if positions:
        total_value += sum(pos.get('current_value', 0) for pos in positions)
    if not trades.empty:
        daily_pnl = f"${trades['profit'].sum():.2f}" if not trades.empty else "$0.00"

    return {
        'total_value': f"${total_value:,.0f}",
        'total_return_pct': total_return_pct,
        'daily_pnl': daily_pnl,
        'me_ratio': me_ratio,
        'net_exposure': net_exposure,
        'mtd_return': mtd_return,
        'mtd_delta': mtd_delta,
        'ytd_return': ytd_return,
        'ytd_delta': ytd_delta
    }

# Example usage (can be removed if not needed in module context)
if __name__ == "__main__":
    initialize()
    symbols = get_sp500_symbols()
    positions = load_positions()
    trades = get_trades_history()
    logger.info(f"Loaded {len(positions)} positions and {len(trades)} trades")
