# daily_update.py - Section 1: Imports and Setup
def get_sp500_symbols():
    with open("data/sp500_symbols.txt") as f:
        return [line.strip() for line in f if line.strip()]
import pandas as pd
import numpy as np
import requests
import logging
import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Union

from data_manager import init_metadata, get_sp500_symbols, get_active_symbols, save_price_data, clean_old_data
from nGS_Strategy import NGSStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("daily_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Polygon.io API configuration
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')  # Set your API key as environment variable
if not POLYGON_API_KEY:
    POLYGON_API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"  # Your API key here
    logger.warning("No Polygon API key found. Please set POLYGON_API_KEY environment variable.")
    
# Polygon API base URL
POLYGON_BASE_URL = "https://api.polygon.io"

# Configuration
CONFIG = {
    "max_workers": 8,        # Max parallel downloads
    "history_days": 200,     # Days of history to download
    "batch_size": 100,       # Symbols to process in one batch
    "retry_attempts": 3,     # API retry attempts
    "rate_limit_pause": 12,  # Seconds to pause when rate limited
}

def load_config():
    """Load configuration from file if exists."""
    config_file = "config/system_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Update our config with user values
                for key, value in user_config.items():
                    if key in CONFIG:
                        CONFIG[key] = value
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return CONFIG
# daily_update.py - Section 3: Parallel Download and Symbol Processing
def download_data_parallel(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download data for multiple symbols in parallel."""
    data = {}
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        future_to_symbol = {executor.submit(get_polygon_daily_data, symbol, CONFIG["history_days"]): symbol for symbol in symbols}
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if not df.empty:
                    data[symbol] = df
                    # Save the raw data immediately
                    save_price_data(symbol, df)
            except Exception as e:
                logger.error(f"Error processing {symbol} in parallel download: {e}")
    return data

def update_sp500_list() -> List[str]:
    """Update the SP500 symbols list from a reliable source."""
    try:
        # First try to download from Polygon API's reference data
        if POLYGON_API_KEY:
            try:
                endpoint = f"{POLYGON_BASE_URL}/v3/reference/tickers"
                params = {
                    'market': 'stocks',
                    'active': 'true',
                    'limit': 1000,  # Get maximum allowed
                    'apiKey': POLYGON_API_KEY
                }
                response = requests.get(endpoint, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data:
                        # Filter to likely SP500 symbols
                        # This is a simplification; in real-world we'd need a more precise way
                        # to identify SP500 constituents from Polygon data
                        symbols = []
                        for ticker in data['results']:
                            # Look for US equities traded on major exchanges
                            if (ticker.get('type') == 'CS' and  # Common Stock
                                ticker.get('market') == 'stocks' and
                                ticker.get('primary_exchange') in ['XNYS', 'XNAS']):
                                symbols.append(ticker['ticker'])
                        
                        # If we found a reasonable number of symbols
                        if len(symbols) >= 450:
                            logger.info(f"Retrieved {len(symbols)} potential SP500 symbols from Polygon")
                            pd.DataFrame({"symbol": symbols}).to_csv("data/sp500_symbols.csv", index=False)
                            return symbols
            except Exception as e:
                logger.warning(f"Polygon SP500 retrieval failed: {e}")
        
        # Fallback to Wikipedia
        logger.info("Falling back to Wikipedia for SP500 list")
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
# daily_update.py - Section 4: Main Function and Entry Point
def main():
    """Run the daily update process."""
    logger.info("Starting daily update process")
    
    # Load configuration
    load_config()
    
    # Update SP500 list
    update_sp500_list()
    
    # Get symbols to process
    active_symbols = get_active_symbols()
    all_symbols = get_sp500_symbols()
    
    # Process active symbols first
    logger.info(f"Processing {len(active_symbols)} active symbols")
    active_data = download_data_parallel(active_symbols)
    
    # Process a batch of other symbols
    remaining_symbols = [s for s in all_symbols if s not in active_symbols]
    batch_size = CONFIG["batch_size"]  # Process batch_size inactive symbols per day
    batch_symbols = remaining_symbols[:batch_size]
    logger.info(f"Processing batch of {len(batch_symbols)} inactive symbols")
    batch_data = download_data_parallel(batch_symbols)
    
    # Combine the data
    all_data = {**active_data, **batch_data}
    
    # Run the strategy if we have data
    if all_data:
        strategy = NGSStrategy()
        results = strategy.run(all_data)
        logger.info(f"Strategy run complete with {len(results)} symbols processed")
    else:
        logger.warning("No data available to run strategy")
    
    # Clean old data
    clean_old_data()
    
    logger.info("Daily update process complete")

def download_single_symbol(symbol: str) -> Optional[pd.DataFrame]:
    """Utility function to download data for a single symbol."""
    logger.info(f"Downloading data for {symbol}")
    df = get_polygon_daily_data(symbol, CONFIG["history_days"])
    if not df.empty:
        save_price_data(symbol, df)
        return df
    return None

if __name__ == "__main__":
    # Check if API key is set
    if not POLYGON_API_KEY:
        print("ERROR: Polygon API key not set. Please set the POLYGON_API_KEY environment variable.")
        print("Example: export POLYGON_API_KEY='your_api_key_here'")
        exit(1)
        
    main()