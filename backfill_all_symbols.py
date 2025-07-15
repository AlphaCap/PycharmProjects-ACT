from nGS_Revised_Strategy import NGSStrategy
import pandas as pd
from datetime import datetime, timedelta
import logging
from data_manager import get_sp500_symbols, get_historical_data
from polygon import RESTClient
import os
import backoff

# Configure logging to handle Unicode
logging.basicConfig(level=logging.INFO, encoding='utf-8', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Backoff decorator for rate limit handling
@backoff.on_exception(backoff.expo, (ConnectionError, ValueError), max_tries=8, max_time=60)
def fetch_data_with_backoff(polygon_client, symbol, start_date, end_date):
    return get_historical_data(polygon_client, symbol, start_date, end_date)

def backfill_all_symbols():
    """
    Backfill historical data for all S&P 500 symbols using the NGSStrategy.

    This function initializes the strategy, processes symbols in batches, and handles rate limits.
    """
    # Initialize Polygon client
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if not polygon_api_key:
        logging.error("Polygon API key not configured. Set POLYGON_API_KEY environment variable.")
        return
    polygon_client = RESTClient(polygon_api_key)

    # Initialize the strategy
    strategy = NGSStrategy()

    # Define historical date range (avoid future dates)
    end_date = datetime.now() - timedelta(days=1)  # Yesterday to ensure available data
    start_date = end_date - timedelta(days=30)    # Last 30 days of historical data

    # Fetch all S&P 500 symbols from data_manager
    try:
        symbols = get_sp500_symbols()
        logging.info(f"Retrieved {len(symbols)} S&P 500 symbols for backfill")
    except ImportError:
        logging.error("data_manager.get_sp500_symbols not found. Please ensure data_manager.py is updated.")
        return
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        return

    # Process symbols in batches to manage rate limits
    BATCH_SIZE = 50
    for i in range(0, len(symbols), BATCH_SIZE):
        batch_symbols = symbols[i:i + BATCH_SIZE]
        logging.info(f"Processing batch {i // BATCH_SIZE + 1} of {len(symbols) // BATCH_SIZE + 1}: {batch_symbols[:5]}...")
        
        for symbol in batch_symbols:
            try:
                data = fetch_data_with_backoff(polygon_client, symbol, start_date, end_date)
                if data is not None and not data.empty:
                    strategy.backfill_symbol(symbol, data)
                    logging.info(f"✔ Backfilled data for {symbol} from {start_date.date()} to {end_date.date()}")
                else:
                    logging.warning(f"No data available for {symbol}")
            except Exception as e:
                logging.error(f"Error backfilling {symbol}: {e}")
            # Add a small delay between requests within a batch
            import time
            time.sleep(1)  # 1-second delay to reduce load

        # Larger delay between batches
        time.sleep(10)  # 10-second delay between batches

    # Check if finalize_backfill exists before calling
    if hasattr(strategy, 'finalize_backfill'):
        strategy.finalize_backfill()
        logging.info("Backfill process finalized.")
    else:
        logging.info("NGSStrategy has no finalize_backfill method. No finalization performed.")

if __name__ == "__main__":
    backfill_all_symbols()