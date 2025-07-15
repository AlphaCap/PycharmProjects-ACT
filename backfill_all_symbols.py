from nGS_Revised_Strategy import NGSStrategy
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging to handle Unicode
logging.basicConfig(level=logging.INFO, encoding='utf-8', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def backfill_all_symbols():
    """
    Backfill historical data for all S&P 500 symbols using the NGSStrategy.

    This function initializes the strategy and processes data for all symbols
    retrieved from the data manager.
    """
    # Initialize the strategy
    strategy = NGSStrategy()

    # Define date range for backfill (e.g., last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Fetch all S&P 500 symbols from data_manager
    try:
        from data_manager import get_sp500_symbols  # Adjust if the function name differs
        symbols = get_sp500_symbols()
        logging.info(f"Retrieved {len(symbols)} S&P 500 symbols for backfill")
    except ImportError:
        logging.error("data_manager.get_sp500_symbols not found. Please implement or adjust import.")
        return
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        return

    # Attempt to import get_historical_data, with fallback if missing
    try:
        from data_manager import get_historical_data
        logging.info("Successfully imported get_historical_data from data_manager")
    except ImportError:
        logging.warning("data_manager.get_historical_data not found. Skipping data fetch.")
        return

    # Process each symbol
    for symbol in symbols:
        try:
            data = get_historical_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                # Process data with the strategy
                strategy.backfill_symbol(symbol, data)
                logging.info(f"✔ Backfilled data for {symbol} from {start_date.date()} to {end_date.date()}")
            else:
                logging.warning(f"No data available for {symbol}")
        except Exception as e:
            logging.error(f"Error backfilling {symbol}: {e}")

    # Check if finalize_backfill exists before calling
    if hasattr(strategy, 'finalize_backfill'):
        strategy.finalize_backfill()
        logging.info("Backfill process finalized.")
    else:
        logging.warning("NGSStrategy has no finalize_backfill method. Skipping finalization.")

if __name__ == "__main__":
    backfill_all_symbols()
