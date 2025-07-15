from nGS_Revised_Strategy import NGSStrategy
import pandas as pd
from datetime import datetime, timedelta

def backfill_all_symbols():
    """
    Backfill historical data for all symbols using the NGSStrategy.

    This function initializes the strategy and processes data for a list of symbols.
    Customize the symbol list and data source as needed.
    """
    # Initialize the strategy
    strategy = NGSStrategy()

    # Example list of symbols (replace with your data source, e.g., S&P 500 symbols)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # Define date range for backfill (e.g., last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Placeholder for data fetching (replace with your data provider API or file)
    for symbol in symbols:
        # Simulate fetching data (e.g., from a data manager or API)
        # Example: Assume a function get_historical_data exists in data_manager
        try:
            from data_manager import get_historical_data
            data = get_historical_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                # Process data with the strategy
                strategy.backfill_symbol(symbol, data)
                print(f"Backfilled data for {symbol} from {start_date.date()} to {end_date.date()}")
            else:
                print(f"No data available for {symbol}")
        except ImportError:
            print(f"Error: data_manager.get_historical_data not found for {symbol}. Please implement or adjust import.")
        except Exception as e:
            print(f"Error backfilling {symbol}: {e}")

    # Finalize or save results (customize as needed)
    strategy.finalize_backfill()
    print("Backfill process completed for all symbols.")

if __name__ == "__main__":
    backfill_all_symbols()
