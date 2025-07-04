import os
import pandas as pd
from datetime import datetime, timedelta
import time
import schedule
from utils.polygon_api import PolygonClient
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, api_key=None):
        """Initialize with API key and create Polygon client."""
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        
        self.client = PolygonClient(self.api_key)
        self.sp500_symbols = self._get_sp500_symbols()
        
    def _get_sp500_symbols(self):
        """Get S&P 500 symbols - placeholder for actual implementation."""
        # In a real implementation, you would fetch the current S&P 500 components
        # For now, we'll use a small sample
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    def fetch_daily_data(self, days_back=5, save=True):
        """Fetch daily data for all S&P 500 stocks."""
        logger.info(f"Starting daily data fetch for {len(self.sp500_symbols)} symbols")
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        all_data = {}
        for symbol in self.sp500_symbols:
            try:
                logger.info(f"Fetching daily data for {symbol}")
                symbol_data = self.client.get_bars(
                    symbol,
                    timespan="day",
                    multiplier=1,
                    from_date=from_date,
                    to_date=to_date
                )
                
                if not symbol_data.empty:
                    all_data[symbol] = symbol_data
                    
                    # Save individual symbol data
                    if save:
                        self._save_symbol_data(symbol, symbol_data, "daily")
                
                # Rate limiting to avoid API throttling
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        logger.info(f"Completed daily data fetch. Retrieved data for {len(all_data)} symbols")
        return all_data
    
    def fetch_minute_data(self, days_back=1, save=True):
        """Fetch 1-minute data for all S&P 500 stocks for recent days."""
        logger.info(f"Starting minute data fetch for {len(self.sp500_symbols)} symbols")
        
        all_data = {}
        for symbol in self.sp500_symbols:
            try:
                logger.info(f"Fetching minute data for {symbol}")
                symbol_data = self.client.get_minute_bars(symbol, days_back=days_back)
                
                if not symbol_data.empty:
                    all_data[symbol] = symbol_data
                    
                    # Save individual symbol data
                    if save:
                        self._save_symbol_data(symbol, symbol_data, "minute")
                
                # Rate limiting to avoid API throttling
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching minute data for {symbol}: {e}")
        
        logger.info(f"Completed minute data fetch. Retrieved data for {len(all_data)} symbols")
        return all_data
    
    def _save_symbol_data(self, symbol, df, timeframe):
        """Save data for a symbol to CSV."""
        # Create data directory if it doesn't exist
        data_dir = os.path.join("data", timeframe)
        os.makedirs(data_dir, exist_ok=True)
        
        # Save to CSV
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {timeframe} data for {symbol} to {file_path}")

    def schedule_daily_fetch(self, time_str="16:30"):
        """Schedule daily data fetching after market close."""
        logger.info(f"Scheduling daily data fetch at {time_str}")
        schedule.every().monday.at(time_str).do(self.fetch_daily_data)
        schedule.every().tuesday.at(time_str).do(self.fetch_daily_data)
        schedule.every().wednesday.at(time_str).do(self.fetch_daily_data)
        schedule.every().thursday.at(time_str).do(self.fetch_daily_data)
        schedule.every().friday.at(time_str).do(self.fetch_daily_data)
        
        # Return the scheduler for the caller to run
        return schedule

# Example usage for running as a standalone script
if __name__ == "__main__":
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("Please set the POLYGON_API_KEY environment variable")
        exit(1)
    
    fetcher = DataFetcher(api_key)
    
    # Example: Fetch daily data for the past 5 days
    fetcher.fetch_daily_data(days_back=5)
    
    # Example: Set up scheduled fetching and run continuously
    fetcher.schedule_daily_fetch(time_str="16:30")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
