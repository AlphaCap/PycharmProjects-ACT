import os
import pandas as pd
from datetime import datetime, timedelta
import time
import schedule
from utils.polygon_api import PolygonClient
import logging
import json
from config import POLYGON_API_KEY  # Import API key from config

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
        self.api_key = api_key or POLYGON_API_KEY  # Use config API key instead of environment variable
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        
        self.client = PolygonClient(self.api_key)
        self.sp500_symbols = self._get_sp500_symbols()
        
        # Rest of the code remains the same
        
        # Ensure data directories exist
        os.makedirs(os.path.join("data", "daily"), exist_ok=True)
        os.makedirs(os.path.join("data", "minute"), exist_ok=True)
        os.makedirs(os.path.join("data", "trades"), exist_ok=True)
        
        # Load last fetch dates
        self.last_fetch_dates = self._load_last_fetch_dates()
        
    def _get_sp500_symbols(self):
        """Get S&P 500 symbols."""
        # Try to load from a local file first
        sp500_file = os.path.join("data", "sp500_symbols.json")
        try:
            if os.path.exists(sp500_file):
                with open(sp500_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load S&P 500 symbols from file: {e}")
        
        # Fallback to hardcoded sample
        # In production, you would fetch these from an official source
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.A", "UNH", "JNJ"]
    
    def _load_last_fetch_dates(self):
        """Load the last fetch dates from a file."""
        last_fetch_file = os.path.join("data", "last_fetch_dates.json")
        if os.path.exists(last_fetch_file):
            try:
                with open(last_fetch_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading last fetch dates: {e}")
        
        # Default values if file doesn't exist
        return {
            "daily": {
                "full_history_date": None,
                "last_update": None
            },
            "minute": {
                "full_history_date": None,
                "last_update": None
            }
        }
    
    def _save_last_fetch_dates(self):
        """Save the last fetch dates to a file."""
        last_fetch_file = os.path.join("data", "last_fetch_dates.json")
        with open(last_fetch_file, 'w') as f:
            json.dump(self.last_fetch_dates, f, indent=2)
    
    def fetch_daily_data(self, force_full_history=False):
        """
        Fetch daily data for all S&P 500 stocks.
        Maintains a 6-month rolling historical database.
        
        Args:
            force_full_history: If True, fetch full 6 months regardless of last fetch date
        """
        logger.info(f"Starting daily data fetch for {len(self.sp500_symbols)} symbols")
        
        # Determine fetch period
        today = datetime.now()
        # For full history (6 months)
        six_months_ago = (today - timedelta(days=180)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        # Check if we've already fetched full history
        if self.last_fetch_dates["daily"]["full_history_date"] and not force_full_history:
            # We've already fetched full history, just get updates
            last_update = self.last_fetch_dates["daily"]["last_update"]
            # Allow 1-day overlap for data corrections
            if last_update:
                from_date = (datetime.strptime(last_update, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                from_date = six_months_ago
        else:
            # First time or forced full history
            from_date = six_months_ago
            self.last_fetch_dates["daily"]["full_history_date"] = to_date
        
        logger.info(f"Fetching daily data from {from_date} to {to_date}")
        
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
                    
                    # If we're updating existing data, merge with it
                    existing_file = os.path.join("data", "daily", f"{symbol}.csv")
                    if os.path.exists(existing_file) and not force_full_history:
                        try:
                            existing_data = pd.read_csv(existing_file)
                            # Convert timestamp to datetime for merging
                            if 'timestamp' in existing_data.columns:
                                existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
                            if 'timestamp' in symbol_data.columns:
                                symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
                            
                            # Combine data, drop duplicates, and keep only last 180 days
                            combined_data = pd.concat([existing_data, symbol_data]).drop_duplicates(subset=['timestamp'])
                            # Sort by timestamp and keep only last 180 days
                            combined_data = combined_data.sort_values('timestamp')
                            cutoff_date = (today - timedelta(days=180))
                            combined_data = combined_data[combined_data['timestamp'] >= cutoff_date]
                            
                            # Save to CSV
                            combined_data.to_csv(existing_file, index=False)
                            logger.info(f"Updated daily data for {symbol} with {len(symbol_data)} new bars")
                        except Exception as e:
                            logger.error(f"Error updating data for {symbol}: {e}")
                            # If update fails, save the new data
                            symbol_data.to_csv(existing_file, index=False)
                    else:
                        # Save new data
                        symbol_data.to_csv(existing_file, index=False)
                        logger.info(f"Saved daily data for {symbol} with {len(symbol_data)} bars")
                
                # Rate limiting to avoid API throttling
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        # Update last fetch date
        self.last_fetch_dates["daily"]["last_update"] = to_date
        self._save_last_fetch_dates()
        
        logger.info(f"Completed daily data fetch. Retrieved data for {len(all_data)} symbols")
        return all_data
    
    def fetch_minute_data(self, force_full_history=False):
        """
        Fetch 1-minute data for all S&P 500 stocks.
        Maintains a 5-day rolling historical database.
        
        Args:
            force_full_history: If True, fetch full 5 days regardless of last fetch date
        """
        logger.info(f"Starting minute data fetch for {len(self.sp500_symbols)} symbols")
        
        # Determine fetch period
        today = datetime.now()
        # For full history (5 days)
        five_days_ago = (today - timedelta(days=5)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        # Check if we've already fetched full history
        if self.last_fetch_dates["minute"]["full_history_date"] and not force_full_history:
            # We've already fetched full history, just get updates
            last_update = self.last_fetch_dates["minute"]["last_update"]
            # Allow overlap for data corrections
            if last_update:
                from_date = (datetime.strptime(last_update, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                from_date = five_days_ago
        else:
            # First time or forced full history
            from_date = five_days_ago
            self.last_fetch_dates["minute"]["full_history_date"] = to_date
        
        logger.info(f"Fetching minute data from {from_date} to {to_date}")
        
        all_data = {}
        for symbol in self.sp500_symbols:
            try:
                logger.info(f"Fetching minute data for {symbol}")
                # Need to use get_bars with minute timespan
                symbol_data = self.client.get_bars(
                    symbol,
                    timespan="minute",
                    multiplier=1,
                    from_date=from_date,
                    to_date=to_date,
                    limit=50000  # Higher limit for minute data
                )
                
                if not symbol_data.empty:
                    all_data[symbol] = symbol_data
                    
                    # If we're updating existing data, merge with it
                    existing_file = os.path.join("data", "minute", f"{symbol}.csv")
                    if os.path.exists(existing_file) and not force_full_history:
                        try:
                            existing_data = pd.read_csv(existing_file)
                            # Convert timestamp to datetime for merging
                            if 'timestamp' in existing_data.columns:
                                existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
                            if 'timestamp' in symbol_data.columns:
                                symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
                            
                            # Combine data, drop duplicates, and keep only last 5 days
                            combined_data = pd.concat([existing_data, symbol_data]).drop_duplicates(subset=['timestamp'])
                            # Sort by timestamp and keep only last 5 days
                            combined_data = combined_data.sort_values('timestamp')
                            cutoff_date = (today - timedelta(days=5))
                            combined_data = combined_data[combined_data['timestamp'] >= cutoff_date]
                            
                            # Save to CSV
                            combined_data.to_csv(existing_file, index=False)
                            logger.info(f"Updated minute data for {symbol} with {len(symbol_data)} new bars")
                        except Exception as e:
                            logger.error(f"Error updating data for {symbol}: {e}")
                            # If update fails, save the new data
                            symbol_data.to_csv(existing_file, index=False)
                    else:
                        # Save new data
                        symbol_data.to_csv(existing_file, index=False)
                        logger.info(f"Saved minute data for {symbol} with {len(symbol_data)} bars")
                
                # Rate limiting to avoid API throttling
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching minute data for {symbol}: {e}")
        
        # Update last fetch date
        self.last_fetch_dates["minute"]["last_update"] = to_date
        self._save_last_fetch_dates()
        
        logger.info(f"Completed minute data fetch. Retrieved data for {len(all_data)} symbols")
        return all_data
    
    def save_trade(self, trade_data):
        """
        Save a trade to the historical trades database.
        Trade data is appended to preserve historical record.
        
        Args:
            trade_data: Dictionary with trade information
        """
        trades_file = os.path.join("data", "trades", "trades.csv")
        
        # Create trades file with headers if it doesn't exist
        if not os.path.exists(trades_file):
            headers = ["timestamp", "symbol", "side", "price", "quantity", 
                       "strategy", "entry_date", "exit_date", "profit", "comment"]
            pd.DataFrame(columns=headers).to_csv(trades_file, index=False)
        
        # Add timestamp if not present
        if "timestamp" not in trade_data:
            trade_data["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Append trade to CSV
        trade_df = pd.DataFrame([trade_data])
        trade_df.to_csv(trades_file, mode='a', header=False, index=False)
        
        logger.info(f"Trade recorded for {trade_data.get('symbol')} - {trade_data.get('side')}")
        
        return True
    
    def schedule_daily_fetches(self):
        """Schedule data fetching tasks."""
        logger.info("Scheduling data fetching tasks")
        
        # Daily data after market close at 4:30 PM ET
        schedule.every().monday.at("16:30").do(self.fetch_daily_data)
        schedule.every().tuesday.at("16:30").do(self.fetch_daily_data)
        schedule.every().wednesday.at("16:30").do(self.fetch_daily_data)
        schedule.every().thursday.at("16:30").do(self.fetch_daily_data)
        schedule.every().friday.at("16:30").do(self.fetch_daily_data)
        
        # Minute data every hour during trading day
        trading_hours = ["09:35", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:05"]
        for hour in trading_hours:
            schedule.every().monday.at(hour).do(self.fetch_minute_data)
            schedule.every().tuesday.at(hour).do(self.fetch_minute_data)
            schedule.every().wednesday.at(hour).do(self.fetch_minute_data)
            schedule.every().thursday.at(hour).do(self.fetch_minute_data)
            schedule.every().friday.at(hour).do(self.fetch_minute_data)
        
        # Return the scheduler for the caller to run
        return schedule