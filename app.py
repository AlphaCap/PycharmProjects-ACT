import os
import time
from utils.data_fetcher import DataFetcher
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ACT Data Fetcher")
    parser.add_argument("--init", action="store_true", 
                      help="Initialize with full historical data")
    parser.add_argument("--daily", action="store_true",
                      help="Fetch daily data only")
    parser.add_argument("--minute", action="store_true",
                      help="Fetch minute data only")
    parser.add_argument("--continuous", action="store_true",
                      help="Run continuously with scheduled fetches")
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        exit(1)
    
    try:
        # Create data fetcher
        fetcher = DataFetcher(api_key)
        
        # Initial data fetch based on arguments
        if args.init:
            logger.info("Initializing full historical databases")
            fetcher.fetch_daily_data(force_full_history=True)  # 6-month history
            fetcher.fetch_minute_data(force_full_history=True)  # 5-day history
        elif args.daily:
            fetcher.fetch_daily_data()
        elif args.minute:
            fetcher.fetch_minute_data()
        
        # Run continuously if requested
        if args.continuous:
            scheduler = fetcher.schedule_daily_fetches()
            logger.info("Data fetcher scheduled. Running continuously...")
            
            while True:
                scheduler.run_pending()
                time.sleep(60)  # Check every minute
        
    except KeyboardInterrupt:
        logger.info("Data fetcher stopped by user")
    except Exception as e:
        logger.error(f"Error in data fetcher: {e}")