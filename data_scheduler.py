from data_fetcher import DataFetcher
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting data scheduler")
    
    try:
        # Initialize data fetcher
        fetcher = DataFetcher()
        
        # Set up schedule
        schedule = fetcher.schedule_daily_fetches()
        
        logger.info("Scheduler initialized. Running initial data fetch...")
        
        # Run initial fetch
        fetcher.fetch_daily_data()
        fetcher.fetch_minute_data()
        
        logger.info("Initial data fetch complete. Starting scheduled fetches.")
        
        # Keep the script running and check for scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Error in scheduler: {e}")

if __name__ == "__main__":
    main()
