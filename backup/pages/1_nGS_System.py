from data_fetcher import DataFetcher
import time
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scheduler.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    logger.info(
        f"Starting historical data scheduler - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    try:
        # Initialize data fetcher
        fetcher = DataFetcher()

        # Ensure data directories exist
        os.makedirs(os.path.join("data", "daily"), exist_ok=True)
        os.makedirs(os.path.join("data", "minute"), exist_ok=True)

        # Run initial data fetch (both daily and minute)
        logger.info("Starting initial data fetch...")

        # Fetch daily historical data (6 months)
        logger.info("Fetching daily historical data...")
        daily_data = fetcher.fetch_daily_data(force_full_history=True)
        logger.info(
            f"Completed daily data fetch for {len(daily_data) if daily_data else 0} symbols"
        )

        # Fetch minute historical data (5 days)
        logger.info("Fetching minute historical data...")
        minute_data = fetcher.fetch_minute_data(force_full_history=True)
        logger.info(
            f"Completed minute data fetch for {len(minute_data) if minute_data else 0} symbols"
        )

        logger.info("Initial data fetch complete.")

        # Set up schedule and run continuously
        logger.info("Setting up scheduled data fetches...")
        schedule = fetcher.schedule_daily_fetches()

        logger.info("Scheduler running. Press Ctrl+C to stop.")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Error in scheduler: {e}", exc_info=True)


if __name__ == "__main__":
    main()


