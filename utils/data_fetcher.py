import os
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import json
from utils.polygon_api import PolygonClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_fetcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, api_key=None):
        """Initialize with API key and create Polygon client."""
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key is required")

        self.client = PolygonClient(self.api_key)
        self.sp500_symbols = self._get_sp500_symbols()

        # Ensure data directories exist
        os.makedirs(os.path.join("data", "daily"), exist_ok=True)

        # Load last fetch dates
        self.last_fetch_dates = self._load_last_fetch_dates()

    def _get_sp500_symbols(self):
        """Get S&P 500 symbols."""
        sp500_csv_file = os.path.join("data", "sp500_symbols.csv")
        try:
            if os.path.exists(sp500_csv_file):
                df = pd.read_csv(sp500_csv_file)
                symbol_cols = ["Symbol", "symbol", "ticker", "Ticker"]
                for col in symbol_cols:
                    if col in df.columns:
                        symbols = df[col].dropna().tolist()
                        break
                else:
                    symbols = df.iloc[:, 0].dropna().tolist()
                symbols = [
                    str(symbol).strip().upper()
                    for symbol in symbols
                    if str(symbol).strip()
                ]
                if symbols and len(symbols) > 100:
                    logger.info(f"Loaded {len(symbols)} S&P 500 symbols from CSV file")
                    return symbols
        except Exception as e:
            logger.warning(f"Could not load S&P 500 symbols from CSV file: {e}")

        # Try JSON as backup
        sp500_json_file = os.path.join("data", "sp500_symbols.json")
        try:
            if os.path.exists(sp500_json_file):
                with open(sp500_json_file, "r") as f:
                    symbols = json.load(f)
                    if symbols and len(symbols) > 100:
                        logger.info(
                            f"Loaded {len(symbols)} S&P 500 symbols from JSON file"
                        )
                        return symbols
        except Exception as e:
            logger.warning(f"Could not load S&P 500 symbols from JSON file: {e}")

        logger.info("Using hardcoded S&P 500 symbols list")
        return [
            # Add hardcoded list of symbols here for fallback
            "MSFT",
            "NVDA",
            "AAPL",
            "AMZN",
            "GOOG",
            "META",
            "AVGO",
            "TSLA",
            "BRK.B",
            "V",
            "JNJ",
            "PG",
            "JPM",
        ]

    def _load_last_fetch_dates(self):
        """Load the last fetch dates from a file."""
        last_fetch_file = os.path.join("data", "last_fetch_dates.json")
        if os.path.exists(last_fetch_file):
            try:
                with open(last_fetch_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading last fetch dates: {e}")
        # Default if file doesn't exist
        return {"daily": {"full_history_date": None, "last_update": None}}

    def _save_last_fetch_dates(self):
        """Save the last fetch dates to a file."""
        last_fetch_file = os.path.join("data", "last_fetch_dates.json")
        with open(last_fetch_file, "w") as f:
            json.dump(self.last_fetch_dates, f, indent=2)

    def fetch_daily_data(self, force_full_history=False):
        """
        Fetch daily data for all S&P 500 stocks.
        Maintains a 6-month rolling historical database.

        Args:
            force_full_history: If True, fetch full 6 months regardless of last fetch date
        """
        logger.info(f"Starting daily data fetch for {len(self.sp500_symbols)} symbols")

        today = datetime.now()
        six_months_ago = (today - timedelta(days=180)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")

        # Check if we've already fetched full history
        if (
            self.last_fetch_dates["daily"]["full_history_date"]
            and not force_full_history
        ):
            last_update = self.last_fetch_dates["daily"]["last_update"]
            if last_update:
                from_date = (
                    datetime.strptime(last_update, "%Y-%m-%d") - timedelta(days=1)
                ).strftime("%Y-%m-%d")
            else:
                from_date = six_months_ago
        else:
            from_date = six_months_ago
            self.last_fetch_dates["daily"]["full_history_date"] = to_date

        logger.info(f"Fetching daily data from {from_date} to {to_date}")

        for symbol in self.sp500_symbols:
            try:
                logger.info(f"Fetching daily data for {symbol}")
                symbol_data = self.client.get_daily_bars(symbol, from_date, to_date)
                if not symbol_data.empty:
                    self._update_or_save_data(symbol, symbol_data, "daily", 180)
                # Rate limiting
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        self.last_fetch_dates["daily"]["last_update"] = to_date
        self._save_last_fetch_dates()
        logger.info(f"Completed daily data fetch for {len(self.sp500_symbols)} symbols")

    def _update_or_save_data(self, symbol, new_data, data_type, retention_days):
        """Update or save the fetched data to a CSV file."""
        data_dir = os.path.join("data", data_type)
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        try:
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path)
                if "timestamp" in existing_data.columns:
                    existing_data["timestamp"] = pd.to_datetime(
                        existing_data["timestamp"]
                    )
                if "timestamp" in new_data.columns:
                    new_data["timestamp"] = pd.to_datetime(new_data["timestamp"])
                combined_data = pd.concat([existing_data, new_data]).drop_duplicates(
                    subset=["timestamp"]
                )
                combined_data = combined_data.sort_values("timestamp")
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                combined_data = combined_data[combined_data["timestamp"] >= cutoff_date]
                combined_data.to_csv(file_path, index=False)
                logger.info(
                    f"Updated daily data for {symbol} with {len(new_data)} new bars"
                )
            else:
                new_data.to_csv(file_path, index=False)
                logger.info(
                    f"Saved new daily data for {symbol} with {len(new_data)} bars"
                )
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")


