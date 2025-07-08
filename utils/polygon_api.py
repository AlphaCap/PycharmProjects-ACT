import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# --- HARDCODED POLYGON API KEY ---
POLYGON_API_KEY = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"

logger = logging.getLogger(__name__)

class PolygonClient:
    """Client for interacting with the Polygon.io API (daily data only)."""

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key=None):
        """
        Initialize with API key.

        Args:
            api_key: Polygon.io API key, or None to use hardcoded key
        """
        self.api_key = api_key or POLYGON_API_KEY

        if not self.api_key:
            raise ValueError("No Polygon API key provided")

        self.session = requests.Session()

    def _handle_rate_limit(self, response):
        """Handle rate limiting by waiting if needed."""
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limit hit. Waiting for {retry_after} seconds.")
            time.sleep(retry_after)
            return True
        return False

    def _make_request(self, endpoint, params=None):
        """
        Make a request to the Polygon API.

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            JSON response
        """
        if params is None:
            params = {}

        params['apiKey'] = self.api_key

        url = f"{self.BASE_URL}{endpoint}"
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.session.get(url, params=params)

                if self._handle_rate_limit(response):
                    retry_count += 1
                    continue

                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    return None

                return response.json()

            except Exception as e:
                logger.error(f"Request error: {e}")
                retry_count += 1
                time.sleep(2)

        logger.error(f"Failed after {max_retries} retries")
        return None

    def get_daily_bars(self, symbol, from_date, to_date, limit=5000):
        """
        Get aggregated daily bars for a symbol.

        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results to return

        Returns:
            DataFrame with daily bars data
        """
        # Polygon expects dates in YYYY-MM-DD format
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"

        params = {"limit": limit, "sort": "asc"}
        response_data = self._make_request(endpoint, params)

        if not response_data or response_data.get('status') != 'OK' or 'results' not in response_data:
            logger.error(f"Error getting daily bars for {symbol}: {response_data}")
            return pd.DataFrame()

        bars = response_data['results']
        if not bars:
            logger.info(f"No bars returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(bars)

        column_map = {
            'v': 'volume',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            't': 'timestamp',
            'vw': 'vwap',
            'n': 'transactions'
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Only keep relevant columns for daily bars
        keep_cols = [col for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] if col in df.columns]
        return df[keep_cols]

# Example usage (in your other scripts):
# client = PolygonClient()
# df = client.get_daily_bars("AAPL", "2025-07-01", "2025-07-07")