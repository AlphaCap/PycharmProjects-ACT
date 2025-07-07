import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import os

# Import from config file if available, otherwise use environment variable
try:
    from utils.config import POLYGON_API_KEY as DEFAULT_API_KEY
except ImportError:
    DEFAULT_API_KEY = None

logger = logging.getLogger(__name__)

class PolygonClient:
    """Client for interacting with the Polygon.io API."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key=None):
        """
        Initialize with API key.
        
        Args:
            api_key: Polygon.io API key, or None to use default from config
        """
        # Try to use provided key, then config file, then environment variable
        self.api_key = api_key or DEFAULT_API_KEY or os.environ.get('POLYGON_API_KEY')
        
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
        
        # Add API key to params
        params['apiKey'] = self.api_key
        
        # Make request
        url = f"{self.BASE_URL}{endpoint}"
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.session.get(url, params=params)
                
                # Handle rate limiting
                if self._handle_rate_limit(response):
                    retry_count += 1
                    continue
                
                # Check for errors
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
    
    def get_bars(self, symbol, multiplier=1, timespan="day", 
                 from_date=None, to_date=None, limit=5000):
        """
        Get aggregated bars for a symbol.
        
        Args:
            symbol: Stock symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Number of results to return
            
        Returns:
            DataFrame with bars data
        """
        # Format dates if provided
        if from_date:
            from_date = from_date.replace("-", "")
        if to_date:
            to_date = to_date.replace("-", "")
        
        # Build endpoint
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        # Make request
        params = {"limit": limit, "sort": "asc"}
        response_data = self._make_request(endpoint, params)
        
        if not response_data or response_data.get('status') != 'OK' or 'results' not in response_data:
            logger.error(f"Error getting bars for {symbol}: {response_data}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        bars = response_data['results']
        if not bars:
            logger.info(f"No bars returned for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        
        # Rename columns to be more readable
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
        
        # Convert timestamp from milliseconds to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def get_ticker_details(self, symbol):
        """
        Get details for a ticker symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{symbol}"
        response_data = self._make_request(endpoint)
        
        if not response_data or 'results' not in response_data:
            logger.error(f"Error getting ticker details for {symbol}: {response_data}")
            return {}
        
        return response_data['results']
    
    def get_news(self, symbol, limit=10):
        """
        Get news articles for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Number of news articles to return
            
        Returns:
            List of news articles
        """
        endpoint = "/v2/reference/news"
        params = {"ticker": symbol, "limit": limit, "sort": "published_utc"}
        response_data = self._make_request(endpoint, params)
        
        if not response_data or 'results' not in response_data:
            logger.error(f"Error getting news for {symbol}: {response_data}")
            return []
        
        return response_data['results']
    
    def get_last_quote(self, symbol):
        """
        Get the last quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote information
        """
        endpoint = f"/v2/last/nbbo/{symbol}"
        response_data = self._make_request(endpoint)
        
        if not response_data or 'results' not in response_data:
            logger.error(f"Error getting last quote for {symbol}: {response_data}")
            return {}
        
        return response_data['results']
    
    def get_minute_bars(self, symbol, days_back=1):
        """
        Helper method to get minute bars for the last X days.
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            DataFrame with minute bars
        """
        today = datetime.now()
        from_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = today.strftime('%Y-%m-%d')
        
        return self.get_bars(
            symbol=symbol,
            multiplier=1,
            timespan="minute",
            from_date=from_date,
            to_date=to_date,
            limit=10000  # Higher limit for minute data
        )
