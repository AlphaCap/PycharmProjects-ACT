"""
Enhanced Data Fetcher for gSTDayTrader
Handles batch fetching of 100+ symbols with rate limiting and caching
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class GSTEnhancedFetcher:
    """Enhanced data fetcher for gap trading with 100+ symbols"""
    
    def __init__(self, api_key: str, cache_dir: str = 'data/cache'):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting for free tier (25 requests/day)
        self.requests_made = 0
        self.max_requests = 20  # Leave some buffer
        self.request_delay = 12  # 12 seconds between requests (5 per minute max)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_db = os.path.join(cache_dir, 'gst_cache.db')
        self._init_cache_db()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_prices (
                symbol TEXT,
                date TEXT,
                close REAL,
                volume INTEGER,
                last_updated TEXT,
                PRIMARY KEY (symbol, date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intraday_data (
                symbol TEXT,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                last_updated TEXT,
                PRIMARY KEY (symbol, datetime)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_previous_close_from_cache(self, symbol: str) -> Optional[float]:
        """Load previous close from cache if recent"""
        conn = sqlite3.connect(self.cache_db)
        
        try:
            # Look for data from last 2 days
            cutoff_time = datetime.now() - timedelta(hours=48)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT close FROM daily_prices 
                WHERE symbol = ? AND last_updated > ?
                ORDER BY date DESC LIMIT 1
            ''', (symbol, cutoff_str))
            
            result = cursor.fetchone()
            if result:
                self.logger.info(f"Loaded {symbol} previous close from cache: ${result[0]:.2f}")
                return float(result[0])
                
        except Exception as e:
            self.logger.error(f"Cache load error for {symbol}: {e}")
        finally:
            conn.close()
        
        return None
    
    def _save_previous_close_to_cache(self, symbol: str, close_price: float):
        """Save previous close to cache"""
        conn = sqlite3.connect(self.cache_db)
        
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            today = datetime.now().strftime('%Y-%m-%d')
            
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO daily_prices 
                (symbol, date, close, volume, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, today, close_price, 0, now))
            
            conn.commit()
            self.logger.info(f"Cached {symbol} previous close: ${close_price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Cache save error for {symbol}: {e}")
        finally:
            conn.close()
    
    def _load_intraday_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load intraday data from cache if recent"""
        conn = sqlite3.connect(self.cache_db)
        
        try:
            # Look for data from last 4 hours
            cutoff_time = datetime.now() - timedelta(hours=4)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            query = '''
                SELECT datetime, open, high, low, close, volume 
                FROM intraday_data 
                WHERE symbol = ? AND last_updated > ?
                ORDER BY datetime
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, cutoff_str))
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                self.logger.info(f"Loaded {symbol} intraday from cache ({len(df)} rows)")
                return df
                
        except Exception as e:
            self.logger.error(f"Cache load error for {symbol}: {e}")
        finally:
            conn.close()
        
        return None
    
    def _save_intraday_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save intraday data to cache"""
        conn = sqlite3.connect(self.cache_db)
        
        try:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Clear old data for this symbol
            cursor = conn.cursor()
            cursor.execute('DELETE FROM intraday_data WHERE symbol = ?', (symbol,))
            
            # Insert new data
            for timestamp, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO intraday_data 
                    (symbol, datetime, open, high, low, close, volume, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    float(row['open']), float(row['high']), float(row['low']),
                    float(row['close']), int(row['volume']), now
                ))
            
            conn.commit()
            self.logger.info(f"Cached {symbol} intraday data ({len(df)} rows)")
            
        except Exception as e:
            self.logger.error(f"Cache save error for {symbol}: {e}")
        finally:
            conn.close()
    
    def get_previous_close_smart(self, symbol: str) -> Optional[float]:
        """Get previous close with smart caching"""
        # Try cache first
        cached_close = self._load_previous_close_from_cache(symbol)
        if cached_close is not None:
            return cached_close
        
        # Check if we've hit API limit
        if self.requests_made >= self.max_requests:
            self.logger.warning(f"API limit reached ({self.requests_made}/{self.max_requests}), using fallback")
            return None
        
        # Fetch from API
        self.logger.info(f"Fetching {symbol} previous close from API...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            self.requests_made += 1
            
            if 'Time Series (Daily)' in data:
                daily_data = data['Time Series (Daily)']
                
                if daily_data:
                    # Get most recent close
                    latest_date = sorted(daily_data.keys(), reverse=True)[0]
                    close_price = float(daily_data[latest_date]['4. close'])
                    
                    # Cache the result
                    self._save_previous_close_to_cache(symbol, close_price)
                    
                    # Add delay to respect rate limits
                    time.sleep(self.request_delay)
                    
                    return close_price
            
            elif 'Note' in data:
                self.logger.warning(f"API limit hit for {symbol}: {data['Note']}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error fetching previous close for {symbol}: {e}")
        
        return None
    
    def get_intraday_data_smart(self, symbol: str, interval: str = "1min") -> Optional[pd.DataFrame]:
        """Get intraday data with smart caching"""
        # Try cache first
        cached_df = self._load_intraday_from_cache(symbol)
        if cached_df is not None and len(cached_df) > 50:
            return cached_df
        
        # Check if we've hit API limit
        if self.requests_made >= self.max_requests:
            self.logger.warning(f"API limit reached ({self.requests_made}/{self.max_requests}), using cached data only")
            return cached_df  # Return whatever cache we have
        
        # Fetch from API
        self.logger.info(f"Fetching {symbol} intraday from API...")
        
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            self.requests_made += 1
            
            time_series_key = f'Time Series ({interval})'
            if time_series_key in data:
                time_series = data[time_series_key]
                
                if time_series:
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(time_series, orient='index')
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    df.index = pd.to_datetime(df.index)
                    df = df.astype(float)
                    df.sort_index(inplace=True)
                    
                    # Cache the result
                    self._save_intraday_to_cache(symbol, df)
                    
                    # Add delay to respect rate limits
                    time.sleep(self.request_delay)
                    
                    return df
            
            elif 'Note' in data:
                self.logger.warning(f"API limit hit for {symbol}: {data['Note']}")
                return cached_df  # Return cached data if available
            
        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {e}")
        
        return cached_df  # Return cached data if API fails
    
    def batch_get_symbols_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Batch get data for multiple symbols with smart rate limiting"""
        results = {}
        
        self.logger.info(f"Starting batch fetch for {len(symbols)} symbols")
        self.logger.info(f"API requests available: {self.max_requests - self.requests_made}")
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Processing {i}/{len(symbols)}: {symbol}")
            
            try:
                # Get previous close
                previous_close = self.get_previous_close_smart(symbol)
                
                # Get intraday data
                intraday_df = self.get_intraday_data_smart(symbol)
                
                results[symbol] = {
                    'previous_close': previous_close,
                    'intraday_data': intraday_df,
                    'has_data': previous_close is not None and intraday_df is not None
                }
                
                # Stop if we've hit our API limit
                if self.requests_made >= self.max_requests:
                    self.logger.warning(f"API limit reached after {i} symbols")
                    break
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {
                    'previous_close': None,
                    'intraday_data': None,
                    'has_data': False
                }
        
        successful = sum(1 for r in results.values() if r['has_data'])
        self.logger.info(f"Batch fetch complete: {successful}/{len(results)} symbols with data")
        
        return results
    
    def get_top_100_symbols(self) -> List[str]:
        """Get top 100 symbols for gap trading (focus on high-volume stocks)"""
        # Top 100 most liquid stocks for gap trading
        symbols = [
            # Mega caps (>$500B)
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            
            # Large caps ($100B-$500B)
            'BRK.B', 'UNH', 'JNJ', 'XOM', 'V', 'PG', 'JPM', 'MA', 'HD', 'CVX',
            'LLY', 'ABBV', 'AVGO', 'WMT', 'BAC', 'ORCL', 'KO', 'PFE', 'TMO',
            'COST', 'MRK', 'ABT', 'ACN', 'CSCO', 'DHR', 'VZ', 'ADBE', 'NKE',
            'TXN', 'DIS', 'CRM', 'QCOM', 'BMY', 'LIN', 'PM', 'NEE', 'RTX',
            
            # Mid caps with high volume
            'HON', 'T', 'NFLX', 'UPS', 'LOW', 'SPGI', 'GS', 'DE', 'MDT',
            'INTC', 'CAT', 'AMD', 'BLK', 'ELV', 'SBUX', 'AMT', 'PLD', 'BKNG',
            'AXP', 'CVS', 'TJX', 'GILD', 'MDLZ', 'ADP', 'CI', 'CB', 'MMC',
            'ISRG', 'SYK', 'ZTS', 'MO', 'SO', 'PGR', 'DUK', 'ITW', 'NOC',
            
            # Popular trading stocks
            'UBER', 'PYPL', 'SQ', 'ROKU', 'ZM', 'PTON', 'SNAP', 'TWTR',
            'F', 'GM', 'GE', 'MU', 'INTC', 'IBM', 'ORCL', 'CRM', 'NFLX',
            'DIS', 'BABA', 'JD', 'NIO', 'XPEV', 'LI'
        ]
        
        return symbols[:100]  # Return exactly 100 symbols
    
    def cleanup_old_cache(self, days_old: int = 3):
        """Clean up cache entries older than specified days"""
        conn = sqlite3.connect(self.cache_db)
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
            
            cursor = conn.cursor()
            
            # Clean daily prices
            cursor.execute('DELETE FROM daily_prices WHERE last_updated < ?', (cutoff_str,))
            daily_deleted = cursor.rowcount
            
            # Clean intraday data
            cursor.execute('DELETE FROM intraday_data WHERE last_updated < ?', (cutoff_str,))
            intraday_deleted = cursor.rowcount
            
            conn.commit()
            
            self.logger.info(f"Cache cleanup: {daily_deleted} daily, {intraday_deleted} intraday records deleted")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {e}")
        finally:
            conn.close()

# Example usage
if __name__ == "__main__":
    # Test the enhanced fetcher
    api_key = "D4NJ9SDT2NS2L6UX"
    fetcher = GSTEnhancedFetcher(api_key)
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'TSLA', 'MSFT']
    
    print("Testing Enhanced Fetcher...")
    results = fetcher.batch_get_symbols_data(test_symbols)
    
    for symbol, data in results.items():
        print(f"{symbol}: {'✅' if data['has_data'] else '❌'}")
        if data['has_data']:
            print(f"  Previous close: ${data['previous_close']:.2f}")
            print(f"  Intraday points: {len(data['intraday_data'])}")

    print(f"API requests used: {fetcher.requests_made}")