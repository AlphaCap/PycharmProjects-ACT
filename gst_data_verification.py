"""
gSTDayTrader Data Verification Script
- Fixed gap thresholds (0.2% to 0.8%)
- Pull 100 symbols with 1-minute data
- Display all indicators in column format
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json
import os

class GSTDataVerifier:
    """Data verification for gSTDayTrader with corrected gap thresholds"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # CORRECTED GAP THRESHOLDS
        self.min_gap_threshold = 0.002  # 0.2% minimum gap (was 2%)
        self.max_gap_threshold = 0.008  # 0.8% maximum gap (was 8%)
        
        # Other parameters
        self.min_volume_threshold = 500000
        self.min_price = 20
        self.max_price = 1000
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # API rate limiting with batching
        self.requests_made = 0
        self.max_requests_per_batch = 10  # 10 requests per batch
        self.max_daily_requests = 25      # Free tier daily limit
        self.batch_delay = 60             # 60 seconds between batches
        self.request_delay = 12           # 12 seconds between individual requests
        
        print(f"📊 GSTDataVerifier initialized")
        print(f"🎯 Gap thresholds: {self.min_gap_threshold:.1%} to {self.max_gap_threshold:.1%}")
        print(f"📈 Volume threshold: {self.min_volume_threshold:,}")
        print(f"💰 Price range: ${self.min_price} - ${self.max_price}")
    
    def get_top_100_symbols(self) -> List[str]:
        """Get top 100 liquid symbols for gap trading"""
        return [
            # Mega caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH',
            'JNJ', 'XOM', 'V', 'PG', 'JPM', 'MA', 'HD', 'CVX', 'LLY', 'ABBV',
            
            # Large caps
            'AVGO', 'WMT', 'BAC', 'ORCL', 'KO', 'PFE', 'TMO', 'COST', 'MRK', 'ABT',
            'ACN', 'CSCO', 'DHR', 'VZ', 'ADBE', 'NKE', 'TXN', 'DIS', 'CRM', 'QCOM',
            'BMY', 'LIN', 'PM', 'NEE', 'RTX', 'HON', 'T', 'NFLX', 'UPS', 'LOW',
            
            # Mid caps with high volume
            'SPGI', 'GS', 'DE', 'MDT', 'INTC', 'CAT', 'AMD', 'BLK', 'ELV', 'SBUX',
            'AMT', 'PLD', 'BKNG', 'AXP', 'CVS', 'TJX', 'GILD', 'MDLZ', 'ADP', 'CI',
            'CB', 'MMC', 'ISRG', 'SYK', 'ZTS', 'MO', 'SO', 'PGR', 'DUK', 'ITW',
            
            # High volume trading stocks
            'NOC', 'UBER', 'PYPL', 'SQ', 'ROKU', 'ZM', 'SNAP', 'F', 'GM', 'GE',
            'MU', 'IBM', 'ORCL', 'CRM', 'NFLX', 'DIS', 'BABA', 'JD', 'NIO', 'XPEV',
            'LI', 'PLTR', 'COIN', 'HOOD', 'SOFI', 'RIVN', 'LCID', 'CHPT', 'TLRY', 'SNDL'
        ]
    
    def get_intraday_data(self, symbol: str, interval: str = "1min") -> Optional[pd.DataFrame]:
        """Get 1-minute intraday data with rate limiting"""
        if self.requests_made >= self.max_daily_requests:
            self.logger.warning(f"Daily API limit reached ({self.requests_made}/{self.max_daily_requests})")
            return None
            
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            self.requests_made += 1
            
            if 'Error Message' in data:
                self.logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                self.logger.warning(f"API Limit for {symbol}: {data['Note']}")
                return None
                
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                self.logger.warning(f"No time series data for {symbol}")
                return None
                
            time_series = data[time_series_key]
            
            if not time_series:
                self.logger.warning(f"Empty time series for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.sort_index(inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Add delay for rate limiting
            time.sleep(self.request_delay)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous close price with rate limiting"""
        if self.requests_made >= self.max_daily_requests:
            return None
            
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            self.requests_made += 1
            
            if 'Time Series (Daily)' in data:
                daily_data = data['Time Series (Daily)']
                if daily_data:
                    latest_date = sorted(daily_data.keys(), reverse=True)[0]
                    return float(daily_data[latest_date]['4. close'])
                    
        except Exception as e:
            self.logger.error(f"Error getting previous close for {symbol}: {e}")
            
        return None
    
    def calculate_indicators(self, df: pd.DataFrame, symbol: str, previous_close: float) -> pd.DataFrame:
        """Calculate all indicators and gap analysis"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Basic indicators
        df['prev_close'] = previous_close
        df['gap_pct'] = ((df['open'] - previous_close) / previous_close).round(4)
        df['gap_direction'] = df['gap_pct'].apply(lambda x: 'up' if x > 0 else 'down' if x < 0 else 'flat')
        
        # Price ranges and spreads
        df['spread'] = (df['high'] - df['low']).round(2)
        df['body'] = abs(df['close'] - df['open']).round(2)
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)).round(2)
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']).round(2)
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean().round(2)
        df['sma_10'] = df['close'].rolling(10).mean().round(2)
        df['sma_20'] = df['close'].rolling(20).mean().round(2)
        
        # Volatility indicators
        df['price_change'] = (df['close'] - df['close'].shift(1)).round(2)
        df['price_change_pct'] = (df['price_change'] / df['close'].shift(1) * 100).round(2)
        df['volatility_5min'] = df['price_change_pct'].rolling(5).std().round(2)
        
        # Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(10).mean().round(0)
        df['volume_ratio'] = (df['volume'] / df['volume_sma_10']).round(2)
        
        # Gap classification
        df['gap_size'] = df['gap_pct'].abs()
        df['gap_valid'] = (
            (df['gap_size'] >= self.min_gap_threshold) & 
            (df['gap_size'] <= self.max_gap_threshold)
        )
        
        # Trading signals (simplified)
        df['signal'] = 'none'
        gap_up_condition = (df['gap_pct'] > self.min_gap_threshold) & (df['gap_pct'] <= self.max_gap_threshold)
        gap_down_condition = (df['gap_pct'] < -self.min_gap_threshold) & (df['gap_pct'] >= -self.max_gap_threshold)
        
        df.loc[gap_up_condition, 'signal'] = 'short'  # Gap up -> short (expect fill)
        df.loc[gap_down_condition, 'signal'] = 'long'   # Gap down -> long (expect fill)
        
        # Risk metrics
        df['stop_loss_long'] = (df['close'] * 0.99).round(2)   # 1% stop loss
        df['stop_loss_short'] = (df['close'] * 1.01).round(2)  # 1% stop loss
        df['target_long'] = (df['close'] * 1.02).round(2)      # 2% target
        df['target_short'] = (df['close'] * 0.98).round(2)     # 2% target
        
        return df
    
    def process_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Process a single symbol and return data with indicators"""
        print(f"📊 Processing {symbol}...", end=" ")
        
        # Get previous close (uses 1 API call)
        previous_close = self.get_previous_close(symbol)
        if previous_close is None:
            print(f"❌ No previous close")
            return None
        
        # Get intraday data (uses 1 API call)
        df = self.get_intraday_data(symbol)
        if df is None:
            print(f"❌ No intraday data")
            return None
        
        # Calculate indicators
        df_with_indicators = self.calculate_indicators(df, symbol, previous_close)
        
        print(f"✅ {len(df_with_indicators)} points, gap: {df_with_indicators.iloc[0]['gap_pct']:.3%}")
        
        return df_with_indicators
    
    def verify_data_pull_batched(self, max_symbols: int = 100) -> pd.DataFrame:
        """Verify data pull for 100 symbols using batch processing"""
        symbols = self.get_top_100_symbols()[:max_symbols]
        
        print(f"\n🚀 Starting BATCHED data verification for {len(symbols)} symbols")
        print(f"📦 Batch size: {self.max_requests_per_batch} symbols per batch")
        print(f"⏱️  Batch delay: {self.batch_delay} seconds between batches")
        print(f"📡 Daily API limit: {self.max_daily_requests} requests")
        print("=" * 80)
        
        all_data = []
        successful = 0
        batch_num = 1
        
        # Process in batches
        for i in range(0, len(symbols), self.max_requests_per_batch):
            batch_symbols = symbols[i:i + self.max_requests_per_batch]
            batch_start_requests = self.requests_made
            
            print(f"\n📦 BATCH {batch_num} ({len(batch_symbols)} symbols)")
            print(f"🎯 Symbols: {', '.join(batch_symbols)}")
            print(f"📊 API requests used so far: {self.requests_made}/{self.max_daily_requests}")
            print("-" * 60)
            
            # Process batch
            batch_successful = 0
            for j, symbol in enumerate(batch_symbols, 1):
                print(f"[{i+j:3}/{len(symbols)}] ", end="")
                
                # Check if we're approaching daily limit
                if self.requests_made >= self.max_daily_requests - 2:  # Leave 2 requests buffer
                    print(f"⚠️ Approaching daily API limit, stopping at {symbol}")
                    break
                
                df = self.process_symbol_data(symbol)
                if df is not None:
                    # Take last 3 rows for verification
                    recent_data = df.tail(3).copy()
                    all_data.append(recent_data)
                    successful += 1
                    batch_successful += 1
            
            batch_requests_used = self.requests_made - batch_start_requests
            print(f"\n✅ Batch {batch_num} complete: {batch_successful}/{len(batch_symbols)} successful")
            print(f"📊 API requests used in batch: {batch_requests_used}")
            
            # Check if we should continue
            if self.requests_made >= self.max_daily_requests - 2:
                print(f"\n⚠️ Stopping: Near daily API limit ({self.requests_made}/{self.max_daily_requests})")
                break
            
            # Delay between batches (except for last batch)
            if i + self.max_requests_per_batch < len(symbols):
                print(f"⏳ Waiting {self.batch_delay} seconds before next batch...")
                
                # Show countdown
                for remaining in range(self.batch_delay, 0, -10):
                    print(f"   ⏰ {remaining} seconds remaining...", end="\r")
                    time.sleep(10)
                print(f"   ✅ Ready for next batch!      ")
            
            batch_num += 1
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n🎉 BATCHED DATA PULL COMPLETE!")
            print(f"✅ Total successful: {successful}/{len(symbols)} symbols")
            print(f"📊 API requests used: {self.requests_made}/{self.max_daily_requests}")
            print(f"📦 Batches processed: {batch_num - 1}")
            return combined_df
        else:
            print(f"\n❌ No data retrieved")
            return pd.DataFrame()
    
    def display_data_summary(self, df: pd.DataFrame):
        """Display data in column format for verification"""
        if df.empty:
            print("No data to display")
            return
        
        print(f"\n📊 DATA VERIFICATION SUMMARY")
        print("=" * 80)
        
        # Key columns to display
        display_cols = [
            'symbol', 'open', 'high', 'low', 'close', 'volume',
            'prev_close', 'gap_pct', 'gap_direction', 'gap_valid', 'signal',
            'sma_5', 'sma_10', 'spread', 'volume_ratio'
        ]
        
        # Filter to available columns
        available_cols = [col for col in display_cols if col in df.columns]
        display_df = df[available_cols].copy()
        
        # Format for better display
        if 'gap_pct' in display_df.columns:
            display_df['gap_pct'] = (display_df['gap_pct'] * 100).round(2)  # Convert to percentage
        
        # Display by symbol
        for symbol in display_df['symbol'].unique():
            symbol_data = display_df[display_df['symbol'] == symbol].tail(3)  # Last 3 rows per symbol
            
            print(f"\n🔍 {symbol}:")
            print("-" * 60)
            
            for _, row in symbol_data.iterrows():
                print(f"  Time: {row.name if hasattr(row, 'name') else 'N/A'}")
                print(f"  OHLC: ${row['open']:.2f} / ${row['high']:.2f} / ${row['low']:.2f} / ${row['close']:.2f}")
                print(f"  Volume: {row['volume']:,.0f}")
                print(f"  Gap: {row['gap_pct']:.2f}% ({row['gap_direction']}) - Valid: {row['gap_valid']}")
                print(f"  Signal: {row['signal']}")
                print(f"  SMA5/10: ${row['sma_5']:.2f} / ${row['sma_10']:.2f}")
                print(f"  Spread: ${row['spread']:.2f}, Vol Ratio: {row['volume_ratio']:.1f}")
                print()
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print("=" * 50)
        print(f"Total data points: {len(display_df)}")
        print(f"Unique symbols: {display_df['symbol'].nunique()}")
        print(f"Gap range: {display_df['gap_pct'].min():.2f}% to {display_df['gap_pct'].max():.2f}%")
        print(f"Valid gaps: {display_df['gap_valid'].sum()}")
        print(f"Long signals: {(display_df['signal'] == 'long').sum()}")
        print(f"Short signals: {(display_df['signal'] == 'short').sum()}")
        
        # Show columns available
        print(f"\n📋 AVAILABLE COLUMNS:")
        print("=" * 50)
        for i, col in enumerate(display_df.columns, 1):
            print(f"{i:2}. {col}")

def main():
    """Main verification function with batch processing"""
    api_key = "D4NJ9SDT2NS2L6UX"
    verifier = GSTDataVerifier(api_key)
    
    print("🔍 gSTDayTrader Data Verification - BATCH MODE")
    print("=" * 60)
    print("📋 Checking:")
    print("   ✓ Corrected gap thresholds (0.2% - 0.8%)")
    print("   ✓ 1-minute data retrieval")
    print("   ✓ All indicators calculation")
    print("   ✓ Column format display")
    print("   ✓ Batch processing for 100 symbols")
    
    print(f"\n🔧 BATCH CONFIGURATION:")
    print(f"   📦 Batch size: {verifier.max_requests_per_batch} symbols per batch")
    print(f"   ⏱️  Batch delay: {verifier.batch_delay} seconds between batches")
    print(f"   📡 Daily API limit: {verifier.max_daily_requests} requests")
    print(f"   🚀 For 100 symbols: ~10 batches over ~10 minutes")
    
    # Get user choice
    try:
        print("\nChoose verification mode:")
        print("1. Quick test (5 symbols)")
        print("2. Medium test (25 symbols)")
        print("3. Full test (100 symbols in batches)")
        
        choice = input("Enter choice (1-3, default 1): ").strip() or "1"
        
        if choice == "1":
            max_symbols = 5
            use_batching = False
        elif choice == "2":
            max_symbols = 25
            use_batching = True
        elif choice == "3":
            max_symbols = 100
            use_batching = True
        else:
            max_symbols = 5
            use_batching = False
            
    except (KeyboardInterrupt, EOFError):
        print("\nTest cancelled by user")
        return
    
    # Run verification
    if use_batching and max_symbols > 10:
        df = verifier.verify_data_pull_batched(max_symbols)
    else:
        # Use original method for small tests
        symbols = verifier.get_top_100_symbols()[:max_symbols]
        print(f"\n🚀 Starting quick verification for {len(symbols)} symbols")
        
        all_data = []
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i:2}/{len(symbols)}] ", end="")
            df_symbol = verifier.process_symbol_data(symbol)
            if df_symbol is not None:
                all_data.append(df_symbol.tail(3))
        
        df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    # Display results
    if not df.empty:
        verifier.display_data_summary(df)
        
        # Save verification results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data_verification_{max_symbols}symbols_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Verification data saved to: {filename}")
        
        # Show gap analysis
        valid_gaps = df[df['gap_valid'] == True]
        if not valid_gaps.empty:
            print(f"\n🎯 VALID GAPS FOUND:")
            print("-" * 40)
            for _, row in valid_gaps.iterrows():
                print(f"   {row['symbol']}: {row['gap_pct']:.2f}% {row['gap_direction']} - Signal: {row['signal']}")
        else:
            print(f"\n📊 No valid gaps found (range: {verifier.min_gap_threshold:.1%} - {verifier.max_gap_threshold:.1%})")
        
    else:
        print("\n❌ Verification failed - no data retrieved")
    
    print(f"\n📊 FINAL API USAGE: {verifier.requests_made}/{verifier.max_daily_requests} requests used")
    remaining = verifier.max_daily_requests - verifier.requests_made
    print(f"📈 Remaining requests today: {remaining}")
    
    if remaining > 0:
        print(f"💡 You can process ~{remaining//2} more symbols today (2 API calls per symbol)")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> c3b6458da22b719f87e33e86cea00599c7a24a0f
