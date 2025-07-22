"""
ETF Historical Data Downloader
Downloads 4+ years of historical data for sector ETFs using Polygon API.
Separate from main strategy data for ML optimization purposes.
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Import existing configuration if available
try:
    from polygon_config import POLYGON_API_KEY
    POLYGON_CONFIGURED = True
    print("✅ Using polygon_config for API key")
except ImportError:
    try:
        # Try to get from environment
        import os
        POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
        if POLYGON_API_KEY:
            POLYGON_CONFIGURED = True
            print("✅ Using environment variable for API key")
        else:
            POLYGON_CONFIGURED = False
            print("⚠️  No Polygon API key found. Set POLYGON_API_KEY environment variable.")
    except:
        POLYGON_CONFIGURED = False
        print("⚠️  Polygon API configuration not found")

class ETFHistoricalDownloader:
    """
    Downloads and manages historical ETF data for ML optimization.
    
    Features:
    - Downloads 4+ years of data for sector ETFs
    - Stores in separate directory from live trading data
    - Handles API rate limits and retries
    - Updates existing data incrementally
    - Validates data quality
    """
    
    def __init__(self, data_dir: str = "data/etf_historical", years_back: int = 4):
        self.data_dir = data_dir
        self.years_back = years_back
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Sector ETF mappings (from parameter manager)
        self.sector_etfs = {
            'Technology': 'XLK',
            'Financials': 'XLF',
            'Healthcare': 'XLV',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Industrials': 'XLI',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Consumer Staples': 'XLP',
            'Communication Services': 'XLC'
        }
        
        # Calculate date ranges
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * years_back + 30)  # Extra buffer
        
        # API settings
        self.api_delay = 12.1  # Polygon free tier: 5 calls per minute
        self.max_retries = 3
        self.chunk_size = 1000  # Records per API call
        
        print(f"📊 ETF Historical Downloader initialized")
        print(f"📅 Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"📁 Data directory: {data_dir}")
        print(f"🎯 Target ETFs: {len(self.sector_etfs)}")
        
        if not POLYGON_CONFIGURED:
            print("⚠️  WARNING: Polygon API not configured. Some features may not work.")
    
    def get_etf_filename(self, etf_symbol: str) -> str:
        """Get filename for ETF historical data"""
        return os.path.join(self.data_dir, f"{etf_symbol}_historical.csv")
    
    def load_existing_data(self, etf_symbol: str) -> Optional[pd.DataFrame]:
        """Load existing historical data if available"""
        filename = self.get_etf_filename(etf_symbol)
        
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                print(f"📂 Loaded existing data for {etf_symbol}: {len(df)} records")
                print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
                
                return df
            except Exception as e:
                print(f"❌ Error loading existing data for {etf_symbol}: {e}")
                return None
        else:
            print(f"📭 No existing data found for {etf_symbol}")
            return None
    
    def download_polygon_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Download data from Polygon API
        """
        if not POLYGON_CONFIGURED:
            print(f"❌ Cannot download {symbol} - Polygon API not configured")
            return None
        
        try:
            # Polygon API endpoint for daily aggregates
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            
            params = {
                'apikey': POLYGON_API_KEY,
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000  # Max records per call
            }
            
            print(f"📡 Downloading {symbol} from {start_date} to {end_date}...")
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'results' in data and data['results']:
                    # Convert to DataFrame
                    df = pd.DataFrame(data['results'])
                    
                    # Rename columns to match existing format
                    df = df.rename(columns={
                        't': 'timestamp',
                        'o': 'Open',
                        'h': 'High',
                        'l': 'Low',
                        'c': 'Close',
                        'v': 'Volume'
                    })
                    
                    # Convert timestamp to date
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Select and reorder columns
                    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    
                    # Sort by date
                    df = df.sort_values('Date').reset_index(drop=True)
                    
                    print(f"✅ Downloaded {len(df)} records for {symbol}")
                    
                    return df
                else:
                    print(f"❌ No data returned for {symbol}")
                    return None
            
            elif response.status_code == 429:
                print(f"⏳ Rate limit hit for {symbol}, waiting...")
                time.sleep(60)  # Wait 1 minute for rate limit reset
                return None
            
            else:
                print(f"❌ API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            return None
    
    def update_etf_data(self, etf_symbol: str, force_full_download: bool = False) -> bool:
        """
        Update historical data for a single ETF
        
        Args:
            etf_symbol: ETF ticker (e.g., 'XLK')
            force_full_download: If True, download all data regardless of existing files
        """
        print(f"\n🔄 Updating {etf_symbol}...")
        
        # Load existing data
        existing_df = None if force_full_download else self.load_existing_data(etf_symbol)
        
        if existing_df is not None and not existing_df.empty:
            # Determine what new data we need
            last_date = existing_df['Date'].max()
            
            # If data is recent enough, just update from last date
            if last_date > (datetime.now() - timedelta(days=7)):
                print(f"✅ {etf_symbol} data is up to date (last: {last_date.strftime('%Y-%m-%d')})")
                return True
            
            # Download only missing data
            download_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            download_end = self.end_date.strftime('%Y-%m-%d')
            
            print(f"📈 Updating {etf_symbol} from {download_start}")
            
        else:
            # Download full historical range
            download_start = self.start_date.strftime('%Y-%m-%d')
            download_end = self.end_date.strftime('%Y-%m-%d')
            
            print(f"📈 Full download for {etf_symbol}")
        
        # Download new data
        new_data = self.download_polygon_data(etf_symbol, download_start, download_end)
        
        if new_data is None or new_data.empty:
            print(f"❌ No new data downloaded for {etf_symbol}")
            return False
        
        # Combine with existing data
        if existing_df is not None and not existing_df.empty:
            # Merge datasets
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['Date'])
            combined_df = combined_df.sort_values('Date').reset_index(drop=True)
            
            print(f"📊 Combined data: {len(existing_df)} existing + {len(new_data)} new = {len(combined_df)} total")
        else:
            combined_df = new_data
        
        # Validate data quality
        if not self.validate_data_quality(combined_df, etf_symbol):
            print(f"❌ Data quality check failed for {etf_symbol}")
            return False
        
        # Save updated data
        filename = self.get_etf_filename(etf_symbol)
        try:
            combined_df.to_csv(filename, index=False)
            print(f"💾 Saved {len(combined_df)} records to {filename}")
            
            # Show final date range
            date_range = f"{combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}"
            print(f"📅 Final date range: {date_range}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving data for {etf_symbol}: {e}")
            return False
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate data quality for downloaded ETF data
        """
        if df is None or df.empty:
            print(f"❌ {symbol}: Empty dataset")
            return False
        
        # Check required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ {symbol}: Missing columns: {missing_columns}")
            return False
        
        # Check for reasonable data ranges
        if len(df) < 200:  # At least ~1 year of trading days
            print(f"⚠️  {symbol}: Only {len(df)} records (might be insufficient)")
        
        # Check for obvious data errors
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (df[col] <= 0).any():
                print(f"❌ {symbol}: Invalid prices found in {col}")
                return False
            
            if (df[col] > 10000).any():  # ETFs shouldn't be > $10k
                print(f"⚠️  {symbol}: Unusually high prices in {col}")
        
        # Check High >= Low
        if (df['High'] < df['Low']).any():
            print(f"❌ {symbol}: High < Low detected")
            return False
        
        # Check for reasonable volume
        if (df['Volume'] < 0).any():
            print(f"❌ {symbol}: Negative volume detected")
            return False
        
        print(f"✅ {symbol}: Data quality validation passed")
        return True
    
    def download_all_etfs(self, force_full_download: bool = False) -> Dict[str, bool]:
        """
        Download historical data for all sector ETFs
        
        Args:
            force_full_download: If True, redownload all data regardless of existing files
        """
        print(f"\n🚀 Starting ETF historical data download")
        print(f"📊 ETFs to download: {len(self.sector_etfs)}")
        print(f"🕒 Estimated time: {len(self.sector_etfs) * 0.5:.1f} minutes (with API delays)")
        
        results = {}
        successful_downloads = 0
        
        for i, (sector, etf_symbol) in enumerate(self.sector_etfs.items(), 1):
            print(f"\n📊 Processing {i}/{len(self.sector_etfs)}: {etf_symbol} ({sector})")
            
            try:
                success = self.update_etf_data(etf_symbol, force_full_download)
                results[etf_symbol] = success
                
                if success:
                    successful_downloads += 1
                    print(f"✅ {etf_symbol} completed successfully")
                else:
                    print(f"❌ {etf_symbol} failed")
                
                # Rate limiting - wait between API calls
                if i < len(self.sector_etfs) and POLYGON_CONFIGURED:
                    print(f"⏳ Waiting {self.api_delay:.1f}s for API rate limit...")
                    time.sleep(self.api_delay)
                    
            except Exception as e:
                print(f"❌ Error processing {etf_symbol}: {e}")
                results[etf_symbol] = False
        
        # Summary
        print(f"\n📊 DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"✅ Successful: {successful_downloads}/{len(self.sector_etfs)}")
        print(f"❌ Failed: {len(self.sector_etfs) - successful_downloads}/{len(self.sector_etfs)}")
        
        for etf, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {etf}")
        
        return results
    
    def get_download_summary(self) -> pd.DataFrame:
        """
        Get summary of all downloaded ETF data
        """
        summary_data = []
        
        for sector, etf_symbol in self.sector_etfs.items():
            filename = self.get_etf_filename(etf_symbol)
            
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    summary_data.append({
                        'Sector': sector,
                        'ETF': etf_symbol,
                        'Records': len(df),
                        'Start Date': df['Date'].min().strftime('%Y-%m-%d'),
                        'End Date': df['Date'].max().strftime('%Y-%m-%d'),
                        'Years': round((df['Date'].max() - df['Date'].min()).days / 365.25, 1),
                        'File Size': f"{os.path.getsize(filename) / 1024:.1f} KB",
                        'Status': 'Available'
                    })
                except Exception as e:
                    summary_data.append({
                        'Sector': sector,
                        'ETF': etf_symbol,
                        'Records': 0,
                        'Start Date': 'Error',
                        'End Date': 'Error', 
                        'Years': 0,
                        'File Size': '0 KB',
                        'Status': f'Error: {str(e)[:30]}'
                    })
            else:
                summary_data.append({
                    'Sector': sector,
                    'ETF': etf_symbol,
                    'Records': 0,
                    'Start Date': 'Not Downloaded',
                    'End Date': 'Not Downloaded',
                    'Years': 0,
                    'File Size': '0 KB',
                    'Status': 'Not Downloaded'
                })
        
        return pd.DataFrame(summary_data)
    
    def cleanup_old_data(self, keep_years: int = 5):
        """
        Remove data older than specified years to manage disk space
        """
        cutoff_date = datetime.now() - timedelta(days=365 * keep_years)
        
        for sector, etf_symbol in self.sector_etfs.items():
            filename = self.get_etf_filename(etf_symbol)
            
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Keep only recent data
                    df_filtered = df[df['Date'] >= cutoff_date].copy()
                    
                    if len(df_filtered) < len(df):
                        df_filtered.to_csv(filename, index=False)
                        print(f"🧹 Cleaned {etf_symbol}: {len(df)} → {len(df_filtered)} records")
                    else:
                        print(f"✅ {etf_symbol}: No cleanup needed")
                        
                except Exception as e:
                    print(f"❌ Error cleaning {etf_symbol}: {e}")

# CLI interface and testing
if __name__ == "__main__":
    print("📊 ETF Historical Data Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = ETFHistoricalDownloader(years_back=4)
    
    # Show current status
    print("\n📋 Current ETF Data Status:")
    summary = downloader.get_download_summary()
    print(summary.to_string(index=False))
    
    # Ask user what to do
    print(f"\nOptions:")
    print(f"1. Download missing ETF data only")
    print(f"2. Force full redownload of all ETFs")
    print(f"3. Update existing data (add recent days)")
    print(f"4. Show summary and exit")
    
    if POLYGON_CONFIGURED:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Download only missing ETFs
            missing_etfs = summary[summary['Status'] == 'Not Downloaded']['ETF'].tolist()
            if missing_etfs:
                print(f"\n📥 Downloading {len(missing_etfs)} missing ETFs...")
                for etf in missing_etfs:
                    downloader.update_etf_data(etf)
            else:
                print("✅ No missing ETFs found")
                
        elif choice == "2":
            print(f"\n🔄 Force downloading ALL ETFs (this will take ~{len(downloader.sector_etfs) * 0.5:.1f} minutes)...")
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                downloader.download_all_etfs(force_full_download=True)
            else:
                print("Cancelled")
                
        elif choice == "3":
            print(f"\n📈 Updating all existing ETF data...")
            downloader.download_all_etfs(force_full_download=False)
            
        elif choice == "4":
            print("📋 Summary shown above")
            
        else:
            print("Invalid choice")
    else:
        print("\n⚠️  Polygon API not configured. Cannot download data.")
        print("Set POLYGON_API_KEY environment variable or create polygon_config.py")
    
    # Show final summary
    print(f"\n📊 Final Summary:")
    final_summary = downloader.get_download_summary()
    print(final_summary.to_string(index=False))
