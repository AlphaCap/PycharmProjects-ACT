# OPTION 3: Quick Fix for Original File
# Just replace the load_etf_data method in your existing sector_etf_optimizer.py

# FIND THIS METHOD in sector_etf_optimizer.py (around line 80):
def load_etf_data(self, etf_symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Load price data for ETF symbol
    Uses data_manager if available, fallback otherwise
    """
    try:
        if DATA_MANAGER_AVAILABLE:
            # Use your existing data loading
            df = load_price_data(etf_symbol)
            if df is not None and not df.empty:
                # Ensure Date column is datetime
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                return df
        else:
            # Fallback data loading (you could implement alternative source)
            print(f"⚠️  No data available for {etf_symbol} - data_manager not found")
            return None
            
    except Exception as e:
        print(f"❌ Error loading data for {etf_symbol}: {e}")
        return None

# REPLACE WITH THIS:
def load_etf_data(self, etf_symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Load price data for ETF symbol - FIXED VERSION
    Loads directly from downloaded CSV files
    """
    try:
        # Load directly from CSV files
        filepath = os.path.join("data/etf_historical", f"{etf_symbol}_historical.csv")
        
        if not os.path.exists(filepath):
            print(f"❌ ETF data file not found: {filepath}")
            print(f"💡 Make sure you've run etf_historical_downloader.py first")
            return None
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        if df.empty:
            print(f"❌ Empty data file for {etf_symbol}")
            return None
        
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Filter by date range if provided
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        if len(df) == 0:
            print(f"❌ No data for {etf_symbol} in specified date range")
            return None
        
        # Validate required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns in {etf_symbol}: {missing_columns}")
            return None
        
        print(f"✅ Loaded {etf_symbol}: {len(df)} records ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')})")
        return df[required_columns]
        
    except Exception as e:
        print(f"❌ Error loading {etf_symbol}: {e}")
        return None
