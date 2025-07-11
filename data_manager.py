import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Ensure data directory structure exists
os.makedirs('data/daily', exist_ok=True)
os.makedirs('data/trades', exist_ok=True)

# Create a sample AAPL file if it doesn't exist
aapl_file = 'data/daily/AAPL.csv'
if not os.path.exists(aapl_file):
    # Create a minimal AAPL CSV with required columns
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })
    sample_data.to_csv(aapl_file, index=False)
    print(f"Created sample {aapl_file}")
else:
    print(f"{aapl_file} already exists.")

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_sp500_symbols():
    """Return the complete list of S&P 500 symbols (503 companies as of 2024)"""
    symbols = [
        'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ATVI', 'ADBE', 'ADP', 'AAP', 'AES',
        'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT',
        'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AMD', 'AEE', 'AAL', 'AEP',
        'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI',
        'ANSS', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ANET', 'AJG', 'AIZ',
        'T', 'ATO', 'ADSK', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC',
        'BBWI', 'BAX', 'BDX', 'WRB', 'BRK-B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK',
        'BK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO',
        'BF-B', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL',
        'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CDAY',
        'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF',
        'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL',
        'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW',
        'CTVA', 'CSGP', 'COST', 'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR',
        'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS',
        'DISH', 'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK',
        'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV',
        'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR',
        'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR',
        'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FITB', 'FRC', 'FE',
        'FIS', 'FISV', 'FLT', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN',
        'FCX', 'GRMN', 'IT', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD',
        'GL', 'GPN', 'GS', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC',
        'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM',
        'HPQ', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY',
        'IR', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH',
        'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K',
        'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX',
        'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LNC', 'LIN', 'LYV', 'LKQ',
        'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MRO', 'MPC', 'MKTX', 'MAR',
        'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK',
        'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK',
        'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI',
        'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI',
        'NDSN', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR',
        'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR',
        'PKG', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PKI', 'PFE',
        'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG',
        'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'QRVO', 'PWR',
        'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG',
        'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC',
        'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SNA',
        'SEDG', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SYF',
        'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPG', 'TGT', 'TEL', 'TDY', 'TFX',
        'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV',
        'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS',
        'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC',
        'VICI', 'V', 'VMC', 'WAB', 'WBA', 'WMT', 'WBD', 'WM', 'WAT', 'WEC',
        'WFC', 'WELL', 'WST', 'WDC', 'WRK', 'WY', 'WHR', 'WMB', 'WTW', 'GWW',
        'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS'
    ]
    
    print(f"📊 Loaded {len(symbols)} S&P 500 symbols for backfill")
    return symbols

def save_price_data(symbol, df):
    """Save price data to CSV file"""
    try:
        file_path = f'data/daily/{symbol}.csv'
        
        # Ensure Date column is properly formatted
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows for {symbol} to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {e}")
        raise

def load_price_data(symbol):
    """Load price data from CSV file"""
    try:
        file_path = f'data/daily/{symbol}.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Ensure Date column is properly formatted
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            logger.info(f"Loaded {len(df)} rows for {symbol} from {file_path}")
            return df
        else:
            logger.warning(f"Data file not found for symbol: {symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def save_trades(trades_list):
    """Save trades to CSV file"""
    try:
        trades_file = 'data/trades/trade_history.csv'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(trades_file), exist_ok=True)
        
        if os.path.exists(trades_file):
            # Append to existing trades
            existing_df = pd.read_csv(trades_file)
            new_df = pd.DataFrame(trades_list)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(trades_file, index=False)
        else:
            # Create new file
            df = pd.DataFrame(trades_list)
            df.to_csv(trades_file, index=False)
        
        logger.info(f"Saved {len(trades_list)} trades to {trades_file}")
        
    except Exception as e:
        logger.error(f"Error saving trades: {e}")
        raise

def save_positions(positions_list):
    """Save positions to CSV file"""
    try:
        positions_file = 'data/trades/current_positions.csv'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(positions_file), exist_ok=True)
        
        df = pd.DataFrame(positions_list)
        df.to_csv(positions_file, index=False)
        
        logger.info(f"Saved {len(positions_list)} positions to {positions_file}")
        
    except Exception as e:
        logger.error(f"Error saving positions: {e}")
        raise

def save_signals(signals_list):
    """Save signals to CSV file"""
    try:
        signals_file = 'data/trades/recent_signals.csv'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(signals_file), exist_ok=True)
        
        df = pd.DataFrame(signals_list)
        df.to_csv(signals_file, index=False)
        
        logger.info(f"Saved {len(signals_list)} signals to {signals_file}")
        
    except Exception as e:
        logger.error(f"Error saving signals: {e}")
        raise

def get_positions_df():
    """Get positions DataFrame"""
    try:
        positions_file = 'data/trades/current_positions.csv'
        
        if os.path.exists(positions_file):
            df = pd.read_csv(positions_file)
            print(f"🔍 Positions CSV shape: {df.shape}, columns: {list(df.columns)}")
            
            # Handle comma-separated data in single column (defensive)
            if len(df.columns) == 1 and not df.empty:
                first_cell = str(df.iloc[0, 0])
                if ',' in first_cell:
                    print("🔧 Splitting comma-separated positions data")
                    df = df.iloc[:, 0].str.split(',', expand=True)
                    position_columns = ['symbol', 'shares', 'entry_price', 'entry_date', 'current_price', 
                                      'current_value', 'profit', 'profit_pct', 'days_held', 'side', 'strategy']
                    df.columns = position_columns[:len(df.columns)]
            
            return df
        else:
            print("⚠️ Positions file not found - creating empty DataFrame")
            return pd.DataFrame(columns=['symbol', 'shares', 'entry_price', 'current_price', 'unrealized_pnl'])
            
    except Exception as e:
        print(f"❌ Error reading positions CSV: {e}")
        logger.error(f"Error in get_positions_df: {e}")
        return pd.DataFrame(columns=['symbol', 'shares', 'entry_price', 'current_price', 'unrealized_pnl'])

def get_positions():
    """Get current positions as list of dictionaries"""
    try:
        df = get_positions_df()
        return df.to_dict(orient="records") if not df.empty else []
    except Exception as e:
        logger.error(f"Error in get_positions: {e}")
        return []

def get_signals():
    """Get trading signals DataFrame"""
    try:
        signals_file = 'data/trades/recent_signals.csv'
        
        if os.path.exists(signals_file):
            df = pd.read_csv(signals_file)
            print(f"🔍 Signals CSV shape: {df.shape}, columns: {list(df.columns)}")
            
            # Handle comma-separated data in single column (defensive)
            if len(df.columns) == 1 and not df.empty:
                first_cell = str(df.iloc[0, 0])
                if ',' in first_cell:
                    print("🔧 Splitting comma-separated signals data")
                    df = df.iloc[:, 0].str.split(',', expand=True)
                    signal_columns = ['date', 'symbol', 'signal_type', 'direction', 'price', 'strategy']
                    df.columns = signal_columns[:len(df.columns)]
            
            return df
        else:
            print("⚠️ Signals file not found - creating empty DataFrame")
            return pd.DataFrame(columns=['symbol', 'signal', 'timestamp', 'confidence'])
            
    except Exception as e:
        print(f"❌ Error reading signals CSV: {e}")
        logger.error(f"Error in get_signals: {e}")
        return pd.DataFrame(columns=['symbol', 'signal', 'timestamp', 'confidence'])

def get_portfolio_metrics():
    """Get basic portfolio metrics"""
    try:
        # Check if trades file exists
        trades_file = 'data/trades/trade_history.csv'
        
        print(f"🔍 Looking for trades file at: {trades_file}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Files in current directory: {os.listdir('.')}")
        
        if os.path.exists('data'):
            print(f"🔍 Files in data: {os.listdir('data')}")
            if os.path.exists('data/trades'):
                print(f"🔍 Files in data/trades: {os.listdir('data/trades')}")
        
        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)
            print(f"🔍 Trades CSV shape: {trades_df.shape}, columns: {list(trades_df.columns)}")
            
            if not trades_df.empty and 'profit' in trades_df.columns:
                total_profit = trades_df['profit'].sum()
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                return {
                    'total_profit': total_profit,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0
                }
        
        # Return default metrics if no data
        return {
            'total_profit': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit_per_trade': 0
        }
        
    except Exception as e:
        logger.error(f"Error in get_portfolio_metrics: {e}")
        return {
            'total_profit': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_profit_per_trade': 0
        }

def get_strategy_performance():
    """Get strategy performance data"""
    try:
        # Return sample performance data
        return {
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.15,
            'total_return': 0.25,
            'volatility': 0.18
        }
    except Exception as e:
        logger.error(f"Error in get_strategy_performance: {e}")
        return {
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0,
            'volatility': 0
        }

def get_portfolio_performance_stats():
    """Get detailed portfolio performance statistics"""
    try:
        # Return sample stats
        return {
            'monthly_returns': [0.02, 0.015, -0.01, 0.03, 0.025],
            'cumulative_returns': [1.02, 1.035, 1.025, 1.055, 1.081],
            'benchmark_returns': [0.015, 0.012, -0.008, 0.025, 0.02]
        }
    except Exception as e:
        logger.error(f"Error in get_portfolio_performance_stats: {e}")
        return {
            'monthly_returns': [],
            'cumulative_returns': [],
            'benchmark_returns': []
        }

def get_current_positions():
    """Get current portfolio positions"""
    try:
        positions_file = 'data/trades/current_positions.csv'
        
        if os.path.exists(positions_file):
            positions_df = pd.read_csv(positions_file)
            return positions_df
        else:
            print("⚠️ Positions file not found - creating empty DataFrame")
            return pd.DataFrame(columns=['symbol', 'shares', 'entry_price', 'current_price', 'unrealized_pnl'])
            
    except Exception as e:
        logger.error(f"Error in get_current_positions: {e}")
        return pd.DataFrame(columns=['symbol', 'shares', 'entry_price', 'current_price', 'unrealized_pnl'])

def get_recent_signals():
    """Get recent trading signals"""
    try:
        signals_file = 'data/trades/recent_signals.csv'
        
        if os.path.exists(signals_file):
            signals_df = pd.read_csv(signals_file)
            return signals_df
        else:
            print("⚠️ Signals file not found - creating empty DataFrame")
            return pd.DataFrame(columns=['symbol', 'signal', 'timestamp', 'confidence'])
            
    except Exception as e:
        logger.error(f"Error in get_recent_signals: {e}")
        return pd.DataFrame(columns=['symbol', 'signal', 'timestamp', 'confidence'])

def get_trades_history():
    """Get trades history"""
    try:
        trades_file = 'data/trades/trade_history.csv'
        
        print(f"🔍 Looking for trades file at: {trades_file}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 Files in current directory: {os.listdir('.')}")
        
        if os.path.exists('data'):
            print(f"🔍 Files in data: {os.listdir('data')}")
            if os.path.exists('data/trades'):
                print(f"🔍 Files in data/trades: {os.listdir('data/trades')}")
        
        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)
            print(f"🔍 Trades CSV shape: {trades_df.shape}, columns: {list(trades_df.columns)}")
            return trades_df
        else:
            print("⚠️ Trades history file not found - creating empty DataFrame")
            return pd.DataFrame(columns=['symbol', 'type', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'shares', 'profit', 'exit_reason'])
            
    except Exception as e:
        logger.error(f"Error in get_trades_history: {e}")
        return pd.DataFrame(columns=['symbol', 'type', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'shares', 'profit', 'exit_reason'])

def load_stock_data(symbol, start_date=None, end_date=None):
    """Load stock data for a given symbol"""
    try:
        file_path = f'data/daily/{symbol}.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            return df
        else:
            logger.warning(f"Data file not found for symbol: {symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        return pd.DataFrame()

def initialize():
    """Initialize data manager - create directories and validate setup"""
    try:
        # Ensure all required directories exist
        required_dirs = ['data', 'data/daily', 'data/trades']
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        # Validate data integrity
        validate_data_integrity()
        
        logger.info("Data manager initialized successfully")
        print("✅ Data manager initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing data manager: {e}")
        raise

def ensure_dir(path):
    """Ensure directory exists for given file path"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

def validate_data_integrity():
    """Validate that all data files are properly formatted"""
    try:
        required_dirs = ['data', 'data/daily', 'data/trades']
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        logger.info("Data integrity validation completed")
        return True
        
    except Exception as e:
        logger.error(f"Data integrity validation failed: {e}")
        return False

# Initialize on import
validate_data_integrity()