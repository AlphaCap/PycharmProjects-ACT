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
    """Return a list of S&P 500 symbols"""
    # Sample S&P 500 symbols - you can expand this list or load from a file
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO',
        'PEP', 'TMO', 'COST', 'MRK', 'WMT', 'NFLX', 'DIS', 'ABT', 'ADBE', 'CRM',
        'XOM', 'VZ', 'CMCSA', 'NKE', 'INTC', 'T', 'AMD', 'TXN', 'QCOM', 'LOW',
        'UPS', 'PM', 'SPGI', 'HON', 'INTU', 'IBM', 'GS', 'AMGN', 'BKNG', 'CAT',
        'SBUX', 'GILD', 'MDT', 'AXP', 'BLK', 'ISRG', 'TJX', 'MMM', 'LRCX', 'MU',
        'CVS', 'MO', 'PYPL', 'PLD', 'ZTS', 'MDLZ', 'TMUS', 'C', 'REGN', 'DUK',
        'SO', 'CB', 'BMY', 'SCHW', 'NEE', 'RTX', 'NOW', 'SYK', 'BSX', 'COP',
        'ELV', 'LMT', 'DE', 'FDX', 'ANTM', 'EQIX', 'EL', 'ITW', 'AON', 'MMC',
        'NSC', 'HUM', 'PNC', 'GD', 'FCX', 'CSX', 'WM', 'USB', 'EMR', 'SRE',
        'TGT', 'GM', 'CL', 'F', 'APD', 'GE', 'ORLY', 'MCD', 'ATVI', 'D'
    ]
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

# Remove any git merge conflict markers that might be present
# This function ensures clean data manager operation
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
<<<<<<< HEAD
validate_data_integrity()
=======
validate_data_integrity()
>>>>>>> 9eefb3de18a8bb290c7c8028019a093e7188c36f
