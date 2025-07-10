import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# --- CONFIG ---
DATA_DIR = "data"
DAILY_DIR = os.path.join(DATA_DIR, "daily")
TRADES_DIR = os.path.join(DATA_DIR, "trades")
POSITIONS_FILE = os.path.join(TRADES_DIR, "positions.csv")
TRADES_HISTORY_FILE = os.path.join(TRADES_DIR, "trade_history.csv")
SIGNALS_FILE = os.path.join(TRADES_DIR, "recent_signals.csv")
SYSTEM_STATUS_FILE = os.path.join(DATA_DIR, "system_status.csv")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
SP500_SYMBOLS_FILE = os.path.join(DATA_DIR, "sp500_symbols.csv")

RETENTION_DAYS = 180
PRIMARY_TIER_DAYS = 30
MAX_THREADS = 8
HISTORY_DAYS = 200  # Rolling window for daily data

PRICE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
INDICATOR_COLUMNS = [
    "BBAvg", "BBSDev", "UpperBB", "LowerBB", 
    "High_Low", "High_Close", "Low_Close", "TR", "ATR", "ATRma",
    "LongPSAR", "ShortPSAR", "PSAR_EP", "PSAR_AF", "PSAR_IsLong",
    "oLRSlope", "oLRAngle", "oLRIntercept", "TSF", 
    "oLRSlope2", "oLRAngle2", "oLRIntercept2", "TSF5", 
    "Value1", "ROC", "LRV", "LinReg", 
    "oLRValue", "oLRValue2", "SwingLow", "SwingHigh"
]
ALL_COLUMNS = PRICE_COLUMNS + INDICATOR_COLUMNS

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- FILE UTILS ---
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# --- S&P 500 SYMBOLS ---
def get_sp500_symbols() -> list:
    """
    Load S&P 500 symbols from the saved CSV file.
    Returns a list of symbol strings.
    """
    if os.path.exists(SP500_SYMBOLS_FILE):
        df = pd.read_csv(SP500_SYMBOLS_FILE)
        # Try to auto-detect the symbol column
        if 'symbol' in df.columns:
            return df['symbol'].dropna().astype(str).tolist()
        else:
            # If only one column, use the first column
            return df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        logger.warning(f"S&P 500 symbols file not found: {SP500_SYMBOLS_FILE}")
        return []

# --- PRICE + INDICATOR DATA ---
def save_price_data(symbol: str, df: pd.DataFrame, history_days: int = HISTORY_DAYS):
    """
    Save the DataFrame with price + indicator columns for a symbol.
    Only the most recent `history_days` rows are retained.
    """
    filename = os.path.join(DAILY_DIR, f"{symbol}.csv")
    ensure_dir(filename)
    if not df.empty:
        # Ensure correct columns and order
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ALL_COLUMNS]
        df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").tail(history_days)
        df.to_csv(filename, index=False)
    else:
        logger.warning(f"No data to save for {symbol}")

def load_price_data(symbol: str) -> pd.DataFrame:
    filename = os.path.join(DAILY_DIR, f"{symbol}.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, parse_dates=["Date"])
    else:
        return pd.DataFrame(columns=ALL_COLUMNS)

# --- TRADES, POSITIONS, SIGNALS, METADATA, INITIALIZATION ---

TRADE_COLUMNS = [
    "symbol", "type", "entry_date", "exit_date", 
    "entry_price", "exit_price", "shares", "profit", "exit_reason"
]
POSITION_COLUMNS = [
    "symbol", "shares", "entry_price", "entry_date", "current_price", 
    "current_value", "profit", "profit_pct", "days_held", "side", "strategy"
]
SIGNAL_COLUMNS = [
    "date", "symbol", "signal_type", "direction", "price", "strategy"
]
SYSTEM_STATUS_COLUMNS = [
    "timestamp", "system", "message"
]

# --- TRADE HISTORY ---
def get_trades_history():
    import os
    print(f"🔍 Looking for trades file at: {TRADES_HISTORY_FILE}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Files in current directory: {os.listdir('.')}")
    if os.path.exists("data"):
        print(f"🔍 Files in data: {os.listdir('data')}")
        if os.path.exists("data/trades"):
            print(f"🔍 Files in data/trades: {os.listdir('data/trades')}")
    
    if os.path.exists(TRADES_HISTORY_FILE):
        try:
            df = pd.read_csv(TRADES_HISTORY_FILE)
            # Debug: Print what we actually got
            print(f"🔍 Trades CSV shape: {df.shape}, columns: {list(df.columns)}")
            
            # Handle comma-separated data in single column
            if len(df.columns) == 1 and not df.empty:
                first_cell = str(df.iloc[0, 0])
                if ',' in first_cell:
                    print("🔧 Splitting comma-separated trades data")
                    # Split comma-separated data
                    df = df.iloc[:, 0].str.split(',', expand=True)
                    df.columns = TRADE_COLUMNS[:len(df.columns)]
            
            return df
        except Exception as e:
            print(f"❌ Error reading trades CSV: {e}")
            return pd.DataFrame(columns=TRADE_COLUMNS)
    else:
        print("⚠️ Trades history file not found - creating empty DataFrame")
        return pd.DataFrame(columns=TRADE_COLUMNS)

def save_trades(trades_list: List[Dict]):
    ensure_dir(TRADES_HISTORY_FILE)
    if os.path.exists(TRADES_HISTORY_FILE):
        # Append to existing trades
        existing_df = pd.read_csv(TRADES_HISTORY_FILE)
        new_df = pd.DataFrame(trades_list)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(TRADES_HISTORY_FILE, index=False)
    else:
        # Create new file
        df = pd.DataFrame(trades_list)
        df.to_csv(TRADES_HISTORY_FILE, index=False)

# --- POSITIONS ---
def get_positions_df():
    if os.path.exists(POSITIONS_FILE):
        try:
            df = pd.read_csv(POSITIONS_FILE)
            # Debug: Print what we actually got
            print(f"🔍 Positions CSV shape: {df.shape}, columns: {list(df.columns)}")
            
            # Handle comma-separated data in single column
            if len(df.columns) == 1 and not df.empty:
                first_cell = str(df.iloc[0, 0])
                if ',' in first_cell:
                    print("🔧 Splitting comma-separated positions data")
                    # Split comma-separated data
                    df = df.iloc[:, 0].str.split(',', expand=True)
                    df.columns = POSITION_COLUMNS[:len(df.columns)]
            
            return df
        except Exception as e:
            print(f"❌ Error reading positions CSV: {e}")
            return pd.DataFrame(columns=POSITION_COLUMNS)
    else:
        print("⚠️ Positions file not found - creating empty DataFrame")
        return pd.DataFrame(columns=POSITION_COLUMNS)

def save_positions(positions_list: List[Dict]):
    ensure_dir(POSITIONS_FILE)
    df = pd.DataFrame(positions_list)
    df.to_csv(POSITIONS_FILE, index=False)

def get_positions():
    df = get_positions_df()
    return df.to_dict(orient="records") if not df.empty else []

# --- SIGNALS ---
def get_signals():
    if os.path.exists(SIGNALS_FILE):
        try:
            df = pd.read_csv(SIGNALS_FILE)
            # Debug: Print what we actually got
            print(f"🔍 Signals CSV shape: {df.shape}, columns: {list(df.columns)}")
            
            # Handle comma-separated data in single column
            if len(df.columns) == 1 and not df.empty:
                first_cell = str(df.iloc[0, 0])
                if ',' in first_cell:
                    print("🔧 Splitting comma-separated signals data")
                    # Split comma-separated data
                    df = df.iloc[:, 0].str.split(',', expand=True)
                    df.columns = SIGNAL_COLUMNS[:len(df.columns)]
            
            return df
        except Exception as e:
            print(f"❌ Error reading signals CSV: {e}")
            return pd.DataFrame(columns=SIGNAL_COLUMNS)
    else:
        print("⚠️ Signals file not found - creating empty DataFrame")
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

def save_signals(signals: List[Dict]):
    ensure_dir(SIGNALS_FILE)
    df = pd.DataFrame(signals)
    df.to_csv(SIGNALS_FILE, index=False)

# --- SYSTEM STATUS ---
def get_system_status():
    if os.path.exists(SYSTEM_STATUS_FILE):
        return pd.read_csv(SYSTEM_STATUS_FILE)
    return pd.DataFrame(columns=SYSTEM_STATUS_COLUMNS)

def save_system_status(message: str, system: str = "nGS"):
    ensure_dir(SYSTEM_STATUS_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = pd.DataFrame([{"timestamp": now, "system": system, "message": message}])
    if os.path.exists(SYSTEM_STATUS_FILE):
        df = pd.read_csv(SYSTEM_STATUS_FILE)
        df = pd.concat([new_row, df], ignore_index=True)
    else:
        df = new_row
    df.to_csv(SYSTEM_STATUS_FILE, index=False)

# --- METADATA ---
def init_metadata():
    if not os.path.exists(METADATA_FILE):
        metadata = {
            "created": datetime.now().isoformat(),
            "retention_days": RETENTION_DAYS,
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    return metadata

def update_metadata(key: str, value):
    metadata = init_metadata()
    if "." in key:
        parts = key.split(".")
        d = metadata
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value
    else:
        metadata[key] = value
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
# --- DASHBOARD FUNCTIONS FOR LONG/SHORT SYSTEM ---

def get_portfolio_metrics(initial_portfolio_value: float = 100000) -> Dict:
    """
    Calculate portfolio metrics for long/short system.
    
    Args:
        initial_portfolio_value: Starting portfolio value
        
    Returns:
        Dictionary with portfolio metrics including M/E ratio
    """
    try:
        # Get current positions
        positions_df = get_positions_df()
        trades_df = get_trades_history()
        
        if positions_df.empty:
            return {
                'total_value': f"${initial_portfolio_value:.2f}",
                'total_return_pct': "0.00%",
                'daily_pnl': "$0.00",
                'mtd_return': "$0.00",
                'mtd_delta': "0.00%",
                'ytd_return': "$0.00",
                'ytd_delta': "0.00%",
                'me_ratio': "0.00%",
                'long_exposure': "$0.00",
                'short_exposure': "$0.00",
                'net_exposure': "$0.00"
            }
        
        # Calculate exposures
        long_positions = positions_df[positions_df['shares'] > 0]
        short_positions = positions_df[positions_df['shares'] < 0]
        
        long_exposure = long_positions['current_value'].sum() if not long_positions.empty else 0
        short_exposure = abs(short_positions['current_value'].sum()) if not short_positions.empty else 0
        net_exposure = long_exposure - short_exposure
        total_margin = long_exposure + short_exposure  # Total margin used
        
        # Portfolio calculations
        total_trade_profit = trades_df['profit'].sum() if not trades_df.empty else 0
        unrealized_pnl = positions_df['profit'].sum() if not positions_df.empty else 0
        
        # Current portfolio value
        current_portfolio_value = initial_portfolio_value + total_trade_profit + unrealized_pnl
        
        # M/E Ratio (Margin to Equity)
        me_ratio = (total_margin / current_portfolio_value * 100) if current_portfolio_value > 0 else 0
        
        # Returns
        total_return = current_portfolio_value - initial_portfolio_value
        total_return_pct = f"{(total_return / initial_portfolio_value * 100):.2f}%" if initial_portfolio_value > 0 else "0.00%"
        
        # Daily P&L (unrealized from current positions)
        daily_pnl = unrealized_pnl
        
        # MTD and YTD (simplified)
        mtd_return = f"${total_return:.2f}"
        mtd_delta = total_return_pct
        ytd_return = f"${total_return:.2f}" 
        ytd_delta = total_return_pct
        
        return {
            'total_value': f"${current_portfolio_value:,.0f}",  # No cents, with commas
            'total_return_pct': total_return_pct,
            'daily_pnl': f"${daily_pnl:.2f}",
            'mtd_return': mtd_return,
            'mtd_delta': mtd_delta,
            'ytd_return': ytd_return,
            'ytd_delta': ytd_delta,
            'me_ratio': f"{me_ratio:.1f}%",
            'long_exposure': f"${long_exposure:,.0f}",  # No cents
            'short_exposure': f"${short_exposure:,.0f}",  # No cents
            'net_exposure': f"${net_exposure:,.0f}"  # No cents
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {
            'total_value': f"${initial_portfolio_value:,.0f}",  # No cents
            'total_return_pct': "0.00%",
            'daily_pnl': "$0.00",
            'mtd_return': "$0.00",
            'mtd_delta': "0.00%",
            'ytd_return': "$0.00",
            'ytd_delta': "0.00%",
            'me_ratio': "0.00%",
            'long_exposure': "$0",
            'short_exposure': "$0",
            'net_exposure': "$0"
        }

def get_strategy_performance(initial_portfolio_value: float = 100000) -> pd.DataFrame:
    """
    Get strategy performance summary.
    
    Args:
        initial_portfolio_value: Starting portfolio value
        
    Returns:
        DataFrame with strategy performance
    """
    try:
        trades_df = get_trades_history()
        
        if trades_df.empty:
            return pd.DataFrame(columns=['Strategy', 'Trades', 'Win Rate', 'Total Profit', 'Avg Profit'])
        
        # Group by strategy type (using signal type from trades)
        strategy_stats = []
        
        # Overall nGS Strategy stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = trades_df['profit'].sum()
        avg_profit = trades_df['profit'].mean()
        
        strategy_stats.append({
            'Strategy': 'nGS System',
            'Trades': total_trades,
            'Win Rate': f"{win_rate:.1%}",
            'Total Profit': f"${total_profit:.2f}",
            'Avg Profit': f"${avg_profit:.2f}"
        })
        
        # Breakdown by trade type if available
        if 'type' in trades_df.columns:
            for trade_type in trades_df['type'].unique():
                type_trades = trades_df[trades_df['type'] == trade_type]
                if not type_trades.empty:
                    type_total = len(type_trades)
                    type_wins = len(type_trades[type_trades['profit'] > 0])
                    type_win_rate = type_wins / type_total if type_total > 0 else 0
                    type_profit = type_trades['profit'].sum()
                    type_avg = type_trades['profit'].mean()
                    
                    strategy_stats.append({
                        'Strategy': f'nGS {trade_type.title()}',
                        'Trades': type_total,
                        'Win Rate': f"{type_win_rate:.1%}",
                        'Total Profit': f"${type_profit:.2f}",
                        'Avg Profit': f"${type_avg:.2f}"
                    })
        
        return pd.DataFrame(strategy_stats)
        
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return pd.DataFrame(columns=['Strategy', 'Trades', 'Win Rate', 'Total Profit', 'Avg Profit'])

def get_portfolio_performance_stats() -> pd.DataFrame:
    """
    Get detailed portfolio performance statistics for display.
    
    Returns:
        DataFrame with performance statistics
    """
    try:
        trades_df = get_trades_history()
        positions_df = get_positions_df()
        
        if trades_df.empty:
            return pd.DataFrame(columns=['Metric', 'Value'])
        
        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = trades_df['profit'].sum()
        avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        total_wins = trades_df[trades_df['profit'] > 0]['profit'].sum()
        total_losses = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Current positions
        current_positions = len(positions_df) if not positions_df.empty else 0
        unrealized_pnl = positions_df['profit'].sum() if not positions_df.empty else 0
        
        # Create stats DataFrame
        stats = [
            ['Total Trades', f"{total_trades}"],
            ['Win Rate', f"{win_rate:.1%}"],
            ['Total Profit', f"${total_profit:.2f}"],
            ['Avg Win', f"${avg_win:.2f}"],
            ['Avg Loss', f"${avg_loss:.2f}"],
            ['Profit Factor', f"{profit_factor:.2f}"],
            ['Open Positions', f"{current_positions}"],
            ['Unrealized P&L', f"${unrealized_pnl:.2f}"]
        ]
        
        return pd.DataFrame(stats, columns=['Metric', 'Value'])
        
    except Exception as e:
        logger.error(f"Error getting portfolio performance stats: {e}")
        return pd.DataFrame(columns=['Metric', 'Value'])

def get_long_positions_formatted() -> pd.DataFrame:
    """
    Get formatted long positions for dashboard display.
    
    Returns:
        DataFrame with formatted long positions
    """
    try:
        positions_df = get_positions_df()
        
        if positions_df.empty:
            print("⚠️ No positions data available")
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Convert shares column to numeric if it's string
        if 'shares' in positions_df.columns:
            positions_df['shares'] = pd.to_numeric(positions_df['shares'], errors='coerce')
        
        # Filter for long positions
        long_positions = positions_df[positions_df['shares'] > 0].copy()
        
        if long_positions.empty:
            print("⚠️ No long positions found")
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Convert numeric columns
        for col in ['entry_price', 'current_price', 'profit', 'profit_pct', 'days_held']:
            if col in long_positions.columns:
                long_positions[col] = pd.to_numeric(long_positions[col], errors='coerce')
        
        # Format for display
        formatted = pd.DataFrame({
            'Symbol': long_positions['symbol'],
            'Shares': long_positions['shares'].astype(int),
            'Entry Price': long_positions['entry_price'].apply(lambda x: f"${x:.2f}"),
            'Current Price': long_positions['current_price'].apply(lambda x: f"${x:.2f}"),
            'P&L': long_positions['profit'].apply(lambda x: f"${x:.2f}"),
            'P&L %': long_positions['profit_pct'].apply(lambda x: f"{x:.1f}%"),
            'Days': long_positions['days_held'].astype(int)
        })
        
        return formatted.reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error getting formatted long positions: {e}")
        logger.error(f"Error getting formatted long positions: {e}")
        return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])

def get_short_positions_formatted() -> pd.DataFrame:
    """
    Get formatted short positions for dashboard display.
    
    Returns:
        DataFrame with formatted short positions
    """
    try:
        positions_df = get_positions_df()
        
        if positions_df.empty:
            print("⚠️ No positions data available")
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Convert shares column to numeric if it's string
        if 'shares' in positions_df.columns:
            positions_df['shares'] = pd.to_numeric(positions_df['shares'], errors='coerce')
        
        # Filter for short positions
        short_positions = positions_df[positions_df['shares'] < 0].copy()
        
        if short_positions.empty:
            print("⚠️ No short positions found")
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Convert numeric columns
        for col in ['entry_price', 'current_price', 'profit', 'profit_pct', 'days_held']:
            if col in short_positions.columns:
                short_positions[col] = pd.to_numeric(short_positions[col], errors='coerce')
        
        # Format for display (show absolute shares for shorts)
        formatted = pd.DataFrame({
            'Symbol': short_positions['symbol'],
            'Shares': short_positions['shares'].abs().astype(int),
            'Entry Price': short_positions['entry_price'].apply(lambda x: f"${x:.2f}"),
            'Current Price': short_positions['current_price'].apply(lambda x: f"${x:.2f}"),
            'P&L': short_positions['profit'].apply(lambda x: f"${x:.2f}"),
            'P&L %': short_positions['profit_pct'].apply(lambda x: f"{x:.1f}%"),
            'Days': short_positions['days_held'].astype(int)
        })
        
        return formatted.reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error getting formatted short positions: {e}")
        logger.error(f"Error getting formatted short positions: {e}")
        return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])

def get_long_positions() -> List[Dict]:
    """
    Get current long positions.
    
    Returns:
        List of long position dictionaries
    """
    try:
        positions_df = get_positions_df()
        
        if positions_df.empty:
            return []
        
        # Filter for long positions (positive shares)
        long_positions = positions_df[positions_df['shares'] > 0]
        
        # Convert to list of dictionaries for Streamlit
        return long_positions.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error getting long positions: {e}")
        return []

def get_short_positions() -> List[Dict]:
    """
    Get current short positions.
    
    Returns:
        List of short position dictionaries
    """
    try:
        positions_df = get_positions_df()
        
        if positions_df.empty:
            return []
        
        # Filter for short positions (negative shares)
        short_positions = positions_df[positions_df['shares'] < 0]
        
        # Convert to list of dictionaries for Streamlit
        return short_positions.to_dict('records')
        
    except Exception as e:
        logger.error(f"Error getting short positions: {e}")
        return []

def inspect_stock_data(symbol: str):
    """Inspect a specific stock's data for debugging"""
    print(f"\n🔍 INSPECTING {symbol}")
    print("-" * 40)
    
    filename = os.path.join(DAILY_DIR, f"{symbol}.csv")
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return
    
    try:
        # Read the CSV
        df = pd.read_csv(filename, parse_dates=["Date"])
        
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Show latest data in column format
        print(f"\n📈 Latest 3 rows:")
        latest = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(3)
        for _, row in latest.iterrows():
            print(f"  {row['Date'].strftime('%Y-%m-%d')}: O=${row['Open']:.2f} H=${row['High']:.2f} L=${row['Low']:.2f} C=${row['Close']:.2f} V={row['Volume']:,.0f}")
        
        # Check indicators
        indicator_cols = ['BBAvg', 'UpperBB', 'LowerBB', 'ATR']
        print(f"\n📊 Indicators:")
        for col in indicator_cols:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                latest_value = df[col].iloc[-1] if non_null_count > 0 else None
                print(f"  {col}: {non_null_count}/{len(df)} values, Latest: {latest_value:.2f if pd.notna(latest_value) else 'N/A'}")
            else:
                print(f"  {col}: Not found")
                
    except Exception as e:
        print(f"❌ Error reading {symbol}: {e}")

# --- INITIALIZE ---
def initialize():
    ensure_dir(POSITIONS_FILE)
    ensure_dir(TRADES_HISTORY_FILE)
    ensure_dir(SIGNALS_FILE)
    ensure_dir(SYSTEM_STATUS_FILE)
    ensure_dir(METADATA_FILE)
    init_metadata()
    logger.info("Data manager initialized")

if __name__ == "__main__":
    initialize()
    logger.info("data_manager.py loaded successfully")