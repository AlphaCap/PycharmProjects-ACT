import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# --- CONFIG ---
DATA_DIR = "."  # Use current directory
DAILY_DIR = os.path.join("data", "daily")  # Added missing DAILY_DIR
POSITIONS_FILE = "positions.csv"
TRADES_HISTORY_FILE = "data/trades/trade_history.csv"  # Updated path
SIGNALS_FILE = "recent_signals.csv"
SYSTEM_STATUS_FILE = "system_status.csv"
METADATA_FILE = "metadata.json"
SP500_SYMBOLS_FILE = "sp500_symbols.txt" 

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
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# --- FORMATTING UTILS ---
def format_dollars(value):
    """Format dollar amounts without cents"""
    if isinstance(value, str) and '$' in value:
        # Already formatted
        return value
    try:
        return f"${float(value):,.0f}"
    except:
        return "$0"

# --- S&P 500 SYMBOLS ---
def get_sp500_symbols() -> list:
    """
    Load S&P 500 symbols from the saved txt file.
    Returns a list of symbol strings.
    """
    if os.path.exists(SP500_SYMBOLS_FILE):
        with open(SP500_SYMBOLS_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
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
    if os.path.exists(TRADES_HISTORY_FILE):
        return pd.read_csv(TRADES_HISTORY_FILE)
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
        return pd.read_csv(POSITIONS_FILE)
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
        return pd.read_csv(SIGNALS_FILE)
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

# --- M/E RATIO CALCULATIONS ---
def calculate_historical_me_ratio(trades_df: pd.DataFrame, initial_value: float = 100000) -> float:
    """
    Calculate historical M/E ratio as rolling average of daily position values.
    M/E = (Total Position Value / Portfolio Equity) * 100
    """
    # First try to load pre-calculated M/E ratio
    try:
        if os.path.exists('me_ratio_summary.json'):
            with open('me_ratio_summary.json', 'r') as f:
                summary = json.load(f)
                return summary.get('average_me_ratio', 150.0)
    except:
        pass
    
    # If no pre-calculated data, calculate from trades
    if trades_df.empty:
        return 150.0  # Default for margin trading
    
    try:
        # Convert dates
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Get date range
        start_date = trades_df['entry_date'].min()
        end_date = trades_df['exit_date'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Calculate daily M/E ratios
        daily_me_ratios = []
        cumulative_profit = 0
        
        for current_date in date_range:
            # Find all positions open on this date
            open_positions = trades_df[
                (trades_df['entry_date'] <= current_date) & 
                (trades_df['exit_date'] >= current_date)
            ]
            
            if not open_positions.empty:
                # Calculate total position value
                # For margin accounts, this is the full value of positions
                long_positions = open_positions[open_positions['shares'] > 0]
                short_positions = open_positions[open_positions['shares'] < 0]
                
                # Long value = price * shares
                long_value = (long_positions['entry_price'] * long_positions['shares']).sum() if not long_positions.empty else 0
                
                # Short value = price * abs(shares)
                short_value = (short_positions['entry_price'] * short_positions['shares'].abs()).sum() if not short_positions.empty else 0
                
                # Total position value
                total_position_value = long_value + short_value
                
                # Calculate portfolio equity up to this date
                closed_trades = trades_df[trades_df['exit_date'] < current_date]
                cumulative_profit = closed_trades['profit'].sum() if not closed_trades.empty else 0
                portfolio_equity = initial_value + cumulative_profit
                
                # Calculate M/E ratio
                me_ratio = (total_position_value / portfolio_equity) * 100
                daily_me_ratios.append(me_ratio)
        
        # Return average M/E ratio
        if daily_me_ratios:
            avg_me = np.mean(daily_me_ratios)
            return max(avg_me, 100.0)  # Should be at least 100% for active trading
        else:
            return 150.0  # Default
            
    except Exception as e:
        logger.error(f"Error calculating historical M/E ratio: {e}")
        return 150.0  # Default for margin trading

def calculate_ytd_return(trades_df: pd.DataFrame, initial_value: float) -> tuple:
    """Calculate Year-to-Date return from closed trades"""
    if trades_df.empty:
        return "$0", "0.00%"
    
    # Get current year trades
    current_year = datetime.now().year
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Filter for current year trades
    ytd_trades = trades_df[trades_df['exit_date'].dt.year == current_year]
    ytd_profit = ytd_trades['profit'].sum() if not ytd_trades.empty else 0
    
    # Calculate percentage
    ytd_pct = (ytd_profit / initial_value * 100) if initial_value > 0 else 0
    
    return format_dollars(ytd_profit), f"{ytd_pct:.2f}%"

def calculate_mtd_return(trades_df: pd.DataFrame, initial_value: float) -> tuple:
    """Calculate Month-to-Date return from closed trades"""
    if trades_df.empty:
        return "$0", "0.00%"
    
    # Get current month trades
    current_date = datetime.now()
    current_month_start = datetime(current_date.year, current_date.month, 1)
    
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Filter for current month trades
    mtd_trades = trades_df[trades_df['exit_date'] >= current_month_start]
    mtd_profit = mtd_trades['profit'].sum() if not mtd_trades.empty else 0
    
    # Calculate percentage
    mtd_pct = (mtd_profit / initial_value * 100) if initial_value > 0 else 0
    
    return format_dollars(mtd_profit), f"{mtd_pct:.2f}%"

# --- DASHBOARD FUNCTIONS FOR LONG/SHORT SYSTEM ---
def get_portfolio_metrics(initial_portfolio_value: float = 100000, is_historical: bool = False) -> Dict:
    """
    Calculate portfolio metrics for long/short system.
    
    Args:
        initial_portfolio_value: Starting portfolio value
        is_historical: True for historical page, False for current trading page
        
    Returns:
        Dictionary with portfolio metrics including M/E ratio
    """
    try:
        # Get data
        positions_df = get_positions_df()
        trades_df = get_trades_history()
        
        # Calculate from historical trades
        total_trade_profit = trades_df['profit'].sum() if not trades_df.empty else 0
        
        # Current portfolio value from closed trades
        current_portfolio_value = initial_portfolio_value + total_trade_profit
        
        # Calculate exposures from open positions
        if not positions_df.empty:
            long_positions = positions_df[positions_df['shares'] > 0]
            short_positions = positions_df[positions_df['shares'] < 0]
            
            # Calculate position values (full value, not just margin)
            long_value = (long_positions['current_price'] * long_positions['shares']).sum() if not long_positions.empty else 0
            short_value = (short_positions['current_price'] * short_positions['shares'].abs()).sum() if not short_positions.empty else 0
            
            # Total position value (both long and short)
            total_position_value = long_value + short_value
            
            # Current M/E Ratio (for live trading page)
            # This represents current leverage/margin usage
            current_me_ratio = (total_position_value / current_portfolio_value * 100) if current_portfolio_value > 0 else 0
            
            # Net exposure
            net_exposure = long_value - short_value
            
            # Daily P&L (unrealized from current positions)
            daily_pnl = positions_df['profit'].sum()
        else:
            # No open positions
            long_value = 0
            short_value = 0
            net_exposure = 0
            current_me_ratio = 0
            daily_pnl = 0
        
        # Historical M/E Ratio (rolling average for historical page)
        historical_me_ratio = calculate_historical_me_ratio(trades_df, initial_portfolio_value) if not trades_df.empty else 150.0
        
        # Returns based on closed trades
        total_return = total_trade_profit
        total_return_pct = f"{(total_return / initial_portfolio_value * 100):.2f}%" if initial_portfolio_value > 0 else "0.00%"
        
        # Calculate proper MTD and YTD
        mtd_return, mtd_pct = calculate_mtd_return(trades_df, initial_portfolio_value)
        ytd_return, ytd_pct = calculate_ytd_return(trades_df, initial_portfolio_value)
        
        # Format all dollar amounts without cents
        metrics = {
            'total_value': format_dollars(current_portfolio_value),
            'total_return_pct': total_return_pct,
            'daily_pnl': format_dollars(daily_pnl),
            'mtd_return': mtd_return,
            'mtd_delta': mtd_pct,
            'ytd_return': ytd_return,
            'ytd_delta': ytd_pct,
            'me_ratio': f"{current_me_ratio:.1f}%",
            'historical_me_ratio': f"{historical_me_ratio:.1f}%",
            'long_exposure': format_dollars(long_value),
            'short_exposure': format_dollars(short_value),
            'net_exposure': format_dollars(net_exposure)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {
            'total_value': format_dollars(initial_portfolio_value),
            'total_return_pct': "0.00%",
            'daily_pnl': "$0",
            'mtd_return': "$0",
            'mtd_delta': "0.00%",
            'ytd_return': "$0",
            'ytd_delta': "0.00%",
            'me_ratio': "0.0%",
            'historical_me_ratio': "150.0%",  # Default for margin account
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
            'Total Profit': format_dollars(total_profit),
            'Avg Profit': format_dollars(avg_profit)
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
                        'Total Profit': format_dollars(type_profit),
                        'Avg Profit': format_dollars(type_avg)
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
        
        # Create stats DataFrame - format without cents
        stats = [
            ['Total Trades', f"{total_trades}"],
            ['Win Rate', f"{win_rate:.1%}"],
            ['Total Profit', format_dollars(total_profit)],
            ['Avg Win', format_dollars(avg_win)],
            ['Avg Loss', format_dollars(avg_loss)],
            ['Profit Factor', f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"],
            ['Open Positions', f"{current_positions}"],
            ['Unrealized P&L', format_dollars(unrealized_pnl)]
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
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Filter for long positions
        long_positions = positions_df[positions_df['shares'] > 0].copy()
        
        if long_positions.empty:
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Format for display - no cents on P&L
        formatted = pd.DataFrame({
            'Symbol': long_positions['symbol'],
            'Shares': long_positions['shares'].astype(int),
            'Entry Price': long_positions['entry_price'].apply(lambda x: f"${x:.2f}"),
            'Current Price': long_positions['current_price'].apply(lambda x: f"${x:.2f}"),
            'P&L': long_positions['profit'].apply(lambda x: format_dollars(x)),
            'P&L %': long_positions['profit_pct'].apply(lambda x: f"{x:.1f}%"),
            'Days': long_positions['days_held'].astype(int)
        })
        
        return formatted.reset_index(drop=True)
        
    except Exception as e:
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
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Filter for short positions
        short_positions = positions_df[positions_df['shares'] < 0].copy()
        
        if short_positions.empty:
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Current Price', 'P&L', 'P&L %', 'Days'])
        
        # Format for display (show absolute shares for shorts) - no cents on P&L
        formatted = pd.DataFrame({
            'Symbol': short_positions['symbol'],
            'Shares': short_positions['shares'].abs().astype(int),
            'Entry Price': short_positions['entry_price'].apply(lambda x: f"${x:.2f}"),
            'Current Price': short_positions['current_price'].apply(lambda x: f"${x:.2f}"),
            'P&L': short_positions['profit'].apply(lambda x: format_dollars(x)),
            'P&L %': short_positions['profit_pct'].apply(lambda x: f"{x:.1f}%"),
            'Days': short_positions['days_held'].astype(int)
        })
        
        return formatted.reset_index(drop=True)
        
    except Exception as e:
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

# --- INITIALIZE ---
def initialize():
    ensure_dir(POSITIONS_FILE)
    ensure_dir(TRADES_HISTORY_FILE)
    ensure_dir(SIGNALS_FILE)
    ensure_dir(SYSTEM_STATUS_FILE)
    ensure_dir(METADATA_FILE)
    ensure_dir(DAILY_DIR)  # Ensure daily directory exists
    init_metadata()
    logger.info("Data manager initialized")

if __name__ == "__main__":
    initialize()
    logger.info("data_manager.py loaded successfully")
