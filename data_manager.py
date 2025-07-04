import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join("data")
SP500_DIR = os.path.join(DATA_DIR, "SP500")
DAILY_DIR = os.path.join(DATA_DIR, "daily")
INDICATOR_DIR = os.path.join(DATA_DIR, "indicators")
TRADES_DIR = os.path.join(DATA_DIR, "trades")
POSITIONS_FILE = os.path.join(TRADES_DIR, "positions.csv")
TRADES_HISTORY_FILE = os.path.join(TRADES_DIR, "trade_history.csv")
SIGNALS_FILE = os.path.join(TRADES_DIR, "recent_signals.csv")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
SP500_SYMBOLS_FILE = os.path.join(DATA_DIR, "sp500_symbols.csv")

# Maximum days of data to keep
RETENTION_DAYS = 180
# Days of data to keep for all symbols
PRIMARY_TIER_DAYS = 30
# Number of parallel threads for data processing
MAX_THREADS = 8

# Define data column structures
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
TRADE_COLUMNS = [
    "symbol", "type", "entry_date", "exit_date", 
    "entry_price", "exit_price", "shares", "profit", "exit_reason"
]
POSITION_COLUMNS = [
    "symbol", "shares", "entry_price", "entry_date", "current_price", 
    "current_value", "profit", "profit_pct", "days_held"
]
SIGNAL_COLUMNS = [
    "symbol", "date", "price", "signal_type", "direction", "strength"
]

def ensure_directories():
    """Create all necessary directories for data storage."""
    directories = [DATA_DIR, SP500_DIR, DAILY_DIR, INDICATOR_DIR, TRADES_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    return True

def init_metadata():
    """Initialize or load metadata tracking file."""
    if not os.path.exists(METADATA_FILE):
        metadata = {
            "last_update": datetime.now().strftime("%Y-%m-%d"),
            "active_symbols": [],
            "data_stats": {
                "total_symbols": 0,
                "primary_tier_symbols": 0,
                "secondary_tier_symbols": 0,
                "symbols_with_positions": 0,
                "total_trades": 0
            },
            "version": "1.0.0"
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Created new metadata file: {METADATA_FILE}")
    else:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        logger.debug(f"Loaded existing metadata from: {METADATA_FILE}")
    return metadata

def update_metadata(key: str, value):
    """Update a specific metadata key."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
        else:
            metadata = init_metadata()
        
        # Handle nested keys using dot notation
        if "." in key:
            parts = key.split(".")
            current = metadata
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            metadata[key] = value
        
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error updating metadata key '{key}': {e}")
        return False

def get_sp500_symbols() -> List[str]:
    """Get list of current SP500 symbols."""
    try:
        if os.path.exists(SP500_SYMBOLS_FILE):
            symbols_df = pd.read_csv(SP500_SYMBOLS_FILE)
            symbols = symbols_df["symbol"].tolist()
            logger.debug(f"Loaded {len(symbols)} symbols from {SP500_SYMBOLS_FILE}")
            return symbols
        else:
            logger.warning(f"SP500 symbols file not found: {SP500_SYMBOLS_FILE}")
            # Return a default list or fetch from an API
            return []
    except Exception as e:
        logger.error(f"Error getting SP500 symbols: {e}")
        return []

def get_active_symbols() -> List[str]:
    """Get list of symbols with active positions or recent signals."""
    try:
        metadata = init_metadata()
        active_symbols = metadata.get("active_symbols", [])
        
        # Also check positions file
        if os.path.exists(POSITIONS_FILE):
            positions_df = pd.read_csv(POSITIONS_FILE)
            active_symbols.extend(positions_df["symbol"].tolist())
        
        # Make unique and return
        return list(set(active_symbols))
    except Exception as e:
        logger.error(f"Error getting active symbols: {e}")
        return []

def save_price_data(symbol: str, df: pd.DataFrame) -> bool:
    """Save price data for a symbol to CSV, maintaining the rolling window."""
    try:
        # Ensure dataframe has the right columns and types
        if "Date" not in df.columns:
            logger.error(f"Missing Date column in dataframe for {symbol}")
            return False
        
        df = df.copy()
        
        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Sort by date
        df = df.sort_values("Date")
        
        # Apply the rolling window
        cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
        df = df[df["Date"] >= cutoff_date]
        
        # Round price columns
        price_cols = [col for col in PRICE_COLUMNS if col != "Date" and col in df.columns]
        for col in price_cols:
            if col in df.columns:
                if col == "Volume":
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].round(2)
        
        # Create directory if it doesn't exist
        os.makedirs(DAILY_DIR, exist_ok=True)
        
        # Save to CSV
        file_path = os.path.join(DAILY_DIR, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        logger.debug(f"Saved {len(df)} rows of price data for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Error saving price data for {symbol}: {e}")
        return False

def save_indicator_data(symbol: str, df: pd.DataFrame) -> bool:
    """Save indicator data for a symbol to CSV, maintaining the rolling window."""
    try:
        # Ensure dataframe has the right columns and types
        if "Date" not in df.columns:
            logger.error(f"Missing Date column in indicator dataframe for {symbol}")
            return False
        
        df = df.copy()
        
        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Sort by date
        df = df.sort_values("Date")
        
        # Apply the rolling window
        cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
        df = df[df["Date"] >= cutoff_date]
        
        # Round indicator columns
        indicator_cols = [col for col in INDICATOR_COLUMNS if col in df.columns]
        for col in indicator_cols:
            if col in df.columns and df[col].dtype in [np.float64, float]:
                df[col] = df[col].round(2)
        
        # Create directory if it doesn't exist
        os.makedirs(INDICATOR_DIR, exist_ok=True)
        
        # Save to CSV
        file_path = os.path.join(INDICATOR_DIR, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        logger.debug(f"Saved {len(df)} rows of indicator data for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Error saving indicator data for {symbol}: {e}")
        return False

def load_price_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load price data for a symbol from CSV."""
    try:
        file_path = os.path.join(DAILY_DIR, f"{symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            logger.debug(f"Loaded {len(df)} rows of price data for {symbol}")
            return df
        else:
            logger.warning(f"Price data file not found for {symbol}: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading price data for {symbol}: {e}")
        return None

def load_indicator_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load indicator data for a symbol from CSV."""
    try:
        file_path = os.path.join(INDICATOR_DIR, f"{symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            logger.debug(f"Loaded {len(df)} rows of indicator data for {symbol}")
            return df
        else:
            logger.warning(f"Indicator data file not found for {symbol}: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading indicator data for {symbol}: {e}")
        return None

def load_combined_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load both price and indicator data, combining them if both exist."""
    price_df = load_price_data(symbol)
    indicator_df = load_indicator_data(symbol)
    
    if price_df is None:
        return indicator_df  # Might still be None
    
    if indicator_df is None:
        return price_df
    
    # Both exist, merge them
    try:
        combined_df = price_df.merge(indicator_df, on="Date", how="outer")
        logger.debug(f"Created combined dataframe for {symbol} with {len(combined_df)} rows")
        return combined_df
    except Exception as e:
        logger.error(f"Error combining data for {symbol}: {e}")
        return price_df  # Fall back to just price data

def save_trade(trade_dict: Dict) -> bool:
    """Save a trade to the permanent trade history."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(TRADES_DIR, exist_ok=True)
        
        # Validate required fields
        required_fields = ["symbol", "entry_date", "exit_date", "entry_price", "exit_price", "shares"]
        for field in required_fields:
            if field not in trade_dict:
                logger.error(f"Missing required field '{field}' in trade: {trade_dict}")
                return False
        
        # Format numeric fields
        if "entry_price" in trade_dict:
            trade_dict["entry_price"] = round(float(trade_dict["entry_price"]), 2)
        if "exit_price" in trade_dict:
            trade_dict["exit_price"] = round(float(trade_dict["exit_price"]), 2)
        if "shares" in trade_dict:
            trade_dict["shares"] = int(round(float(trade_dict["shares"])))
        if "profit" in trade_dict:
            trade_dict["profit"] = round(float(trade_dict["profit"]), 2)
        
        # Create dataframe with just this trade
        trade_df = pd.DataFrame([trade_dict])
        
        # Add timestamp if not present
        if "timestamp" not in trade_dict:
            trade_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to the trade history file
        if os.path.exists(TRADES_HISTORY_FILE):
            trade_df.to_csv(TRADES_HISTORY_FILE, mode="a", header=False, index=False)
        else:
            trade_df.to_csv(TRADES_HISTORY_FILE, index=False)
        
        # Update metadata
        try:
            metadata = init_metadata()
            metadata["data_stats"]["total_trades"] += 1
            # Make sure symbol is in active symbols list
            if trade_dict["symbol"] not in metadata["active_symbols"]:
                metadata["active_symbols"].append(trade_dict["symbol"])
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Error updating metadata after saving trade: {e}")
        
        logger.info(f"Saved trade for {trade_dict['symbol']}: {trade_dict['type'] if 'type' in trade_dict else 'trade'}")
        return True
    except Exception as e:
        logger.error(f"Error saving trade: {e}")
        return False

def update_positions(positions_list: List[Dict]) -> bool:
    """Update the current positions file with a list of position dictionaries."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(TRADES_DIR, exist_ok=True)
        
        if not positions_list:
            # If positions list is empty, create empty dataframe
            positions_df = pd.DataFrame(columns=POSITION_COLUMNS)
        else:
            # Format numeric fields in each position
            for pos in positions_list:
                if "entry_price" in pos:
                    pos["entry_price"] = round(float(pos["entry_price"]), 2)
                if "current_price" in pos:
                    pos["current_price"] = round(float(pos["current_price"]), 2)
                if "current_value" in pos:
                    pos["current_value"] = round(float(pos["current_value"]), 2)
                if "profit" in pos:
                    pos["profit"] = round(float(pos["profit"]), 2)
                if "profit_pct" in pos:
                    pos["profit_pct"] = round(float(pos["profit_pct"]), 2)
                if "shares" in pos:
                    pos["shares"] = int(round(float(pos["shares"])))
            
            # Create dataframe with all positions
            positions_df = pd.DataFrame(positions_list)
        
        # Save to positions file
        positions_df.to_csv(POSITIONS_FILE, index=False)
        
        # Update metadata
        try:
            metadata = init_metadata()
            metadata["data_stats"]["symbols_with_positions"] = len(positions_list)
            # Make sure all symbols in positions are in active symbols list
            for pos in positions_list:
                if pos["symbol"] not in metadata["active_symbols"]:
                    metadata["active_symbols"].append(pos["symbol"])
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Error updating metadata after saving positions: {e}")
        
        logger.info(f"Updated positions file with {len(positions_list)} positions")
        return True
    except Exception as e:
        logger.error(f"Error updating positions: {e}")
        return False

def get_positions() -> List[Dict]:
    """Get current positions from positions file."""
    try:
        if os.path.exists(POSITIONS_FILE):
            positions_df = pd.read_csv(POSITIONS_FILE)
            positions_list = positions_df.to_dict("records")
            logger.debug(f"Loaded {len(positions_list)} positions from {POSITIONS_FILE}")
            return positions_list
        else:
            logger.debug(f"Positions file not found: {POSITIONS_FILE}")
            return []
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return []

def get_trades_history(symbol: Optional[str] = None, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """Get trade history, optionally filtered by symbol and date range."""
    try:
        if os.path.exists(TRADES_HISTORY_FILE):
            trades_df = pd.read_csv(TRADES_HISTORY_FILE)
            
            # Apply filters
            if symbol:
                trades_df = trades_df[trades_df["symbol"] == symbol]
            
            if "exit_date" in trades_df.columns:
                trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
                if start_date:
                    trades_df = trades_df[trades_df["exit_date"] >= pd.to_datetime(start_date)]
                if end_date:
                    trades_df = trades_df[trades_df["exit_date"] <= pd.to_datetime(end_date)]
            
            logger.debug(f"Loaded {len(trades_df)} trades from history")
            return trades_df
        else:
            logger.debug(f"Trades history file not found: {TRADES_HISTORY_FILE}")
            return pd.DataFrame(columns=TRADE_COLUMNS)
    except Exception as e:
        logger.error(f"Error getting trades history: {e}")
        return pd.DataFrame(columns=TRADE_COLUMNS)

def save_signals(signals_df: pd.DataFrame) -> bool:
    """Save recent signals to a CSV file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(TRADES_DIR, exist_ok=True)
        
        # Format the dataframe
        if not signals_df.empty:
            signals_df = signals_df.copy()
            
            # Ensure date column is proper format
            if "date" in signals_df.columns:
                signals_df["date"] = pd.to_datetime(signals_df["date"]).dt.strftime("%Y-%m-%d")
            
            # Round price column if it exists
            if "price" in signals_df.columns:
                signals_df["price"] = signals_df["price"].round(2)
            
            # Save to CSV
            signals_df.to_csv(SIGNALS_FILE, index=False)
            logger.info(f"Saved {len(signals_df)} signals to {SIGNALS_FILE}")
            
            # Update active symbols in metadata
            try:
                if "symbol" in signals_df.columns:
                    metadata = init_metadata()
                    signal_symbols = signals_df["symbol"].unique().tolist()
                    for symbol in signal_symbols:
                        if symbol not in metadata["active_symbols"]:
                            metadata["active_symbols"].append(symbol)
                    with open(METADATA_FILE, "w") as f:
                        json.dump(metadata, f, indent=4)
            except Exception as e:
                logger.error(f"Error updating metadata after saving signals: {e}")
        else:
            # Create empty file
            pd.DataFrame(columns=SIGNAL_COLUMNS).to_csv(SIGNALS_FILE, index=False)
            logger.info(f"Saved empty signals file to {SIGNALS_FILE}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving signals: {e}")
        return False

def get_signals() -> pd.DataFrame:
    """Get recent signals from signals file."""
    try:
        if os.path.exists(SIGNALS_FILE):
            signals_df = pd.read_csv(SIGNALS_FILE)
            logger.debug(f"Loaded {len(signals_df)} signals from {SIGNALS_FILE}")
            return signals_df
        else:
            logger.debug(f"Signals file not found: {SIGNALS_FILE}")
            return pd.DataFrame(columns=SIGNAL_COLUMNS)
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

def clean_old_data() -> bool:
    """Remove data older than the retention period."""
    try:
        cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")
        
        # Get list of symbols to process
        all_symbols = get_sp500_symbols()
        active_symbols = get_active_symbols()
        
        # Process active symbols first (keep full history)
        for symbol in active_symbols:
            try:
                # Handle price data
                price_df = load_price_data(symbol)
                if price_df is not None and not price_df.empty:
                    price_df = price_df[price_df["Date"] >= cutoff_date]
                    save_price_data(symbol, price_df)
                
                # Handle indicator data
                indicator_df = load_indicator_data(symbol)
                if indicator_df is not None and not indicator_df.empty:
                    indicator_df = indicator_df[indicator_df["Date"] >= cutoff_date]
                    save_indicator_data(symbol, indicator_df)
            except Exception as e:
                logger.error(f"Error cleaning data for active symbol {symbol}: {e}")
                continue
        
        # Process remaining symbols (keep only PRIMARY_TIER_DAYS)
        for symbol in [s for s in all_symbols if s not in active_symbols]:
            try:
                # For inactive symbols, keep only PRIMARY_TIER_DAYS
                primary_cutoff = datetime.now() - timedelta(days=PRIMARY_TIER_DAYS)
                
                # Handle price data
                price_df = load_price_data(symbol)
                if price_df is not None and not price_df.empty:
                    price_df = price_df[price_df["Date"] >= primary_cutoff]
                    save_price_data(symbol, price_df)
                
                # Handle indicator data
                indicator_df = load_indicator_data(symbol)
                if indicator_df is not None and not indicator_df.empty:
                    indicator_df = indicator_df[indicator_df["Date"] >= primary_cutoff]
                    save_indicator_data(symbol, indicator_df)
            except Exception as e:
                logger.error(f"Error cleaning data for inactive symbol {symbol}: {e}")
                continue
        
        # Update metadata
        try:
            metadata = init_metadata()
            metadata["last_update"] = datetime.now().strftime("%Y-%m-%d")
            metadata["data_stats"]["total_symbols"] = len(all_symbols)
            metadata["data_stats"]["primary_tier_symbols"] = len(all_symbols) - len(active_symbols)
            metadata["data_stats"]["secondary_tier_symbols"] = len(active_symbols)
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Error updating metadata after cleaning old data: {e}")
        
        logger.info(f"Cleaned data older than {cutoff_str}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning old data: {e}")
        return False

def batch_process_symbols(symbols: List[str], 
                         process_func, 
                         batch_size: int = 50, 
                         max_workers: int = MAX_THREADS) -> Dict[str, bool]:
    """Process symbols in batches using parallel execution."""
    results = {}
    
    # Process in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {symbol: executor.submit(process_func, symbol) for symbol in batch}
            
            # Collect results
            for symbol, future in futures.items():
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {symbol} in batch: {e}")
                    results[symbol] = False
    
    return results

def initialize():
    """Initialize the data manager."""
    ensure_directories()
    init_metadata()
    logger.info("Data manager initialized")

if __name__ == "__main__":
    # Basic test to make sure everything is working
    initialize()
    logger.info("Data manager module loaded successfully")
