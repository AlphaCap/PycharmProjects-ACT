import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from polygon import RESTClient

# Optional imports with fallbacks for sector functionality
try:
    import requests

    HAS_REQUESTS: bool = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup

    HAS_BEAUTIFULSOUP: bool = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

# --- CONFIG ---
DATA_DIR: str = "."
DAILY_DIR: str = os.path.join("data", "daily")
POSITIONS_FILE: str = "positions.csv"
TRADES_HISTORY_FILE: str = "trade_history.csv"
SIGNALS_FILE: str = "recent_signals.csv"
SYSTEM_STATUS_FILE: str = "system_status.csv"
METADATA_FILE: str = "metadata.json"
SP500_SYMBOLS_FILE: str = os.path.join("data", "sp500_symbols.txt")

SECTOR_DATA_FILE: str = os.path.join("data", "sp500_sectors.json")
SECTOR_CACHE_HOURS: int = 24

RETENTION_DAYS: int = 180
PRIMARY_TIER_DAYS: int = 30
MAX_THREADS: int = 8
HISTORY_DAYS: int = 200

SP500_EXPECTED_COUNT: int = 500
SP500_MINIMUM_COUNT: int = 490

PRICE_COLUMNS: List[str] = ["Date", "Open", "High", "Low", "Close", "Volume"]
INDICATOR_COLUMNS: List[str] = [
    "BBAvg",
    "BBSDev",
    "UpperBB",
    "LowerBB",
    "High_Low",
    "High_Close",
    "Low_Close",
    "TR",
    "ATR",
    "ATRma",
    "LongPSAR",
    "ShortPSAR",
    "PSAR_EP",
    "PSAR_AF",
    "PSAR_IsLong",
    "oLRSlope",
    "oLRAngle",
    "oLRIntercept",
    "TSF",
    "oLRSlope2",
    "oLRAngle2",
    "oLRIntercept2",
    "TSF5",
    "Value1",
    "ROC",
    "LRV",
    "LinReg",
    "oLRValue",
    "oLRValue2",
    "SwingLow",
    "SwingHigh",
    "ME_Ratio",
]
ALL_COLUMNS: List[str] = PRICE_COLUMNS + INDICATOR_COLUMNS

SP500_SECTORS: Dict[str, List[str]] = {
    "Information Technology": [],
    "Health Care": [],
    "Financials": [],
    "Communication Services": [],
    "Consumer Discretionary": [],
    "Industrials": [],
    "Consumer Staples": [],
    "Energy": [],
    "Utilities": [],
    "Real Estate": [],
    "Materials": [],
}

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_manager.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# --- UTILITY FUNCTIONS ---
def get_cutoff_date() -> datetime:
    return datetime.now() - timedelta(days=RETENTION_DAYS)


def parse_date_flexibly(date_str: Any) -> Union[pd.Timestamp, pd.NaT]:
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str)
    date_part = date_str.split()[0]
    try:
        return pd.to_datetime(date_part, format="%Y-%m-%d")
    except Exception:
        try:
            return pd.to_datetime(date_str)
        except Exception:
            logger.warning(f"Could not parse date: {date_str}")
            return pd.NaT


def format_dollars(value: Any) -> str:
    if isinstance(value, str) and "$" in value:
        return value
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def filter_by_retention_period(
    df: pd.DataFrame, date_column: str = "Date"
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df
    cutoff_date = get_cutoff_date()
    df = df.copy()
    try:
        df[date_column] = df[date_column].apply(parse_date_flexibly)
    except Exception as e:
        logger.warning(f"Could not parse dates in column {date_column}: {e}")
        return df
    original_count = len(df)
    df = df[df[date_column] >= cutoff_date].copy()
    filtered_count = len(df)
    logger.debug(
        f"Filtered {date_column}: {original_count} → {filtered_count} rows (cutoff: {cutoff_date.strftime('%Y-%m-%d')})"
    )
    return df


def ensure_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


# --- S&P 500 SYMBOLS ---
def get_sp500_symbols() -> List[str]:
    if os.path.exists(SP500_SYMBOLS_FILE):
        with open(SP500_SYMBOLS_FILE, "r") as f:
            symbols = [
                line.strip().upper()
                for line in f
                if line.strip()
                and (
                    line.strip().isalpha() or "." in line.strip() or "-" in line.strip()
                )
            ]
            logger.info(
                f"Loaded {len(symbols)} S&P 500 symbols from {SP500_SYMBOLS_FILE}"
            )
            if symbols:
                logger.info(f"Sample symbols: {symbols[:5]}...")
            return symbols
    else:
        logger.warning(f"S&P 500 symbols file not found: {SP500_SYMBOLS_FILE}")
        return []


def filter_to_sp500(symbols: List[str]) -> List[str]:
    sp500_set: Set[str] = set(get_sp500_symbols())
    return [s for s in symbols if s.upper() in sp500_set]


def verify_sp500_coverage() -> bool:
    symbols = get_sp500_symbols()
    if not symbols:
        logger.error(" No S&P 500 symbols loaded!")
        return False
    symbol_count = len(symbols)
    logger.info(f"S&P 500 symbols loaded: {symbol_count}")
    if symbol_count < SP500_MINIMUM_COUNT:
        logger.error(f" Symbol count too low: {symbol_count} < {SP500_MINIMUM_COUNT}")
        return False
    elif symbol_count > 510:
        logger.warning(
            f" Symbol count high: {symbol_count} > 510 (may include additional securities)"
        )
    must_have = ["AAPL", "MSFT", "GOOGL"]
    missing = [s for s in must_have if s not in symbols]
    if missing:
        logger.error(f" Missing major symbols: {missing}")
        return False
    duplicate_count = len(symbols) - len(set(symbols))
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate symbols")
    else:
        logger.info(" No duplicate symbols found")
    logger.info(" S&P 500 symbol validation passed")
    return True


def get_sp500_symbol_stats() -> Dict[str, Any]:
    symbols = get_sp500_symbols()
    stats: Dict[str, Any] = {
        "total_count": len(symbols),
        "expected_count": SP500_EXPECTED_COUNT,
        "minimum_count": SP500_MINIMUM_COUNT,
        "coverage_complete": len(symbols) == SP500_EXPECTED_COUNT,
        "coverage_acceptable": len(symbols) >= SP500_MINIMUM_COUNT,
        "duplicate_count": len(symbols) - len(set(symbols)) if symbols else 0,
        "sample_symbols": symbols[:10] if symbols else [],
    }
    return stats


# --- SECTOR MANAGEMENT ---
def get_sp500_sector_data(force_refresh: bool = False) -> Dict[str, Any]:
    if os.path.exists(SECTOR_DATA_FILE) and not force_refresh:
        try:
            with open(SECTOR_DATA_FILE, "r") as f:
                cached_data = json.load(f)
            cache_time = datetime.fromisoformat(
                cached_data.get("last_updated", "2000-01-01")
            )
            if datetime.now() - cache_time < timedelta(hours=SECTOR_CACHE_HOURS):
                logger.info(
                    f"Using cached sector data from {cache_time.strftime('%Y-%m-%d %H:%M')}"
                )
                return cached_data
        except Exception as e:
            logger.warning(f"Error reading sector cache: {e}")
    logger.info("Fetching fresh S&P 500 sector data...")
    sector_data = fetch_sp500_sectors()
    sector_data["last_updated"] = datetime.now().isoformat()
    ensure_dir(SECTOR_DATA_FILE)
    with open(SECTOR_DATA_FILE, "w") as f:
        json.dump(sector_data, f, indent=2)
    logger.info(
        f"Cached sector data for {len(sector_data.get('symbol_to_sector', {}))} symbols"
    )
    return sector_data


def fetch_sp500_sectors() -> Dict[str, Any]:
    polygon_data = fetch_sectors_from_polygon()
    if polygon_data:
        return polygon_data
    wiki_data = fetch_sectors_from_wikipedia()
    if wiki_data:
        return wiki_data
    logger.warning("Using static sector mapping - may be outdated")
    return get_static_sector_mapping()


def fetch_sectors_from_polygon() -> Optional[Dict[str, Any]]:
    try:
        polygon_api_key = os.getenv("POLYGON_API_KEY")
        if not polygon_api_key:
            return None
        symbols = get_sp500_symbols()
        if not symbols:
            return None
        polygon_client = RESTClient(polygon_api_key)
        sectors: Dict[str, List[str]] = {}
        symbol_to_sector: Dict[str, str] = {}
        for i in range(0, len(symbols), 50):
            batch = symbols[i : i + 50]
            for symbol in batch:
                try:
                    ticker_details = polygon_client.get_ticker_details(symbol)
                    if ticker_details and hasattr(ticker_details, "sic_description"):
                        sector = map_sic_to_sector(ticker_details.sic_description)
                        if sector:
                            if sector not in sectors:
                                sectors[sector] = []
                            sectors[sector].append(symbol)
                            symbol_to_sector[symbol] = sector
                except Exception as e:
                    logger.debug(f"Error getting sector for {symbol}: {e}")
            time.sleep(0.1)
        if sectors:
            return format_sector_data(sectors, symbol_to_sector)
    except Exception as e:
        logger.warning(f"Polygon sector fetch failed: {e}")
    return None


def fetch_sectors_from_wikipedia() -> Optional[Dict[str, Any]]:
    try:
        if not HAS_REQUESTS or not HAS_BEAUTIFULSOUP:
            return None
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if not table:
            table = soup.find("table", class_="wikitable")
        if not table:
            return None
        sectors: Dict[str, List[str]] = {}
        symbol_to_sector: Dict[str, str] = {}
        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 4:
                symbol = cells[0].get_text().strip()
                sector = cells[3].get_text().strip()
                if sector and symbol:
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(symbol)
                    symbol_to_sector[symbol] = sector
        if sectors:
            logger.info(
                f"Wikipedia: Found {len(symbol_to_sector)} symbols in {len(sectors)} sectors"
            )
            return format_sector_data(sectors, symbol_to_sector)
    except Exception as e:
        logger.warning(f"Wikipedia sector fetch failed: {e}")
    return None


def get_static_sector_mapping() -> Dict[str, Any]:
    # [ ... unchanged list of symbols ... ]
    static_sectors: Dict[str, List[str]] = {
        # ... (see previous section for actual content; omitted here for brevity) ...
    }
    symbol_to_sector: Dict[str, str] = {}
    for sector, symbols in static_sectors.items():
        for symbol in symbols:
            symbol_to_sector[symbol] = sector
    return format_sector_data(static_sectors, symbol_to_sector)


def format_sector_data(
    sectors: Dict[str, List[str]], symbol_to_sector: Dict[str, str]
) -> Dict[str, Any]:
    sector_info: Dict[str, Dict[str, Any]] = {}
    total_symbols = len(symbol_to_sector)
    for sector, symbols in sectors.items():
        sector_info[sector] = {
            "count": len(symbols),
            "weight": len(symbols) / total_symbols if total_symbols > 0 else 0,
            "symbols": sorted(symbols),
        }
    return {
        "sectors": sectors,
        "symbol_to_sector": symbol_to_sector,
        "sector_info": sector_info,
        "total_symbols": total_symbols,
        "sector_count": len(sectors),
    }


# --- SECTOR ACCESS FUNCTIONS ---
def get_sector_symbols(sector_name: str) -> List[str]:
    sector_data = get_sp500_sector_data()
    return sector_data.get("sectors", {}).get(sector_name, [])


def get_symbol_sector(symbol: str) -> str:
    sector_data = get_sp500_sector_data()
    return sector_data.get("symbol_to_sector", {}).get(symbol, "Unknown")


def get_all_sectors() -> List[str]:
    sector_data = get_sp500_sector_data()
    return list(sector_data.get("sectors", {}).keys())


def get_sector_weights() -> Dict[str, float]:
    sector_data = get_sp500_sector_data()
    return {
        sector: info.get("weight", 0)
        for sector, info in sector_data.get("sector_info", {}).items()
    }

    target_weights: Optional[Dict[str, float]] = (None,)

    def my_function() -> Dict[str, float]:
        if target_weights is None:
            return get_sector_weights()

    return target_weights


def get_sector_rebalance_targets(
    target_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Returns the sector rebalance targets dict. If target_weights is not provided, defaults to current sector weights.
    """
    if target_weights is not None:
        return target_weights
    return get_sector_weights()


def calculate_sector_rebalance_needs(
    positions_df: pd.DataFrame,
    target_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    # Current sector exposure based on positions
    current_exposure = get_portfolio_sector_exposure(positions_df)

    # Retrieve sector rebalance targets
    targets = get_sector_rebalance_targets(target_weights)

    # Initialize rebalance needs dictionary
    rebalance_needs: Dict[str, Dict[str, Any]] = {}

    # Calculate total value from positions DataFrame
    total_value = positions_df["current_value"].sum() if not positions_df.empty else 0

    # Iterate through each sector in targets
    for sector in targets:
        # Get the current weight for the sector (default to 0 if not found)
        current_weight = current_exposure.get(sector, {}).get("weight", 0)

        # Target weight for the sector
        target_weight = targets[sector]

        # Calculate the difference between target and current weights
        difference = target_weight - current_weight

        # Populate rebalance_needs with details
        rebalance_needs[sector] = {
            "current_weight": current_weight,
            "target_weight": target_weight,
            "difference": difference,
            "dollar_adjustment": difference * total_value if total_value > 0 else 0,
            "action": "buy" if difference > 0 else "sell" if difference < 0 else "hold",
        }
    # Return the rebalance needs dictionary
    return rebalance_needs


def save_price_data(
    symbol: str, df: pd.DataFrame, history_days: int = HISTORY_DAYS
) -> None:
    filename = os.path.join(DAILY_DIR, f"{symbol}.csv")
    ensure_dir(filename)
    if not df.empty:
        df = filter_by_retention_period(df, "Date")
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[ALL_COLUMNS]
        df = (
            df.sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .tail(history_days)
        )
        df.to_csv(filename, index=False)
        if "ME_Ratio" in df.columns:
            legacy_me_path = os.path.join("data", "me_ratio_history.csv")
            try:
                df[["Date", "ME_Ratio"]].to_csv(legacy_me_path, index=False)
                logger.info(f"Copied ME ratio history to legacy path: {legacy_me_path}")
            except Exception as e:
                logger.warning(
                    f"Could not copy ME ratio file for legacy dashboard: {e}"
                )
    else:
        logger.warning(f"No data to save for {symbol}")


def load_price_data(symbol: str) -> pd.DataFrame:
    filename = os.path.join(DAILY_DIR, f"{symbol}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, parse_dates=["Date"])
        return filter_by_retention_period(df, "Date")
    else:
        return pd.DataFrame(columns=ALL_COLUMNS)


TRADE_COLUMNS: List[str] = [
    "symbol",
    "type",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "shares",
    "profit",
    "exit_reason",
    "side",
    "strategy",
]
POSITION_COLUMNS: List[str] = [
    "symbol",
    "shares",
    "entry_price",
    "entry_date",
    "current_price",
    "current_value",
    "profit",
    "profit_pct",
    "days_held",
    "side",
    "strategy",
]
SIGNAL_COLUMNS: List[str] = [
    "date",
    "symbol",
    "signal_type",
    "direction",
    "price",
    "strategy",
]
SYSTEM_STATUS_COLUMNS: List[str] = ["timestamp", "system", "message"]


def get_trades_history() -> pd.DataFrame:
    try:
        if not os.path.exists(TRADES_HISTORY_FILE):
            logger.warning(f"Trade history file not found: {TRADES_HISTORY_FILE}")
            logger.info(f"Looking for file at: {os.path.abspath(TRADES_HISTORY_FILE)}")
            return pd.DataFrame(columns=TRADE_COLUMNS)
        file_size = os.path.getsize(TRADES_HISTORY_FILE)
        if file_size == 0:
            logger.warning(f"Trade history file is empty: {TRADES_HISTORY_FILE}")
            return pd.DataFrame(columns=TRADE_COLUMNS)
        logger.info(
            f"Loading trade history from {TRADES_HISTORY_FILE} ({file_size} bytes)"
        )
        df = pd.read_csv(TRADES_HISTORY_FILE)
        if df.empty:
            logger.warning("Trade history file contains no data")
            return pd.DataFrame(columns=TRADE_COLUMNS)
        missing_columns = [
            col
            for col in ["symbol", "entry_date", "exit_date", "profit"]
            if col not in df.columns
        ]
        if missing_columns:
            logger.error(
                f"Missing required columns in trade history: {missing_columns}"
            )
            logger.info(f"Available columns: {list(df.columns)}")
            return pd.DataFrame(columns=TRADE_COLUMNS)
        try:
            df["entry_date"] = df["entry_date"].apply(parse_date_flexibly)
            df["exit_date"] = df["exit_date"].apply(parse_date_flexibly)
        except Exception as e:
            logger.error(f"Error parsing dates in trade history: {e}")
        cutoff_date = get_cutoff_date()
        original_count = len(df)
        df = df[df["exit_date"] >= cutoff_date].copy()
        filtered_count = len(df)
        logger.info(
            f"Trade history filtered: {original_count} → {filtered_count} trades (cutoff: {cutoff_date.strftime('%Y-%m-%d')})"
        )
        if not df.empty:
            logger.info(
                f"Trade date range: {df['exit_date'].min()} to {df['exit_date'].max()}"
            )
        return df
    except FileNotFoundError:
        logger.warning(f"Trade history file not found: {TRADES_HISTORY_FILE}")
        return pd.DataFrame(columns=TRADE_COLUMNS)
    except pd.errors.EmptyDataError:
        logger.warning(
            f"Trade history file is empty or contains no valid data: {TRADES_HISTORY_FILE}"
        )
        return pd.DataFrame(columns=TRADE_COLUMNS)
    except Exception as e:
        logger.error(f"Unexpected error loading trade history: {e}")
        logger.error(f"File path: {os.path.abspath(TRADES_HISTORY_FILE)}")
        return pd.DataFrame(columns=TRADE_COLUMNS)


def get_trades_history_formatted() -> pd.DataFrame:
    try:
        trades_df = get_trades_history()
        if trades_df.empty:
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Symbol",
                    "Type",
                    "Shares",
                    "Entry",
                    "Exit",
                    "P&L",
                    "Days",
                ]
            )
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        trades_df["days_held"] = (
            trades_df["exit_date"] - trades_df["entry_date"]
        ).dt.days
        formatted = pd.DataFrame(
            {
                "Date": trades_df["exit_date"].dt.strftime("%Y-%m-%d"),
                "Symbol": trades_df["symbol"],
                "Type": (
                    trades_df["type"].str.title()
                    if "type" in trades_df.columns
                    else "Long"
                ),
                "Shares": (
                    trades_df["shares"].astype(int)
                    if "shares" in trades_df.columns
                    else 0
                ),
                "Entry": (
                    trades_df["entry_price"].apply(lambda x: f"${x:.2f}")
                    if "entry_price" in trades_df.columns
                    else "$0.00"
                ),
                "Exit": (
                    trades_df["exit_price"].apply(lambda x: f"${x:.2f}")
                    if "exit_price" in trades_df.columns
                    else "$0.00"
                ),
                "P&L": trades_df["profit"].apply(lambda x: format_dollars(x)),
                "Days": trades_df["days_held"],
            }
        )
        return formatted.sort_values("Date", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error formatting trade history: {e}")
        return pd.DataFrame(
            columns=["Date", "Symbol", "Type", "Shares", "Entry", "Exit", "P&L", "Days"]
        )


def get_me_ratio_history(file_path="data/me_ratio_history.csv") -> pd.DataFrame:
    """
    Retrieve the historical Margin-to-Equity (M/E) ratio data from the specified file.
    """
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            me_ratio_df = pd.read_csv(file_path)
            me_ratio_df["Date"] = pd.to_datetime(me_ratio_df["Date"], errors="coerce")
            return me_ratio_df.dropna(subset=["ME_Ratio"])
        else:
            logger.warning(
                f"File not found: {file_path}. Create it to track M/E ratios."
            )
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading M/E ratio history: {e}")
        return pd.DataFrame()


def save_trades(trades_list: List[Dict[str, Any]]) -> None:
    ensure_dir(TRADES_HISTORY_FILE)
    cutoff_date = get_cutoff_date()
    filtered_trades: List[Dict[str, Any]] = []
    for trade in trades_list:
        try:
            exit_date = parse_date_flexibly(trade["exit_date"])
            if exit_date >= cutoff_date:
                trade_copy = trade.copy()
                trade_copy["exit_date"] = exit_date.strftime("%Y-%m-%d")
                trade_copy["entry_date"] = parse_date_flexibly(
                    trade["entry_date"]
                ).strftime("%Y-%m-%d")
                if "side" not in trade_copy:
                    trade_copy["side"] = (
                        "long" if trade_copy.get("shares", 0) > 0 else "short"
                    )
                if "strategy" not in trade_copy:
                    trade_copy["strategy"] = "nGS"
                filtered_trades.append(trade_copy)
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid trade date in trade: {trade} - {e}")

    if not filtered_trades:
        logger.info("No trades within retention period to save")
        return

    if os.path.exists(TRADES_HISTORY_FILE):
        existing_df = pd.read_csv(TRADES_HISTORY_FILE)
        if not existing_df.empty:
            if "exit_date" in existing_df.columns:
                existing_df["exit_date"] = (
                    existing_df["exit_date"]
                    .apply(parse_date_flexibly)
                    .dt.strftime("%Y-%m-%d")
                )
            if "entry_date" in existing_df.columns:
                existing_df["entry_date"] = (
                    existing_df["entry_date"]
                    .apply(parse_date_flexibly)
                    .dt.strftime("%Y-%m-%d")
                )
        existing_df = filter_by_retention_period(existing_df, "exit_date")
        new_df = pd.DataFrame(filtered_trades)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(TRADES_HISTORY_FILE, index=False)
        logger.info(
            f"Saved {len(filtered_trades)} new trades, total: {len(combined_df)}"
        )
    else:
        df = pd.DataFrame(filtered_trades)
        df.to_csv(TRADES_HISTORY_FILE, index=False)
        logger.info(f"Created new trade history with {len(filtered_trades)} trades")


def get_positions_df() -> pd.DataFrame:
    if os.path.exists(POSITIONS_FILE):
        df = pd.read_csv(POSITIONS_FILE)
        if not df.empty and "entry_date" in df.columns:
            df = filter_by_retention_period(df, "entry_date")
        return df
    return pd.DataFrame(columns=POSITION_COLUMNS)


def save_positions(positions_list: List[Dict[str, Any]]) -> None:
    ensure_dir(POSITIONS_FILE)
    cutoff_date = get_cutoff_date()
    filtered_positions: List[Dict[str, Any]] = []
    for pos in positions_list:
        try:
            entry_date = parse_date_flexibly(pos["entry_date"])
            if entry_date >= cutoff_date:
                pos_copy = pos.copy()
                pos_copy["entry_date"] = entry_date.strftime("%Y-%m-%d")
                if "side" not in pos_copy:
                    pos_copy["side"] = (
                        "long" if pos_copy.get("shares", 0) > 0 else "short"
                    )
                if "strategy" not in pos_copy:
                    pos_copy["strategy"] = "nGS"
                filtered_positions.append(pos_copy)
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid entry date in position: {pos} - {e}")

    df = pd.DataFrame(filtered_positions)
    df.to_csv(POSITIONS_FILE, index=False)
    logger.info(f"Saved {len(filtered_positions)} positions within retention period")


def get_positions() -> List[Dict[str, Any]]:
    df = get_positions_df()
    return df.to_dict(orient="records") if not df.empty else []


def get_signals() -> pd.DataFrame:
    if os.path.exists(SIGNALS_FILE):
        df = pd.read_csv(SIGNALS_FILE)
        if not df.empty and "date" in df.columns:
            df = filter_by_retention_period(df, "date")
        return df
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def save_signals(signals: List[Dict[str, Any]]) -> None:
    ensure_dir(SIGNALS_FILE)
    cutoff_date = get_cutoff_date()
    filtered_signals: List[Dict[str, Any]] = []
    for signal in signals:
        try:
            signal_date = parse_date_flexibly(signal["date"])
            if signal_date >= cutoff_date:
                signal_copy = signal.copy()
                signal_copy["date"] = signal_date.strftime("%Y-%m-%d")
                filtered_signals.append(signal_copy)
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid date in signal: {signal} - {e}")

    df = pd.DataFrame(filtered_signals)
    df.to_csv(SIGNALS_FILE, index=False)
    logger.info(f"Saved {len(filtered_signals)} signals within retention period")


def save_system_status(message: str, system: str = "nGS") -> None:
    ensure_dir(SYSTEM_STATUS_FILE)
    now = datetime.now()
    cutoff_date = get_cutoff_date()
    if now >= cutoff_date:
        new_row = pd.DataFrame(
            [
                {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M"),
                    "system": system,
                    "message": message,
                }
            ]
        )
        if os.path.exists(SYSTEM_STATUS_FILE):
            df = pd.read_csv(SYSTEM_STATUS_FILE)
            if not df.empty:
                df["timestamp"] = df["timestamp"].apply(parse_date_flexibly)
                df = df[df["timestamp"] >= cutoff_date].copy()
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            df = pd.concat([new_row, df], ignore_index=True)
        else:
            df = new_row
        df.to_csv(SYSTEM_STATUS_FILE, index=False)


def init_metadata() -> Dict[str, Any]:
    if not os.path.exists(METADATA_FILE):
        metadata = {
            "created": datetime.now().isoformat(),
            "retention_days": RETENTION_DAYS,
            "last_cleanup": datetime.now().isoformat(),
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        if metadata.get("retention_days") != RETENTION_DAYS:
            metadata["retention_days"] = RETENTION_DAYS
            metadata["last_cleanup"] = datetime.now().isoformat()
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)
    return metadata


def update_metadata(key: str, value: Any) -> None:
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


def calculate_current_me_ratio(
    positions_df: pd.DataFrame, portfolio_value: float
) -> float:
    if positions_df.empty or portfolio_value <= 0:
        return 0.0
    try:
        long_positions = positions_df[positions_df["shares"] > 0]
        short_positions = positions_df[positions_df["shares"] < 0]
        long_value = (
            (long_positions["current_price"] * long_positions["shares"]).sum()
            if not long_positions.empty
            else 0
        )
        short_value = (
            (short_positions["current_price"] * short_positions["shares"].abs()).sum()
            if not short_positions.empty
            else 0
        )
        total_position_value = long_value + short_value
        me_ratio = (total_position_value / portfolio_value) * 100
        return max(me_ratio, 0.0)
    except Exception as e:
        logger.error(f"Error calculating current M/E ratio: {e}")
        return 0.0


def calculate_historical_me_ratio(
    trades_df: pd.DataFrame, initial_value: float = 1000000
) -> float:
    try:
        symbols = get_sp500_symbols()
        if not symbols:
            logger.warning("No S&P 500 symbols found for M/E calculation")
            return 18.5
        logger.info(f"Calculating historical M/E from {len(symbols)} S&P 500 symbols")
        me_ratios: List[float] = []
        symbols_with_data = 0
        sample_symbols = symbols[::5]
        for symbol in sample_symbols:
            try:
                df = load_price_data(symbol)
                if not df.empty and "ME_Ratio" in df.columns:
                    valid_me = df["ME_Ratio"][df["ME_Ratio"] > 0]
                    if not valid_me.empty:
                        me_ratios.extend(valid_me.tolist())
                        symbols_with_data += 1
            except Exception as e:
                logger.debug(f"Could not load M/E data for {symbol}: {e}")
                continue
        if me_ratios:
            avg_historical_me = np.mean(me_ratios)
            logger.info(
                f"Historical M/E calculated from {symbols_with_data} symbols: {avg_historical_me:.1f}%"
            )
            return avg_historical_me
        else:
            logger.warning("No M/E ratio data found in daily indicators")
    except Exception as e:
        logger.warning(f"Could not load historical M/E from indicators: {e}")
    return 18.5


def calculate_ytd_return(
    trades_df: pd.DataFrame, initial_value: float
) -> Tuple[str, str]:
    if trades_df.empty:
        return "$0", "0.00%"
    try:
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        current_year = datetime.now().year
        previous_year_trades = trades_df[trades_df["exit_date"].dt.year < current_year]
        if previous_year_trades.empty:
            ytd_profit = trades_df["profit"].sum()
        else:
            ytd_trades = trades_df[trades_df["exit_date"].dt.year == current_year]
            ytd_profit = ytd_trades["profit"].sum() if not ytd_trades.empty else 0
        ytd_pct = (ytd_profit / initial_value * 100) if initial_value > 0 else 0
        return format_dollars(ytd_profit), f"{ytd_pct:.2f}%"
    except Exception as e:
        logger.error(f"Error calculating YTD return: {e}")
        return "$0", "0.00%"


def calculate_mtd_return(
    trades_df: pd.DataFrame, initial_value: float
) -> Tuple[str, str]:
    if trades_df.empty:
        return "$0", "0.00%"
    current_date = datetime.now()
    current_month_start = datetime(current_date.year, current_date.month, 1)
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
    mtd_trades = trades_df[trades_df["exit_date"] >= current_month_start]
    mtd_profit = mtd_trades["profit"].sum() if not mtd_trades.empty else 0
    mtd_pct = (mtd_profit / initial_value * 100) if initial_value > 0 else 0
    return format_dollars(mtd_profit), f"{mtd_pct:.2f}%"


def get_portfolio_metrics(
    initial_portfolio_value: float = 1000000, is_historical: bool = False
) -> Dict[str, Any]:
    try:
        positions_df = get_positions_df()
        trades_df = get_trades_history()
        total_trade_profit = trades_df["profit"].sum() if not trades_df.empty else 0
        current_portfolio_value = initial_portfolio_value + total_trade_profit

        if not positions_df.empty:
            long_positions = positions_df[positions_df["shares"] > 0]
            short_positions = positions_df[positions_df["shares"] < 0]
            long_value = (
                (long_positions["current_price"] * long_positions["shares"]).sum()
                if not long_positions.empty
                else 0
            )
            short_value = (
                (
                    short_positions["current_price"] * short_positions["shares"].abs()
                ).sum()
                if not short_positions.empty
                else 0
            )
            net_exposure = long_value - short_value
            daily_pnl = positions_df["profit"].sum()
        else:
            long_value = 0
            short_value = 0
            net_exposure = 0
            daily_pnl = 0

        current_me_ratio = calculate_current_me_ratio(
            positions_df, current_portfolio_value
        )
        historical_me_ratio = calculate_historical_me_ratio(
            trades_df, initial_portfolio_value
        )

        total_return = total_trade_profit
        total_return_pct = (
            f"{(total_return / initial_portfolio_value * 100):.2f}%"
            if initial_portfolio_value > 0
            else "0.00%"
        )
        mtd_return, mtd_pct = calculate_mtd_return(trades_df, initial_portfolio_value)
        ytd_return, ytd_pct = calculate_ytd_return(trades_df, initial_portfolio_value)

        metrics: Dict[str, Any] = {
            "total_value": format_dollars(current_portfolio_value),
            "total_return_pct": total_return_pct,
            "daily_pnl": format_dollars(daily_pnl),
            "mtd_return": mtd_return,
            "mtd_delta": mtd_pct,
            "ytd_return": ytd_return,
            "ytd_delta": ytd_pct,
            "me_ratio": f"{current_me_ratio:.1f}%",
            "historical_me_ratio": f"{historical_me_ratio:.1f}%",
            "long_exposure": format_dollars(long_value),
            "short_exposure": format_dollars(short_value),
            "net_exposure": format_dollars(net_exposure),
        }
        return metrics
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {
            "total_value": format_dollars(initial_portfolio_value),
            "total_return_pct": "0.00%",
            "daily_pnl": "$0",
            "mtd_return": "$0",
            "mtd_delta": "0.00%",
            "ytd_return": "$0",
            "ytd_delta": "0.00%",
            "me_ratio": "0.0%",
            "historical_me_ratio": "18.5%",
            "long_exposure": "$0",
            "short_exposure": "$0",
            "net_exposure": "$0",
        }


def get_strategy_performance(initial_portfolio_value: float = 1000000) -> pd.DataFrame:
    try:
        trades_df = get_trades_history()
        if trades_df.empty:
            return pd.DataFrame(
                columns=["Strategy", "Trades", "Win Rate", "Total Profit", "Avg Profit"]
            )
        strategy_stats: List[Dict[str, Any]] = []
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["profit"] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = trades_df["profit"].sum()
        avg_profit = trades_df["profit"].mean()
        strategy_stats.append(
            {
                "Strategy": "nGS System",
                "Trades": total_trades,
                "Win Rate": f"{win_rate:.1%}",
                "Total Profit": format_dollars(total_profit),
                "Avg Profit": format_dollars(avg_profit),
            }
        )
        if "type" in trades_df.columns:
            for trade_type in trades_df["type"].unique():
                type_trades = trades_df[trades_df["type"] == trade_type]
                if not type_trades.empty:
                    type_total = len(type_trades)
                    type_wins = len(type_trades[type_trades["profit"] > 0])
                    type_win_rate = type_wins / type_total if type_total > 0 else 0
                    type_profit = type_trades["profit"].sum()
                    type_avg = type_trades["profit"].mean()
                    strategy_stats.append(
                        {
                            "Strategy": f"nGS {trade_type.title()}",
                            "Trades": type_total,
                            "Win Rate": f"{type_win_rate:.1%}",
                            "Total Profit": format_dollars(type_profit),
                            "Avg Profit": format_dollars(type_avg),
                        }
                    )
        return pd.DataFrame(strategy_stats)
    except Exception as e:
        logger.error(f"Error getting strategy performance: {e}")
        return pd.DataFrame(
            columns=["Strategy", "Trades", "Win Rate", "Total Profit", "Avg Profit"]
        )


def get_portfolio_performance_stats() -> pd.DataFrame:
    try:
        trades_df = get_trades_history()
        positions_df = get_positions_df()
        if trades_df.empty:
            return pd.DataFrame(columns=["Metric", "Value"])
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["profit"] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_profit = trades_df["profit"].sum()
        avg_win = (
            trades_df[trades_df["profit"] > 0]["profit"].mean()
            if winning_trades > 0
            else 0
        )
        avg_loss = (
            trades_df[trades_df["profit"] < 0]["profit"].mean()
            if losing_trades > 0
            else 0
        )
        total_wins = trades_df[trades_df["profit"] > 0]["profit"].sum()
        total_losses = abs(trades_df[trades_df["profit"] < 0]["profit"].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        current_positions = len(positions_df) if not positions_df.empty else 0
        unrealized_pnl = positions_df["profit"].sum() if not positions_df.empty else 0
        stats = [
            ["Total Trades", f"{total_trades}"],
            ["Win Rate", f"{win_rate:.1%}"],
            ["Total Profit", format_dollars(total_profit)],
            ["Avg Win", format_dollars(avg_win)],
            ["Avg Loss", format_dollars(avg_loss)],
            [
                "Profit Factor",
                f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞",
            ],
            ["Open Positions", f"{current_positions}"],
            ["Unrealized P&L", format_dollars(unrealized_pnl)],
        ]
        return pd.DataFrame(stats, columns=["Metric", "Value"])
    except Exception as e:
        logger.error(f"Error getting portfolio performance stats: {e}")
        return pd.DataFrame(columns=["Metric", "Value"])


def get_long_positions_formatted() -> pd.DataFrame:
    try:
        positions_df = get_positions_df()
        if positions_df.empty:
            return pd.DataFrame(
                columns=[
                    "Symbol",
                    "Shares",
                    "Entry Price",
                    "Current Price",
                    "P&L",
                    "P&L %",
                    "Days",
                ]
            )
        long_positions = positions_df[positions_df["shares"] > 0].copy()
        if long_positions.empty:
            return pd.DataFrame(
                columns=[
                    "Symbol",
                    "Shares",
                    "Entry Price",
                    "Current Price",
                    "P&L",
                    "P&L %",
                    "Days",
                ]
            )
        formatted = pd.DataFrame(
            {
                "Symbol": long_positions["symbol"],
                "Shares": long_positions["shares"].astype(int),
                "Entry Price": long_positions["entry_price"].apply(
                    lambda x: f"${x:.2f}"
                ),
                "Current Price": long_positions["current_price"].apply(
                    lambda x: f"${x:.2f}"
                ),
                "P&L": long_positions["profit"].apply(lambda x: format_dollars(x)),
                "P&L %": long_positions["profit_pct"].apply(lambda x: f"{x:.1f}%"),
                "Days": long_positions["days_held"].astype(int),
            }
        )
        return formatted.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error getting formatted long positions: {e}")
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Shares",
                "Entry Price",
                "Current Price",
                "P&L",
                "P&L %",
                "Days",
            ]
        )


def get_short_positions_formatted() -> pd.DataFrame:
    try:
        positions_df = get_positions_df()
        if positions_df.empty:
            return pd.DataFrame(
                columns=[
                    "Symbol",
                    "Shares",
                    "Entry Price",
                    "Current Price",
                    "P&L",
                    "P&L %",
                    "Days",
                ]
            )
        short_positions = positions_df[positions_df["shares"] < 0].copy()
        if short_positions.empty:
            return pd.DataFrame(
                columns=[
                    "Symbol",
                    "Shares",
                    "Entry Price",
                    "Current Price",
                    "P&L",
                    "P&L %",
                    "Days",
                ]
            )
        formatted = pd.DataFrame(
            {
                "Symbol": short_positions["symbol"],
                "Shares": short_positions["shares"].abs().astype(int),
                "Entry Price": short_positions["entry_price"].apply(
                    lambda x: f"${x:.2f}"
                ),
                "Current Price": short_positions["current_price"].apply(
                    lambda x: f"${x:.2f}"
                ),
                "P&L": short_positions["profit"].apply(lambda x: format_dollars(x)),
                "P&L %": short_positions["profit_pct"].apply(lambda x: f"{x:.1f}%"),
                "Days": short_positions["days_held"].astype(int),
            }
        )
        return formatted.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error getting formatted short positions: {e}")
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Shares",
                "Entry Price",
                "Current Price",
                "P&L",
                "P&L %",
                "Days",
            ]
        )


def get_long_positions() -> List[Dict[str, Any]]:
    try:
        positions_df = get_positions_df()
        if positions_df.empty:
            return []
        long_positions = positions_df[positions_df["shares"] > 0]
        return long_positions.to_dict("records")
    except Exception as e:
        logger.error(f"Error getting long positions: {e}")
        return []


def get_short_positions() -> List[Dict[str, Any]]:
    try:
        positions_df = get_positions_df()
        if positions_df.empty:
            return []
        short_positions = positions_df[positions_df["shares"] < 0]
        return short_positions.to_dict("records")
    except Exception as e:
        logger.error(f"Error getting short positions: {e}")
        return []


def initialize() -> None:
    ensure_dir(POSITIONS_FILE)
    ensure_dir(TRADES_HISTORY_FILE)
    ensure_dir(SIGNALS_FILE)
    ensure_dir(SYSTEM_STATUS_FILE)
    ensure_dir(METADATA_FILE)
    ensure_dir(DAILY_DIR)
    init_metadata()
    coverage_ok = verify_sp500_coverage()
    stats = get_sp500_symbol_stats()
    logger.info(f"S&P 500 Symbol Statistics:")
    logger.info(f"  Loaded: {stats['total_count']}/{stats['expected_count']} symbols")
    logger.info(f"  Coverage Complete: {stats['coverage_complete']}")
    logger.info(f"  Coverage Acceptable: {stats['coverage_acceptable']}")
    logger.info(f"  Duplicates: {stats['duplicate_count']}")
    if not coverage_ok:
        logger.warning(
            "  S&P 500 symbol coverage is incomplete - system performance may be affected"
        )
    logger.info("Initializing sector classification data...")
    try:
        sector_data = get_sp500_sector_data()
        sector_count = len(sector_data.get("sectors", {}))
        symbol_count = len(sector_data.get("symbol_to_sector", {}))
        logger.info(f" Loaded {symbol_count} symbols across {sector_count} sectors")
        for sector, info in sector_data.get("sector_info", {}).items():
            logger.info(f"  {sector}: {info['count']} symbols ({info['weight']:.1%})")
    except Exception as e:
        logger.error(f" Error initializing sector data: {e}")
        logger.warning("Sector-based features will be limited")
    logger.info(f"Data retention policy: {RETENTION_DAYS} days (6 months)")
    logger.info(f"Current cutoff date: {get_cutoff_date().strftime('%Y-%m-%d')}")
    logger.info("Data manager initialized with 6-month retention and sector support")


def get_historical_data(
    polygon_client: Any, symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    if not polygon_client:
        logger.error("Polygon API client not provided. Ensure POLYGON_API_KEY is set.")
        return pd.DataFrame()
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        agg = polygon_client.get_aggs(symbol, 1, "day", start_str, end_str)
        if not agg:
            logger.warning(
                f"No historical data available for {symbol} from {start_str} to {end_str}"
            )
            return pd.DataFrame()
        data = pd.DataFrame(agg)
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
        data = data.rename(
            columns={
                "timestamp": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
        for col in INDICATOR_COLUMNS:
            data[col] = np.nan
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        logger.info(
            f" Fetched and formatted historical data for {symbol}: {len(data)} rows"
        )
        logger.debug(f"Columns: {list(data.columns)}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def map_sic_to_sector(sic_description: str) -> Optional[str]:
    if not sic_description:
        return None
    description = sic_description.lower()
    mapping = {
        "technology": "Information Technology",
        "software": "Information Technology",
        "semiconductor": "Information Technology",
        "health": "Health Care",
        "pharma": "Health Care",
        "financial": "Financials",
        "bank": "Financials",
        "insurance": "Financials",
        "communication": "Communication Services",
        "telecom": "Communication Services",
        "media": "Communication Services",
        "consumer discretionary": "Consumer Discretionary",
        "retail": "Consumer Discretionary",
        "auto": "Consumer Discretionary",
        "industrial": "Industrials",
        "machinery": "Industrials",
        "manufacturing": "Industrials",
        "consumer staples": "Consumer Staples",
        "food": "Consumer Staples",
        "beverage": "Consumer Staples",
        "energy": "Energy",
        "oil": "Energy",
        "gas": "Energy",
        "utility": "Utilities",
        "real estate": "Real Estate",
        "material": "Materials",
        "chemical": "Materials",
        "mining": "Materials",
    }
    for keyword, sector in mapping.items():
        if keyword in description:
            return sector
    return None


__all__ = [
    "get_sp500_symbols",
    "filter_to_sp500",
    "verify_sp500_coverage",
    "get_sp500_symbol_stats",
    "get_sp500_sector_data",
    "fetch_sp500_sectors",
    "fetch_sectors_from_polygon",
    "fetch_sectors_from_wikipedia",
    "get_static_sector_mapping",
    "format_sector_data",
    "get_sector_symbols",
    "get_symbol_sector",
    "get_all_sectors",
    "get_sector_weights",
    "get_portfolio_sector_exposure",
    "get_sector_rebalance_targets",
    "calculate_sector_rebalance_needs",
    "save_price_data",
    "load_price_data",
    "get_trades_history",
    "get_trades_history_formatted",
    "get_me_ratio_history",
    "save_trades",
    "get_positions_df",
    "save_positions",
    "get_positions",
    "get_signals",
    "save_signals",
    "save_system_status",
    "init_metadata",
    "update_metadata",
    "calculate_current_me_ratio",
    "calculate_historical_me_ratio",
    "calculate_ytd_return",
    "calculate_mtd_return",
    "get_portfolio_metrics",
    "get_strategy_performance",
    "get_portfolio_performance_stats",
    "get_long_positions_formatted",
    "get_short_positions_formatted",
    "get_long_positions",
    "get_short_positions",
    "initialize",
    "get_historical_data",
    "map_sic_to_sector",
]

if __name__ == "__main__":
    initialize()
    logger.info(
        "data_manager.py loaded successfully with 6-month data retention and sector management"
    )
