import os
import pandas as pd
import numpy as np
import json
import logging
import time
import re
from datetime import datetime, timedelta
from polygon import RESTClient
from typing import Dict, List, Optional, Union, Set

# Optional imports with fallbacks for sector functionality
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup

    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

# --- CONFIG ---
DATA_DIR = "."  # Use current directory
DAILY_DIR = os.path.join("data", "daily")  # Added missing DAILY_DIR
POSITIONS_FILE = "positions.csv"
TRADES_HISTORY_FILE = "trade_history.csv"
SIGNALS_FILE = "recent_signals.csv"
SYSTEM_STATUS_FILE = "system_status.csv"
METADATA_FILE = "metadata.json"
SP500_SYMBOLS_FILE = os.path.join("data", "sp500_symbols.txt")

# Sector Configuration
SECTOR_DATA_FILE = os.path.join("data", "sp500_sectors.json")
SECTOR_CACHE_HOURS = 24  # Refresh sector data daily

RETENTION_DAYS = 180  # 6 months data retention
PRIMARY_TIER_DAYS = 30
MAX_THREADS = 8
HISTORY_DAYS = 200  # Rolling window for daily data

# S&P 500 Configuration
SP500_EXPECTED_COUNT = 500  # Exact count for S&P 500
SP500_MINIMUM_COUNT = 490  # Minimum acceptable count (allowing for recent changes)

PRICE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
INDICATOR_COLUMNS = [
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
    "ME_Ratio",  # ADDED: M/E ratio as daily indicator
]
ALL_COLUMNS = PRICE_COLUMNS + INDICATOR_COLUMNS

# Standard S&P 500 Sectors (GICS - Global Industry Classification Standard)
SP500_SECTORS = {
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
    encoding="utf-8",  # Added to handle Unicode characters
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_manager.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# --- UTILITY FUNCTIONS ---
def get_cutoff_date():
    """Get the cutoff date for 6-month retention"""
    return datetime.now() - timedelta(days=RETENTION_DAYS)


# Python
def parse_date_flexibly(date_str):
    """
    Parse date string flexibly, handling both 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS' formats.
    Returns a datetime object.
    """
    if pd.isna(date_str):
        return pd.NaT

    # Convert to string if not already
    date_str = str(date_str)

    # Take only date part if time is included
    date_part = date_str.split()[0]

    try:
        # Try parsing as date only
        return pd.to_datetime(date_part, format="%Y-%m-%d")
    except Exception:
        try:
            # Fallback to pandas flexible parsing
            return pd.to_datetime(date_str)
        except Exception:
            logger.warning(f"Could not parse date: {date_str}")
            return pd.NaT

def format_dollars(value):
    """Format dollar amounts without cents"""
    if isinstance(value, str) and "$" in value:
        # Already formatted
        return value
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"

def filter_by_retention_period(
    df: pd.DataFrame, date_column: str = "Date"
) -> pd.DataFrame:
    """Filter DataFrame to only include data within retention period"""
    if df is None or df.empty:
        return df

    if date_column not in df.columns:
        logger.warning(f"Date column '{date_column}' not found in DataFrame")
        return df

    cutoff_date = get_cutoff_date()
    df = df.copy()

    # Handle mixed date formats
    try:
        # Apply flexible date parsing
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


# --- FILE UTILS ---
def ensure_dir(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


# --- FORMATTING UTILS ---
def format_dollars(value):
    """Format dollar amounts without cents"""
    if isinstance(value, str) and "$" in value:
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
    Returns a list of uppercase symbol strings.
    """
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
            # Log first few symbols for verification
            if symbols:
                logger.info(f"Sample symbols: {symbols[:5]}...")
            return symbols
    else:
        logger.warning(f"S&P 500 symbols file not found: {SP500_SYMBOLS_FILE}")
        return []


def filter_to_sp500(symbols: list) -> list:
    """
    Filter a list of symbols to only those present in the S&P 500 universe.
    """
    sp500_set = set(get_sp500_symbols())
    return [s for s in symbols if s.upper() in sp500_set]


def verify_sp500_coverage():
    """Simplified S&P 500 validation"""
    symbols = get_sp500_symbols()

    if not symbols:
        logger.error(" No S&P 500 symbols loaded!")
        return False

    symbol_count = len(symbols)
    logger.info(f"S&P 500 symbols loaded: {symbol_count}")

    # Simple count validation (S&P 500 should be ~500)
    if symbol_count < SP500_MINIMUM_COUNT:
        logger.error(f" Symbol count too low: {symbol_count} < {SP500_MINIMUM_COUNT}")
        return False
    elif symbol_count > 510:
        logger.warning(
            f" Symbol count high: {symbol_count} > 510 (may include additional securities)"
        )

    # Quick sanity check - just verify a few major symbols exist
    must_have = ["AAPL", "MSFT", "GOOGL"]  # Reduced to just 3 essential ones
    missing = [s for s in must_have if s not in symbols]

    if missing:
        logger.error(f" Missing major symbols: {missing}")
        return False

    # Check for duplicates
    duplicate_count = len(symbols) - len(set(symbols))
    if duplicate_count > 0:
        logger.warning(f"Found {duplicate_count} duplicate symbols")
    else:
        logger.info(" No duplicate symbols found")

    logger.info(" S&P 500 symbol validation passed")
    return True


def get_sp500_symbol_stats():
    """Get detailed statistics about S&P 500 symbol coverage"""
    symbols = get_sp500_symbols()

    stats = {
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
def get_sp500_sector_data(force_refresh: bool = False) -> Dict[str, Dict]:
    """
    Get S&P 500 sector classifications with symbol mappings.
    Returns: {
        "sectors": {"Technology": ["AAPL", "MSFT", ...], ...},
        "symbol_to_sector": {"AAPL": "Technology", ...},
        "sector_info": {"Technology": {"count": 75, "weight": 0.28}, ...}
    }
    """
    # Check if we have cached data
    if os.path.exists(SECTOR_DATA_FILE) and not force_refresh:
        try:
            with open(SECTOR_DATA_FILE, "r") as f:
                cached_data = json.load(f)

            # Check if cache is still fresh
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

    # Fetch fresh sector data
    logger.info("Fetching fresh S&P 500 sector data...")
    sector_data = fetch_sp500_sectors()

    # Save to cache
    sector_data["last_updated"] = datetime.now().isoformat()
    ensure_dir(SECTOR_DATA_FILE)
    with open(SECTOR_DATA_FILE, "w") as f:
        json.dump(sector_data, f, indent=2)

    logger.info(
        f"Cached sector data for {len(sector_data.get('symbol_to_sector', {}))} symbols"
    )
    return sector_data


def fetch_sp500_sectors() -> Dict[str, Dict]:
    """
    Fetch S&P 500 sector data from multiple sources with fallbacks.
    Returns structured sector data.
    """
    # Method 1: Try Polygon API if available
    polygon_data = fetch_sectors_from_polygon()
    if polygon_data:
        return polygon_data

    # Method 2: Try web scraping Wikipedia
    wiki_data = fetch_sectors_from_wikipedia()
    if wiki_data:
        return wiki_data

    # Method 3: Use built-in static mapping (fallback)
    logger.warning("Using static sector mapping - may be outdated")
    return get_static_sector_mapping()


def fetch_sectors_from_polygon() -> Dict[str, Dict]:
    """Fetch sector data using Polygon API"""
    try:
        polygon_api_key = os.getenv("POLYGON_API_KEY")
        if not polygon_api_key:
            return None

        # Get S&P 500 symbols
        symbols = get_sp500_symbols()
        if not symbols:
            return None

        polygon_client = RESTClient(polygon_api_key)
        sectors = {}
        symbol_to_sector = {}

        # Batch requests for efficiency
        for i in range(0, len(symbols), 50):  # Process in batches of 50
            batch = symbols[i : i + 50]
            for symbol in batch:
                try:
                    # Get ticker details
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

            time.sleep(0.1)  # Rate limiting

        if sectors:
            return format_sector_data(sectors, symbol_to_sector)

    except Exception as e:
        logger.warning(f"Polygon sector fetch failed: {e}")

    return None


def fetch_sectors_from_wikipedia() -> Dict[str, Dict]:
    """Fetch sector data from Wikipedia S&P 500 page"""
    try:
        if not HAS_REQUESTS or not HAS_BEAUTIFULSOUP:
            return None

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        # Find the main S&P 500 table
        table = soup.find("table", {"id": "constituents"})
        if not table:
            table = soup.find("table", class_="wikitable")

        if not table:
            return None

        sectors = {}
        symbol_to_sector = {}

        rows = table.find_all("tr")[1:]  # Skip header
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 4:
                symbol = cells[0].get_text().strip()
                sector = cells[3].get_text().strip()  # GICS Sector column

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


def get_static_sector_mapping() -> Dict[str, Dict]:
    """Static sector mapping as fallback"""
    static_sectors = {
        "Information Technology": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "GOOG",
            "META",
            "TSLA",
            "NVDA",
            "CRM",
            "ORCL",
            "ADBE",
            "NFLX",
            "INTC",
            "CSCO",
            "AMD",
            "IBM",
            "NOW",
            "TXN",
            "QCOM",
            "AMAT",
            "MU",
        ],
        "Health Care": [
            "UNH",
            "JNJ",
            "PFE",
            "ABT",
            "TMO",
            "LLY",
            "ABBV",
            "MRK",
            "MDT",
            "BMY",
            "AMGN",
            "GILD",
            "CVS",
            "DHR",
            "SYK",
            "BDX",
            "REGN",
            "VRTX",
            "ELV",
            "CI",
        ],
        "Financials": [
            "BRK.B",
            "JPM",
            "V",
            "MA",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "AXP",
            "BLK",
            "SPGI",
            "C",
            "MMC",
            "CB",
            "PGR",
            "ICE",
            "TFC",
            "AON",
            "USB",
            "COF",
        ],
        "Communication Services": [
            "GOOGL",
            "GOOG",
            "META",
            "NFLX",
            "DIS",
            "CMCSA",
            "VZ",
            "T",
            "TMUS",
            "CHTR",
        ],
        "Consumer Discretionary": [
            "AMZN",
            "TSLA",
            "HD",
            "MCD",
            "NKE",
            "LOW",
            "SBUX",
            "TJX",
            "BKNG",
            "F",
            "GM",
            "MAR",
            "HLT",
            "ABNB",
            "CMG",
            "LRCX",
            "ORLY",
            "TGT",
            "AZO",
            "ROST",
        ],
        "Industrials": [
            "BA",
            "CAT",
            "UNP",
            "RTX",
            "HON",
            "UPS",
            "LMT",
            "DE",
            "ADP",
            "GE",
            "MMM",
            "FDX",
            "NOC",
            "WM",
            "EMR",
            "ITW",
            "CSX",
            "GD",
            "NSC",
            "RSG",
        ],
        "Consumer Staples": [
            "PG",
            "KO",
            "PEP",
            "WMT",
            "COST",
            "MDLZ",
            "CL",
            "KMB",
            "GIS",
            "K",
            "SYY",
            "HSY",
            "MKC",
            "CPB",
            "CAG",
            "CLX",
            "TSN",
            "HRL",
            "LW",
            "CHD",
        ],
        "Energy": [
            "XOM",
            "CVX",
            "COP",
            "EOG",
            "SLB",
            "PSX",
            "VLO",
            "MPC",
            "KMI",
            "OKE",
            "WMB",
            "BKR",
            "HAL",
            "DVN",
            "FANG",
            "APA",
            "EQT",
            "CTRA",
            "MRO",
            "OXY",
        ],
        "Utilities": [
            "NEE",
            "SO",
            "DUK",
            "AEP",
            "SRE",
            "D",
            "PEG",
            "EXC",
            "XEL",
            "ED",
            "WEC",
            "AWK",
            "ES",
            "DTE",
            "ETR",
            "FE",
            "EIX",
            "PPL",
            "ATO",
            "CMS",
        ],
        "Real Estate": [
            "PLD",
            "AMT",
            "CCI",
            "EQIX",
            "PSA",
            "EXR",
            "AVB",
            "EQR",
            "WELL",
            "DLR",
            "SBAC",
            "VTR",
            "ESS",
            "MAA",
            "ARE",
            "INVH",
            "UDR",
            "CPT",
            "HST",
            "REG",
        ],
        "Materials": [
            "LIN",
            "APD",
            "SHW",
            "FCX",
            "NUE",
            "NEM",
            "DOW",
            "VMC",
            "MLM",
            "PPG",
            "ECL",
            "CTVA",
            "DD",
            "ALB",
            "IFF",
            "PKG",
            "BALL",
            "AMCR",
            "AVY",
            "CF",
        ],
    }

    symbol_to_sector = {}
    for sector, symbols in static_sectors.items():
        for symbol in symbols:
            symbol_to_sector[symbol] = sector

    return format_sector_data(static_sectors, symbol_to_sector)


def format_sector_data(
    sectors: Dict[str, List], symbol_to_sector: Dict[str, str]
) -> Dict:
    """Format sector data into standard structure"""
    sector_info = {}
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
    """Get all symbols in a specific sector"""
    sector_data = get_sp500_sector_data()
    return sector_data.get("sectors", {}).get(sector_name, [])


def get_symbol_sector(symbol: str) -> str:
    """Get the sector for a specific symbol"""
    sector_data = get_sp500_sector_data()
    return sector_data.get("symbol_to_sector", {}).get(symbol, "Unknown")


def get_all_sectors() -> List[str]:
    """Get list of all sector names"""
    sector_data = get_sp500_sector_data()
    return list(sector_data.get("sectors", {}).keys())


def get_sector_weights() -> Dict[str, float]:
    """Get sector weights (percentage of S&P 500)"""
    sector_data = get_sp500_sector_data()
    return {
        sector: info.get("weight", 0)
        for sector, info in sector_data.get("sector_info", {}).items()
    }


def get_portfolio_sector_exposure(positions_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Calculate current portfolio exposure by sector
    Args:
        positions_df: DataFrame with position data including 'symbol' and 'current_value' columns
    Returns:
        Dict with sector exposure analysis
    """
    if positions_df.empty:
        return {}

    sector_data = get_sp500_sector_data()
    symbol_to_sector = sector_data.get("symbol_to_sector", {})

    sector_exposure = {}
    total_portfolio_value = positions_df["current_value"].sum()

    for _, position in positions_df.iterrows():
        symbol = position["symbol"]
        value = position["current_value"]
        sector = symbol_to_sector.get(symbol, "Unknown")

        if sector not in sector_exposure:
            sector_exposure[sector] = {
                "value": 0,
                "weight": 0,
                "symbols": [],
                "count": 0,
            }

        sector_exposure[sector]["value"] += value
        sector_exposure[sector]["symbols"].append(symbol)
        sector_exposure[sector]["count"] += 1

    # Calculate weights
    for sector in sector_exposure:
        if total_portfolio_value > 0:
            sector_exposure[sector]["weight"] = (
                sector_exposure[sector]["value"] / total_portfolio_value
            )

    return sector_exposure


def get_sector_rebalance_targets(
    target_weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Get target sector weights for rebalancing
    Args:
        target_weights: Custom sector weights, defaults to S&P 500 sector weights
    Returns:
        Dict of sector -> target weight
    """
    if target_weights is None:
        # Use S&P 500 sector weights as default
        return get_sector_weights()

    return target_weights


def calculate_sector_rebalance_needs(
    positions_df: pd.DataFrame, target_weights: Dict[str, float] = None
) -> Dict[str, Dict]:
    """
    Calculate how much rebalancing is needed per sector
    """
    current_exposure = get_portfolio_sector_exposure(positions_df)
    targets = get_sector_rebalance_targets(target_weights)

    rebalance_needs = {}
    total_value = positions_df["current_value"].sum() if not positions_df.empty else 0

    for sector in targets:
        current_weight = current_exposure.get(sector, {}).get("weight", 0)
        target_weight = targets[sector]
        difference = target_weight - current_weight

        rebalance_needs[sector] = {
            "current_weight": current_weight,
            "target_weight": target_weight,
            "difference": difference,
            "dollar_adjustment": difference * total_value if total_value > 0 else 0,
            "action": "buy" if difference > 0 else "sell" if difference < 0 else "hold",
        }

    return rebalance_needs


# --- PRICE + INDICATOR DATA ---
def save_price_data(symbol: str, df: pd.DataFrame, history_days: int = HISTORY_DAYS):
    """
    Save the DataFrame with price + indicator columns for a symbol.
    Only the most recent `history_days` rows are retained.
    """
    filename = os.path.join(DAILY_DIR, f"{symbol}.csv")
    ensure_dir(filename)
    if not df.empty:
        # Apply 6-month filtering first
        df = filter_by_retention_period(df, "Date")

        # Ensure correct columns and order
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

        # Legacy dashboard compatibility: copy to data/me_ratio_history.csv
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
        # Apply 6-month filtering when loading
        return filter_by_retention_period(df, "Date")
    else:
        return pd.DataFrame(columns=ALL_COLUMNS)


# --- TRADES, POSITIONS, SIGNALS, METADATA, INITIALIZATION ---

TRADE_COLUMNS = [
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
POSITION_COLUMNS = [
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
SIGNAL_COLUMNS = ["date", "symbol", "signal_type", "direction", "price", "strategy"]
SYSTEM_STATUS_COLUMNS = ["timestamp", "system", "message"]


# --- TRADE HISTORY ---
def get_trades_history():
    """
    Get trade history with 6-month filtering and improved error handling.

    Returns:
        DataFrame with trade history filtered to last 6 months
    """
    try:
        # Check if file exists
        if not os.path.exists(TRADES_HISTORY_FILE):
            logger.warning(f"Trade history file not found: {TRADES_HISTORY_FILE}")
            logger.info(f"Looking for file at: {os.path.abspath(TRADES_HISTORY_FILE)}")
            return pd.DataFrame(columns=TRADE_COLUMNS)

        # Check file size
        file_size = os.path.getsize(TRADES_HISTORY_FILE)
        if file_size == 0:
            logger.warning(f"Trade history file is empty: {TRADES_HISTORY_FILE}")
            return pd.DataFrame(columns=TRADE_COLUMNS)

        logger.info(
            f"Loading trade history from {TRADES_HISTORY_FILE} ({file_size} bytes)"
        )

        # Read the file
        df = pd.read_csv(TRADES_HISTORY_FILE)

        # Validate columns
        if df.empty:
            logger.warning("Trade history file contains no data")
            return pd.DataFrame(columns=TRADE_COLUMNS)

        # Check for required columns
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

        # Convert dates with flexible parsing
        try:
            df["entry_date"] = df["entry_date"].apply(parse_date_flexibly)
            df["exit_date"] = df["exit_date"].apply(parse_date_flexibly)
        except Exception as e:
            logger.error(f"Error parsing dates in trade history: {e}")
            # Continue anyway, dates might still be usable as strings

        # Apply 6-month filtering based on exit_date
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
    """
    Get formatted trade history for dashboard display (6-month filtered).

    Returns:
        DataFrame with formatted trade history
    """
    try:
        trades_df = get_trades_history()  # Already 6-month filtered

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

        # Calculate days held
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])
        trades_df["days_held"] = (
            trades_df["exit_date"] - trades_df["entry_date"]
        ).dt.days

        # Format for display
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

        # Sort by exit date descending (most recent first)
        return formatted.sort_values("Date", ascending=False).reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error formatting trade history: {e}")
        return pd.DataFrame(
            columns=["Date", "Symbol", "Type", "Shares", "Entry", "Exit", "P&L", "Days"]
        )


def get_me_ratio_history() -> pd.DataFrame:
    """
    Get M/E ratio history from dedicated portfolio ME file.

    Returns:
        DataFrame with Date and ME_Ratio columns
    """
    try:
        filename = "data/me_ratio_history.csv"

        if not os.path.exists(filename):
            logger.warning(f"Portfolio M/E history file not found: {filename}")
            return pd.DataFrame(columns=["Date", "ME_Ratio"])

        df = pd.read_csv(filename, parse_dates=["Date"])

        # Filter to last 6 months
        df = filter_by_retention_period(df, "Date")

        if df.empty:
            logger.warning("No M/E ratio data after filtering")
            return pd.DataFrame(columns=["Date", "ME_Ratio"])

        # Select only needed columns and sort
        history_df = df[["Date", "ME_Ratio"]].sort_values("Date").reset_index(drop=True)

        logger.info(f"Loaded M/E ratio history: {len(history_df)} days")
        return history_df

    except Exception as e:
        logger.error(f"Error loading M/E ratio history: {e}")
        return pd.DataFrame(columns=["Date", "ME_Ratio"])


def save_trades(trades_list: List[Dict]):
    ensure_dir(TRADES_HISTORY_FILE)

    # Filter new trades to retention period
    cutoff_date = get_cutoff_date()
    filtered_trades = []

    for trade in trades_list:
        try:
            # Handle date format flexibly
            exit_date = parse_date_flexibly(trade["exit_date"])

            if exit_date >= cutoff_date:
                # Ensure dates are in consistent format
                trade_copy = trade.copy()
                trade_copy["exit_date"] = exit_date.strftime("%Y-%m-%d")
                trade_copy["entry_date"] = parse_date_flexibly(
                    trade["entry_date"]
                ).strftime("%Y-%m-%d")
                # Ensure 'side' and 'strategy' fields exist for dashboard/strategy compatibility
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
        # Append to existing trades
        existing_df = pd.read_csv(TRADES_HISTORY_FILE)

        # Convert existing dates to consistent format
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

        # Filter existing trades to retention period too
        existing_df = filter_by_retention_period(existing_df, "exit_date")

        new_df = pd.DataFrame(filtered_trades)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(TRADES_HISTORY_FILE, index=False)
        logger.info(
            f"Saved {len(filtered_trades)} new trades, total: {len(combined_df)}"
        )
    else:
        # Create new file
        df = pd.DataFrame(filtered_trades)
        df.to_csv(TRADES_HISTORY_FILE, index=False)
        logger.info(f"Created new trade history with {len(filtered_trades)} trades")


# --- POSITIONS ---
def get_positions_df():
    if os.path.exists(POSITIONS_FILE):
        df = pd.read_csv(POSITIONS_FILE)
        # Filter positions to retention period based on entry_date
        if not df.empty and "entry_date" in df.columns:
            df = filter_by_retention_period(df, "entry_date")
        return df
    return pd.DataFrame(columns=POSITION_COLUMNS)


def save_positions(positions_list: List[Dict]):
    ensure_dir(POSITIONS_FILE)

    # Filter positions to retention period
    cutoff_date = get_cutoff_date()
    filtered_positions = []

    for pos in positions_list:
        try:
            entry_date = parse_date_flexibly(pos["entry_date"])
            if entry_date >= cutoff_date:
                # Ensure date is in consistent format
                pos_copy = pos.copy()
                pos_copy["entry_date"] = entry_date.strftime("%Y-%m-%d")
                # Ensure 'side' and 'strategy' fields exist for dashboard/strategy compatibility
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


def get_positions():
    df = get_positions_df()
    return df.to_dict(orient="records") if not df.empty else []


# --- SIGNALS ---
def get_signals():
    if os.path.exists(SIGNALS_FILE):
        df = pd.read_csv(SIGNALS_FILE)
        # Filter signals to retention period
        if not df.empty and "date" in df.columns:
            df = filter_by_retention_period(df, "date")
        return df
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def save_signals(signals: List[Dict]):
    ensure_dir(SIGNALS_FILE)

    # Filter signals to retention period
    cutoff_date = get_cutoff_date()
    filtered_signals = []

    for signal in signals:
        try:
            signal_date = parse_date_flexibly(signal["date"])
            if signal_date >= cutoff_date:
                # Ensure date is in consistent format
                signal_copy = signal.copy()
                signal_copy["date"] = signal_date.strftime("%Y-%m-%d")
                filtered_signals.append(signal_copy)
        except (KeyError, ValueError) as e:
            logger.warning(f"Invalid date in signal: {signal} - {e}")

    df = pd.DataFrame(filtered_signals)
    df.to_csv(SIGNALS_FILE, index=False)
    logger.info(f"Saved {len(filtered_signals)} signals within retention period")


# --- SYSTEM STATUS ---
def save_system_status(message: str, system: str = "nGS"):
    ensure_dir(SYSTEM_STATUS_FILE)
    now = datetime.now()

    # Only save if within retention period (should always be true for new status)
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
            # Filter existing status to retention period
            if not df.empty:
                df["timestamp"] = df["timestamp"].apply(parse_date_flexibly)
                df = df[df["timestamp"] >= cutoff_date].copy()
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
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
            "last_cleanup": datetime.now().isoformat(),
        }
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        # Update retention days if changed
        if metadata.get("retention_days") != RETENTION_DAYS:
            metadata["retention_days"] = RETENTION_DAYS
            metadata["last_cleanup"] = datetime.now().isoformat()
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)
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
def calculate_current_me_ratio(
    positions_df: pd.DataFrame, portfolio_value: float
) -> float:
    """
    Calculate current M/E ratio from open positions.
    M/E = (Total Position Value / Portfolio Equity) * 100
    """
    if positions_df.empty or portfolio_value <= 0:
        return 0.0

    try:
        # Calculate total position value (both long and short)
        long_positions = positions_df[positions_df["shares"] > 0]
        short_positions = positions_df[positions_df["shares"] < 0]

        # Long value = price * shares
        long_value = (
            (long_positions["current_price"] * long_positions["shares"]).sum()
            if not long_positions.empty
            else 0
        )

        # Short value = price * abs(shares)
        short_value = (
            (short_positions["current_price"] * short_positions["shares"].abs()).sum()
            if not short_positions.empty
            else 0
        )

        # Total position value (represents margin usage)
        total_position_value = long_value + short_value

        # M/E ratio
        me_ratio = (total_position_value / portfolio_value) * 100

        return max(me_ratio, 0.0)

    except Exception as e:
        logger.error(f"Error calculating current M/E ratio: {e}")
        return 0.0


def calculate_historical_me_ratio(
    trades_df: pd.DataFrame, initial_value: float = 1000000
) -> float:
    """
    Get historical M/E ratio from stored daily indicator data across all S&P 500 symbols.
    """
    try:
        # Get all S&P 500 symbols for comprehensive M/E history
        symbols = get_sp500_symbols()

        if not symbols:
            logger.warning("No S&P 500 symbols found for M/E calculation")
            return 18.5  # Reasonable fallback

        logger.info(f"Calculating historical M/E from {len(symbols)} S&P 500 symbols")

        me_ratios = []
        symbols_with_data = 0

        # Sample a reasonable number of symbols to avoid performance issues
        # Use every 5th symbol for efficiency while maintaining representativeness
        sample_symbols = symbols[::5]  # Every 5th symbol

        for symbol in sample_symbols:
            try:
                df = load_price_data(symbol)  # Already 6-month filtered
                if not df.empty and "ME_Ratio" in df.columns:
                    # Get valid M/E ratios (greater than 0)
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

    # Fallback to reasonable estimate (slightly higher than current)
    return 18.5  # Reasonable historical average


def calculate_ytd_return(trades_df: pd.DataFrame, initial_value: float) -> tuple:
    """
    Calculate Year-to-Date return from closed trades (6-month filtered).
    """
    if trades_df.empty:
        return "$0", "0.00%"

    try:
        # Convert dates
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

        # Get current year
        current_year = datetime.now().year

        # Check if there are any trades from previous years
        previous_year_trades = trades_df[trades_df["exit_date"].dt.year < current_year]

        if previous_year_trades.empty:
            # No previous year trades - YTD should equal Total Return
            ytd_profit = trades_df["profit"].sum()
        else:
            # There are previous year trades - calculate YTD only
            ytd_trades = trades_df[trades_df["exit_date"].dt.year == current_year]
            ytd_profit = ytd_trades["profit"].sum() if not ytd_trades.empty else 0

        # Calculate percentage
        ytd_pct = (ytd_profit / initial_value * 100) if initial_value > 0 else 0

        return format_dollars(ytd_profit), f"{ytd_pct:.2f}%"

    except Exception as e:
        logger.error(f"Error calculating YTD return: {e}")
        return "$0", "0.00%"


def calculate_mtd_return(trades_df: pd.DataFrame, initial_value: float) -> tuple:
    """Calculate Month-to-Date return from closed trades (6-month filtered)"""
    if trades_df.empty:
        return "$0", "0.00%"

    # Get current month trades
    current_date = datetime.now()
    current_month_start = datetime(current_date.year, current_date.month, 1)

    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

    # Filter for current month trades
    mtd_trades = trades_df[trades_df["exit_date"] >= current_month_start]
    mtd_profit = mtd_trades["profit"].sum() if not mtd_trades.empty else 0

    # Calculate percentage
    mtd_pct = (mtd_profit / initial_value * 100) if initial_value > 0 else 0

    return format_dollars(mtd_profit), f"{mtd_pct:.2f}%"


# --- DASHBOARD FUNCTIONS FOR LONG/SHORT SYSTEM ---
def get_portfolio_metrics(
    initial_portfolio_value: float = 1000000, is_historical: bool = False
) -> Dict:
    """
    Calculate portfolio metrics with proper current vs historical M/E ratios (6-month filtered).

    Args:
        initial_portfolio_value: Starting portfolio value
        is_historical: True for historical page, False for current trading page

    Returns:
        Dictionary with portfolio metrics including proper M/E ratios
    """
    try:
        # Get data (already 6-month filtered)
        positions_df = get_positions_df()
        trades_df = get_trades_history()

        # Calculate from historical trades
        total_trade_profit = trades_df["profit"].sum() if not trades_df.empty else 0

        # Current portfolio value from closed trades
        current_portfolio_value = initial_portfolio_value + total_trade_profit

        # Calculate exposures from open positions
        if not positions_df.empty:
            long_positions = positions_df[positions_df["shares"] > 0]
            short_positions = positions_df[positions_df["shares"] < 0]

            # Calculate position values (full value, not just margin)
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

            # Net exposure
            net_exposure = long_value - short_value

            # Daily P&L (unrealized from current positions)
            daily_pnl = positions_df["profit"].sum()
        else:
            # No open positions
            long_value = 0
            short_value = 0
            net_exposure = 0
            daily_pnl = 0

        # Current M/E Ratio (for main page - based on current open positions)
        current_me_ratio = calculate_current_me_ratio(
            positions_df, current_portfolio_value
        )

        # Historical M/E Ratio (for historical page - from daily indicator data)
        historical_me_ratio = calculate_historical_me_ratio(
            trades_df, initial_portfolio_value
        )

        # Returns based on closed trades
        total_return = total_trade_profit
        total_return_pct = (
            f"{(total_return / initial_portfolio_value * 100):.2f}%"
            if initial_portfolio_value > 0
            else "0.00%"
        )

        # Calculate proper MTD and YTD
        mtd_return, mtd_pct = calculate_mtd_return(trades_df, initial_portfolio_value)
        ytd_return, ytd_pct = calculate_ytd_return(trades_df, initial_portfolio_value)

        # Format all dollar amounts without cents
        metrics = {
            "total_value": format_dollars(current_portfolio_value),
            "total_return_pct": total_return_pct,
            "daily_pnl": format_dollars(daily_pnl),
            "mtd_return": mtd_return,
            "mtd_delta": mtd_pct,
            "ytd_return": ytd_return,
            "ytd_delta": ytd_pct,
            "me_ratio": f"{current_me_ratio:.1f}%",  # Current M/E for main page
            "historical_me_ratio": f"{historical_me_ratio:.1f}%",  # Historical M/E for historical page
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
            "historical_me_ratio": "18.5%",  # Default reasonable value
            "long_exposure": "$0",
            "short_exposure": "$0",
            "net_exposure": "$0",
        }


def get_strategy_performance(initial_portfolio_value: float = 1000000) -> pd.DataFrame:
    """
    Get strategy performance summary (6-month filtered).

    Args:
        initial_portfolio_value: Starting portfolio value

    Returns:
        DataFrame with strategy performance
    """
    try:
        trades_df = get_trades_history()  # Already 6-month filtered

        if trades_df.empty:
            return pd.DataFrame(
                columns=["Strategy", "Trades", "Win Rate", "Total Profit", "Avg Profit"]
            )

        # Group by strategy type (using signal type from trades)
        strategy_stats = []

        # Overall nGS Strategy stats
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

        # Breakdown by trade type if available
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
    """
    Get detailed portfolio performance statistics for display (6-month filtered).

    Returns:
        DataFrame with performance statistics
    """
    try:
        trades_df = get_trades_history()  # Already 6-month filtered
        positions_df = get_positions_df()  # Already 6-month filtered

        if trades_df.empty:
            return pd.DataFrame(columns=["Metric", "Value"])

        # Calculate statistics
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

        # Profit factor
        total_wins = trades_df[trades_df["profit"] > 0]["profit"].sum()
        total_losses = abs(trades_df[trades_df["profit"] < 0]["profit"].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # Current positions
        current_positions = len(positions_df) if not positions_df.empty else 0
        unrealized_pnl = positions_df["profit"].sum() if not positions_df.empty else 0

        # Create stats DataFrame - format without cents
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
    """
    Get formatted long positions for dashboard display (6-month filtered).

    Returns:
        DataFrame with formatted long positions
    """
    try:
        positions_df = get_positions_df()  # Already 6-month filtered

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

        # Filter for long positions
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

        # Format for display - no cents on P&L
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
    """
    Get formatted short positions for dashboard display (6-month filtered).

    Returns:
        DataFrame with formatted short positions
    """
    try:
        positions_df = get_positions_df()  # Already 6-month filtered

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

        # Filter for short positions
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

        # Format for display (show absolute shares for shorts) - no cents on P&L
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


def get_long_positions() -> List[Dict]:
    """
    Get current long positions (6-month filtered).

    Returns:
        List of long position dictionaries
    """
    try:
        positions_df = get_positions_df()  # Already 6-month filtered

        if positions_df.empty:
            return []

        # Filter for long positions (positive shares)
        long_positions = positions_df[positions_df["shares"] > 0]

        # Convert to list of dictionaries for Streamlit
        return long_positions.to_dict("records")

    except Exception as e:
        logger.error(f"Error getting long positions: {e}")
        return []


def get_short_positions() -> List[Dict]:
    """
    Get current short positions (6-month filtered).

    Returns:
        List of short position dictionaries
    """
    try:
        positions_df = get_positions_df()  # Already 6-month filtered

        if positions_df.empty:
            return []

        # Filter for short positions (negative shares)
        short_positions = positions_df[positions_df["shares"] < 0]

        # Convert to list of dictionaries for Streamlit
        return short_positions.to_dict("records")

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

    # Verify S&P 500 coverage on initialization
    coverage_ok = verify_sp500_coverage()

    # Log symbol statistics
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

    # Initialize sector data
    logger.info("Initializing sector classification data...")
    try:
        sector_data = get_sp500_sector_data()
        sector_count = len(sector_data.get("sectors", {}))
        symbol_count = len(sector_data.get("symbol_to_sector", {}))
        logger.info(f" Loaded {symbol_count} symbols across {sector_count} sectors")

        # Log sector summary
        for sector, info in sector_data.get("sector_info", {}).items():
            logger.info(f"  {sector}: {info['count']} symbols ({info['weight']:.1%})")

    except Exception as e:
        logger.error(f" Error initializing sector data: {e}")
        logger.warning("Sector-based features will be limited")

    # Log retention policy
    logger.info(f"Data retention policy: {RETENTION_DAYS} days (6 months)")
    logger.info(f"Current cutoff date: {get_cutoff_date().strftime('%Y-%m-%d')}")

    logger.info("Data manager initialized with 6-month retention and sector support")


if __name__ == "__main__":
    initialize()
    logger.info(
        "data_manager.py loaded successfully with 6-month data retention and sector management"
    )


# --- HISTORICAL DATA WITH POLYGON (FIXED COLUMN CASE) ---
def get_historical_data(
    polygon_client, symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol using Polygon API with proper column formatting.

    Args:
        polygon_client: Initialized RESTClient instance.
        symbol (str): Stock symbol (e.g., "AAPL").
        start_date (datetime): Start date for data.
        end_date (datetime): End date for data.

    Returns:
        pd.DataFrame: Historical data with properly formatted columns.
    """
    if not polygon_client:
        logger.error("Polygon API client not provided. Ensure POLYGON_API_KEY is set.")
        return pd.DataFrame()

    try:
        # Convert datetime to string format expected by Polygon
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Fetch aggregated daily data from Polygon
        agg = polygon_client.get_aggs(symbol, 1, "day", start_str, end_str)
        if not agg:
            logger.warning(
                f"No historical data available for {symbol} from {start_str} to {end_str}"
            )
            return pd.DataFrame()

        # Convert Polygon response to DataFrame
        data = pd.DataFrame(agg)
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")

        # FIXED: Rename columns to match system expectations (capitalized)
        data = data.rename(
            columns={
                "timestamp": "Date",
                "open": "Open",  # Fix: capitalize from lowercase
                "high": "High",  # Fix: capitalize from lowercase
                "low": "Low",  # Fix: capitalize from lowercase
                "close": "Close",  # Fix: capitalize from lowercase
                "volume": "Volume",  # Fix: capitalize from lowercase
            }
        )

        # Ensure Date is in string format, not datetime index
        data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")

        # Add missing indicator columns with NaN values
        # The system expects these columns to exist
        for col in INDICATOR_COLUMNS:
            data[col] = np.nan

        # Ensure numeric columns are proper type
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


