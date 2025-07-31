import pandas as pd
import numpy as np
import logging
from typing import Callable, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ngs_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NGSStrategy:
    """
    Neural Grid Strategy (nGS) implementation using AI decision-making.
    """

    def __init__(self, data_manager_funcs: Dict[str, Callable], ai_model: Callable, account_size: float = 100000):
        """
        Initialize the nGS trading strategy with AI model.

        Args:
            data_manager_funcs: Dictionary of data manager functions (e.g., save_trade, update_positions).
            ai_model: Callable AI model for decision-making.
            account_size: Starting account value.
        """
        self.account_size = round(float(account_size), 2)
        self.cash = round(float(account_size), 2)
        self.positions = {}  # {symbol: {shares, entry_price, entry_date, bars_since_entry, profit}}
        self._trades = []  # Use backing variable for property

        # Injected functions from data_manager
        self.save_trade = data_manager_funcs['save_trade']
        self.update_positions = data_manager_funcs['update_positions']
        self.get_positions = data_manager_funcs['get_positions']

        # AI model for decision-making
        self.ai_decision_maker = ai_model

        # Load current positions
        self._load_positions()

    def _load_positions(self) -> None:
        """Load current positions using injected data manager function."""
        positions_list = self.get_positions()
        for pos in positions_list:
            symbol = pos.get('symbol')
            if symbol:
                self.positions[symbol] = {
                    'shares': int(pos.get('shares', 0)),
                    'entry_price': float(pos.get('entry_price', 0)),
                    'entry_date': pos.get('entry_date'),
                    'bars_since_entry': int(pos.get('days_held', 0)),
                    'profit': float(pos.get('profit', 0))
                }
        logger.info(f"Loaded {len(positions_list)} positions from data manager")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators required for the strategy.

        Args:
            df: DataFrame with price data.

        Returns:
            DataFrame with added indicators.
        """
        # Call AI model for indicator calculations
        indicators = self.ai_decision_maker(df, task="calculate_indicators")
        return indicators

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals using AI model.

        Args:
            df: DataFrame with price data and indicators.

        Returns:
            DataFrame with added signals.
        """
        signals = self.ai_decision_maker(df, task="generate_signals")
        return signals

    def manage_positions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Manage position exits and updates using AI model.

        Args:
            df: DataFrame with price data and signals.
            symbol: Stock symbol.

        Returns:
            DataFrame with updated positions.
        """
        decisions = self.ai_decision_maker(df, task="manage_positions")
        return decisions

# Example usage:
# data_manager_funcs = {'save_trade': save_trade, 'update_positions': update_positions, 'get_positions': get_positions}
# ai_model = some_ai_model_function
# strategy = NGSStrategy(data_manager_funcs, ai_model)