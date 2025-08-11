import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory"""
    return Path("tests/test_data")

@pytest.fixture
def sample_price_data():
    """Fixture providing sample price data for testing"""
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='B')
    data = {
        'Date': dates,
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_sector_data():
    """Fixture providing mock sector mapping data"""
    return {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLY': 'Consumer Discretionary'
    }

@pytest.fixture
def mock_positions():
    """Fixture providing sample trading positions"""
    return [
        {'symbol': 'AAPL', 'quantity': 100, 'entry_price': 150.0, 'side': 'long'},
        {'symbol': 'MSFT', 'quantity': -50, 'entry_price': 200.0, 'side': 'short'}
    ]

@pytest.fixture
def mock_trades_history():
    """Fixture providing sample trading history"""
    return pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'entry_date': pd.date_range(start='2025-01-01', periods=3),
        'exit_date': pd.date_range(start='2025-01-15', periods=3),
        'entry_price': [150.0, 200.0, 2500.0],
        'exit_price': [160.0, 190.0, 2600.0],
        'quantity': [100, -50, 10],
        'pnl': [1000.0, 500.0, 1000.0],
        'side': ['long', 'short', 'long']
    })

@pytest.fixture
def mock_me_calculator(sample_price_data, mock_positions):
    """Fixture providing initialized ME ratio calculator"""
    from nGS_Revised_Strategy import DailyMERatioCalculator
    calculator = DailyMERatioCalculator(initial_portfolio_value=1000000.0)
    for position in mock_positions:
        calculator.update_position(
            symbol=position['symbol'],
            quantity=position['quantity'],
            price=position['entry_price']
        )
    return calculator

@pytest.fixture
def mock_ngs_strategy():
    """Fixture providing initialized NGS strategy"""
    from nGS_Revised_Strategy import NGSStrategy
    return NGSStrategy(
        account_size=1000000.0,
        data_dir="tests/test_data",
        retention_days=20
    )

@pytest.fixture
def mock_sector_adapter():
    """Fixture providing initialized NGS sector adapter"""
    from ngs_sector_adapter import NGSSectorAdapter
    return NGSSectorAdapter(
        strategy_config={
            'account_size': 1000000.0,
            'data_dir': 'tests/test_data'
        }
    )

