import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Union

from data_manager import (
    save_trades, save_positions, load_price_data,
    save_signals, get_positions, initialize as init_data_manager,
    RETENTION_DAYS  # Import the 6-month retention setting
)
# Add M/E ratio tracking imports here
try:
    from me_ratio_calculator import (
        update_me_ratio_for_trade, 
        add_realized_profit, 
        get_current_risk_assessment,
        get_me_calculator
    )
    ME_TRACKING_AVAILABLE = True
except ImportError:
    ME_TRACKING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class NGSStrategy:
    """
    Neural Grid Strategy (nGS) implementation.
    Handles both signal generation and position management with 6-month data retention.
    """
    def __init__(self, account_size: float = 100000, data_dir: str = 'data'):
        self.account_size = round(float(account_size), 2)
        self.cash = round(float(account_size), 2)
        self.positions = {}
        self._trades = []
        self.data_dir = data_dir
        self.retention_days = RETENTION_DAYS  # Use the 6-month retention from data_manager
        self.cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Add M/E tracking - separate current vs historical
        self.historical_me_ratios = []    # Historical daily M/E values during backtest
        
        self.inputs = {
            'Length': 25,
            'NumDevs': 2,
            'MinPrice': 10,
            'MaxPrice': 500,
            'AfStep': 0.05,
            'AfLimit': 0.21,
            'PositionSize': 5000
        }
        init_data_manager()
        self._load_positions()
        
        logger.info(f"nGS Strategy initialized with {self.retention_days}-day data retention")
        logger.info(f"Data cutoff date: {self.cutoff_date.strftime('%Y-%m-%d')}")

    def calculate_current_me_ratio(self) -> float:
        """
        Calculate CURRENT M/E ratio for all existing positions.
        This is the live M/E ratio for portfolio management.
        """
        total_open_trade_equity = 0.0
        
        # Calculate Total Open Trade Equity for all current positions
        for symbol, position in self.positions.items():
            if position['shares'] != 0:
                # Use entry price for current calculation
                current_price = position['entry_price']
                position_equity = current_price * abs(position['shares'])
                total_open_trade_equity += position_equity
        
        # Current Account Value = current cash
        account_value = self.cash
        
        # Current M/E Ratio = Total Open Trade Equity / Account Value × 100
        me_ratio_pct = (total_open_trade_equity / account_value * 100) if account_value > 0 else 0.0
        
        return round(me_ratio_pct, 2)

    def calculate_historical_me_ratio(self, current_prices: Dict[str, float] = None) -> float:
        """
        Calculate HISTORICAL M/E ratio during backtest for daily tracking.
        This tracks M/E progression during the 6-month backtest period.
        """
        total_open_trade_equity = 0.0
        
        # Calculate Total Open Trade Equity for positions during backtest
        for symbol, position in self.positions.items():
            if position['shares'] != 0:
                # Use current price if available, otherwise entry price
                if current_prices and symbol in current_prices:
                    current_price = current_prices[symbol]
                else:
                    current_price = position['entry_price']
                
                position_equity = current_price * abs(position['shares'])
                total_open_trade_equity += position_equity
        
        # Historical Account Value = cash at that point in time
        account_value = self.cash
        
        # Historical M/E Ratio = Total Open Trade Equity / Account Value × 100
        me_ratio_pct = (total_open_trade_equity / account_value * 100) if account_value > 0 else 0.0
        
        return round(me_ratio_pct, 2)

    def record_historical_me_ratio(self, date_str: str, trade_occurred: bool = False, current_prices: Dict[str, float] = None):
        """
        Record daily M/E ratio for historical tracking during backtest.
        """
        if trade_occurred and current_prices:
            # Recalculate M/E ratio because a trade happened during backtest
            historical_me = self.calculate_historical_me_ratio(current_prices)
        else:
            # No trade - carry forward last value (0.0 if no previous trades)
            historical_me = self.historical_me_ratios[-1]['me_ratio_pct'] if self.historical_me_ratios else 0.0
        
        # Record the historical M/E ratio for this date
        self.historical_me_ratios.append({
            'date': date_str,
            'me_ratio_pct': historical_me,
            'trade_occurred': trade_occurred
        })

    def _filter_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to only include data from the last 6 months (retention period).
        """
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter to retention period
            df = df[df['Date'] >= self.cutoff_date].copy()
            
            logger.debug(f"Filtered data to last {self.retention_days} days: {len(df)} rows remaining")
        
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"Insufficient data for indicator calculation: {len(df) if df is not None else 'None'} rows")
            return None
            
        # Filter to 6-month retention period
        df = self._filter_recent_data(df)
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"Insufficient data after 6-month filtering: {len(df) if df is not None else 'None'} rows")
            return None
            
        df = df.copy()
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None
            
        indicator_columns = [
            'BBAvg', 'BBSDev', 'UpperBB', 'LowerBB', 'High_Low', 'High_Close', 'Low_Close', 'TR', 'ATR', 'ATRma',
            'LongPSAR', 'ShortPSAR', 'PSAR_EP', 'PSAR_AF', 'PSAR_IsLong', 'oLRSlope', 'oLRAngle', 'oLRIntercept',
            'TSF', 'oLRSlope2', 'oLRAngle2', 'oLRIntercept2', 'TSF5', 'Value1', 'ROC', 'LRV', 'LinReg', 'oLRValue',
            'oLRValue2', 'SwingLow', 'SwingHigh'
        ]
        for col in indicator_columns:
            if col not in df.columns:
                df[col] = np.nan
                
        # Bollinger Bands
        try:
            df['BBAvg'] = df['Close'].rolling(window=self.inputs['Length']).mean().round(2)
            df['BBSDev'] = df['Close'].rolling(window=self.inputs['Length']).std().round(2)
            df['UpperBB'] = (df['BBAvg'] + self.inputs['NumDevs'] * df['BBSDev']).round(2)
            df['LowerBB'] = (df['BBAvg'] - self.inputs['NumDevs'] * df['BBSDev']).round(2)
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation error: {e}")
            
        # ATR (Average True Range)
        try:
            df['High_Low'] = (df['High'] - df['Low']).round(2)
            df['High_Close'] = abs(df['High'] - df['Close'].shift(1)).round(2)
            df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1)).round(2)
            df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1).round(2)
            df['ATR'] = df['TR'].rolling(window=5).mean().round(2)
            df['ATRma'] = df['ATR'].rolling(window=13).mean().round(2)
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            
        # Parabolic SAR
        try:
            self._calculate_psar(df)
        except Exception as e:
            logger.warning(f"PSAR calculation error: {e}")
            
        # Linear Regression indicators
        try:
            self._calculate_linear_regression(df)
        except Exception as e:
            logger.warning(f"Linear regression calculation error: {e}")
            
        # Additional indicators
        try:
            df['Value1'] = (df['Close'].rolling(window=5).mean() - df['Close'].rolling(window=35).mean()).round(2)
            df['ROC'] = (df['Value1'] - df['Value1'].shift(3)).round(2)
            self._calculate_lrv(df)
        except Exception as e:
            logger.warning(f"Additional indicators calculation error: {e}")
            
        # Swing High/Low
        try:
            df['SwingLow'] = df['Close'].rolling(window=4).min().round(2)
            df['SwingHigh'] = df['Close'].rolling(window=4).max().round(2)
        except Exception as e:
            logger.warning(f"Swing High/Low calculation error: {e}")
            
        return df

    def _calculate_psar(self, df: pd.DataFrame) -> None:
        df['LongPSAR'] = 0.0
        df['ShortPSAR'] = 0.0
        df['PSAR_EP'] = 0.0
        df['PSAR_AF'] = self.inputs['AfStep']
        df['PSAR_IsLong'] = (df['Close'] > df['Open']).astype('Int64')
        df.loc[df['PSAR_IsLong'] == 1, 'PSAR_EP'] = df.loc[df['PSAR_IsLong'] == 1, 'High']
        df.loc[df['PSAR_IsLong'] == 1, 'LongPSAR'] = df.loc[df['PSAR_IsLong'] == 1, 'Low']
        df.loc[df['PSAR_IsLong'] == 1, 'ShortPSAR'] = df.loc[df['PSAR_IsLong'] == 1, 'High']
        df.loc[df['PSAR_IsLong'] == 0, 'PSAR_EP'] = df.loc[df['PSAR_IsLong'] == 0, 'Low']
        df.loc[df['PSAR_IsLong'] == 0, 'LongPSAR'] = df.loc[df['PSAR_IsLong'] == 0, 'High']
        df.loc[df['PSAR_IsLong'] == 0, 'ShortPSAR'] = df.loc[df['PSAR_IsLong'] == 0, 'Low']
        for i in range(1, len(df)):
            if pd.isna(df['Close'].iloc[i]) or pd.isna(df['Open'].iloc[i]):
                continue
            try:
                prev_is_long = int(df['PSAR_IsLong'].iloc[i-1]) if not pd.isna(df['PSAR_IsLong'].iloc[i-1]) else 1
                if prev_is_long == 1:
                    long_psar = df['LongPSAR'].iloc[i-1] + df['PSAR_AF'].iloc[i-1] * (df['PSAR_EP'].iloc[i-1] - df['LongPSAR'].iloc[i-1])
                    long_psar = min(long_psar, df['Low'].iloc[i], df['Low'].iloc[i-1] if i > 1 else df['Low'].iloc[i])
                    df.loc[df.index[i], 'LongPSAR'] = round(long_psar, 2)
                    if df['High'].iloc[i] > df['PSAR_EP'].iloc[i-1]:
                        df.loc[df.index[i], 'PSAR_EP'] = round(df['High'].iloc[i], 2)
                        df.loc[df.index[i], 'PSAR_AF'] = round(min(df['PSAR_AF'].iloc[i-1] + self.inputs['AfStep'], self.inputs['AfLimit']), 2)
                    else:
                        df.loc[df.index[i], 'PSAR_EP'] = df['PSAR_EP'].iloc[i-1]
                        df.loc[df.index[i], 'PSAR_AF'] = df['PSAR_AF'].iloc[i-1]
                    if df['Low'].iloc[i] <= long_psar:
                        df.loc[df.index[i], 'PSAR_IsLong'] = 0
                        df.loc[df.index[i], 'ShortPSAR'] = round(df['PSAR_EP'].iloc[i-1], 2)
                        df.loc[df.index[i], 'PSAR_EP'] = round(df['Low'].iloc[i], 2)
                        df.loc[df.index[i], 'PSAR_AF'] = round(self.inputs['AfStep'], 2)
                    else:
                        df.loc[df.index[i], 'PSAR_IsLong'] = 1
                        df.loc[df.index[i], 'ShortPSAR'] = df['ShortPSAR'].iloc[i-1]
                else:
                    short_psar = df['ShortPSAR'].iloc[i-1] - df['PSAR_AF'].iloc[i-1] * (df['ShortPSAR'].iloc[i-1] - df['PSAR_EP'].iloc[i-1])
                    short_psar = max(short_psar, df['High'].iloc[i], df['High'].iloc[i-1] if i > 1 else df['High'].iloc[i])
                    df.loc[df.index[i], 'ShortPSAR'] = round(short_psar, 2)
                    if df['Low'].iloc[i] < df['PSAR_EP'].iloc[i-1]:
                        df.loc[df.index[i], 'PSAR_EP'] = round(df['Low'].iloc[i], 2)
                        df.loc[df.index[i], 'PSAR_AF'] = round(min(df['PSAR_AF'].iloc[i-1] + self.inputs['AfStep'], self.inputs['AfLimit']), 2)
                    else:
                        df.loc[df.index[i], 'PSAR_EP'] = df['PSAR_EP'].iloc[i-1]
                        df.loc[df.index[i], 'PSAR_AF'] = df['PSAR_AF'].iloc[i-1]
                    if df['High'].iloc[i] >= short_psar:
                        df.loc[df.index[i], 'PSAR_IsLong'] = 1
                        df.loc[df.index[i], 'LongPSAR'] = round(df['PSAR_EP'].iloc[i-1], 2)
                        df.loc[df.index[i], 'PSAR_EP'] = round(df['High'].iloc[i], 2)
                        df.loc[df.index[i], 'PSAR_AF'] = round(self.inputs['AfStep'], 2)
                    else:
                        df.loc[df.index[i], 'PSAR_IsLong'] = 0
                        df.loc[df.index[i], 'LongPSAR'] = df['LongPSAR'].iloc[i-1]
            except Exception as e:
                logger.warning(f"PSAR calculation error at index {i}: {e}")

    def _calculate_linear_regression(self, df: pd.DataFrame) -> None:
        def linear_reg(series, period, shift):
            if len(series) < period or series.isna().any():
                return np.nan, np.nan, np.nan, np.nan
            try:
                x = np.arange(period)
                slope, intercept = np.polyfit(x, series[-period:], 1)
                value = slope * (period + shift - 1) + intercept
                return round(slope, 2), round(np.degrees(np.arctan(slope)), 2), round(intercept, 2), round(value, 2)
            except Exception as e:
                logger.warning(f"Linear regression calculation error: {e}")
                return np.nan, np.nan, np.nan, np.nan
        for i in range(len(df)):
            try:
                if i >= 3 and not df['Close'].iloc[max(0, i-3):i+1].isna().any():
                    slope, angle, intercept, value = linear_reg(df['Close'].iloc[max(0, i-3):i+1], 3, -2)
                    df.loc[df.index[i], 'oLRSlope'] = slope
                    df.loc[df.index[i], 'oLRAngle'] = angle
                    df.loc[df.index[i], 'oLRIntercept'] = intercept
                    df.loc[df.index[i], 'TSF'] = value
                if i >= 5 and not df['Close'].iloc[max(0, i-5):i+1].isna().any():
                    slope2, angle2, intercept2, value2 = linear_reg(df['Close'].iloc[max(0, i-5):i+1], 5, -3)
                    df.loc[df.index[i], 'oLRSlope2'] = slope2
                    df.loc[df.index[i], 'oLRAngle2'] = angle2
                    df.loc[df.index[i], 'oLRIntercept2'] = intercept2
                    df.loc[df.index[i], 'TSF5'] = value2
            except Exception as e:
                logger.warning(f"Linear regression error at index {i}: {e}")

    def _calculate_lrv(self, df: pd.DataFrame) -> None:
        def linear_reg_value(series, period, shift):
            if len(series) < period or series.isna().any():
                return np.nan
            try:
                x = np.arange(period)
                slope, intercept = np.polyfit(x, series[-period:], 1)
                return round(slope * (period + shift - 1) + intercept, 2)
            except Exception as e:
                logger.warning(f"Linear regression value calculation error: {e}")
                return np.nan
        for i in range(len(df)):
            try:
                if i >= 8 and not df['ROC'].iloc[max(0, i-8):i+1].isna().any():
                    df.loc[df.index[i], 'LRV'] = linear_reg_value(df['ROC'].iloc[max(0, i-8):i+1], 8, 0)
                if i >= 8 and not df['Close'].iloc[max(0, i-8):i+1].isna().any():
                    df.loc[df.index[i], 'LinReg'] = linear_reg_value(df['Close'].iloc[max(0, i-8):i+1], 8, 0)
                if i >= 3 and not df['LRV'].iloc[max(0, i-3):i+1].isna().any():
                    df.loc[df.index[i], 'oLRValue'] = linear_reg_value(df['LRV'].iloc[max(0, i-3):i+1], 3, -2)
                if i >= 5 and not df['LRV'].iloc[max(0, i-5):i+1].isna().any():
                    df.loc[df.index[i], 'oLRValue2'] = linear_reg_value(df['LRV'].iloc[max(0, i-5):i+1], 5, -3)
            except Exception as e:
                logger.warning(f"LRV calculation error at index {i}: {e}")

    def _check_long_signals(self, df: pd.DataFrame, i: int) -> None:
        # Engulfing Long pattern
        if (df['Open'].iloc[i] < df['Close'].iloc[i-1] and
            df['Close'].iloc[i] > df['Open'].iloc[i-1] and
            df['Close'].iloc[i] > df['Open'].iloc[i] and
            df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
            abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
            df['High'].iloc[i] - df['Close'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
            df['Open'].iloc[i-1] - df['Close'].iloc[i-1] > 0.05 and
            not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
            df['oLRValue'].iloc[i] >= df['oLRValue2'].iloc[i] and
            df['Low'].iloc[i] <= df['LowerBB'].iloc[i] * 1.02 and
            df['Close'].iloc[i] <= df['UpperBB'].iloc[i] * 0.95):
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'SignalType'] = 'Engf L'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
        # Engulfing Long with New Low pattern
        elif (df['Open'].iloc[i] < df['Close'].iloc[i-1] and
              df['Close'].iloc[i] > df['Open'].iloc[i-1] and
              df['Close'].iloc[i] > df['Open'].iloc[i] and
              df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
              abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
              df['High'].iloc[i] - df['Close'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
              df['Open'].iloc[i-1] - df['Close'].iloc[i-1] > 0.05 and
              df['Close'].iloc[i-1] <= df['SwingLow'].iloc[i] and
              df['Low'].iloc[i] <= df['LowerBB'].iloc[i] * 1.02 and
              df['Close'].iloc[i] <= df['UpperBB'].iloc[i] * 0.95):
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'SignalType'] = 'Engf L NuLo'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
        # Semi-Engulfing Long pattern
        elif (df['Open'].iloc[i] <= df['Close'].iloc[i-1] * 1.001 and
              df['Open'].iloc[i] > df['Close'].iloc[i-1] and
              df['Close'].iloc[i] > df['Open'].iloc[i-1] + (df['ATR'].iloc[i] * 0.5) and
              abs((df['Open'].iloc[i-1] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]) >= 0.003 and
              df['Close'].iloc[i] > df['Open'].iloc[i] and
              df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
              df['High'].iloc[i] - df['Close'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
              df['Open'].iloc[i-1] - df['Close'].iloc[i-1] > 0.05 and
              abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
              not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
              df['oLRValue'].iloc[i] >= df['oLRValue2'].iloc[i] and
              df['Low'].iloc[i] <= df['LowerBB'].iloc[i] * 1.02 and
              df['Close'].iloc[i] <= df['UpperBB'].iloc[i] * 0.95):
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'SignalType'] = 'SemiEng L'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
        # Semi-Engulfing Long with New Low pattern
        elif (df['Open'].iloc[i] <= df['Close'].iloc[i-1] * 1.001 and
              df['Open'].iloc[i] > df['Close'].iloc[i-1] and
              df['Close'].iloc[i] > df['Open'].iloc[i-1] + (df['ATR'].iloc[i] * 0.5) and
              abs((df['Open'].iloc[i-1] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]) >= 0.003 and
              df['Close'].iloc[i] > df['Open'].iloc[i] and
              df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
              df['High'].iloc[i] - df['Close'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
              df['Open'].iloc[i-1] - df['Close'].iloc[i-1] > 0.05 and
              abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
              df['Close'].iloc[i-1] <= df['SwingLow'].iloc[i] and
              df['Low'].iloc[i] <= df['LowerBB'].iloc[i] * 1.02 and
              df['Close'].iloc[i] <= df['UpperBB'].iloc[i] * 0.95):
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'SignalType'] = 'SemiEng L NuLo'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))

    def _check_short_signals(self, df: pd.DataFrame, i: int) -> None:
        # Engulfing Short pattern
        if (df['Open'].iloc[i] > df['Close'].iloc[i-1] and
            df['Close'].iloc[i] < df['Open'].iloc[i-1] and
            df['Close'].iloc[i] < df['Open'].iloc[i] and
            df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
            df['Close'].iloc[i-1] - df['Low'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
            df['Close'].iloc[i-1] - df['Open'].iloc[i-1] > 0.05 and
            abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
            not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
            df['oLRValue'].iloc[i] <= df['oLRValue2'].iloc[i] and
            df['High'].iloc[i] >= df['UpperBB'].iloc[i] * 0.98 and
            df['Close'].iloc[i] >= df['LowerBB'].iloc[i] * 1.05):
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'SignalType'] = 'Engf S'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
        # Engulfing Short with New High pattern
        elif (df['Open'].iloc[i] > df['Close'].iloc[i-1] and
              df['Close'].iloc[i] < df['Open'].iloc[i-1] and
              df['Close'].iloc[i] < df['Open'].iloc[i] and
              df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
              df['Close'].iloc[i-1] - df['Low'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
              df['Close'].iloc[i-1] - df['Open'].iloc[i-1] > 0.05 and
              df['Close'].iloc[i-1] >= df['SwingHigh'].iloc[i] and
              abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
              df['High'].iloc[i] >= df['UpperBB'].iloc[i] * 0.98 and
              df['Close'].iloc[i] >= df['LowerBB'].iloc[i] * 1.05):
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'SignalType'] = 'Engf S NuHu3'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
        # Semi-Engulfing Short pattern
        elif (df['Open'].iloc[i] >= df['Close'].iloc[i-1] * 0.999 and
              df['Open'].iloc[i] < df['Close'].iloc[i-1] and
              df['Close'].iloc[i] < df['Open'].iloc[i-1] and
              df['Close'].iloc[i] < df['Open'].iloc[i] and
              df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
              df['Close'].iloc[i-1] - df['Low'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
              df['Close'].iloc[i-1] - df['Open'].iloc[i-1] > 0.05 and
              abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
              not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
              df['oLRValue'].iloc[i] <= df['oLRValue2'].iloc[i] and
              df['High'].iloc[i] >= df['UpperBB'].iloc[i] * 0.98 and
              df['Close'].iloc[i] >= df['LowerBB'].iloc[i] * 1.05):
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'SignalType'] = 'SemiEng S'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
        # Semi-Engulfing Short with New High pattern
        elif (df['Open'].iloc[i] >= df['Close'].iloc[i-1] * 0.999 and
              df['Open'].iloc[i] < df['Close'].iloc[i-1] and
              df['Close'].iloc[i] < df['Open'].iloc[i-1] and
              df['Close'].iloc[i] < df['Open'].iloc[i] and
              df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
              df['Close'].iloc[i-1] - df['Low'].iloc[i-1] <= df['ATR'].iloc[i] * 2 and
              df['Close'].iloc[i-1] - df['Open'].iloc[i-1] > 0.05 and
              df['Close'].iloc[i-1] >= df['SwingHigh'].iloc[i] and
              abs(df['Close'].iloc[i-1] - df['Low'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.05 and
              not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
              df['oLRValue'].iloc[i] <= df['oLRValue2'].iloc[i] and
              df['High'].iloc[i] >= df['UpperBB'].iloc[i] * 0.98 and
              df['Close'].iloc[i] >= df['LowerBB'].iloc[i] * 1.05):
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'SignalType'] = 'SemiEng S NuHi'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))

    def _check_long_exits(self, df: pd.DataFrame, i: int, position: Dict) -> None:
        if (position['profit'] > 0 and position['bars_since_entry'] > 1 and
            (df['Open'].iloc[i] >= df['Close'].iloc[i-1] * 1.05 or
             (not pd.isna(df['UpperBB'].iloc[i]) and df['Open'].iloc[i] >= df['UpperBB'].iloc[i]))):
            df.loc[df.index[i], 'ExitSignal'] = -1
            df.loc[df.index[i], 'ExitType'] = 'Gap out L' if df['Open'].iloc[i] >= df['Close'].iloc[i-1] * 1.05 else 'Target L'
        elif (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
              df['Close'].iloc[i] >= position['entry_price'] + df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()):
            df.loc[df.index[i], 'ExitSignal'] = -1
            df.loc[df.index[i], 'ExitType'] = 'BE L'
        elif (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
              df['Close'].iloc[i] >= position['entry_price'] + df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max() * 1.5):
            df.loc[df.index[i], 'ExitSignal'] = -1
            df.loc[df.index[i], 'ExitType'] = 'L ATR X'
        elif (position['bars_since_entry'] > 5 and
              not pd.isna(df['LinReg'].iloc[i]) and not pd.isna(df['LinReg'].iloc[i-1]) and
              df['LinReg'].iloc[i] < df['LinReg'].iloc[i-1] and
              position['profit'] < 0 and
              not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
              df['oLRValue'].iloc[i] < df['oLRValue2'].iloc[i] and
              not pd.isna(df['ATR'].iloc[i]) and not pd.isna(df['ATRma'].iloc[i]) and
              df['ATR'].iloc[i] > df['ATRma'].iloc[i]):
            df.loc[df.index[i], 'ExitSignal'] = -1
            df.loc[df.index[i], 'ExitType'] = 'S 2 L'
            df.loc[df.index[i], 'Signal'] = -1
            df.loc[df.index[i], 'SignalType'] = 'S 2 L'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] * 2 / df['Close'].iloc[i]))
        elif df['Close'].iloc[i] < position['entry_price'] * 0.9:
            df.loc[df.index[i], 'ExitSignal'] = -1
            df.loc[df.index[i], 'ExitType'] = 'Hard Stop S'

    def _check_short_exits(self, df: pd.DataFrame, i: int, position: Dict) -> None:
        if (position['profit'] > 0 and position['bars_since_entry'] > 1 and
            (df['Open'].iloc[i] <= df['Close'].iloc[i-1] * 0.95 or
             (not pd.isna(df['LowerBB'].iloc[i]) and df['Open'].iloc[i] <= df['LowerBB'].iloc[i]))):
            df.loc[df.index[i], 'ExitSignal'] = 1
            df.loc[df.index[i], 'ExitType'] = 'Gap out S' if df['Open'].iloc[i] <= df['Close'].iloc[i-1] * 0.95 else 'Target S'
        elif (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
              df['Close'].iloc[i] <= position['entry_price'] - df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()):
            df.loc[df.index[i], 'ExitSignal'] = 1
            df.loc[df.index[i], 'ExitType'] = 'BE S'
        elif (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
              df['Close'].iloc[i] <= position['entry_price'] - df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max() * 1.5):
            df.loc[df.index[i], 'ExitSignal'] = 1
            df.loc[df.index[i], 'ExitType'] = 'S ATR X'
        elif (position['bars_since_entry'] > 5 and
              not pd.isna(df['LinReg'].iloc[i]) and not pd.isna(df['LinReg'].iloc[i-1]) and
              df['LinReg'].iloc[i] > df['LinReg'].iloc[i-1] and
              position['profit'] < 0 and
              not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
              df['oLRValue'].iloc[i] > df['oLRValue2'].iloc[i] and
              not pd.isna(df['ATR'].iloc[i]) and not pd.isna(df['ATRma'].iloc[i]) and
              df['ATR'].iloc[i] > df['ATRma'].iloc[i]):
            df.loc[df.index[i], 'ExitSignal'] = 1
            df.loc[df.index[i], 'ExitType'] = 'L 2 S'
            df.loc[df.index[i], 'Signal'] = 1
            df.loc[df.index[i], 'SignalType'] = 'L 2 S'
            df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] * 2 / df['Close'].iloc[i]))
        elif df['Close'].iloc[i] > position['entry_price'] * 1.1:
            df.loc[df.index[i], 'ExitSignal'] = 1
            df.loc[df.index[i], 'ExitType'] = 'Hard Stop L'

    def _process_exit(self, df: pd.DataFrame, i: int, symbol: str, position: Dict) -> None:
        exit_price = round(float(df['Close'].iloc[i]), 2)
        profit = round(float(
            (exit_price - position['entry_price']) * position['shares'] if position['shares'] > 0
            else (position['entry_price'] - exit_price) * abs(position['shares'])
        ), 2)
        
        # Ensure exit date is within retention period
        exit_date = str(df['Date'].iloc[i])[:10]
        exit_datetime = datetime.strptime(exit_date, '%Y-%m-%d')
        
        if exit_datetime >= self.cutoff_date:
            trade = {
                'symbol': symbol,
                'type': 'long' if position['shares'] > 0 else 'short',
                'entry_date': position['entry_date'],
                'exit_date': exit_date,
                'entry_price': round(float(position['entry_price']), 2),
                'exit_price': exit_price,
                'shares': int(round(abs(position['shares']))),
                'profit': profit,
                'exit_reason': df['ExitType'].iloc[i]
            }
            if all(k in trade for k in ['symbol', 'type', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'shares', 'profit', 'exit_reason']):
                self._trades.append(trade)
                save_trades([trade])
            else:
                logger.warning(f"Invalid trade skipped: {trade}")
        
        # Add M/E tracking - close position and add realized profit (no alerts)
        if ME_TRACKING_AVAILABLE:
            try:
                update_me_ratio_for_trade(symbol, 0, 0, 0, 'long')  # Close position
                add_realized_profit(profit)  # Add realized profit
            except Exception as e:
                logger.debug(f"M/E tracking error: {e}")
        
        self.cash = round(float(self.cash + position['shares'] * exit_price), 2)
        
        # *** RECORD HISTORICAL M/E RATIO AFTER EXIT ***
        current_prices = {symbol: exit_price}  # At minimum, we have this symbol's price
        self.record_historical_me_ratio(exit_date, trade_occurred=True, current_prices=current_prices)
        
        logger.info(f"Exit {symbol}: {df['ExitType'].iloc[i]} at {exit_price}, profit: {profit}")

    def _process_entry(self, df: pd.DataFrame, i: int, symbol: str, position: Dict) -> None:
        shares = int(round(df['Shares'].iloc[i])) if df['Signal'].iloc[i] > 0 else -int(round(df['Shares'].iloc[i]))
        cost = round(float(shares * df['Close'].iloc[i]), 2)
        
        # Ensure entry date is within retention period
        entry_date = str(df['Date'].iloc[i])[:10]
        entry_datetime = datetime.strptime(entry_date, '%Y-%m-%d')
        
        if entry_datetime >= self.cutoff_date and abs(cost) <= self.cash:
            self.cash = round(float(self.cash - cost), 2)
            position = {
                'shares': shares,
                'entry_price': round(float(df['Close'].iloc[i]), 2),
                'entry_date': entry_date,
                'bars_since_entry': 0,
                'profit': 0
            }
            self.positions[symbol] = position
            
            # Add M/E tracking - new position (no alerts)
            if ME_TRACKING_AVAILABLE:
                try:
                    current_price = round(float(df['Close'].iloc[i]), 2)
                    trade_type = 'long' if shares > 0 else 'short'
                    update_me_ratio_for_trade(symbol, shares, current_price, current_price, trade_type)
                except Exception as e:
                    logger.debug(f"M/E tracking error: {e}")
            
            # *** RECORD HISTORICAL M/E RATIO AFTER ENTRY ***
            current_prices = {symbol: df['Close'].iloc[i]}  # At minimum, we have this symbol's price
            self.record_historical_me_ratio(entry_date, trade_occurred=True, current_prices=current_prices)
            
            logger.info(f"Entry {symbol}: {df['SignalType'].iloc[i]} with {shares} shares at {df['Close'].iloc[i]}")
        elif entry_datetime < self.cutoff_date:
            logger.debug(f"Entry {symbol} skipped - outside retention period")
        else:
            logger.warning(f"Insufficient cash for {symbol}: {cost} required, {self.cash} available")

    def manage_positions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        df['ExitSignal'] = 0
        df['ExitType'] = ''
        position = self.positions.get(symbol, {
            'shares': 0,
            'entry_price': 0,
            'entry_date': None,
            'bars_since_entry': 0,
            'profit': 0
        })
        
        for i in range(1, len(df)):
            if pd.isna(df['Close'].iloc[i]) or pd.isna(df['Open'].iloc[i]) or pd.isna(df['ATR'].iloc[i]):
                continue
                
            current_date = str(df['Date'].iloc[i])[:10]
            
            if position['shares'] != 0:
                position['bars_since_entry'] += 1
                position['profit'] = round(
                    (df['Close'].iloc[i] - position['entry_price']) * position['shares']
                    if position['shares'] > 0 else
                    (position['entry_price'] - df['Close'].iloc[i]) * abs(position['shares']),
                    2
                )
                if position['shares'] > 0:
                    self._check_long_exits(df, i, position)
                elif position['shares'] < 0:
                    self._check_short_exits(df, i, position)
            
            if df['ExitSignal'].iloc[i] != 0:
                self._process_exit(df, i, symbol, position)
                position = {'shares': 0, 'entry_price': 0, 'entry_date': None, 'bars_since_entry': 0, 'profit': 0}
            elif df['Signal'].iloc[i] != 0:
                self._process_entry(df, i, symbol, position)
                position = self.positions.get(symbol, {'shares': 0, 'entry_price': 0, 'entry_date': None, 'bars_since_entry': 0, 'profit': 0})
            else:
                # No trade occurred - record historical M/E ratio carrying forward previous value
                self.record_historical_me_ratio(current_date, trade_occurred=False)
        
        self.positions[symbol] = position
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("Empty dataframe provided to generate_signals")
            return df
        df = df.copy()
        df['Signal'] = 0
        df['SignalType'] = ''
        df['Shares'] = 0
        for i in range(1, len(df)):
            if pd.isna(df['Open'].iloc[i]) or pd.isna(df['Close'].iloc[i]) or pd.isna(df['High'].iloc[i]) or pd.isna(df['Low'].iloc[i]):
                continue
            if not (df['Open'].iloc[i] > self.inputs['MinPrice'] and df['Open'].iloc[i] < self.inputs['MaxPrice']):
                continue
            self._check_long_signals(df, i)
            self._check_short_signals(df, i)
        return df

    def _load_positions(self) -> None:
        positions_list = get_positions()
        logger.info(f"Attempting to load {len(positions_list)} positions from data manager")
        
        for pos in positions_list:
            symbol = pos.get('symbol')
            entry_date = pos.get('entry_date')
            
            logger.debug(f"Processing position: {symbol}, entry_date: {entry_date} (type: {type(entry_date)})")
            
            # Only load positions within retention period
            if symbol and entry_date:
                try:
                    # Handle both string and Timestamp formats
                    if isinstance(entry_date, str):
                        entry_datetime = datetime.strptime(entry_date, '%Y-%m-%d')
                        entry_date_str = entry_date
                    else:
                        # Assume it's a pandas Timestamp
                        entry_datetime = entry_date.to_pydatetime() if hasattr(entry_date, 'to_pydatetime') else entry_date
                        entry_date_str = entry_datetime.strftime('%Y-%m-%d')
                    
                    if entry_datetime >= self.cutoff_date:
                        self.positions[symbol] = {
                            'shares': int(pos.get('shares', 0)),
                            'entry_price': float(pos.get('entry_price', 0)),
                            'entry_date': entry_date_str,
                            'bars_since_entry': int(pos.get('days_held', 0)),
                            'profit': float(pos.get('profit', 0))
                        }
                        logger.debug(f"Loaded position for {symbol}")
                    else:
                        logger.debug(f"Position {symbol} outside retention period: {entry_datetime} < {self.cutoff_date}")
                except (ValueError, AttributeError, TypeError) as e:
                    logger.warning(f"Invalid date format for position {symbol}: {entry_date} - {e}")
        
        logger.info(f"Loaded {len(self.positions)} positions within {self.retention_days}-day retention period")

    def _filter_trades_by_retention(self) -> None:
        """Filter trades list to only include trades within retention period"""
        if self._trades:
            filtered_trades = []
            for trade in self._trades:
                try:
                    exit_date = datetime.strptime(trade['exit_date'], '%Y-%m-%d')
                    if exit_date >= self.cutoff_date:
                        filtered_trades.append(trade)
                except (ValueError, KeyError):
                    logger.warning(f"Invalid trade date format: {trade}")
            
            self._trades = filtered_trades
            logger.info(f"Filtered trades to {len(self._trades)} within retention period")

    @property
    def trades(self):
        return self._trades

    @trades.setter
    def trades(self, value):
        self._trades = value

    def get_current_positions(self) -> Tuple[List[Dict], List[Dict]]:
        long_positions = []
        short_positions = []
        for symbol, pos in self.positions.items():
            if pos['shares'] > 0:
                long_positions.append({
                    'symbol': symbol,
                    'shares': int(round(pos['shares'])),
                    'entry_price': round(float(pos['entry_price']), 2),
                    'entry_date': pos['entry_date'],
                    'gap_target': round(float(pos['entry_price'] * 1.05), 2),
                    'bb_target': None
                })
            elif pos['shares'] < 0:
                short_positions.append({
                    'symbol': symbol,
                    'shares': int(round(abs(pos['shares']))),
                    'entry_price': round(float(pos['entry_price']), 2),
                    'entry_date': pos['entry_date'],
                    'gap_target': round(float(pos['entry_price'] * 0.95), 2),
                    'bb_target': None
                })
        return long_positions, short_positions

    def process_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            df_with_indicators = self.calculate_indicators(df)
            if df_with_indicators is None or df_with_indicators.empty:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return None
            df_with_signals = self.generate_signals(df_with_indicators)
            result_df = self.manage_positions(df_with_signals, symbol)
            return result_df
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Initialize for this run
        self.trades = []
        self.daily_me_ratios = []  # Reset daily M/E tracking
        self.current_me_ratio = 0.0  # Start with 0% M/E ratio
        
        # Filter trades by retention period
        self._filter_trades_by_retention()
        
        # Calculate initial M/E ratio for any existing positions
        if self.positions:
            self.current_me_ratio = self.calculate_me_ratio()
            logger.info(f"Initial M/E ratio with {len(self.positions)} existing positions: {self.current_me_ratio:.2f}%")
        
        results = {}
        for symbol, df in data.items():
            result = self.process_symbol(symbol, df)
            if result is not None and not result.empty:
                results[symbol] = result
                logger.info(f"Processed {symbol}: {len(result)} rows")
        
        # Calculate final M/E ratio after strategy run
        final_me_ratio = self.calculate_me_ratio()
        if self.daily_me_ratios:
            # Update the last entry with final M/E calculation
            if len(self.daily_me_ratios) > 0:
                self.daily_me_ratios[-1]['me_ratio_pct'] = final_me_ratio
                self.current_me_ratio = final_me_ratio
        else:
            # If no daily entries, create one final entry
            from datetime import datetime
            self.daily_me_ratios.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'me_ratio_pct': final_me_ratio,
                'trade_occurred': False
            })
            self.current_me_ratio = final_me_ratio
        
        # Save daily M/E ratios to file
        if self.daily_me_ratios:
            me_df = pd.DataFrame(self.daily_me_ratios)
            me_file = os.path.join(self.data_dir, 'daily_me_ratios.csv')
            os.makedirs(self.data_dir, exist_ok=True)
            me_df.to_csv(me_file, index=False)
            
            print(f"\n[M/E] Daily M/E Ratios saved to: {me_file}")
            print(f"M/E data covers {len(self.daily_me_ratios)} trading days")
            
            # Show recent M/E values
            print("\nRecent M/E Ratio values:")
            recent_me = me_df.tail(10)
            for _, row in recent_me.iterrows():
                trade_flag = " *TRADE*" if row['trade_occurred'] else ""
                print(f"{row['date']}: {row['me_ratio_pct']:6.2f}%{trade_flag}")
            
            # Show M/E statistics
            print(f"\nM/E Ratio Statistics:")
            print(f"Average M/E Ratio: {me_df['me_ratio_pct'].mean():.2f}%")
            print(f"Max M/E Ratio: {me_df['me_ratio_pct'].max():.2f}%")
            print(f"Min M/E Ratio: {me_df['me_ratio_pct'].min():.2f}%")
            print(f"Days with trades: {me_df['trade_occurred'].sum()}")
            print(f"Current M/E Ratio: {self.current_me_ratio:.2f}%")
        
        long_positions, short_positions = self.get_current_positions()
        all_positions = []
        
        for pos in long_positions:
            if pos['symbol'] in results and not results[pos['symbol']].empty:
                current_price = results[pos['symbol']].iloc[-1]['Close']
            else:
                current_price = pos['entry_price']
            all_positions.append({
                'symbol': pos['symbol'],
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'entry_date': pos['entry_date'],
                'current_price': round(float(current_price), 2),
                'current_value': round(float(current_price * pos['shares']), 2),
                'profit': round(float((current_price - pos['entry_price']) * pos['shares']), 2),
                'profit_pct': round(float((current_price / pos['entry_price'] - 1) * 100), 2),
                'days_held': (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days,
                'side': 'long',
                'strategy': 'nGS'
            })
        
        for pos in short_positions:
            if pos['symbol'] in results and not results[pos['symbol']].empty:
                current_price = results[pos['symbol']].iloc[-1]['Close']
            else:
                current_price = pos['entry_price']
            shares_abs = abs(pos['shares'])
            profit = round(float((pos['entry_price'] - current_price) * shares_abs), 2)
            all_positions.append({
                'symbol': pos['symbol'],
                'shares': -shares_abs,
                'entry_price': pos['entry_price'],
                'entry_date': pos['entry_date'],
                'current_price': round(float(current_price), 2),
                'current_value': round(float(current_price * shares_abs), 2),
                'profit': profit,
                'profit_pct': round(float((pos['entry_price'] / current_price - 1) * 100), 2),
                'days_held': (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days,
                'side': 'short',
                'strategy': 'nGS'
            })
        
        save_positions(all_positions)
        
        # Final M/E status with position details for verification
        print(f"\nFinal M/E Status: {self.current_me_ratio:.2f}%")
        
        # Debug: Show M/E calculation details
        total_equity = 0
        for symbol, position in self.positions.items():
            if position['shares'] != 0:
                equity = position['entry_price'] * abs(position['shares'])
                total_equity += equity
        
        print(f"\nM/E Calculation Details:")
        print(f"Total Open Trade Equity: ${total_equity:,.2f}")
        print(f"Account Value (Cash): ${self.cash:,.2f}")
        print(f"Calculated M/E: {(total_equity/self.cash*100):.2f}% (should match Final M/E Status)")
        
        # Final M/E status
        if ME_TRACKING_AVAILABLE:
            try:
                me_calc = get_me_calculator()
                final_risk = get_current_risk_assessment()
                print(f"M/E Risk Status:      {final_risk['risk_level']}")
                print(f"M/E Realized P&L:     ${me_calc.total_realized_pnl:.2f}")
                print(f"M/E Active Positions: {len(me_calc.positions)}")
            except Exception as e:
                logger.debug(f"M/E status error: {e}")
        
        logger.info(f"Strategy run complete. Processed {len(data)} symbols, currently have {len(all_positions)} positions")
        logger.info(f"Data retention: {self.retention_days} days, cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
        return results

def load_polygon_data(symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load EOD data from Polygon.io with automatic 6-month filtering
    Replace this function with your actual Polygon data fetching code
    """
    # Default to 6-month data range if not specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=RETENTION_DAYS + 30)).strftime('%Y-%m-%d')  # Extra buffer for indicators
    
    data = {}
    
    # EXAMPLE - Replace with your actual Polygon implementation
    try:
        # Example using polygon library (install with: pip install polygon-api-client)
        # import polygon
        # from polygon import RESTClient
        # 
        # # Initialize client with your API key
        # client = RESTClient("YOUR_POLYGON_API_KEY")
        
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        for symbol in symbols:
            try:
                logger.debug(f"Loading {symbol} from Polygon...")
                
                # REPLACE THIS SECTION WITH YOUR ACTUAL POLYGON CODE:
                # Example Polygon API call:
                # aggs = client.get_aggs(
                #     ticker=symbol,
                #     multiplier=1,
                #     timespan="day", 
                #     from_=start_date,
                #     to=end_date
                # )
                # 
                # # Convert to DataFrame
                # df_data = []
                # for agg in aggs:
                #     df_data.append({
                #         'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                #         'Open': agg.open,
                #         'High': agg.high,
                #         'Low': agg.low,
                #         'Close': agg.close,
                #         'Volume': agg.volume
                #     })
                # 
                # df = pd.DataFrame(df_data)
                # df = df.sort_values('Date').reset_index(drop=True)
                
                # FOR NOW - Using synthetic data as fallback
                # Remove this when you implement real Polygon loading
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                base_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'TSLA': 200, 'AMZN': 100, 'META': 250, 'NVDA': 400}
                base_price = base_prices.get(symbol, 100)
                
                # Generate realistic price series
                returns = np.random.normal(0, 0.015, len(dates))  # 1.5% daily volatility
                prices = [base_price]
                for ret in returns[1:]:
                    prices.append(max(1, prices[-1] * (1 + ret)))
                
                df_data = []
                for i, (date, close) in enumerate(zip(dates, prices)):
                    if i == 0:
                        open_price = close
                    else:
                        open_price = prices[i-1] * (1 + np.random.normal(0, 0.005))
                    
                    daily_range = close * 0.025  # 2.5% daily range
                    high = max(open_price, close) + daily_range * np.random.random()
                    low = min(open_price, close) - daily_range * np.random.random()
                    volume = np.random.randint(1000000, 10000000)
                    
                    df_data.append({
                        'Date': date,
                        'Open': round(open_price, 2),
                        'High': round(high, 2),
                        'Low': round(low, 2),
                        'Close': round(close, 2),
                        'Volume': volume
                    })
                
                df = pd.DataFrame(df_data)
                # END SYNTHETIC DATA SECTION
                
                if df is not None and not df.empty:
                    # Ensure Date column is datetime
                    df['Date'] = pd.to_datetime(df['Date'])
                    data[symbol] = df
                    logger.debug(f"✓ {symbol}: {len(df)} bars loaded")
                else:
                    logger.warning(f"✗ {symbol}: No data received")
                    
            except Exception as e:
                logger.error(f"✗ {symbol}: Error loading - {e}")
                
    except Exception as e:
        logger.error(f"Polygon connection error: {e}")
        logger.info("Using fallback synthetic data...")
    
    return data

if __name__ == "__main__":
    print("nGS Trading Strategy - Neural Grid System")
    print("=" * 50)
    print(f"Data Retention: {RETENTION_DAYS} days (6 months)")
    print("=" * 50)
    
    try:
        strategy = NGSStrategy(account_size=100000)
        
        # Define your trading universe
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA',
            'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'PFE'
        ]
        
        print(f"Loading historical data for {len(symbols)} symbols...")
        
        # Load data using Polygon (or fallback to synthetic) - automatically uses 6-month range
        data = load_polygon_data(symbols)
        
        if not data:
            print("No data loaded - check your Polygon setup")
            exit(1)
        
        # Test M/E integration
        if ME_TRACKING_AVAILABLE:
            try:
                initial_risk = get_current_risk_assessment()
                print(f"+ M/E Integration active - Initial risk: {initial_risk['risk_level']}")
            except Exception as e:
                print(f"! M/E calculator error: {e}")
        else:
            print("! M/E calculator not found - running without M/E tracking")
        
        # Run the strategy
        print(f"Running nGS strategy on {len(data)} symbols...")
        results = strategy.run(data)
        
        # Results summary
        print(f"\n{'='*70}")
        print("STRATEGY BACKTEST RESULTS (Last 6 Months)")
        print(f"{'='*70}")
        
        total_profit = sum(trade['profit'] for trade in strategy.trades)
        winning_trades = sum(1 for trade in strategy.trades if trade['profit'] > 0)
        
        print(f"Starting capital:     ${strategy.account_size:,.2f}")
        print(f"Ending cash:          ${strategy.cash:,.2f}")
        print(f"Total P&L:            ${total_profit:,.2f}")
        print(f"Return:               {((strategy.cash - strategy.account_size) / strategy.account_size * 100):+.2f}%")
        print(f"Total trades:         {len(strategy.trades)}")
        print(f"Data period:          {strategy.cutoff_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        
        if strategy.trades:
            print(f"Winning trades:       {winning_trades}/{len(strategy.trades)} ({winning_trades/len(strategy.trades)*100:.1f}%)")
            
            # Trade statistics
            profits = [trade['profit'] for trade in strategy.trades]
            avg_profit = np.mean(profits)
            max_win = max(profits)
            max_loss = min(profits)
            
            print(f"Average trade:        ${avg_profit:.2f}")
            print(f"Best trade:           ${max_win:.2f}")
            print(f"Worst trade:          ${max_loss:.2f}")
        
        # Show recent trades
        if strategy.trades:
            print(f"\nRecent trades (last 10):")
            for trade in strategy.trades[-10:]:
                print(f"  {trade['symbol']} {trade['type']:5s} | "
                      f"{trade['entry_date']} → {trade['exit_date']} | "
                      f"${trade['entry_price']:7.2f} → ${trade['exit_price']:7.2f} | "
                      f"P&L: ${trade['profit']:+8.2f} | {trade['exit_reason']}")
        
        # Current positions
        long_pos, short_pos = strategy.get_current_positions()
        total_positions = len(long_pos) + len(short_pos)
        print(f"\nCurrent positions: {total_positions} total ({len(long_pos)} long, {len(short_pos)} short)")
        
        for pos in long_pos:
            print(f"  Long  {pos['symbol']:6s}: {pos['shares']:4d} shares @ ${pos['entry_price']:7.2f}")
        for pos in short_pos:
            print(f"  Short {pos['symbol']:6s}: {pos['shares']:4d} shares @ ${pos['entry_price']:7.2f}")
        
        print(f"\n+ Strategy backtest completed successfully!")
        print(f"+ Data retention enforced: {RETENTION_DAYS} days")
        
    except Exception as e:
        logger.error(f"Strategy backtest failed: {e}")
        import traceback
        traceback.print_exc()
