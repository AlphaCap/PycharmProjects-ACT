import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Union

from data_manager import (
    save_trades, save_positions, load_price_data,
    save_signals, get_positions, initialize as init_data_manager
)
# Add M/E ratio tracking imports here
from me_ratio_calculator import (
    update_me_ratio_for_trade, 
    add_realized_profit, 
    get_current_risk_assessment,
    get_me_calculator
)

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
    Handles both signal generation and position management.
    """
    def __init__(self, account_size: float = 100000, data_dir: str = 'data'):
        self.account_size = round(float(account_size), 2)
        self.cash = round(float(account_size), 2)
        self.positions = {}
        self._trades = []
        self.data_dir = data_dir
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 20:
            logger.warning(f"Insufficient data for indicator calculation: {len(df) if df is not None else 'None'} rows")
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
            if df['Signal'].iloc[i] != 0:
                self._process_entry(df, i, symbol, position)
                position = self.positions.get(symbol, {'shares': 0, 'entry_price': 0, 'entry_date': None, 'bars_since_entry': 0, 'profit': 0})
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

    def _process_exit(self, df: pd.DataFrame, i: int, symbol: str, position: Dict) -> None:
        exit_price = round(float(df['Close'].iloc[i]), 2)
        profit = round(float(
            (exit_price - position['entry_price']) * position['shares'] if position['shares'] > 0
            else (position['entry_price'] - exit_price) * abs(position['shares'])
        ), 2)
        trade = {
            'symbol': symbol,
            'type': 'long' if position['shares'] > 0 else 'short',
            'entry_date': position['entry_date'],
            'exit_date': str(df['Date'].iloc[i])[:10],
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
        self.cash = round(float(self.cash + position['shares'] * exit_price), 2)
        logger.info(f"Exit {symbol}: {df['ExitType'].iloc[i]} at {exit_price}, profit: {profit}")

    def _process_entry(self, df: pd.DataFrame, i: int, symbol: str, position: Dict) -> None:
        shares = int(round(df['Shares'].iloc[i])) if df['Signal'].iloc[i] > 0 else -int(round(df['Shares'].iloc[i]))
        cost = round(float(shares * df['Close'].iloc[i]), 2)
        if abs(cost) <= self.cash:
            self.cash = round(float(self.cash - cost), 2)
            position = {
                'shares': shares,
                'entry_price': round(float(df['Close'].iloc[i]), 2),
                'entry_date': str(df['Date'].iloc[i])[:10],
                'bars_since_entry': 0,
                'profit': 0
            }
            self.positions[symbol] = position
            logger.info(f"Entry {symbol}: {df['SignalType'].iloc[i]} with {shares} shares at {df['Close'].iloc[i]}")
        else:
            logger.warning(f"Insufficient cash for {symbol}: {cost} required, {self.cash} available")

    def _load_positions(self) -> None:
        positions_list = get_positions()
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
        self.trades = []
        results = {}
        for symbol, df in data.items():
            result = self.process_symbol(symbol, df)
            if result is not None and not result.empty:
                results[symbol] = result
                logger.info(f"Processed {symbol}: {len(result)} rows")
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
                'days_held': (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
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
                'days_held': (datetime.now() - datetime.strptime(pos['entry_date'], '%Y-%m-%d')).days
            })
        save_positions(all_positions)
        logger.info(f"Strategy run complete. Processed {len(data)} symbols, currently have {len(all_positions)} positions")
        return results


if __name__ == "__main__":
    print("nGS Trading Strategy - Neural Grid System")
    print("=" * 50)
    try:
        import yfinance as yf
        strategy = NGSStrategy()
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        data = {}
        for symbol in test_symbols:
            print(f"Downloading data for {symbol}...")
            df = yf.download(symbol, start="2023-01-01", end="2023-07-01")
            df.reset_index(inplace=True)
            if not df.empty:
                data[symbol] = df
        results = strategy.run(data)
        print("\nTrades executed:")
        for trade in strategy.trades:
            print(f"{trade['symbol']}: {trade['type']} from {trade['entry_date']} to {trade['exit_date']}, "
                  f"Profit: ${trade['profit']:.2f}, Reason: {trade['exit_reason']}")
        long_pos, short_pos = strategy.get_current_positions()
        print(f"\nCurrent positions: {len(long_pos)} long, {len(short_pos)} short")
        for pos in long_pos:
            print(f"Long {pos['symbol']}: {pos['shares']} shares at ${pos['entry_price']:.2f} from {pos['entry_date']}")
        for pos in short_pos:
            print(f"Short {pos['symbol']}: {abs(pos['shares'])} shares at ${pos['entry_price']:.2f} from {pos['entry_date']}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Error during test: {e}")
