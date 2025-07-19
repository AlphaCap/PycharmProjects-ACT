import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Union
import sys

from data_manager import (
    save_trades, save_positions, load_price_data,
    save_signals, get_positions, initialize as init_data_manager,
    RETENTION_DAYS
)

# Integrate me_ratio_calculator logic directly (copy-pasted class for self-contained)
class DailyMERatioCalculator:
    def __init__(self, initial_portfolio_value: float = 1000000):
        self.initial_portfolio_value = initial_portfolio_value
        self.current_positions = {}  # symbol -> position_data
        self.realized_pnl = 0.0
        self.daily_me_history = []
        
    def update_position(self, symbol: str, shares: int, entry_price: float, 
                       current_price: float, trade_type: str = 'long'):
        if shares == 0:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        else:
            self.current_positions[symbol] = {
                'shares': shares,
                'entry_price': entry_price,
                'current_price': current_price,
                'type': trade_type,
                'position_value': abs(shares) * current_price,
                'unrealized_pnl': self._calculate_unrealized_pnl(shares, entry_price, current_price, trade_type)
            }
    
    def _calculate_unrealized_pnl(self, shares: int, entry_price: float, 
                                current_price: float, trade_type: str) -> float:
        if trade_type.lower() == 'long':
            return (current_price - entry_price) * shares
        else:  # short
            return (entry_price - current_price) * abs(shares)
    
    def add_realized_pnl(self, profit: float):
        self.realized_pnl += profit
    
    def calculate_daily_me_ratio(self, date: str = None) -> Dict:
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate position values
        long_value = 0.0
        short_value = 0.0
        total_unrealized_pnl = 0.0
        
        for symbol, pos in self.current_positions.items():
            if pos['type'].lower() == 'long' and pos['shares'] > 0:
                long_value += pos['position_value']
            elif pos['type'].lower() == 'short' and pos['shares'] < 0:
                short_value += pos['position_value']
            
            total_unrealized_pnl += pos['unrealized_pnl']
        
        # Calculate portfolio equity
        portfolio_equity = self.initial_portfolio_value + self.realized_pnl + total_unrealized_pnl
        
        # Calculate total position value (for M/E ratio)
        total_position_value = long_value + short_value
        
        # Calculate M/E ratio
        me_ratio = (total_position_value / portfolio_equity * 100) if portfolio_equity > 0 else 0.0
        
        # Create daily metrics
        daily_metrics = {
            'Date': date,
            'Portfolio_Equity': round(portfolio_equity, 2),
            'Long_Value': round(long_value, 2),
            'Short_Value': round(short_value, 2),
            'Total_Position_Value': round(total_position_value, 2),
            'ME_Ratio': round(me_ratio, 2),
            'Realized_PnL': round(self.realized_pnl, 2),
            'Unrealized_PnL': round(total_unrealized_pnl, 2),
            'Long_Positions': len([p for p in self.current_positions.values() if p['type'].lower() == 'long' and p['shares'] > 0]),
            'Short_Positions': len([p for p in self.current_positions.values() if p['type'].lower() == 'short' and p['shares'] < 0]),
        }
        
        # Store in history
        self.daily_me_history.append(daily_metrics)
        
        return daily_metrics
    
    def get_me_history_df(self) -> pd.DataFrame:
        if not self.daily_me_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.daily_me_history)
    
    def save_daily_me_data(self, data_dir: str = 'data/daily'):
        """
        Save daily M/E data to the daily data directory as portfolio_ME.csv
        """
        import os
        
        # Ensure directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Get current M/E metrics
        current_metrics = self.calculate_daily_me_ratio()
        
        # Create filename for portfolio M/E data
        filename = os.path.join(data_dir, "portfolio_ME.csv")
        
        # Check if file exists
        if os.path.exists(filename):
            # Append to existing data
            existing_df = pd.read_csv(filename, parse_dates=['Date'])
            
            # Remove today's data if it exists (update)
            today = datetime.now().strftime('%Y-%m-%d')
            existing_df = existing_df[existing_df['Date'].dt.strftime('%Y-%m-%d') != today]
            
            # Add new data
            new_row = pd.DataFrame([current_metrics])
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            # Create new file
            updated_df = pd.DataFrame([current_metrics])
        
        # Save updated data
        updated_df.to_csv(filename, index=False)
        
        return filename
    
    def get_risk_assessment(self) -> Dict:
        """
        Get risk assessment based on current M/E ratio
        """
        current_metrics = self.calculate_daily_me_ratio()
        me_ratio = current_metrics['ME_Ratio']
        
        if me_ratio > 100:
            risk_level = "CRITICAL"
            risk_color = "red"
            recommendation = "IMMEDIATE REBALANCING REQUIRED - Reduce position sizes"
        elif me_ratio > 80:
            risk_level = "HIGH"
            risk_color = "orange"
            recommendation = "Consider reducing position sizes"
        elif me_ratio > 60:
            risk_level = "MODERATE"
            risk_color = "yellow"
            recommendation = "Monitor closely, consider position limits"
        else:
            risk_level = "LOW"
            risk_color = "green"
            recommendation = "Within acceptable risk parameters"
        
        return {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'me_ratio': me_ratio,
            'portfolio_equity': current_metrics['Portfolio_Equity'],
            'total_position_value': current_metrics['Total_Position_Value']
        }

# Configure logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set encoding for Windows console to handle Unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class NGSStrategy:
    """
    Neural Grid Strategy (nGS) implementation.
    Handles both signal generation and position management with 6-month data retention.
    """
    def __init__(self, account_size: float = 1000000, data_dir: str = 'data'):
        self.account_size = round(float(account_size), 2)
        self.cash = round(float(account_size), 2)
        self.positions = {}
        self._trades = []
        self.data_dir = data_dir
        self.retention_days = RETENTION_DAYS  # Use the 6-month retention from data_manager
        self.cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        # Initialize M/E calculator
        self.me_calculator = DailyMERatioCalculator(initial_portfolio_value=account_size)
        
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
        return self.me_calculator.calculate_daily_me_ratio()['ME_Ratio']

    def calculate_historical_me_ratio(self, current_prices: Dict[str, float] = None) -> float:
        return self.me_calculator.calculate_daily_me_ratio()['ME_Ratio']  # Historical is now daily snapshot

    def record_historical_me_ratio(self, date_str: str, trade_occurred: bool = False, current_prices: Dict[str, float] = None):
        self.me_calculator.calculate_daily_me_ratio(date_str)  # Automatically appends to history

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
        possible_exits = []

        # Gap out / Target
        if (position['profit'] > 0 and position['bars_since_entry'] > 1 and
            (df['Open'].iloc[i] >= df['Close'].iloc[i-1] * 1.05 or
             (not pd.isna(df['UpperBB'].iloc[i]) and df['Open'].iloc[i] >= df['UpperBB'].iloc[i]))):
            exit_type = 'Gap out L' if df['Open'].iloc[i] >= df['Close'].iloc[i-1] * 1.05 else 'Target L'
            possible_exits.append(('gap_target', exit_type, -1, None))

        # BE L
        if (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
            df['Close'].iloc[i] >= position['entry_price'] + df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()):
            possible_exits.append(('be', 'BE L', -1, None))

        # L ATR X
        if (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
            df['Close'].iloc[i] >= position['entry_price'] + df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max() * 1.5):
            possible_exits.append(('atr_x', 'L ATR X', -1, None))

        # Reversal S 2 L
        if (position['bars_since_entry'] > 5 and
            not pd.isna(df['LinReg'].iloc[i]) and not pd.isna(df['LinReg'].iloc[i-1]) and
            df['LinReg'].iloc[i] < df['LinReg'].iloc[i-1] and
            position['profit'] < 0 and
            not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
            df['oLRValue'].iloc[i] < df['oLRValue2'].iloc[i] and
            not pd.isna(df['ATR'].iloc[i]) and not pd.isna(df['ATRma'].iloc[i]) and
            df['ATR'].iloc[i] > df['ATRma'].iloc[i]):
            possible_exits.append(('reversal', 'S 2 L', -1, -1))

        # Hard Stop
        if df['Close'].iloc[i] < position['entry_price'] * 0.9:
            possible_exits.append(('hard_stop', 'Hard Stop S', -1, None))

        # Prioritize: hard_stop > reversal > atr_x > be > gap_target
        priority_order = ['hard_stop', 'reversal', 'atr_x', 'be', 'gap_target']
        for prio in priority_order:
            for exit_type, exit_label, exit_sig, entry_sig in possible_exits:
                if exit_type == prio:
                    df.loc[df.index[i], 'ExitSignal'] = exit_sig
                    df.loc[df.index[i], 'ExitType'] = exit_label
                    if entry_sig is not None:
                        df.loc[df.index[i], 'Signal'] = entry_sig
                        df.loc[df.index[i], 'SignalType'] = exit_label
                        df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] * 2 / df['Close'].iloc[i]))
                    return  # Apply first in priority

    def _check_short_exits(self, df: pd.DataFrame, i: int, position: Dict) -> None:
        possible_exits = []

        # Gap out / Target
        if (position['profit'] > 0 and position['bars_since_entry'] > 1 and
            (df['Open'].iloc[i] <= df['Close'].iloc[i-1] * 0.95 or
             (not pd.isna(df['LowerBB'].iloc[i]) and df['Open'].iloc[i] <= df['LowerBB'].iloc[i]))):
            exit_type = 'Gap out S' if df['Open'].iloc[i] <= df['Close'].iloc[i-1] * 0.95 else 'Target S'
            possible_exits.append(('gap_target', exit_type, 1, None))

        # BE S
        if (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
            df['Close'].iloc[i] <= position['entry_price'] - df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()):
            possible_exits.append(('be', 'BE S', 1, None))

        # S ATR X
        if (not pd.isna(df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max()) and
            df['Close'].iloc[i] <= position['entry_price'] - df['ATR'].iloc[max(0, i-position['bars_since_entry']):i+1].max() * 1.5):
            possible_exits.append(('atr_x', 'S ATR X', 1, None))

        # Reversal L 2 S
        if (position['bars_since_entry'] > 5 and
            not pd.isna(df['LinReg'].iloc[i]) and not pd.isna(df['LinReg'].iloc[i-1]) and
            df['LinReg'].iloc[i] > df['LinReg'].iloc[i-1] and
            position['profit'] < 0 and
            not pd.isna(df['oLRValue'].iloc[i]) and not pd.isna(df['oLRValue2'].iloc[i]) and
            df['oLRValue'].iloc[i] > df['oLRValue2'].iloc[i] and
            not pd.isna(df['ATR'].iloc[i]) and not pd.isna(df['ATRma'].iloc[i]) and
            df['ATR'].iloc[i] > df['ATRma'].iloc[i]):
            possible_exits.append(('reversal', 'L 2 S', 1, 1))

        # Hard Stop
        if df['Close'].iloc[i] > position['entry_price'] * 1.1:
            possible_exits.append(('hard_stop', 'Hard Stop L', 1, None))

        # Prioritize: hard_stop > reversal > atr_x > be > gap_target
        priority_order = ['hard_stop', 'reversal', 'atr_x', 'be', 'gap_target']
        for prio in priority_order:
            for exit_type, exit_label, exit_sig, entry_sig in possible_exits:
                if exit_type == prio:
                    df.loc[df.index[i], 'ExitSignal'] = exit_sig
                    df.loc[df.index[i], 'ExitType'] = exit_label
                    if entry_sig is not None:
                        df.loc[df.index[i], 'Signal'] = entry_sig
                        df.loc[df.index[i], 'SignalType'] = exit_label
                        df.loc[df.index[i], 'Shares'] = int(round(self.inputs['PositionSize'] * 2 / df['Close'].iloc[i]))
                    return  # Apply first in priority

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
        
        # Compute trade_type explicitly (fix for KeyError 'type')
        trade_type = 'long' if position['shares'] > 0 else 'short'
        self.me_calculator.update_position(symbol, 0, 0, 0, trade_type)  # Close position
        self.me_calculator.add_realized_pnl(profit)  # Add realized profit
        
        self.cash = round(float(self.cash + position['shares'] * exit_price), 2)
        
        # Record historical M/E after exit
        self.record_historical_me_ratio(exit_date, trade_occurred=True)
        
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
            
            current_price = round(float(df['Close'].iloc[i]), 2)
            trade_type = 'long' if shares > 0 else 'short'
            self.me_calculator.update_position(symbol, shares, position['entry_price'], current_price, trade_type)
            
            # Record historical M/E after entry
            self.record_historical_me_ratio(entry_date, trade_occurred=True)
            
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
            if df['Signal'].iloc[i] != 0:
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
                    # Handle multiple date formats
                    if isinstance(entry_date, str):
                        # Try multiple date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
                            try:
                                entry_datetime = datetime.strptime(entry_date.split()[0], '%Y-%m-%d')  # Just take date part
                                entry_date_str = entry_datetime.strftime('%Y-%m-%d')
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format worked, skip this position
                            logger.warning(f"Could not parse date for position {symbol}: {entry_date}")
                            continue
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
                        # Update M/E calculator with loaded position
                        current_price = pos.get('current_price', pos.get('entry_price', 0))
                        trade_type = pos.get('side', 'long')
                        self.me_calculator.update_position(symbol, self.positions[symbol]['shares'], 
                                                          self.positions[symbol]['entry_price'], current_price, trade_type)
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
        self.me_calculator = DailyMERatioCalculator(self.account_size)  # Reset M/E tracking
        
        # Filter trades by retention period
        self._filter_trades_by_retention()
        
        # Calculate initial M/E ratio for any existing positions
        if self.positions:
            initial_me_ratio = self.calculate_current_me_ratio()
            logger.info(f"Initial M/E ratio with {len(self.positions)} existing positions: {initial_me_ratio:.2f}%")
        
        results = {}
        for i, (symbol, df) in enumerate(data.items()):
            result = self.process_symbol(symbol, df)
            if result is not None and not result.empty:
                results[symbol] = result
                logger.info(f"Processed {symbol}: {len(result)} rows ({i+1}/{len(data)})")
        
        # Save M/E history after run
        self.me_calculator.save_daily_me_data()
        
        long_positions, short_positions = self.get_current_positions()
        all_positions = []
        
        for pos in long_positions:
            if pos['symbol'] in results and not results[pos['symbol']].empty:
                current_price = results[pos['symbol']].iloc[-1]['Close']
            else:
                current_price = pos['entry_price']
            
            # Calculate days held using proper datetime import
            entry_dt = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
            days_held = (datetime.now() - entry_dt).days
            
            all_positions.append({
                'symbol': pos['symbol'],
                'shares': pos['shares'],
                'entry_price': pos['entry_price'],
                'entry_date': pos['entry_date'],
                'current_price': round(float(current_price), 2),
                'current_value': round(float(current_price * pos['shares']), 2),
                'profit': round(float((current_price - pos['entry_price']) * pos['shares']), 2),
                'profit_pct': round(float((current_price / pos['entry_price'] - 1) * 100), 2),
                'days_held': days_held,
                'side': 'long',
                'strategy': 'nGS'
            })
        
        for pos in short_positions:
            if pos['symbol'] in results and not results[pos['symbol']].empty:
                current_price = results[pos['symbol']].iloc[-1]['Close']
            else:
                current_price = pos['entry_price']
            
            # Calculate days held and profit using proper datetime import
            shares_abs = abs(pos['shares'])
            profit = round(float((pos['entry_price'] - current_price) * shares_abs), 2)
            entry_dt = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
            days_held = (datetime.now() - entry_dt).days
            
            all_positions.append({
                'symbol': pos['symbol'],
                'shares': -shares_abs,
                'entry_price': pos['entry_price'],
                'entry_date': pos['entry_date'],
                'current_price': round(float(current_price), 2),
                'current_value': round(float(current_price * shares_abs), 2),
                'profit': profit,
                'profit_pct': round(float((pos['entry_price'] / current_price - 1) * 100), 2),
                'days_held': days_held,
                'side': 'short',
                'strategy': 'nGS'
            })
        
        save_positions(all_positions)
        
        # Final M/E status with position details for verification
        print(f"\nFinal M/E Status: {self.calculate_current_me_ratio():.2f}%")
        
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
        risk = self.me_calculator.get_risk_assessment()
        print(f"M/E Risk Status:      {risk['risk_level']}")
        print(f"M/E Realized P&L:     ${self.me_calculator.realized_pnl:.2f}")
        print(f"M/E Active Positions: {len(self.me_calculator.current_positions)}")
        
        logger.info(f"Strategy run complete. Processed {len(data)} symbols, currently have {len(all_positions)} positions")
        logger.info(f"Data retention: {self.retention_days} days, cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
        return results

    def backfill_symbol(self, symbol: str, data: pd.DataFrame):
        if data is not None and not data.empty:
            df = data.reset_index().rename(columns={
                'timestamp': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df_with_indicators = self.calculate_indicators(df)
            if df_with_indicators is not None:
                from data_manager import save_price_data
                save_price_data(symbol, df_with_indicators)
                logger.info(f"Backfilled and saved indicators for {symbol}")
        else:
            logger.warning(f"No data to backfill for {symbol}")

def load_polygon_data(symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load EOD data using your existing data_manager functions.
    This function uses your automated/manual download data.
    NOTE: load_price_data only takes symbol as argument - date filtering is done by data_manager.
    """
    # Default to 6-month data range if not specified
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=RETENTION_DAYS + 30)).strftime('%Y-%m-%d')  # Extra buffer for indicators
    
    data = {}
    logger.info(f"Loading data for {len(symbols)} symbols")
    logger.info(f"Note: data_manager automatically filters to 6-month retention period")
    
    # Progress tracking for large symbol lists
    batch_size = 50
    total_batches = (len(symbols) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(symbols))
        batch_symbols = symbols[start_idx:end_idx]
        
        print(f"\nLoading batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)...")
        
        for i, symbol in enumerate(batch_symbols):
            try:
                # Use your existing load_price_data function from data_manager
                # It only takes symbol as argument and returns 6-month filtered data
                df = load_price_data(symbol)  # FIXED: Only pass symbol
                
                if df is not None and not df.empty:
                    # Ensure Date column is datetime
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Data is already 6-month filtered by data_manager
                    # Just verify it has enough data
                    if len(df) >= 20:  # Minimum needed for indicators
                        data[symbol] = df
                        if (start_idx + i + 1) % 10 == 0:
                            logger.info(f"Progress: {start_idx + i + 1}/{len(symbols)} symbols loaded")
                    else:
                        logger.warning(f"Insufficient data for {symbol}: only {len(df)} rows")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
    
    logger.info(f"\nCompleted loading data. Successfully loaded {len(data)} out of {len(symbols)} symbols")
    return data

if __name__ == "__main__":
    print("nGS Trading Strategy - Neural Grid System")
    print("=" * 50)
    print(f"Data Retention: {RETENTION_DAYS} days (6 months)")
    print("=" * 50)
    
    try:
        strategy = NGSStrategy(account_size=1000000)
        
        # Load ALL S&P 500 symbols from your data files
        sp500_file = os.path.join('data', 'sp500_symbols.txt')
        try:
            with open(sp500_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            print(f"\nTrading Universe: ALL S&P 500 symbols")
            print(f"Total symbols loaded: {len(symbols)}")
            print(f"First 10 symbols: {', '.join(symbols[:10])}...")
            print(f"Last 10 symbols: {', '.join(symbols[-10:])}...")
            
        except FileNotFoundError:
            print(f"\nWARNING: {sp500_file} not found. Using sample symbols instead.")
            # Fallback to sample symbols if file not found
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA',
                'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'PFE'
            ]
            print(f"Using {len(symbols)} sample symbols")
        
        print(f"\nLoading historical data for {len(symbols)} symbols...")
        print("This may take several minutes for 500+ symbols...")
        
        # Load data using your existing data_manager functions
        data = load_polygon_data(symbols)
        
        if not data:
            print("No data loaded - check your data files")
            exit(1)
        
        print(f"\nSuccessfully loaded data for {len(data)} symbols")
        
        # Run the strategy
        print(f"\nRunning nGS strategy on {len(data)} symbols...")
        print("Processing signals and managing positions...")
        
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
        print(f"Symbols processed:    {len(data)}")
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
            
            # Symbol performance
            symbol_profits = {}
            for trade in strategy.trades:
                symbol = trade['symbol']
                if symbol not in symbol_profits:
                    symbol_profits[symbol] = 0
                symbol_profits[symbol] += trade['profit']
            
            # Top performers
            sorted_symbols = sorted(symbol_profits.items(), key=lambda x: x[1], reverse=True)
            print(f"\nTop 5 performing symbols:")
            for symbol, profit in sorted_symbols[:5]:
                print(f"  {symbol:6s}: ${profit:+8.2f}")
            
            print(f"\nBottom 5 performing symbols:")
            for symbol, profit in sorted_symbols[-5:]:
                print(f"  {symbol:6s}: ${profit:+8.2f}")
        
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
        
        if total_positions > 0:
            print(f"\nSample positions (first 10):")
            all_pos = long_pos[:5] + short_pos[:5]
            for pos in all_pos[:10]:
                side = "Long " if pos in long_pos else "Short"
                shares = pos['shares'] if pos in long_pos else pos['shares']
                print(f"  {side} {pos['symbol']:6s}: {shares:4d} shares @ ${pos['entry_price']:7.2f}")
        
        print(f"\n+ Strategy backtest completed successfully!")
        print(f"+ Processed all {len(data)} S&P 500 symbols")
        print(f"+ Data retention enforced: {RETENTION_DAYS} days")
        
    except Exception as e:
        logger.error(f"Strategy backtest failed: {e}")
        import traceback
        traceback.print_exc()
