"""
nGS Dashboard Live Update System
Automatically detects new trades and updates performance metrics in real-time
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import glob
import os
import json
from typing import Dict, List, Tuple, Optional

class nGSDashboardUpdater:
    """
    Real-time dashboard updater for nGS trading system
    Detects new trades, calculates performance, and updates display
    """
    
    def __init__(self):
        self.data_directory = "."  # Current directory where CSV files are located
        self.trade_cache_file = "ngs_trade_cache.json"
        self.performance_cache_file = "ngs_performance_cache.json"
        self.last_update = None
        
        # Performance tracking
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.current_positions = {}
        
    def scan_for_updates(self) -> Dict:
        """
        Scan data directory for new/updated CSV files
        Returns dictionary of updated files and their timestamps
        """
        csv_files = glob.glob(os.path.join(self.data_directory, "*.csv"))
        updates = {}
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            if file_name.endswith('.csv') and len(file_name.split('.')[0]) <= 5:  # Stock symbols
                mod_time = os.path.getmtime(file_path)
                updates[file_name] = {
                    'file_path': file_path,
                    'modified_time': mod_time,
                    'symbol': file_name.replace('.csv', '').upper()
                }
        
        return updates
    
    def detect_new_trades(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """
        Analyze data to detect new trade signals and exits
        Returns list of detected trades
        """
        if df.empty or len(df) < 2:
            return []
        
        trades = []
        
        # Get latest data point
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Trade detection logic based on nGS indicators
        trade_signals = self.analyze_trade_signals(symbol, df, latest, previous)
        
        return trade_signals
    
    def analyze_trade_signals(self, symbol: str, df: pd.DataFrame, latest: pd.Series, previous: pd.Series) -> List[Dict]:
        """
        Analyze technical indicators to detect trade signals
        """
        trades = []
        current_time = datetime.now()
        
        # Entry Signals
        entry_signal = self.detect_entry_signal(df, latest, previous)
        if entry_signal:
            trade = {
                'symbol': symbol,
                'timestamp': current_time,
                'action': entry_signal['action'],
                'entry_price': latest['Close'],
                'entry_reason': entry_signal['reason'],
                'indicators': {
                    'atr': latest.get('ATR', 0),
                    'bb_position': self.get_bb_position(latest),
                    'trend_strength': latest.get('oLRSlope', 0),
                    'volume': latest.get('Volume', 0)
                },
                'status': 'open'
            }
            trades.append(trade)
        
        # Exit Signals for existing positions
        if symbol in self.current_positions:
            exit_signal = self.detect_exit_signal(df, latest, previous, self.current_positions[symbol])
            if exit_signal:
                self.close_position(symbol, latest['Close'], exit_signal['reason'])
        
        return trades
    
    def detect_entry_signal(self, df: pd.DataFrame, latest: pd.Series, previous: pd.Series) -> Optional[Dict]:
        """
        Detect entry signals based on nGS technical indicators
        """
        signals = []
        
        # Bollinger Band Breakout
        if 'UpperBB' in latest and 'LowerBB' in latest:
            if latest['Close'] > latest['UpperBB'] and previous['Close'] <= previous['UpperBB']:
                signals.append({
                    'action': 'long',
                    'reason': 'BB_Breakout_Up',
                    'strength': 0.7
                })
            elif latest['Close'] < latest['LowerBB'] and previous['Close'] >= previous['LowerBB']:
                signals.append({
                    'action': 'short',
                    'reason': 'BB_Breakout_Down',
                    'strength': 0.7
                })
        
        # PSAR Signal
        if 'PSAR_IsLong' in latest and 'PSAR_IsLong' in previous:
            if latest['PSAR_IsLong'] == 1 and previous['PSAR_IsLong'] == 0:
                signals.append({
                    'action': 'long',
                    'reason': 'PSAR_Long',
                    'strength': 0.6
                })
            elif latest['PSAR_IsLong'] == 0 and previous['PSAR_IsLong'] == 1:
                signals.append({
                    'action': 'short',
                    'reason': 'PSAR_Short',
                    'strength': 0.6
                })
        
        # Linear Regression Trend
        if 'oLRSlope' in latest:
            slope = latest['oLRSlope']
            if abs(slope) > 0.5:  # Strong trend
                if slope > 0 and latest['Close'] > latest.get('LinReg', latest['Close']):
                    signals.append({
                        'action': 'long',
                        'reason': 'Strong_Uptrend',
                        'strength': min(abs(slope), 1.0)
                    })
                elif slope < 0 and latest['Close'] < latest.get('LinReg', latest['Close']):
                    signals.append({
                        'action': 'short',
                        'reason': 'Strong_Downtrend',
                        'strength': min(abs(slope), 1.0)
                    })
        
        # Return strongest signal
        if signals:
            return max(signals, key=lambda x: x['strength'])
        
        return None
    
    def detect_exit_signal(self, df: pd.DataFrame, latest: pd.Series, previous: pd.Series, position: Dict) -> Optional[Dict]:
        """
        Detect exit signals for existing positions
        """
        entry_price = position['entry_price']
        action = position['action']
        current_price = latest['Close']
        
        # Calculate P&L
        if action == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Exit conditions
        
        # 1. Profit target (2%)
        if pnl_pct >= 0.02:
            return {'reason': 'Profit_Target', 'pnl_pct': pnl_pct}
        
        # 2. Stop loss (-1.5%)
        if pnl_pct <= -0.015:
            return {'reason': 'Stop_Loss', 'pnl_pct': pnl_pct}
        
        # 3. PSAR reversal
        if 'PSAR_IsLong' in latest and 'PSAR_IsLong' in previous:
            if action == 'long' and latest['PSAR_IsLong'] == 0 and previous['PSAR_IsLong'] == 1:
                return {'reason': 'PSAR_Reversal', 'pnl_pct': pnl_pct}
            elif action == 'short' and latest['PSAR_IsLong'] == 1 and previous['PSAR_IsLong'] == 0:
                return {'reason': 'PSAR_Reversal', 'pnl_pct': pnl_pct}
        
        # 4. Bollinger Band mean reversion
        if 'BBAvg' in latest:
            if action == 'long' and current_price < latest['BBAvg']:
                return {'reason': 'BB_Mean_Reversion', 'pnl_pct': pnl_pct}
            elif action == 'short' and current_price > latest['BBAvg']:
                return {'reason': 'BB_Mean_Reversion', 'pnl_pct': pnl_pct}
        
        return None
    
    def get_bb_position(self, row: pd.Series) -> str:
        """
        Determine position relative to Bollinger Bands
        """
        if 'UpperBB' not in row or 'LowerBB' not in row:
            return 'unknown'
        
        close = row['Close']
        upper = row['UpperBB']
        lower = row['LowerBB']
        
        if close > upper:
            return 'above_upper'
        elif close < lower:
            return 'below_lower'
        elif close > row.get('BBAvg', (upper + lower) / 2):
            return 'upper_half'
        else:
            return 'lower_half'
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """
        Close an existing position and update performance
        """
        if symbol not in self.current_positions:
            return
        
        position = self.current_positions[symbol]
        entry_price = position['entry_price']
        action = position['action']
        
        # Calculate P&L
        if action == 'long':
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
        
        # Update performance
        self.total_pnl += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Log the trade
        trade_record = {
            'symbol': symbol,
            'entry_time': position['timestamp'],
            'exit_time': datetime.now(),
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason
        }
        
        self.save_trade_record(trade_record)
        
        # Remove from current positions
        del self.current_positions[symbol]
    
    def save_trade_record(self, trade: Dict):
        """
        Save trade record to cache file
        """
        try:
            # Load existing trades
            if os.path.exists(self.trade_cache_file):
                with open(self.trade_cache_file, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []
            
            # Add new trade
            trade['timestamp'] = trade['exit_time'].isoformat()
            trades.append(trade)
            
            # Keep only last 100 trades
            trades = trades[-100:]
            
            # Save back to file
            with open(self.trade_cache_file, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
                
        except Exception as e:
            st.error(f"Error saving trade record: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate current performance metrics
        """
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'active_positions': len(self.current_positions),
            'last_update': datetime.now().strftime('%H:%M:%S')
        }
    
    def update_dashboard(self):
        """
        Main update function - call this in Streamlit app
        """
        # Scan for file updates
        file_updates = self.scan_for_updates()
        
        new_trades = []
        
        # Process each updated file
        for file_name, info in file_updates.items():
            try:
                # Read the CSV file
                df = pd.read_csv(info['file_path'])
                
                if not df.empty:
                    # Detect new trades
                    symbol_trades = self.detect_new_trades(info['symbol'], df)
                    new_trades.extend(symbol_trades)
                    
            except Exception as e:
                st.error(f"Error processing {file_name}: {e}")
        
        # Update current positions
        for trade in new_trades:
            if trade['status'] == 'open':
                self.current_positions[trade['symbol']] = trade
        
        return self.get_performance_metrics(), new_trades

def create_ngs_dashboard():
    """
    Create the updated nGS dashboard with live performance data
    """
    st.title("📊 nGS Trading Dashboard - Live Updates")
    
    # Initialize updater
    if 'ngs_updater' not in st.session_state:
        st.session_state.ngs_updater = nGSDashboardUpdater()
    
    updater = st.session_state.ngs_updater
    
    # Auto-refresh every 30 seconds
    if st.button("🔄 Refresh Data") or 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
        
    # Update dashboard
    performance, new_trades = updater.update_dashboard()
    
    # Display performance metrics
    st.subheader("📈 Live Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total P&L", 
            f"${performance['total_pnl']:.2f}",
            f"${performance['daily_pnl']:.2f} today"
        )
    
    with col2:
        st.metric(
            "Total Trades", 
            performance['total_trades'],
            f"{performance['winning_trades']} winners"
        )
    
    with col3:
        st.metric(
            "Win Rate", 
            f"{performance['win_rate']:.1f}%",
            f"{performance['active_positions']} active"
        )
    
    with col4:
        st.metric(
            "Last Update", 
            performance['last_update'],
            "Live"
        )
    
    # Display active positions
    if updater.current_positions:
        st.subheader("📋 Active Positions")
        
        positions_data = []
        for symbol, pos in updater.current_positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Action': pos['action'].upper(),
                'Entry Price': f"${pos['entry_price']:.2f}",
                'Entry Time': pos['timestamp'].strftime('%H:%M:%S'),
                'Reason': pos['entry_reason']
            })
        
        if positions_data:
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    
    # Display recent trades
    st.subheader("🏆 Recent Trades")
    
    if os.path.exists(updater.trade_cache_file):
        try:
            with open(updater.trade_cache_file, 'r') as f:
                trades = json.load(f)
            
            if trades:
                # Show last 10 trades
                recent_trades = trades[-10:]
                trade_df = pd.DataFrame(recent_trades)
                
                if not trade_df.empty:
                    # Format for display
                    display_df = trade_df[['symbol', 'action', 'entry_price', 'exit_price', 'pnl', 'exit_reason']].copy()
                    display_df.columns = ['Symbol', 'Action', 'Entry', 'Exit', 'P&L', 'Exit Reason']
                    display_df['Action'] = display_df['Action'].str.upper()
                    display_df['Entry'] = display_df['Entry'].apply(lambda x: f"${x:.2f}")
                    display_df['Exit'] = display_df['Exit'].apply(lambda x: f"${x:.2f}")
                    display_df['P&L'] = display_df['P&L'].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No trades recorded yet")
                
        except Exception as e:
            st.error(f"Error loading trade history: {e}")
    else:
        st.info("No trade history file found")
    
    # Auto-refresh
    if st.checkbox("🔄 Auto-refresh (30s)", value=True):
        st.rerun()

# Example usage
if __name__ == "__main__":
    create_ngs_dashboard()