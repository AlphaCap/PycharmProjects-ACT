"""
gSTDayTrader - Gap Strategy Day Trading Algorithm
Professional gap trading system with comprehensive risk management
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json
import os

class GSTDayTrader:
    """
    Gap Strategy Day Trader - Professional intraday gap trading system
    
    Strategy Logic:
    1. Identifies significant pre-market gaps (>2% up or down)
    2. Waits for market open confirmation
    3. Enters positions based on gap-fill probability
    4. Manages risk with tight stops and profit targets
    """
    
    def __init__(self, api_key: str, max_risk_per_trade: float = 100):
        self.api_key = api_key
        self.max_risk_per_trade = max_risk_per_trade  # Max $ risk per trade (scalping)
        self.trades = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Strategy parameters
        self.min_gap_threshold = 0.02  # 2% minimum gap
        self.max_gap_threshold = 0.10  # 10% maximum gap (avoid extreme moves)
        self.stop_loss_pct = 0.015     # 1.5% stop loss
        self.profit_target_pct = 0.025 # 2.5% profit target
        self.max_hold_time = 240       # 4 hours max hold time (minutes)
        
        # Risk management
        self.max_daily_loss = -500     # Max $500 daily loss
        self.max_positions = 3         # Max 3 concurrent positions
        self.min_volume_threshold = 100000  # Min 100k volume
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"gSTDayTrader initialized - Max risk per trade: ${max_risk_per_trade}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/gst_trader_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_intraday_data(self, symbol: str, interval: str = "1min") -> Optional[pd.DataFrame]:
        """
        Get intraday data from Alpha Vantage API
        
        Args:
            symbol: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'full'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                self.logger.warning(f"API Limit for {symbol}: {data['Note']}")
                return None
                
            # Parse time series data
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                self.logger.warning(f"No time series data for {symbol}")
                return None
                
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df.sort_index(inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_previous_close(self, symbol: str) -> Optional[float]:
        """
        Get previous day's closing price
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Previous close price or None if failed
        """
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                daily_data = data['Time Series (Daily)']
                # Get the most recent trading day
                latest_date = sorted(daily_data.keys())[-1]
                previous_close = float(daily_data[latest_date]['4. close'])
                return previous_close
                
        except Exception as e:
            self.logger.error(f"Error getting previous close for {symbol}: {e}")
            
        return None
    
    def calculate_gap(self, current_price: float, previous_close: float) -> float:
        """
        Calculate gap percentage
        
        Args:
            current_price: Current price
            previous_close: Previous close price
            
        Returns:
            Gap percentage (positive for gap up, negative for gap down)
        """
        return (current_price - previous_close) / previous_close
    
    def identify_gap_opportunity(self, df: pd.DataFrame, previous_close: float) -> Dict:
        """
        Identify gap trading opportunities
        
        Args:
            df: Intraday price data
            previous_close: Previous day's close price
            
        Returns:
            Dictionary with gap analysis
        """
        if df.empty:
            return {'has_gap': False, 'reason': 'No data'}
            
        # Get market open price (9:30 AM)
        market_open_time = df.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Find closest timestamp to market open
        open_data = df[df.index >= market_open_time].iloc[0] if len(df[df.index >= market_open_time]) > 0 else df.iloc[0]
        
        open_price = open_data['open']
        current_price = df.iloc[-1]['close']  # Most recent price
        volume = df['volume'].sum()
        
        # Calculate gap
        gap_pct = self.calculate_gap(open_price, previous_close)
        
        # Check gap criteria
        gap_analysis = {
            'symbol': None,
            'has_gap': False,
            'gap_pct': gap_pct,
            'gap_direction': 'up' if gap_pct > 0 else 'down',
            'open_price': open_price,
            'current_price': current_price,
            'previous_close': previous_close,
            'volume': volume,
            'reason': None
        }
        
        # Volume check
        if volume < self.min_volume_threshold:
            gap_analysis['reason'] = f'Low volume: {volume:,.0f}'
            return gap_analysis
        
        # Gap size check
        if abs(gap_pct) < self.min_gap_threshold:
            gap_analysis['reason'] = f'Gap too small: {gap_pct:.1%}'
            return gap_analysis
            
        if abs(gap_pct) > self.max_gap_threshold:
            gap_analysis['reason'] = f'Gap too large: {gap_pct:.1%}'
            return gap_analysis
        
        # Valid gap found
        gap_analysis['has_gap'] = True
        gap_analysis['reason'] = f'Valid {gap_analysis["gap_direction"]} gap: {gap_pct:.1%}'
        
        return gap_analysis
    
    def generate_trade_signal(self, gap_analysis: Dict, df: pd.DataFrame) -> Dict:
        """
        Generate trade signal based on gap analysis
        
        Args:
            gap_analysis: Gap analysis results
            df: Intraday price data
            
        Returns:
            Trade signal dictionary
        """
        if not gap_analysis['has_gap']:
            return {'action': 'no_trade', 'reason': gap_analysis['reason']}
        
        current_price = gap_analysis['current_price']
        gap_pct = gap_analysis['gap_pct']
        
        # Gap-fill strategy logic
        if gap_pct > 0:  # Gap up
            # Short position expecting gap fill
            action = 'short'
            entry_price = current_price
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            profit_target = gap_analysis['previous_close']  # Gap fill target
            
        else:  # Gap down
            # Long position expecting gap fill
            action = 'long'
            entry_price = current_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            profit_target = gap_analysis['previous_close']  # Gap fill target
        
        # Calculate position size based on risk amount (SCALPING APPROACH)
        risk_amount = abs(entry_price - stop_loss)
        shares = int(self.max_risk_per_trade / risk_amount) if risk_amount > 0 else 0
        
        # Minimum position size check
        if shares < 10:  # Don't trade if less than 10 shares
            return {'action': 'no_trade', 'reason': 'Position too small (< 10 shares)'}
        
        # Calculate actual position value for logging
        position_value = shares * entry_price
        
        trade_signal = {
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'shares': shares,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'max_risk': self.max_risk_per_trade,
            'expected_profit': abs(profit_target - entry_price),
            'risk_reward_ratio': abs(profit_target - entry_price) / risk_amount if risk_amount > 0 else 0
        }
        
        return trade_signal
    
    def execute_trade(self, symbol: str, trade_signal: Dict, gap_analysis: Dict) -> Dict:
        """
        Execute trade (simulation)
        
        Args:
            symbol: Stock symbol
            trade_signal: Trade signal details
            gap_analysis: Gap analysis results
            
        Returns:
            Trade execution result
        """
        if trade_signal['action'] == 'no_trade':
            return {'executed': False, 'reason': trade_signal['reason']}
        
        # Risk management checks
        if self.daily_pnl <= self.max_daily_loss:
            return {'executed': False, 'reason': 'Daily loss limit reached'}
        
        if len([t for t in self.trades if t.get('status') == 'open']) >= self.max_positions:
            return {'executed': False, 'reason': 'Maximum positions reached'}
        
        # Execute trade
        trade = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'action': trade_signal['action'],
            'entry_price': trade_signal['entry_price'],
            'stop_loss': trade_signal['stop_loss'],
            'profit_target': trade_signal['profit_target'],
            'shares': trade_signal['shares'],
            'gap_pct': gap_analysis['gap_pct'],
            'status': 'open',
            'pnl': 0.0,
            'exit_price': None,
            'exit_reason': None
        }
        
        self.trades.append(trade)
        
        # Simulate immediate exit (for backtesting)
        # In real trading, this would be managed by separate exit logic
        exit_result = self.simulate_exit(trade, gap_analysis)
        
        self.logger.info(f"Trade executed: {symbol} {trade_signal['action']} at {trade_signal['entry_price']:.2f}")
        
        return {'executed': True, 'trade': trade, 'exit_result': exit_result}
    
    def simulate_exit(self, trade: Dict, gap_analysis: Dict) -> Dict:
        """
        Simulate trade exit for backtesting
        
        Args:
            trade: Trade details
            gap_analysis: Gap analysis results
            
        Returns:
            Exit simulation result
        """
        # Simple simulation: assume 60% probability of gap fill
        # In real system, this would monitor live prices
        
        import random
        gap_fill_probability = 0.6
        
        if random.random() < gap_fill_probability:
            # Gap fills - profitable exit
            exit_price = trade['profit_target']
            exit_reason = 'profit_target'
        else:
            # Gap doesn't fill - stop loss hit
            exit_price = trade['stop_loss']
            exit_reason = 'stop_loss'
        
        # Calculate P&L
        if trade['action'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['shares']
        else:  # short
            pnl = (trade['entry_price'] - exit_price) * trade['shares']
        
        # Update trade
        trade['exit_price'] = exit_price
        trade['exit_reason'] = exit_reason
        trade['pnl'] = pnl
        trade['status'] = 'closed'
        
        # Update portfolio
        self.daily_pnl += pnl
        self.total_pnl += pnl
        
        # Update drawdown
        if pnl < 0:
            self.current_drawdown += pnl
            self.max_drawdown = min(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = 0
        
        return {
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl
        }
    
    def process_symbol(self, symbol: str) -> Dict:
        """
        Process a single symbol for gap trading opportunities
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Processing result
        """
        self.logger.info(f"Processing symbol: {symbol}")
        
        try:
            # Get intraday data
            df = self.get_intraday_data(symbol)
            if df is None:
                return {'success': False, 'reason': 'No intraday data'}
            
            # Get previous close
            previous_close = self.get_previous_close(symbol)
            if previous_close is None:
                return {'success': False, 'reason': 'No previous close data'}
            
            # Identify gap opportunity
            gap_analysis = self.identify_gap_opportunity(df, previous_close)
            gap_analysis['symbol'] = symbol
            
            # Generate trade signal
            trade_signal = self.generate_trade_signal(gap_analysis, df)
            
            # Execute trade if signal is valid
            execution_result = self.execute_trade(symbol, trade_signal, gap_analysis)
            
            result = {
                'success': True,
                'symbol': symbol,
                'gap_analysis': gap_analysis,
                'trade_signal': trade_signal,
                'execution_result': execution_result
            }
            
            # Add small delay to avoid API rate limits
            time.sleep(0.2)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return {'success': False, 'reason': str(e)}
    
    def run_strategy(self, symbols: List[str]) -> Dict:
        """
        Run gap trading strategy on multiple symbols
        
        Args:
            symbols: List of stock symbols to analyze
            
        Returns:
            Strategy execution results
        """
        self.logger.info(f"Starting gap trading strategy on {len(symbols)} symbols")
        
        results = {
            'start_time': datetime.now(),
            'symbols_processed': 0,
            'symbols_with_data': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0
        }
        
        for symbol in symbols:
            try:
                result = self.process_symbol(symbol)
                results['symbols_processed'] += 1
                
                if result['success']:
                    results['symbols_with_data'] += 1
                    
                    if result['execution_result']['executed']:
                        results['total_trades'] += 1
                        
                        # Track trade success
                        trade = result['execution_result']['trade']
                        if trade['pnl'] > 0:
                            results['successful_trades'] += 1
                        else:
                            results['failed_trades'] += 1
                
                # Stop if daily loss limit reached
                if self.daily_pnl <= self.max_daily_loss:
                    self.logger.warning("Daily loss limit reached, stopping strategy")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in strategy loop for {symbol}: {e}")
                continue
        
        results['end_time'] = datetime.now()
        results['duration'] = results['end_time'] - results['start_time']
        
        self.logger.info(f"Strategy completed: {results['total_trades']} trades executed")
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary
        
        Returns:
            Performance metrics dictionary
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        trades_df = pd.DataFrame(self.trades)
        closed_trades = trades_df[trades_df['status'] == 'closed']
        
        if len(closed_trades) == 0:
            return {
                'total_trades': len(trades_df),
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = closed_trades['pnl'].sum()
        avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(closed_trades) > 1:
            returns = closed_trades['pnl'] / self.max_risk_per_trade  # Returns as % of risk
            sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': f"{win_rate:.1%}",
            'total_pnl': f"${total_pnl:.2f}",
            'avg_profit_per_trade': f"${avg_profit_per_trade:.2f}",
            'max_drawdown': f"${self.max_drawdown:.2f}",
            'sharpe_ratio': f"{sharpe_ratio:.2f}",
            'best_trade': f"${closed_trades['pnl'].max():.2f}" if not closed_trades.empty else "$0.00",
            'worst_trade': f"${closed_trades['pnl'].min():.2f}" if not closed_trades.empty else "$0.00"
        }
    
    def save_trades_to_csv(self, filename: str = None):
        """
        Save all trades to CSV file
        
        Args:
            filename: Optional custom filename
        """
        if not self.trades:
            self.logger.warning("No trades to save")
            return
        
        if filename is None:
            filename = f"gst_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename, index=False)
        self.logger.info(f"Trades saved to {filename}")
    
    def reset_daily_stats(self):
        """Reset daily statistics for new trading day"""
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.logger.info("Daily statistics reset")

# Example usage
if __name__ == "__main__":
    # Example usage
    api_key = "YOUR_API_KEY"
    trader = GSTDayTrader(api_key, max_risk_per_trade=100)  # Risk $100 per trade
    
    # Test symbols
    symbols = ['AAPL', 'TSLA', 'MSFT']
    
    # Run strategy
    results = trader.run_strategy(symbols)
    
    # Print results
    print("Strategy Results:")
    print(f"Total trades: {results['total_trades']}")
    
    # Performance summary
    performance = trader.get_performance_summary()
    print("\nPerformance Summary:")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # Save trades
    trader.save_trades_to_csv()