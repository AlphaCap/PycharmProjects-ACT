"""
gSTDayTrader - Gap Strategy Day Trading Algorithm (PERMANENT VERSION)
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
    
    FIXED ISSUES:
    - Position sizing now properly limits risk
    - Added maximum position value check
    - Better error handling for invalid prices
    """
    
    def __init__(self, api_key: str, max_risk_per_trade: float = 100):
        self.api_key = api_key
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_value = max_risk_per_trade * 20
        self.trades = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Strategy parameters
        self.min_gap_threshold = 0.02
        self.max_gap_threshold = 0.08
        self.stop_loss_pct = 0.01
        self.profit_target_pct = 0.02
        self.max_hold_time = 240
        
        # Risk management
        self.max_daily_loss = -300
        self.max_positions = 2
        self.min_volume_threshold = 500000
        self.min_price = 20
        self.max_price = 1000
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info(f"gSTDayTrader initialized - Max risk: ${max_risk_per_trade}, Max position: ${self.max_position_value}")
        
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
        """Get intraday data from Alpha Vantage API with better error handling"""
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 'compact'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'Error Message' in data:
                self.logger.error(f"API Error for {symbol}: {data['Error Message']}")
                return None
                
            if 'Note' in data:
                self.logger.warning(f"API Limit for {symbol}: {data['Note']}")
                return None
                
            time_series_key = f'Time Series ({interval})'
            if time_series_key not in data:
                self.logger.warning(f"No time series data for {symbol}")
                return None
                
            time_series = data[time_series_key]
            
            if not time_series:
                self.logger.warning(f"Empty time series for {symbol}")
                return None
            
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)
            
            try:
                df = df.astype(float)
            except ValueError as e:
                self.logger.error(f"Data type conversion error for {symbol}: {e}")
                return None
            
            df.sort_index(inplace=True)
            
            recent_cutoff = datetime.now() - timedelta(days=2)
            df = df[df.index >= recent_cutoff]
            
            if df.empty:
                self.logger.warning(f"No recent data for {symbol}")
                return None
            
            self.logger.info(f"Retrieved {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_previous_close(self, symbol: str) -> Optional[float]:
        """Get previous day's closing price with better validation"""
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
                
                if not daily_data:
                    return None
                
                dates = sorted(daily_data.keys(), reverse=True)
                
                for date in dates:
                    try:
                        previous_close = float(daily_data[date]['4. close'])
                        
                        if self.min_price <= previous_close <= self.max_price:
                            return previous_close
                        else:
                            self.logger.warning(f"{symbol} price ${previous_close:.2f} outside range ${self.min_price}-${self.max_price}")
                            return None
                            
                    except (ValueError, KeyError) as e:
                        self.logger.error(f"Error parsing close price for {symbol}: {e}")
                        continue
                
        except Exception as e:
            self.logger.error(f"Error getting previous close for {symbol}: {e}")
            
        return None
    
    def calculate_gap(self, current_price: float, previous_close: float) -> float:
        """Calculate gap percentage with validation"""
        if previous_close <= 0:
            return 0.0
        return (current_price - previous_close) / previous_close
    
    def identify_gap_opportunity(self, df: pd.DataFrame, previous_close: float) -> Dict:
        """Identify gap trading opportunities with enhanced validation"""
        if df.empty:
            return {'has_gap': False, 'reason': 'No data'}
        
        if previous_close <= 0 or previous_close < self.min_price or previous_close > self.max_price:
            return {'has_gap': False, 'reason': f'Invalid previous close: ${previous_close:.2f}'}
            
        open_price = df.iloc[0]['open']
        current_price = df.iloc[-1]['close']
        volume = df['volume'].sum()
        
        if any(price <= 0 for price in [open_price, current_price]):
            return {'has_gap': False, 'reason': 'Invalid price data'}
        
        gap_pct = self.calculate_gap(open_price, previous_close)
        
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
        
        if volume < self.min_volume_threshold:
            gap_analysis['reason'] = f'Low volume: {volume:,.0f} < {self.min_volume_threshold:,.0f}'
            return gap_analysis
        
        if abs(gap_pct) < self.min_gap_threshold:
            gap_analysis['reason'] = f'Gap too small: {gap_pct:.1%} < {self.min_gap_threshold:.1%}'
            return gap_analysis
            
        if abs(gap_pct) > self.max_gap_threshold:
            gap_analysis['reason'] = f'Gap too large: {gap_pct:.1%} > {self.max_gap_threshold:.1%}'
            return gap_analysis
        
        gap_analysis['has_gap'] = True
        gap_analysis['reason'] = f'Valid {gap_analysis["gap_direction"]} gap: {gap_pct:.1%}'
        
        return gap_analysis
    
    def generate_trade_signal(self, gap_analysis: Dict, df: pd.DataFrame) -> Dict:
        """Generate trade signal with FIXED position sizing"""
        if not gap_analysis['has_gap']:
            return {'action': 'no_trade', 'reason': gap_analysis['reason']}
        
        current_price = gap_analysis['current_price']
        gap_pct = gap_analysis['gap_pct']
        
        if current_price <= 0 or current_price < self.min_price or current_price > self.max_price:
            return {'action': 'no_trade', 'reason': f'Invalid current price: ${current_price:.2f}'}
        
        if gap_pct > 0:
            action = 'short'
            entry_price = current_price
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            profit_target = max(gap_analysis['previous_close'], entry_price * (1 - self.profit_target_pct))
            
        else:
            action = 'long'
            entry_price = current_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            profit_target = min(gap_analysis['previous_close'], entry_price * (1 + self.profit_target_pct))
        
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {'action': 'no_trade', 'reason': 'Invalid risk calculation'}
        
        max_shares_by_risk = int(self.max_risk_per_trade / risk_per_share)
        max_shares_by_value = int(self.max_position_value / entry_price)
        
        shares = min(max_shares_by_risk, max_shares_by_value)
        
        if shares < 10:
            return {'action': 'no_trade', 'reason': f'Position too small: {shares} shares (min 10)'}
        
        if shares > 1000:
            shares = 1000
            self.logger.warning(f"Capping position at 1000 shares for safety")
        
        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        
        if position_value > self.max_position_value:
            return {'action': 'no_trade', 'reason': f'Position value ${position_value:.2f} exceeds limit ${self.max_position_value:.2f}'}
        
        if actual_risk > self.max_risk_per_trade * 1.1:
            return {'action': 'no_trade', 'reason': f'Actual risk ${actual_risk:.2f} exceeds limit ${self.max_risk_per_trade:.2f}'}
        
        expected_profit = abs(profit_target - entry_price) * shares
        risk_reward_ratio = expected_profit / actual_risk if actual_risk > 0 else 0
        
        trade_signal = {
            'action': action,
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'profit_target': round(profit_target, 2),
            'shares': shares,
            'position_value': round(position_value, 2),
            'risk_amount': round(actual_risk, 2),
            'max_risk': self.max_risk_per_trade,
            'expected_profit': round(expected_profit, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2)
        }
        
        return trade_signal
    
    def execute_trade(self, symbol: str, trade_signal: Dict, gap_analysis: Dict) -> Dict:
        """Execute trade (simulation) with enhanced risk management"""
        if trade_signal['action'] == 'no_trade':
            return {'executed': False, 'reason': trade_signal['reason']}
        
        if self.daily_pnl <= self.max_daily_loss:
            return {'executed': False, 'reason': f'Daily loss limit reached: ${self.daily_pnl:.2f}'}
        
        open_positions = [t for t in self.trades if t.get('status') == 'open']
        if len(open_positions) >= self.max_positions:
            return {'executed': False, 'reason': f'Maximum positions reached: {len(open_positions)}/{self.max_positions}'}
        
        if trade_signal['position_value'] > self.max_position_value:
            return {'executed': False, 'reason': f'Position value ${trade_signal["position_value"]:.2f} exceeds limit'}
        
        trade = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'action': trade_signal['action'],
            'entry_price': trade_signal['entry_price'],
            'stop_loss': trade_signal['stop_loss'],
            'profit_target': trade_signal['profit_target'],
            'shares': trade_signal['shares'],
            'position_value': trade_signal['position_value'],
            'risk_amount': trade_signal['risk_amount'],
            'gap_pct': gap_analysis['gap_pct'],
            'status': 'open',
            'pnl': 0.0,
            'exit_price': None,
            'exit_reason': None
        }
        
        self.trades.append(trade)
        
        exit_result = self.simulate_exit(trade, gap_analysis)
        
        self.logger.info(f"Trade executed: {symbol} {trade_signal['action']} {trade_signal['shares']} shares at ${trade_signal['entry_price']:.2f}, Risk: ${trade_signal['risk_amount']:.2f}")
        
        return {'executed': True, 'trade': trade, 'exit_result': exit_result}
    
    def simulate_exit(self, trade: Dict, gap_analysis: Dict) -> Dict:
        """Simulate trade exit with realistic probabilities"""
        import random
        
        gap_abs = abs(gap_analysis['gap_pct'])
        
        if gap_abs <= 0.03:
            gap_fill_probability = 0.75
        elif gap_abs <= 0.05:
            gap_fill_probability = 0.60
        else:
            gap_fill_probability = 0.45
        
        if random.random() < gap_fill_probability:
            exit_price = trade['profit_target']
            exit_reason = 'profit_target'
        else:
            exit_price = trade['stop_loss']
            exit_reason = 'stop_loss'
        
        if trade['action'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['shares']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['shares']
        
        pnl = round(pnl, 2)
        
        trade['exit_price'] = round(exit_price, 2)
        trade['exit_reason'] = exit_reason
        trade['pnl'] = pnl
        trade['status'] = 'closed'
        
        self.daily_pnl = round(self.daily_pnl + pnl, 2)
        self.total_pnl = round(self.total_pnl + pnl, 2)
        
        if pnl < 0:
            self.current_drawdown = round(self.current_drawdown + pnl, 2)
            self.max_drawdown = round(min(self.max_drawdown, self.current_drawdown), 2)
        else:
            self.current_drawdown = 0
        
        self.logger.info(f"Trade closed: {trade['symbol']} {exit_reason} at ${exit_price:.2f}, P&L: ${pnl:.2f}")
        
        return {
            'exit_price': round(exit_price, 2),
            'exit_reason': exit_reason,
            'pnl': pnl
        }
    
    def process_symbol(self, symbol: str) -> Dict:
        """Process a single symbol for gap trading opportunities"""
        self.logger.info(f"Processing symbol: {symbol}")
        
        try:
            previous_close = self.get_previous_close(symbol)
            if previous_close is None:
                return {'success': False, 'reason': 'No previous close data'}
            
            df = self.get_intraday_data(symbol)
            if df is None:
                return {'success': False, 'reason': 'No intraday data'}
            
            gap_analysis = self.identify_gap_opportunity(df, previous_close)
            gap_analysis['symbol'] = symbol
            
            trade_signal = self.generate_trade_signal(gap_analysis, df)
            
            execution_result = self.execute_trade(symbol, trade_signal, gap_analysis)
            
            result = {
                'success': True,
                'symbol': symbol,
                'gap_analysis': gap_analysis,
                'trade_signal': trade_signal,
                'execution_result': execution_result
            }
            
            time.sleep(1.0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return {'success': False, 'reason': str(e)}
    
    def run_strategy(self, symbols: List[str]) -> Dict:
        """Run gap trading strategy on multiple symbols"""
        self.logger.info(f"Starting gap trading strategy on {len(symbols)} symbols")
        
        results = {
            'start_time': datetime.now(),
            'symbols_processed': 0,
            'symbols_with_data': 0,
            'gaps_found': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'api_errors': 0
        }
        
        for i, symbol in enumerate(symbols, 1):
            self.logger.info(f"Processing {i}/{len(symbols)}: {symbol}")
            
            try:
                result = self.process_symbol(symbol)
                results['symbols_processed'] += 1
                
                if result['success']:
                    results['symbols_with_data'] += 1
                    
                    if result['gap_analysis']['has_gap']:
                        results['gaps_found'] += 1
                    
                    if result['execution_result']['executed']:
                        results['total_trades'] += 1
                        
                        trade = result['execution_result']['trade']
                        if trade['pnl'] > 0:
                            results['successful_trades'] += 1
                        else:
                            results['failed_trades'] += 1
                else:
                    if 'API' in result['reason'] or 'Error' in result['reason']:
                        results['api_errors'] += 1
                
                if self.daily_pnl <= self.max_daily_loss:
                    self.logger.warning(f"Daily loss limit reached: ${self.daily_pnl:.2f}, stopping strategy")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in strategy loop for {symbol}: {e}")
                results['api_errors'] += 1
                continue
        
        results['end_time'] = datetime.now()
        results['duration'] = results['end_time'] - results['start_time']
        
        self.logger.info(f"Strategy completed: {results['total_trades']} trades executed, {results['gaps_found']} gaps found")
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': '0.0%',
                'total_pnl': '$0.00',
                'avg_profit_per_trade': '$0.00',
                'max_drawdown': '$0.00',
                'sharpe_ratio': '0.00',
                'best_trade': '$0.00',
                'worst_trade': '$0.00'
            }
        
        trades_df = pd.DataFrame(self.trades)
        closed_trades = trades_df[trades_df['status'] == 'closed']
        
        if len(closed_trades) == 0:
            return {
                'total_trades': len(trades_df),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': '0.0%',
                'total_pnl': '$0.00',
                'avg_profit_per_trade': '$0.00',
                'max_drawdown': '$0.00',
                'sharpe_ratio': '0.00',
                'best_trade': '$0.00',
                'worst_trade': '$0.00'
            }
        
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = closed_trades['pnl'].sum()
        avg_profit_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        if len(closed_trades) > 1:
            returns = closed_trades['pnl'] / self.max_risk_per_trade
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
        """Save all trades to CSV file"""
        if not self.trades:
            self.logger.warning("No trades to save")
            return
        
        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filename, index=False)
        self.logger.info(f"Trades saved to {filename}")
    
    def save_performance_report(self, filename: str = None):
        """Save detailed performance report"""
        if filename is None:
            filename = f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        performance = self.get_performance_summary()
        
        detailed_report = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'results': {
                'symbols_processed': len(set(trade['symbol'] for trade in self.trades)) if self.trades else 0
            },
            'performance': {
                'total_trades': performance['total_trades'],
                'winning_trades': performance['winning_trades'],
                'losing_trades': performance['losing_trades'],
                'win_rate': performance['win_rate'],
                'total_pnl': performance['total_pnl'],
                'avg_profit_per_trade': performance['avg_profit_per_trade'],
                'max_drawdown': performance['max_drawdown'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'best_trade': performance['best_trade'],
                'worst_trade': performance['worst_trade']
            },
            'settings': {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_position_value': self.max_position_value,
                'min_gap_threshold': f"{self.min_gap_threshold:.1%}",
                'max_gap_threshold': f"{self.max_gap_threshold:.1%}",
                'stop_loss_pct': f"{self.stop_loss_pct:.1%}",
                'profit_target_pct': f"{self.profit_target_pct:.1%}",
                'min_volume_threshold': self.min_volume_threshold
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        self.logger.info(f"Performance report saved to {filename}")
        return detailed_report
    
    def reset_daily_stats(self):
        """Reset daily statistics for new trading day"""
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.logger.info("Daily statistics reset")

if __name__ == "__main__":
    api_key = "D4NJ9SDT2NS2L6UX"
    trader = GSTDayTrader(api_key, max_risk_per_trade=50)
    
    test_symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL']
    
    print("🚀 Starting gSTDayTrader with FIXED position sizing...")
    print(f"Max risk per trade: ${trader.max_risk_per_trade}")
    print(f"Max position value: ${trader.max_position_value}")
    print("=" * 60)
    
    results = trader.run_strategy(test_symbols)
    
    print("\n📊 STRATEGY RESULTS:")
    print("=" * 40)
    print(f"Symbols processed:   {results['symbols_processed']}")
    print(f"Symbols with data:   {results['symbols_with_data']}")
    print(f"Gaps found:          {results['gaps_found']}")
    print(f"Total trades:        {results['total_trades']}")
    print(f"Successful trades:   {results['successful_trades']}")
    print(f"Failed trades:       {results['failed_trades']}")
    print(f"API errors:          {results['api_errors']}")
    print(f"Duration:            {results['duration']}")
    
    performance = trader.get_performance_summary()
    print("\n📈 PERFORMANCE SUMMARY:")
    print("=" * 40)
    for key, value in performance.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    trader.save_trades_to_csv()
    trader.save_performance_report()
    
    print("\n✅ gSTDayTrader test completed!")
<<<<<<< HEAD
    print("📁 Results saved to CSV and JSON files")
=======
    print("📁 Results saved to CSV and JSON files")
>>>>>>> c3b6458da22b719f87e33e86cea00599c7a24a0f
