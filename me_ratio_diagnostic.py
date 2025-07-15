# me_ratio_calculator.py - Daily M/E Ratio Calculator for nGS Strategy
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

class DailyMERatioCalculator:
    """
    Calculate daily M/E ratios based on current positions and portfolio equity.
    This integrates with the nGS strategy to provide real-time risk management.
    """
    
    def __init__(self, initial_portfolio_value: float = 100000):
        self.initial_portfolio_value = initial_portfolio_value
        self.current_positions = {}  # symbol -> position_data
        self.realized_pnl = 0.0
        self.daily_me_history = []
        
    def update_position(self, symbol: str, shares: int, entry_price: float, 
                       current_price: float, trade_type: str = 'long'):
        """
        Update position for a symbol with current market price
        """
        if shares == 0:
            # Position closed
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        else:
            # Position open or updated
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
        """Calculate unrealized P&L for a position"""
        if trade_type.lower() == 'long':
            return (current_price - entry_price) * shares
        else:  # short
            return (entry_price - current_price) * abs(shares)
    
    def add_realized_pnl(self, profit: float):
        """Add realized profit/loss from closed trades"""
        self.realized_pnl += profit
    
    def calculate_daily_me_ratio(self, date: str = None) -> Dict:
        """
        Calculate current M/E ratio and portfolio metrics
        """
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
        """Get M/E ratio history as DataFrame"""
        if not self.daily_me_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.daily_me_history)
    
    def save_daily_me_data(self, symbol: str = 'PORTFOLIO', data_dir: str = 'data/daily', save_full_history: bool = False):
        """
        Save daily M/E data to the daily data directory
        """
        import os
        
        # Ensure directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Create filename for portfolio M/E data
        filename = os.path.join(data_dir, f"{symbol}_ME.csv")
        
        if save_full_history:
            # Save full history for backtests
            history_df = self.get_me_history_df()
            if not history_df.empty:
                # Convert Date to datetime if needed and sort/dedup
                history_df['Date'] = pd.to_datetime(history_df['Date'])
                history_df = history_df.sort_values('Date').drop_duplicates('Date', keep='last')
                history_df.to_csv(filename, index=False)
                return filename
            else:
                # If no history, fall back to saving current
                pass
        
        # Get current M/E metrics (default daily behavior)
        current_metrics = self.calculate_daily_me_ratio()
        
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

# Global M/E calculator instance
_me_calculator = None

def get_me_calculator(initial_value: float = 100000) -> DailyMERatioCalculator:
    """Get or create the global M/E calculator instance"""
    global _me_calculator
    if _me_calculator is None:
        _me_calculator = DailyMERatioCalculator(initial_value)
    return _me_calculator

def update_me_ratio_for_trade(symbol: str, shares: int, entry_price: float, 
                             current_price: float, trade_type: str):
    """Update M/E ratio tracking when a trade is made"""
    calculator = get_me_calculator()
    calculator.update_position(symbol, shares, entry_price, current_price, trade_type)
    return calculator.calculate_daily_me_ratio()

def add_realized_profit(profit: float):
    """Add realized profit from a closed trade"""
    calculator = get_me_calculator()
    calculator.add_realized_pnl(profit)

def get_current_me_ratio() -> float:
    """Get current M/E ratio"""
    calculator = get_me_calculator()
    metrics = calculator.calculate_daily_me_ratio()
    return metrics['ME_Ratio']

def get_current_risk_assessment() -> Dict:
    """Get current risk assessment"""
    calculator = get_me_calculator()
    return calculator.get_risk_assessment()

if __name__ == "__main__":
    # Test the M/E calculator
    print("Testing Daily M/E Ratio Calculator")
    
    calculator = DailyMERatioCalculator(100000)
    
    # Simulate some positions
    calculator.update_position('AAPL', 100, 150.0, 155.0, 'long')
    calculator.update_position('MSFT', -50, 300.0, 295.0, 'short')
    calculator.update_position('GOOGL', 80, 120.0, 125.0, 'long')
    
    # Calculate M/E ratio
    metrics = calculator.calculate_daily_me_ratio()
    
    print("Current M/E Ratio Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Get risk assessment
    risk = calculator.get_risk_assessment()
    print(f"\nRisk Assessment: {risk['risk_level']} - {risk['recommendation']}")