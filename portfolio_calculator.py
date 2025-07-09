"""
Portfolio Performance Calculator
Calculates real portfolio metrics from actual trade history
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

def calculate_real_portfolio_metrics(initial_portfolio_value: float = 100000) -> Dict[str, Any]:
    """
    Calculate actual portfolio performance from trade history
    Returns real metrics instead of placeholder values
    """
    
    try:
        # Import the trade history function
        from data_manager import get_trades_history
        trades_df = get_trades_history()
        
        if trades_df.empty:
            # Return initial values if no trades
            return get_placeholder_metrics(initial_portfolio_value)
        
        # Ensure required columns exist
        required_cols = ['profit', 'exit_date', 'entry_date']
        if not all(col in trades_df.columns for col in required_cols):
            return get_placeholder_metrics(initial_portfolio_value)
        
        # Convert dates
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        
        # Calculate cumulative performance
        trades_sorted = trades_df.sort_values('exit_date')
        trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
        
        # Current portfolio metrics
        total_profit = trades_sorted['profit'].sum()
        current_portfolio_value = initial_portfolio_value + total_profit
        total_return_pct = (total_profit / initial_portfolio_value) * 100
        
        # Daily P&L (trades closed today)
        today = datetime.now().date()
        todays_trades = trades_sorted[trades_sorted['exit_date'].dt.date == today]
        daily_pnl = todays_trades['profit'].sum() if not todays_trades.empty else 0.0
        
        # Monthly P&L (this month)
        current_month = datetime.now().replace(day=1).date()
        monthly_trades = trades_sorted[trades_sorted['exit_date'].dt.date >= current_month]
        mtd_profit = monthly_trades['profit'].sum() if not monthly_trades.empty else 0.0
        mtd_return = (mtd_profit / initial_portfolio_value) * 100
        
        # YTD P&L (this year)
        current_year = datetime.now().replace(month=1, day=1).date()
        ytd_trades = trades_sorted[trades_sorted['exit_date'].dt.date >= current_year]
        ytd_profit = ytd_trades['profit'].sum() if not ytd_trades.empty else 0.0
        ytd_return = (ytd_profit / initial_portfolio_value) * 100
        
        # Position exposure calculations
        long_exposure, short_exposure, total_open_equity = calculate_position_exposure()
        
        # M/E Ratio (Margin to Equity = Total Open Trade Equity / Account Size)
        me_ratio = f"{(total_open_equity / current_portfolio_value):.2f}" if current_portfolio_value > 0 and total_open_equity > 0 else "0.00"
        
        # Performance deltas (vs previous period)
        mtd_delta = calculate_mtd_delta(trades_sorted, initial_portfolio_value)
        ytd_delta = calculate_ytd_delta(trades_sorted, initial_portfolio_value)
        
        return {
            'total_value': f"${current_portfolio_value:,.0f}",
            'total_return_pct': f"{total_return_pct:+.1f}%",
            'daily_pnl': f"${daily_pnl:+,.2f}",
            'me_ratio': me_ratio,
            'mtd_return': f"{mtd_return:+.1f}%",
            'mtd_delta': mtd_delta,
            'ytd_return': f"{ytd_return:+.1f}%", 
            'ytd_delta': ytd_delta,
            # Additional metrics for debugging
            'total_trades': len(trades_df),
            'total_profit_raw': total_profit,
            'winning_trades': len(trades_df[trades_df['profit'] > 0]),
            'losing_trades': len(trades_df[trades_df['profit'] <= 0]),
            'long_exposure_raw': long_exposure,
            'short_exposure_raw': short_exposure,
            'total_open_equity': total_open_equity
        }
        
    except Exception as e:
        print(f"Error calculating portfolio metrics: {e}")
        return get_placeholder_metrics(initial_portfolio_value)

def calculate_position_exposure():
    """
    Calculate current position exposure from open positions
    Returns: (long_exposure, short_exposure, total_open_equity)
    """
    try:
        from data_manager import get_positions_df
        positions_df = get_positions_df()
        
        print(f"🔍 Positions Debug: Shape={positions_df.shape}")
        if not positions_df.empty:
            print(f"🔍 Positions Columns: {list(positions_df.columns)}")
            print(f"🔍 First row: {positions_df.iloc[0].to_dict()}")
        
        if positions_df.empty:
            print("⚠️ No positions data - returning zeros")
            return 0.0, 0.0, 0.0
        
        # Ensure required columns exist and are numeric
        if 'current_value' not in positions_df.columns:
            # Calculate current_value from shares and current_price if missing
            if 'shares' in positions_df.columns and 'current_price' in positions_df.columns:
                positions_df['shares'] = pd.to_numeric(positions_df['shares'], errors='coerce').fillna(0)
                positions_df['current_price'] = pd.to_numeric(positions_df['current_price'], errors='coerce').fillna(0)
                positions_df['current_value'] = positions_df['shares'] * positions_df['current_price']
                print(f"🔧 Calculated current_value from shares × price")
            else:
                print("❌ Missing required columns for position calculations")
                return 0.0, 0.0, 0.0
        
        # Make sure current_value is numeric
        positions_df['current_value'] = pd.to_numeric(positions_df['current_value'], errors='coerce').fillna(0)
        positions_df['shares'] = pd.to_numeric(positions_df['shares'], errors='coerce').fillna(0)
        
        # Calculate exposures based on position direction
        long_positions = positions_df[positions_df['shares'] > 0]
        short_positions = positions_df[positions_df['shares'] < 0]
        
        long_exposure = abs(long_positions['current_value'].sum()) if not long_positions.empty else 0.0
        short_exposure = abs(short_positions['current_value'].sum()) if not short_positions.empty else 0.0
        total_open_equity = long_exposure + short_exposure  # Total margin used
        
        print(f"🔍 Exposure Calc: Long=${long_exposure:,.0f}, Short=${short_exposure:,.0f}, Total=${total_open_equity:,.0f}")
        
        return long_exposure, short_exposure, total_open_equity
        
    except Exception as e:
        print(f"❌ Error calculating position exposure: {e}")
        return 0.0, 0.0, 0.0

def calculate_mtd_delta(trades_df: pd.DataFrame, initial_value: float) -> str:
    """Calculate month-to-date performance delta"""
    try:
        # Compare this month vs last month
        now = datetime.now()
        this_month_start = now.replace(day=1).date()
        last_month_start = (now.replace(day=1) - timedelta(days=1)).replace(day=1).date()
        
        # This month's performance
        this_month_trades = trades_df[trades_df['exit_date'].dt.date >= this_month_start]
        this_month_profit = this_month_trades['profit'].sum() if not this_month_trades.empty else 0.0
        
        # Last month's performance  
        last_month_trades = trades_df[
            (trades_df['exit_date'].dt.date >= last_month_start) & 
            (trades_df['exit_date'].dt.date < this_month_start)
        ]
        last_month_profit = last_month_trades['profit'].sum() if not last_month_trades.empty else 0.0
        
        # Calculate delta
        if last_month_profit != 0:
            delta_pct = ((this_month_profit - last_month_profit) / abs(last_month_profit)) * 100
            return f"{delta_pct:+.1f}%"
        else:
            return "+0.0%" if this_month_profit >= 0 else f"{this_month_profit/initial_value*100:.1f}%"
            
    except Exception:
        return "+0.0%"

def calculate_ytd_delta(trades_df: pd.DataFrame, initial_value: float) -> str:
    """Calculate year-to-date performance delta"""
    try:
        # Compare this year vs last year
        now = datetime.now()
        this_year_start = now.replace(month=1, day=1).date()
        last_year_start = this_year_start.replace(year=this_year_start.year - 1)
        
        # This year's performance
        this_year_trades = trades_df[trades_df['exit_date'].dt.date >= this_year_start]
        this_year_profit = this_year_trades['profit'].sum() if not this_year_trades.empty else 0.0
        
        # Last year's performance (same period)
        last_year_end = this_year_start.replace(year=this_year_start.year - 1, month=now.month, day=now.day)
        last_year_trades = trades_df[
            (trades_df['exit_date'].dt.date >= last_year_start) & 
            (trades_df['exit_date'].dt.date <= last_year_end)
        ]
        last_year_profit = last_year_trades['profit'].sum() if not last_year_trades.empty else 0.0
        
        # Calculate delta
        if last_year_profit != 0:
            delta_pct = ((this_year_profit - last_year_profit) / abs(last_year_profit)) * 100
            return f"{delta_pct:+.1f}%"
        else:
            return "+0.0%" if this_year_profit >= 0 else f"{this_year_profit/initial_value*100:.1f}%"
            
    except Exception:
        return "+0.0%"

def get_placeholder_metrics(initial_value: float) -> Dict[str, Any]:
    """Return placeholder metrics when no trade data is available"""
    return {
        'total_value': f"${initial_value:,.0f}",
        'total_return_pct': "+0.0%",
        'daily_pnl': "$0.00",
        'me_ratio': "0.00",
        'mtd_return': "+0.0%",
        'mtd_delta': "+0.0%",
        'ytd_return': "+0.0%",
        'ytd_delta': "+0.0%",
        'total_trades': 0,
        'total_profit_raw': 0.0,
        'winning_trades': 0,
        'losing_trades': 0,
        'long_exposure_raw': 0.0,
        'short_exposure_raw': 0.0,
        'total_open_equity': 0.0
    }

def get_enhanced_strategy_performance(initial_portfolio_value: float = 100000) -> pd.DataFrame:
    """
    Calculate strategy performance with real metrics
    """
    try:
        from data_manager import get_trades_history
        trades_df = get_trades_history()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Group trades by strategy if strategy column exists
        if 'strategy' in trades_df.columns:
            strategy_stats = []
            
            for strategy in trades_df['strategy'].unique():
                strategy_trades = trades_df[trades_df['strategy'] == strategy]
                
                total_profit = strategy_trades['profit'].sum()
                total_trades = len(strategy_trades)
                winning_trades = len(strategy_trades[strategy_trades['profit'] > 0])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                avg_profit = total_profit / total_trades if total_trades > 0 else 0
                
                strategy_stats.append({
                    'Strategy': strategy,
                    'Total Trades': total_trades,
                    'Winning Trades': winning_trades,
                    'Win Rate': f"{win_rate:.1f}%",
                    'Total Profit': f"${total_profit:,.2f}",
                    'Avg Profit/Trade': f"${avg_profit:,.2f}",
                    'Return %': f"{(total_profit/initial_portfolio_value)*100:+.1f}%"
                })
            
            return pd.DataFrame(strategy_stats)
        else:
            # Single strategy summary
            total_profit = trades_df['profit'].sum()
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            return pd.DataFrame([{
                'Strategy': 'nGS Trading System',
                'Total Trades': total_trades,
                'Winning Trades': winning_trades, 
                'Win Rate': f"{win_rate:.1f}%",
                'Total Profit': f"${total_profit:,.2f}",
                'Avg Profit/Trade': f"${avg_profit:,.2f}",
                'Return %': f"{(total_profit/initial_portfolio_value)*100:+.1f}%"
            }])
            
    except Exception as e:
        print(f"Error calculating strategy performance: {e}")
        return pd.DataFrame()

def patch_portfolio_metrics():
    """
    Monkey patch the get_portfolio_metrics function to use real calculations
    """
    import data_manager
    data_manager.get_portfolio_metrics = calculate_real_portfolio_metrics
    data_manager.get_strategy_performance = get_enhanced_strategy_performance