# Update these functions in your data_manager.py

def calculate_historical_me_ratio(trades_df: pd.DataFrame, initial_value: float = 100000) -> float:
    """
    Calculate historical M/E ratio as rolling average of daily position values.
    M/E = (Total Position Value / Portfolio Equity) * 100
    
    For a proper calculation, we need to simulate daily positions based on trades.
    """
    if trades_df.empty:
        return 0.0
    
    try:
        # Convert dates
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Get date range
        start_date = trades_df['entry_date'].min()
        end_date = trades_df['exit_date'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Calculate daily M/E ratios
        daily_me_ratios = []
        cumulative_profit = 0
        
        for current_date in date_range:
            # Find all positions open on this date
            open_positions = trades_df[
                (trades_df['entry_date'] <= current_date) & 
                (trades_df['exit_date'] >= current_date)
            ]
            
            if not open_positions.empty:
                # Calculate total position value (assuming entry price * shares)
                # For both long and short positions
                total_position_value = (open_positions['entry_price'] * open_positions['shares'].abs()).sum()
                
                # Calculate portfolio equity up to this date
                closed_trades = trades_df[trades_df['exit_date'] < current_date]
                cumulative_profit = closed_trades['profit'].sum() if not closed_trades.empty else 0
                portfolio_equity = initial_value + cumulative_profit
                
                # Calculate M/E ratio
                # Using 2:1 leverage assumption for margin accounts
                # This would give values over 100% when fully invested
                me_ratio = (total_position_value / portfolio_equity) * 100
                daily_me_ratios.append(me_ratio)
        
        # Return average M/E ratio
        if daily_me_ratios:
            avg_me = np.mean(daily_me_ratios)
            # Typical margin account with full investment would be 100-200%
            return avg_me
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Error calculating historical M/E ratio: {e}")
        # Return a reasonable default for margin trading
        return 150.0  # Assume average 1.5x leverage

def calculate_ytd_return(trades_df: pd.DataFrame, initial_value: float) -> tuple:
    """Calculate Year-to-Date return from closed trades"""
    if trades_df.empty:
        return "$0", "0.00%"
    
    # Get current year trades
    current_year = datetime.now().year
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Filter for current year trades
    ytd_trades = trades_df[trades_df['exit_date'].dt.year == current_year]
    ytd_profit = ytd_trades['profit'].sum() if not ytd_trades.empty else 0
    
    # Calculate percentage
    ytd_pct = (ytd_profit / initial_value * 100) if initial_value > 0 else 0
    
    return format_dollars(ytd_profit), f"{ytd_pct:.2f}%"

def calculate_mtd_return(trades_df: pd.DataFrame, initial_value: float) -> tuple:
    """Calculate Month-to-Date return from closed trades"""
    if trades_df.empty:
        return "$0", "0.00%"
    
    # Get current month trades
    current_date = datetime.now()
    current_month_start = datetime(current_date.year, current_date.month, 1)
    
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Filter for current month trades
    mtd_trades = trades_df[trades_df['exit_date'] >= current_month_start]
    mtd_profit = mtd_trades['profit'].sum() if not mtd_trades.empty else 0
    
    # Calculate percentage
    mtd_pct = (mtd_profit / initial_value * 100) if initial_value > 0 else 0
    
    return format_dollars(mtd_profit), f"{mtd_pct:.2f}%"

# Update the get_portfolio_metrics function
def get_portfolio_metrics(initial_portfolio_value: float = 100000, is_historical: bool = False) -> Dict:
    """
    Calculate portfolio metrics for long/short system.
    
    Args:
        initial_portfolio_value: Starting portfolio value
        is_historical: True for historical page, False for current trading page
        
    Returns:
        Dictionary with portfolio metrics including M/E ratio
    """
    try:
        # Get data
        positions_df = get_positions_df()
        trades_df = get_trades_history()
        
        # Calculate from historical trades
        total_trade_profit = trades_df['profit'].sum() if not trades_df.empty else 0
        
        # Current portfolio value from closed trades
        current_portfolio_value = initial_portfolio_value + total_trade_profit
        
        # Calculate exposures from open positions
        if not positions_df.empty:
            long_positions = positions_df[positions_df['shares'] > 0]
            short_positions = positions_df[positions_df['shares'] < 0]
            
            # Calculate position values
            long_value = (long_positions['current_price'] * long_positions['shares']).sum() if not long_positions.empty else 0
            short_value = (short_positions['current_price'] * short_positions['shares'].abs()).sum() if not short_positions.empty else 0
            
            # Total position value (both long and short)
            total_position_value = long_value + short_value
            
            # Current M/E Ratio (for live trading page)
            # This represents current leverage/margin usage
            current_me_ratio = (total_position_value / current_portfolio_value * 100) if current_portfolio_value > 0 else 0
            
            # Net exposure
            net_exposure = long_value - short_value
            
            # Daily P&L (unrealized from current positions)
            daily_pnl = positions_df['profit'].sum()
        else:
            # No open positions
            long_value = 0
            short_value = 0
            net_exposure = 0
            current_me_ratio = 0
            daily_pnl = 0
        
        # Historical M/E Ratio (rolling average for historical page)
        historical_me_ratio = calculate_historical_me_ratio(trades_df, initial_portfolio_value) if not trades_df.empty else 0
        
        # Returns based on closed trades
        total_return = total_trade_profit
        total_return_pct = f"{(total_return / initial_portfolio_value * 100):.2f}%" if initial_portfolio_value > 0 else "0.00%"
        
        # Calculate proper MTD and YTD
        mtd_return, mtd_pct = calculate_mtd_return(trades_df, initial_portfolio_value)
        ytd_return, ytd_pct = calculate_ytd_return(trades_df, initial_portfolio_value)
        
        # Format all dollar amounts without cents
        metrics = {
            'total_value': format_dollars(current_portfolio_value),
            'total_return_pct': total_return_pct,
            'daily_pnl': format_dollars(daily_pnl),
            'mtd_return': mtd_return,
            'mtd_delta': mtd_pct,
            'ytd_return': ytd_return,
            'ytd_delta': ytd_pct,
            'me_ratio': f"{current_me_ratio:.1f}%",
            'historical_me_ratio': f"{historical_me_ratio:.1f}%",
            'long_exposure': format_dollars(long_value),
            'short_exposure': format_dollars(short_value),
            'net_exposure': format_dollars(net_exposure)
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {
            'total_value': format_dollars(initial_portfolio_value),
            'total_return_pct': "0.00%",
            'daily_pnl': "$0",
            'mtd_return': "$0",
            'mtd_delta': "0.00%",
            'ytd_return': "$0",
            'ytd_delta': "0.00%",
            'me_ratio': "0.0%",
            'historical_me_ratio': "150.0%",  # Default for margin account
            'long_exposure': "$0",
            'short_exposure': "$0",
            'net_exposure': "$0"
        }
