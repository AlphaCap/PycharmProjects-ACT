import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Union
import sys

from ngs_ai_integration_manager import NGSAIIntegrationManager
from ngs_ai_performance_comparator import NGSAIPerformanceComparator
from data_manager import (
    save_trades, save_positions, load_price_data,
    save_signals, get_positions, initialize as init_data_manager,
    RETENTION_DAYS,
    # Import sector management functions
    get_sector_symbols, 
    get_symbol_sector,
    get_portfolio_sector_exposure,
    calculate_sector_rebalance_needs,
    get_all_sectors,
    get_sector_weights,
    get_positions_df
)

# FIXED: Corrected M/E Ratio Calculator with proper price updates
class DailyMERatioCalculator:
    def __init__(self, initial_portfolio_value: float = 1000000):
        self.initial_portfolio_value = initial_portfolio_value
        self.current_positions = {}  # symbol -> position_data
        self.realized_pnl = 0.0
        self.daily_me_history = []
        
    def update_position(self, symbol: str, shares: int, entry_price: float, 
                       current_price: float, trade_type: str = 'long'):
        """
        FIXED: Update position with current market price
        """
        if shares == 0:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        else:
            # FIXED: Always store absolute position value (market value)
            position_market_value = abs(shares) * current_price
            unrealized_pnl = self._calculate_unrealized_pnl(shares, entry_price, current_price, trade_type)
            
            self.current_positions[symbol] = {
                'shares': shares,
                'entry_price': entry_price,
                'current_price': current_price,  # CRITICAL: Always current market price
                'type': trade_type,
                'position_value': position_market_value,  # Market value (always positive)
                'unrealized_pnl': unrealized_pnl
            }
    
    def _calculate_unrealized_pnl(self, shares: int, entry_price: float, 
                                current_price: float, trade_type: str) -> float:
        if trade_type.lower() == 'long':
            return (current_price - entry_price) * shares
        else:  # short
            return (entry_price - current_price) * abs(shares)
    
    def update_all_positions_with_current_prices(self, current_prices: dict):
        """
        CRITICAL FIX: Update all positions with current market prices
        This should be called regularly to ensure accurate M/E ratios
        """
        for symbol in list(self.current_positions.keys()):
            if symbol in current_prices:
                position = self.current_positions[symbol]
                new_price = current_prices[symbol]
                
                # Update with current market price
                self.update_position(
                    symbol=symbol,
                    shares=position['shares'],
                    entry_price=position['entry_price'],
                    current_price=new_price,
                    trade_type=position['type']
                )
    
    def add_realized_pnl(self, profit: float):
        self.realized_pnl += profit
    
    def calculate_daily_me_ratio(self, date: str = None, current_prices: dict = None) -> Dict:
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # CRITICAL: Update positions with current prices first
        if current_prices:
            self.update_all_positions_with_current_prices(current_prices)
        
        # Calculate position values (sum of all market values)
        total_position_value = 0.0
        long_value = 0.0
        short_value = 0.0
        total_unrealized_pnl = 0.0
        
        for symbol, pos in self.current_positions.items():
            # Position value is always market value (positive)
            position_market_value = pos['position_value']
            total_position_value += position_market_value
            
            # Separate long/short for reporting
            if pos['type'].lower() == 'long' and pos['shares'] > 0:
                long_value += position_market_value
            elif pos['type'].lower() == 'short' and pos['shares'] < 0:
                short_value += position_market_value
            
            # Unrealized P&L (can be positive or negative)
            total_unrealized_pnl += pos['unrealized_pnl']
        
        # FIXED: Portfolio equity calculation
        # Total equity = Starting capital + All P&L (realized + unrealized)
        portfolio_equity = self.initial_portfolio_value + self.realized_pnl + total_unrealized_pnl
        
        # FIXED: M/E ratio = Total position market value / Portfolio equity
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
            existing_df = pd.read_csv(filename)
            
            # Ensure Date column is datetime
            existing_df['Date'] = pd.to_datetime(existing_df['Date'], errors='coerce')
            
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
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set encoding for Windows console to handle Unicode
#if sys.platform == 'win32':
    #import io
    #sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    #sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class NGSStrategy:
    """
    Neural Grid Strategy (nGS) implementation with Active M/E Rebalancing.
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
        
        # Sector Management Configuration (DISABLED - using M/E control instead)
        self.sector_allocation_enabled = False  # Disabled: M/E ratio controls position management
        self.sector_targets = {}  # Not used
        self.max_sector_weight = 1.0  # No sector limits - natural allocation
        self.min_sector_weight = 0.0  # No sector limits - natural allocation
        self.sector_rebalance_threshold = 1.0  # Effectively disabled
        
        # L/S Ratio and M/E Rebalancing Configuration - ENHANCED VERIFICATION
        self.me_rebalancing_enabled = True  # CRITICAL: Enable M/E band rebalancing
        self.me_target_min = 50.0  # Minimum M/E ratio (50%)
        self.me_target_max = 80.0  # Maximum M/E ratio (80%)
        self.min_positions_for_scaling_up = 5  # Minimum positions required for upward scaling
        self.ls_ratio_enabled = True  # Enable L/S ratio adjustments for shorts
        
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
        
        # ENHANCED M/E VERIFICATION LOGGING
        print(f"\n{'='*70}")
        print("M/E REBALANCING SYSTEM STATUS")
        print(f"{'='*70}")
        print(f"M/E Rebalancing:      {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}")
        print(f"M/E Target Range:     {self.me_target_min:.1f}% - {self.me_target_max:.1f}%")
        print(f"Min Positions Scale:  {self.min_positions_for_scaling_up}")
        print(f"L/S Ratio Adjust:     {'ENABLED' if self.ls_ratio_enabled else 'DISABLED'}")
        print(f"Sector Management:    {'ENABLED' if self.sector_allocation_enabled else 'DISABLED'}")
        print(f"{'='*70}")
        
        logger.info(f"nGS Strategy initialized with {self.retention_days}-day data retention")
        logger.info(f"Data cutoff date: {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"M/E rebalancing: {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}")
        logger.info(f"M/E target range: {self.me_target_min}-{self.me_target_max}%")
        logger.info(f"Min positions for scale-up: {self.min_positions_for_scaling_up}")
        logger.info(f"L/S ratio adjustments: {'ENABLED' if self.ls_ratio_enabled else 'DISABLED'}")
        logger.info(f"Note: Sector management DISABLED - using M/E ratio control instead")

    # FIXED: Added method to update M/E calculator with current prices
    def update_me_calculator_with_current_prices(self, current_prices: dict):
        """
        CRITICAL FIX: Update M/E calculator with current market prices
        Call this method regularly to ensure accurate M/E calculations
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices and position['shares'] != 0:
                current_price = current_prices[symbol]
                trade_type = 'long' if position['shares'] > 0 else 'short'
                
                self.me_calculator.update_position(
                    symbol=symbol,
                    shares=position['shares'],
                    entry_price=position['entry_price'],
                    current_price=current_price,
                    trade_type=trade_type
                )

    # --- SECTOR MANAGEMENT METHODS ---
    
    def enable_sector_rebalancing(self, custom_targets: Dict[str, float] = None):
        """Enable sector-based rebalancing with optional custom targets"""
        self.sector_allocation_enabled = True
        
        if custom_targets:
            self.sector_targets = custom_targets
        else:
            # Use S&P 500 sector weights as baseline
            self.sector_targets = get_sector_weights()
        
        logger.info("Sector rebalancing enabled with targets:")
        for sector, weight in self.sector_targets.items():
            logger.info(f"  {sector}: {weight:.1%}")
    
    def disable_sector_rebalancing(self):
        """Disable sector-based rebalancing"""
        self.sector_allocation_enabled = False
        self.sector_targets = {}
        logger.info("Sector rebalancing disabled")
    
    def get_rebalance_candidates(self, positions_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get symbols that need rebalancing by sector
        Returns: {"buy": [symbols], "sell": [symbols], "hold": [symbols]}
        """
        if not self.sector_allocation_enabled:
            return {"buy": [], "sell": [], "hold": []}
        
        rebalance_needs = calculate_sector_rebalance_needs(positions_df, self.sector_targets)
        
        buy_sectors = [sector for sector, data in rebalance_needs.items() 
                      if data["action"] == "buy" and abs(data["difference"]) > self.sector_rebalance_threshold]
        sell_sectors = [sector for sector, data in rebalance_needs.items() 
                       if data["action"] == "sell" and abs(data["difference"]) > self.sector_rebalance_threshold]
        
        buy_candidates = []
        sell_candidates = []
        
        # Get symbols for underweight sectors (buy candidates)
        for sector in buy_sectors:
            sector_symbols = get_sector_symbols(sector)
            # Filter to symbols with good technical signals
            buy_candidates.extend(self.filter_buy_candidates(sector_symbols))
        
        # Get symbols for overweight sectors (sell candidates)  
        current_positions = positions_df['symbol'].tolist() if not positions_df.empty else []
        for sector in sell_sectors:
            sector_positions = [symbol for symbol in current_positions 
                              if get_symbol_sector(symbol) == sector]
            sell_candidates.extend(self.filter_sell_candidates(sector_positions, positions_df))
        
        return {
            "buy": buy_candidates,
            "sell": sell_candidates, 
            "hold": [symbol for symbol in current_positions 
                    if symbol not in buy_candidates + sell_candidates]
        }
    
    def filter_buy_candidates(self, sector_symbols: List[str]) -> List[str]:
        """Filter sector symbols to best buy candidates based on technical analysis"""
        candidates = []
        
        for symbol in sector_symbols:
            # Apply your existing technical filters
            if self.meets_buy_criteria(symbol):
                candidates.append(symbol)
        
        # Sort by signal strength/ranking and return top candidates
        return sorted(candidates, key=lambda x: self.get_signal_strength(x), reverse=True)[:5]
    
    def filter_sell_candidates(self, sector_positions: List[str], positions_df: pd.DataFrame) -> List[str]:
        """Filter sector positions to best sell candidates"""
        candidates = []
        
        for symbol in sector_positions:
            # Check if position should be closed (loss limits, profit targets, etc.)
            if self.meets_sell_criteria(symbol, positions_df):
                candidates.append(symbol)
        
        return candidates
    
    def meets_buy_criteria(self, symbol: str) -> bool:
        """Check if symbol meets technical buy criteria"""
        try:
            df = load_price_data(symbol)
            if df.empty or len(df) < 2:
                return False
            
            # Apply simplified buy criteria - price range check
            current_price = df['Close'].iloc[-1]
            return (self.inputs['MinPrice'] <= current_price <= self.inputs['MaxPrice'])
            
        except Exception as e:
            logger.debug(f"Error checking buy criteria for {symbol}: {e}")
            return False
    
    def meets_sell_criteria(self, symbol: str, positions_df: pd.DataFrame) -> bool:
        """Check if position meets sell criteria"""
        try:
            if positions_df.empty:
                return False
            
            position = positions_df[positions_df['symbol'] == symbol]
            if position.empty:
                return False
            
            # Simple sell criteria - profit target or stop loss
            profit_pct = position['profit_pct'].iloc[0]
            days_held = position['days_held'].iloc[0]
            
            # Sell if profit > 10% or loss > 5% or held > 30 days
            return (profit_pct > 10 or profit_pct < -5 or days_held > 30)
            
        except Exception as e:
            logger.debug(f"Error checking sell criteria for {symbol}: {e}")
            return False
    
    def get_signal_strength(self, symbol: str) -> float:
        """Get signal strength score for ranking (0-100)"""
        try:
            df = load_price_data(symbol)
            if df.empty or len(df) < 5:
                return 0.0
            
            # Simple signal strength based on recent price action
            recent_returns = df['Close'].pct_change().tail(5)
            momentum = recent_returns.mean() * 100
            volatility = recent_returns.std() * 100
            
            # Higher momentum, lower volatility = higher signal strength
            return max(0, momentum - volatility)
            
        except Exception as e:
            logger.debug(f"Error calculating signal strength for {symbol}: {e}")
            return 0.0
    
    def check_sector_limits(self, symbol: str, proposed_position_value: float, 
                          current_portfolio_value: float) -> bool:
        """
        Check if adding this position would violate sector concentration limits
        """
        if not self.sector_allocation_enabled:
            return True
        
        sector = get_symbol_sector(symbol)
        if sector == "Unknown":
            logger.warning(f"Unknown sector for {symbol} - allowing position")
            return True  # Allow unknown sectors for now
        
        # Get current sector exposure
        positions_df = get_positions_df()  # From data_manager
        current_exposure = get_portfolio_sector_exposure(positions_df)
        
        current_sector_value = current_exposure.get(sector, {}).get("value", 0)
        new_sector_value = current_sector_value + abs(proposed_position_value)
        new_sector_weight = new_sector_value / current_portfolio_value if current_portfolio_value > 0 else 0
        
        if new_sector_weight > self.max_sector_weight:
            logger.warning(f"Sector limit exceeded: {sector} would be {new_sector_weight:.1%} > {self.max_sector_weight:.1%}")
            return False
        
        logger.debug(f"Sector check passed: {sector} would be {new_sector_weight:.1%}")
        return True
    
    def generate_sector_report(self) -> Dict:
        """Generate sector allocation report for dashboard"""
        positions_df = get_positions_df()
        current_exposure = get_portfolio_sector_exposure(positions_df)
        
        if self.sector_allocation_enabled:
            rebalance_needs = calculate_sector_rebalance_needs(positions_df, self.sector_targets)
        else:
            rebalance_needs = {}
        
        return {
            "current_exposure": current_exposure,
            "target_weights": self.sector_targets if self.sector_allocation_enabled else {},
            "rebalance_needs": rebalance_needs,
            "sector_allocation_enabled": self.sector_allocation_enabled,
            "max_sector_weight": self.max_sector_weight,
            "min_sector_weight": self.min_sector_weight,
            "rebalance_threshold": self.sector_rebalance_threshold
        }

    # --- L/S RATIO AND M/E REBALANCING METHODS - ENHANCED VERIFICATION ---
    
    def calculate_ls_ratio(self) -> Optional[float]:
        """
        Calculate Long/Short ratio for risk assessment
        Returns: L/S ratio (positive if net long, negative if net short, None if all long or all short)
        """
        try:
            long_count = len([pos for pos in self.positions.values() if pos['shares'] > 0])
            short_count = len([pos for pos in self.positions.values() if pos['shares'] < 0])
            
            # Only calculate if we have both longs and shorts
            if long_count > 0 and short_count > 0:
                if long_count >= short_count:  # Net long
                    ls_ratio = long_count / short_count
                    logger.debug(f"L/S Ratio (net long): {ls_ratio:.2f} ({long_count}L/{short_count}S)")
                    return ls_ratio
                else:  # Net short
                    ls_ratio = -(short_count / long_count)
                    logger.debug(f"L/S Ratio (net short): {ls_ratio:.2f} ({long_count}L/{short_count}S)")
                    return ls_ratio
            else:
                logger.debug(f"L/S Ratio: N/A (only {long_count}L/{short_count}S)")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating L/S ratio: {e}")
            return None
    
    def get_short_position_size(self, base_position_size: float) -> float:
        """
        Adjust short position size based on L/S ratio
        Args:
            base_position_size: Base position size (e.g., $5000)
        Returns:
            Adjusted position size based on risk assessment
        """
        if not self.ls_ratio_enabled:
            return base_position_size
        
        ls_ratio = self.calculate_ls_ratio()
        
        if ls_ratio is None:
            # No L/S adjustment possible (all long or all short)
            return base_position_size
        
        try:
            if ls_ratio > 1.5:
                # Lower risk: Portfolio heavily net long, safer to add shorts
                # Use 50% short margin -> $10,000 position size
                adjusted_size = base_position_size * 2.0  # 50% margin = 2x leverage
                logger.debug(f"L/S {ls_ratio:.2f} > 1.5: Lower risk, 50% margin -> ${adjusted_size:,.0f}")
                return adjusted_size
            elif ls_ratio > -1.5:
                # Higher risk: Portfolio balanced to moderately net short
                # Use 75% short margin -> $3,750 position size
                adjusted_size = base_position_size * 0.75  # 75% margin = reduced size
                logger.debug(f"L/S {ls_ratio:.2f} > -1.5: Higher risk, 75% margin -> ${adjusted_size:,.0f}")
                return adjusted_size
            else:
                # Very high risk: Heavily net short
                # Use standard size with high margin requirement
                logger.debug(f"L/S {ls_ratio:.2f} <= -1.5: Very high risk, standard size")
                return base_position_size
                
        except Exception as e:
            logger.error(f"Error adjusting short position size: {e}")
            return base_position_size
    
    def check_me_rebalancing_needed(self) -> Optional[str]:
        """
        Check if M/E ratio rebalancing is needed - ENHANCED WITH DEBUGGING
        Returns: 'scale_up', 'scale_down', or None
        """
        if not self.me_rebalancing_enabled:
            print(f"M/E REBALANCING DISABLED - No action taken")
            return None
        
        try:
            current_me = self.calculate_current_me_ratio()
            position_count = len([pos for pos in self.positions.values() if pos['shares'] != 0])
            
            # ENHANCED DEBUG OUTPUT
            print(f"\nM/E REBALANCING CHECK:")
            print(f"   Current M/E:        {current_me:.1f}%")
            print(f"   Target Range:       {self.me_target_min:.1f}% - {self.me_target_max:.1f}%")
            print(f"   Position Count:     {position_count}")
            print(f"   Min for Scale-Up:   {self.min_positions_for_scaling_up}")
            
            if current_me < self.me_target_min:
                # Below minimum - need to scale up
                if position_count >= self.min_positions_for_scaling_up:
                    print(f"SCALE UP NEEDED: {current_me:.1f}% < {self.me_target_min}% with {position_count} positions")
                    logger.info(f"M/E {current_me:.1f}% < {self.me_target_min}% with {position_count} positions: SCALE UP needed")
                    return 'scale_up'
                else:
                    print(f"SCALE UP BLOCKED: Only {position_count} positions (need min {self.min_positions_for_scaling_up})")
                    logger.info(f"M/E {current_me:.1f}% < {self.me_target_min}% but only {position_count} positions (min {self.min_positions_for_scaling_up}): No scaling")
                    return None
            elif current_me > self.me_target_max:
                # Above maximum - need to scale down
                print(f"SCALE DOWN NEEDED: {current_me:.1f}% > {self.me_target_max}%")
                logger.info(f"M/E {current_me:.1f}% > {self.me_target_max}%: SCALE DOWN needed")
                return 'scale_down'
            else:
                # Within target range
                print(f"M/E IN TARGET RANGE: {current_me:.1f}% within {self.me_target_min:.1f}%-{self.me_target_max:.1f}%")
                logger.debug(f"M/E {current_me:.1f}% within target range {self.me_target_min}-{self.me_target_max}%")
                return None
                
        except Exception as e:
            logger.error(f"Error checking M/E rebalancing: {e}")
            return None
    
    def perform_me_rebalancing(self, action: str) -> bool:
        """
        Perform M/E ratio rebalancing by scaling positions - FIXED OVERSHOOT ISSUE
        Args:
            action: 'scale_up' or 'scale_down'
        Returns:
            True if rebalancing was performed, False otherwise
        """
        if not self.me_rebalancing_enabled or not action:
            print(f"M/E REBALANCING SKIPPED: {'Disabled' if not self.me_rebalancing_enabled else 'No action specified'}")
            return False
        
        try:
            current_me = self.calculate_current_me_ratio()
            positions_to_rebalance = {symbol: pos for symbol, pos in self.positions.items() if pos['shares'] != 0}
            
            if not positions_to_rebalance:
                print(f"NO POSITIONS TO REBALANCE")
                logger.warning("No positions to rebalance")
                return False
            
            # CRITICAL FIX: More precise scaling calculation to stay within bounds
            if action == 'scale_up':
                # Scale up to reach the MIDDLE of target range, not minimum
                target_me = (self.me_target_min + self.me_target_max) / 2  # 65% instead of 50%
                target_scaling = target_me / current_me if current_me > 0 else 1.3
                # Cap scaling more conservatively
                target_scaling = min(target_scaling, 1.8)  # Max 1.8x instead of 2x
                
            elif action == 'scale_down':
                # Scale down to reach the MIDDLE of target range, not maximum  
                target_me = (self.me_target_min + self.me_target_max) / 2  # 65% instead of 80%
                target_scaling = target_me / current_me if current_me > 0 else 0.75
                # Ensure minimum scaling
                target_scaling = max(target_scaling, 0.6)  # Minimum 60% scaling
                
            else:
                logger.error(f"Invalid rebalancing action: {action}")
                return False
            
            # ENHANCED REBALANCING OUTPUT
            print(f"\nPERFORMING M/E REBALANCING:")
            print(f"   Action:             {action.upper()}")
            print(f"   Target M/E:         {target_me:.1f}% (middle of range)")
            print(f"   Scaling Factor:     {target_scaling:.3f}x")
            print(f"   Before M/E:         {current_me:.1f}%")
            print(f"   Positions:          {len(positions_to_rebalance)}")
            
            logger.info(f"M/E Rebalancing: {action} targeting {target_me:.1f}% with {target_scaling:.2f}x scaling factor")
            logger.info(f"Before: M/E {current_me:.1f}%, {len(positions_to_rebalance)} positions")
            
            # Apply proportional scaling to all positions
            rebalanced_positions = []
            total_value_change = 0
            
            for symbol, position in positions_to_rebalance.items():
                old_shares = position['shares']
                new_shares = int(round(old_shares * target_scaling))
                
                # Ensure we don't go to zero shares (minimum 1 share)
                if old_shares > 0:
                    new_shares = max(1, new_shares)
                elif old_shares < 0:
                    new_shares = min(-1, new_shares)
                
                shares_change = new_shares - old_shares
                value_change = shares_change * position['entry_price']
                
                # Update position
                self.positions[symbol]['shares'] = new_shares
                
                # Update M/E calculator
                current_price = position.get('current_price', position['entry_price'])
                trade_type = 'long' if new_shares > 0 else 'short'
                self.me_calculator.update_position(symbol, new_shares, position['entry_price'], current_price, trade_type)
                
                # Track for cash adjustment
                total_value_change += value_change
                
                rebalanced_positions.append({
                    'symbol': symbol,
                    'old_shares': old_shares,
                    'new_shares': new_shares,
                    'shares_change': shares_change,
                    'value_change': value_change
                })
                
                print(f"   {symbol}: {old_shares:+4d} -> {new_shares:+4d} shares ({shares_change:+3d}) | ${value_change:+8,.0f}")
                logger.debug(f"  {symbol}: {old_shares} -> {new_shares} shares ({shares_change:+d}), value change: ${value_change:+,.2f}")
            
            # Adjust cash for the position changes
            self.cash = round(float(self.cash - total_value_change), 2)
            
            # Verify the rebalancing worked
            new_me = self.calculate_current_me_ratio()
            
            print(f"   After M/E:          {new_me:.1f}%")
            print(f"   Cash Change:        ${total_value_change:+,.0f}")
            
            # CRITICAL: Check if we're still within bounds
            if self.me_target_min <= new_me <= self.me_target_max:
                print(f"M/E REBALANCING COMPLETED: {len(rebalanced_positions)} positions scaled")
                print(f"TARGET ACHIEVED: {new_me:.1f}% is within {self.me_target_min:.1f}%-{self.me_target_max:.1f}% range")
            else:
                print(f"M/E REBALANCING WARNING: {new_me:.1f}% outside target range")
                if new_me > self.me_target_max:
                    print(f"   Still above {self.me_target_max:.1f}% limit - may need further adjustment")
                elif new_me < self.me_target_min:
                    print(f"   Still below {self.me_target_min:.1f}% limit - may need further adjustment")
            
            logger.info(f"After: M/E {new_me:.1f}%, cash change: ${total_value_change:+,.2f}")
            logger.info(f"M/E Rebalancing completed: {len(rebalanced_positions)} positions scaled")
            
            # Record the rebalancing as a system event
            self.record_historical_me_ratio(datetime.now().strftime('%Y-%m-%d'), trade_occurred=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing M/E rebalancing: {e}")
            return False
    
    def end_of_day_rebalancing(self) -> None:
        """
        Perform end-of-day M/E rebalancing check and execution - ENHANCED WITH DEBUGGING
        Call this at the end of each trading day
        """
        try:
            print(f"\n{'='*70}")
            print("END OF DAY M/E REBALANCING CHECK")
            print(f"{'='*70}")
            print(f"M/E Rebalancing Status: {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}")
            
            if not self.me_rebalancing_enabled:
                print(f"M/E REBALANCING DISABLED - Skipping EOD check")
                logger.info("M/E rebalancing disabled - skipping EOD check")
                return
            
            logger.info("=== END OF DAY M/E REBALANCING CHECK ===")
            
            # Check if rebalancing is needed
            rebalancing_action = self.check_me_rebalancing_needed()
            
            if rebalancing_action:
                success = self.perform_me_rebalancing(rebalancing_action)
                if success:
                    print(f"M/E REBALANCING COMPLETED: {rebalancing_action.upper()}")
                    logger.info(f"M/E rebalancing completed: {rebalancing_action}")
                else:
                    print(f"M/E REBALANCING FAILED: {rebalancing_action.upper()}")
                    logger.warning(f"M/E rebalancing failed: {rebalancing_action}")
            else:
                print(f"NO REBALANCING NEEDED")
                logger.info("M/E ratio within target range - no rebalancing needed")
            
            # Generate final status report
            final_me = self.calculate_current_me_ratio()
            position_count = len([pos for pos in self.positions.values() if pos['shares'] != 0])
            ls_ratio = self.calculate_ls_ratio()
            
            print(f"\nEOD STATUS:")
            print(f"   Final M/E:          {final_me:.1f}%")
            print(f"   Positions:          {position_count}")
            print(f"   L/S Ratio:          {ls_ratio if ls_ratio else 'N/A'}")
            print(f"{'='*70}")
            
            logger.info(f"EOD Status: M/E {final_me:.1f}%, {position_count} positions, L/S {ls_ratio if ls_ratio else 'N/A'}")
            logger.info("=== END OF DAY REBALANCING COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Error in end-of-day rebalancing: {e}")

    # --- EXISTING M/E RATIO METHODS (PRESERVED) ---
    
    def calculate_current_me_ratio(self) -> float:
        return self.me_calculator.calculate_daily_me_ratio()['ME_Ratio']

    def calculate_historical_me_ratio(self, current_prices: Dict[str, float] = None) -> float:
        return self.me_calculator.calculate_daily_me_ratio(current_prices=current_prices)['ME_Ratio']

    def record_historical_me_ratio(self, date_str: str, trade_occurred: bool = False, current_prices: Dict[str, float] = None):
        self.me_calculator.calculate_daily_me_ratio(date_str, current_prices=current_prices)

    # --- DATA FILTERING AND INDICATOR CALCULATIONS (PRESERVED) ---
    
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

    # --- SIGNAL GENERATION (PRESERVED WITH SECTOR LIMITS) ---
    
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
            # Apply L/S ratio adjustment for short position sizing
            base_shares = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
            adjusted_position_size = self.get_short_position_size(self.inputs['PositionSize'])
            df.loc[df.index[i], 'Shares'] = int(round(adjusted_position_size / df['Close'].iloc[i]))
        
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
            # Apply L/S ratio adjustment for short position sizing
            base_shares = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
            adjusted_position_size = self.get_short_position_size(self.inputs['PositionSize'])
            df.loc[df.index[i], 'Shares'] = int(round(adjusted_position_size / df['Close'].iloc[i]))
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
            # Apply L/S ratio adjustment for short position sizing
            base_shares = int(round(self.inputs['PositionSize'] / df['Close'].iloc[i]))
            adjusted_position_size = self.get_short_position_size(self.inputs['PositionSize'])
            df.loc[df.index[i], 'Shares'] = int(round(adjusted_position_size / df['Close'].iloc[i]))

    # --- EXIT SIGNAL LOGIC (PRESERVED) ---
    
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

    # --- POSITION MANAGEMENT (ENHANCED WITH SECTOR CHECKS) ---
    
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
            
            # Note: Sector limits disabled - using M/E ratio control instead
            # Position entry controlled by M/E rebalancing, not sector limits
            
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
            
            # Log sector information
            sector = get_symbol_sector(symbol)
            logger.info(f"Entry {symbol} ({sector}): {df['SignalType'].iloc[i]} with {shares} shares at {df['Close'].iloc[i]}")
            
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

    # --- POSITION LOADING AND MANAGEMENT (PRESERVED) ---
    
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
            
            # CRITICAL FIX: Update M/E calculator with current price after processing
            if not result_df.empty and symbol in self.positions:
                current_price = result_df['Close'].iloc[-1]
                if self.positions[symbol]['shares'] != 0:
                    trade_type = 'long' if self.positions[symbol]['shares'] > 0 else 'short'
                    self.me_calculator.update_position(
                        symbol=symbol,
                        shares=self.positions[symbol]['shares'],
                        entry_price=self.positions[symbol]['entry_price'],
                        current_price=current_price,
                        trade_type=trade_type
                    )
            
            return result_df
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    # --- MAIN RUN METHOD (ENHANCED WITH M/E VERIFICATION) ---
    
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Initialize for this run
        self.trades = []
        self.me_calculator = DailyMERatioCalculator(self.account_size)  # Reset M/E tracking
        
        # Filter trades by retention period
        self._filter_trades_by_retention()
        
        # ENHANCED M/E STATUS VERIFICATION
        print(f"\n{'='*70}")
        print("STARTING STRATEGY RUN - M/E VERIFICATION")
        print(f"{'='*70}")
        print(f"M/E Rebalancing:      {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}")
        print(f"M/E Target Range:     {self.me_target_min:.1f}% - {self.me_target_max:.1f}%")
        print(f"Min Positions Scale:  {self.min_positions_for_scaling_up}")
        print(f"Initial Positions:    {len(self.positions)}")
        
        # Calculate initial M/E ratio for any existing positions
        if self.positions:
            initial_me_ratio = self.calculate_current_me_ratio()
            print(f"Initial M/E:          {initial_me_ratio:.2f}%")
            logger.info(f"Initial M/E ratio with {len(self.positions)} existing positions: {initial_me_ratio:.2f}%")
        else:
            print(f"Initial M/E:          0.00% (no positions)")
        
        print(f"{'='*70}")
        
        results = {}
        for i, (symbol, df) in enumerate(data.items()):
            result = self.process_symbol(symbol, df)
            if result is not None and not result.empty:
                results[symbol] = result
                if (i + 1) % 50 == 0:  # Progress update every 50 symbols
                    current_me = self.calculate_current_me_ratio()
                    print(f"Progress: {i+1}/{len(data)} symbols | Current M/E: {current_me:.1f}%")
                logger.info(f"Processed {symbol}: {len(result)} rows ({i+1}/{len(data)})")
        
        # CRITICAL FIX: Update all positions with final prices at end of run
        final_prices = {}
        for symbol, df in results.items():
            if not df.empty:
                final_prices[symbol] = df['Close'].iloc[-1]
        
        # Update M/E calculator with final prices
        self.update_me_calculator_with_current_prices(final_prices)
        
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
        
        # CRITICAL: Perform end-of-day M/E rebalancing
        print(f"\nCALLING END-OF-DAY M/E REBALANCING...")
        self.end_of_day_rebalancing()
        
        # Note: Sector reporting disabled - using M/E ratio control instead
        # if self.sector_allocation_enabled:
        #     sector_report = self.generate_sector_report()
        #     self._display_sector_summary(sector_report)
        
        # FINAL M/E STATUS VERIFICATION - FIXED CALCULATION
        final_me_metrics = self.me_calculator.calculate_daily_me_ratio(
            date=datetime.now().strftime('%Y-%m-%d'),
            current_prices=final_prices
        )
        final_me = final_me_metrics['ME_Ratio']
        
        print(f"\n{'='*70}")
        print("FINAL M/E STATUS")
        print(f"{'='*70}")
        print(f"Final M/E Ratio:      {final_me:.2f}%")
        print(f"Target Range:         {self.me_target_min:.1f}% - {self.me_target_max:.1f}%")
        print(f"Final Positions:      {len(all_positions)}")
        print(f"Rebalancing System:   {'ACTIVE' if self.me_rebalancing_enabled else 'INACTIVE'}")
        
        # FIXED: Show M/E calculation details with correct formula
        print(f"\nM/E Calculation Details:")
        print(f"Portfolio Equity:       ${final_me_metrics['Portfolio_Equity']:,.2f}")
        print(f"Total Position Value:   ${final_me_metrics['Total_Position_Value']:,.2f}")
        print(f"Long Value:             ${final_me_metrics['Long_Value']:,.2f}")
        print(f"Short Value:            ${final_me_metrics['Short_Value']:,.2f}")
        print(f"Unrealized P&L:         ${final_me_metrics['Unrealized_PnL']:,.2f}")
        print(f"Realized P&L:           ${final_me_metrics['Realized_PnL']:,.2f}")
        print(f"Starting Account:       ${self.account_size:,.2f}")
        
        # Final M/E status
        risk = self.me_calculator.get_risk_assessment()
        print(f"M/E Risk Status:        {risk['risk_level']}")
        print(f"M/E Active Positions:   {len(self.me_calculator.current_positions)}")
        print(f"{'='*70}")
        
        logger.info(f"Strategy run complete. Processed {len(data)} symbols, currently have {len(all_positions)} positions")
        logger.info(f"Data retention: {self.retention_days} days, cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"M/E rebalancing: {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}")
        logger.info(f"Final M/E ratio: {final_me:.2f}%")
        logger.info(f"L/S ratio adjustments: {'ENABLED' if self.ls_ratio_enabled else 'DISABLED'}")
        logger.info(f"Note: Sector management DISABLED - using M/E ratio control instead")
        
        return results

    def _display_sector_summary(self, sector_report: Dict):
        """Display sector allocation summary"""
        print(f"\n{'='*60}")
        print("SECTOR ALLOCATION SUMMARY")
        print(f"{'='*60}")
        
        current_exposure = sector_report['current_exposure']
        target_weights = sector_report['target_weights']
        rebalance_needs = sector_report['rebalance_needs']
        
        if current_exposure:
            print(f"\nCurrent Sector Exposure:")
            for sector, data in current_exposure.items():
                target_weight = target_weights.get(sector, 0) * 100
                current_weight = data['weight'] * 100
                print(f"  {sector:22s}: {current_weight:5.1f}% (target: {target_weight:5.1f}%) - {data['count']} positions")
        
        if rebalance_needs:
            needs_rebalancing = [sector for sector, data in rebalance_needs.items() 
                               if abs(data['difference']) > self.sector_rebalance_threshold]
            if needs_rebalancing:
                print(f"\nSectors needing rebalancing (>{self.sector_rebalance_threshold:.0%} threshold):")
                for sector in needs_rebalancing:
                    data = rebalance_needs[sector]
                    print(f"  {sector:22s}: {data['action']:4s} {data['difference']:+5.1%} "
                          f"(${data['dollar_adjustment']:+,.0f})")
        
        print(f"\nSector Limits: Max {self.max_sector_weight:.0%}, Min {self.min_sector_weight:.0%}")
        print(f"{'='*60}")

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

# --- DATA LOADING FUNCTION (PRESERVED) ---

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

def run_ngs_automated_reporting(comparison=None):
    from ngs_ai_backtesting_system import NGSAIBacktestingSystem
    import pandas as pd
    import os
    import json

    # 1. Load your universe
    symbols = []
    sp500_file = os.path.join('data', 'sp500_symbols.txt')
    if os.path.exists(sp500_file):
        with open(sp500_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        symbols = ["AAPL", "MSFT", "GOOGL"]

    # Load the price data
    data = load_polygon_data(symbols)

    # Now run the strategy
    # Create an instance of NGSStrategy first
    strategy = NGSStrategy(account_size=1_000_000)
    # Then run the strategy on the data
    results = strategy.run(data)
    # Save trades to CSV for dashboard
    trade_history_path = 'data/trade_history.csv'
    if os.path.exists(trade_history_path):
        prior_trades = pd.read_csv(trade_history_path)
    else:
        prior_trades = pd.DataFrame()
    new_trades_df = pd.DataFrame([{
        'symbol': trade['symbol'],
        'entry_date': trade['entry_date'],
        'exit_date': trade['exit_date'],
        'entry_price': trade['entry_price'],
        'exit_price': trade['exit_price'],
        'profit_loss': trade['profit']
    } for trade in strategy.trades])
    all_trades_df = pd.concat([prior_trades, new_trades_df], ignore_index=True)
    all_trades_df = all_trades_df.drop_duplicates(subset=['symbol', 'entry_date', 'exit_date'])
    all_trades_df.to_csv(trade_history_path, index=False)

    # 6. Save summary stats for dashboard if comparison is provided
    if comparison is not None:
        with open('data/summary_stats.json', 'w') as f:
            json.dump(comparison.summary_stats, f, indent=2)
        print("✅ Trades and summary stats exported for Streamlit dashboard.")
    else:
        print("✅ Trades exported for Streamlit dashboard (no summary stats).")

if __name__ == "__main__":
    print("🚀 nGS Trading Strategy with AI SELECTION ENABLED")
    print("=" * 70)
    print(f"Data Retention: {RETENTION_DAYS} days (6 months)")
    print("=" * 70)
    
    try:
        # STEP 1: Initialize AI systems
        print("\n🧠 Initializing AI Strategy Selection System...")
        
        AI_AVAILABLE = True
        print("✅ AI modules imported successfully")
            
            # Initialize AI systems
            ai_integration_manager = NGSAIIntegrationManager(
                account_size=1000000,
                data_dir='data'
            )
            
            performance_comparator = NGSAIPerformanceComparator(
                account_size=1000000,
                data_dir='data'
            )
            print("✅ AI systems initialized")
            
        except Exception as e:
            print(f"⚠️  AI initialization failed: {e}")
            print("Falling back to original nGS strategy...")
            AI_AVAILABLE = False
        
        # STEP 2: Load data (same as before)
        sp500_file = os.path.join('data', 'sp500_symbols.txt')
        try:
            with open(sp500_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            print(f"📊 Loaded {len(symbols)} S&P 500 symbols")
        except FileNotFoundError:
            print(f"⚠️  {sp500_file} not found. Using sample symbols.")
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA',
                'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'PFE'
            ]
        
        print(f"🔄 Loading market data for {len(symbols)} symbols...")
        data = load_polygon_data(symbols)
        
        if not data:
            print("❌ No data loaded - check your data files")
            exit(1)
        
        print(f"✅ Successfully loaded data for {len(data)} symbols")
        
        # STEP 3: AI Strategy Selection or Fallback
        if AI_AVAILABLE:
            # AI-POWERED EXECUTION
            print(f"\n🔍 AI ANALYZING STRATEGY OPTIONS...")
            print("This will compare your original nGS vs AI-optimized variants")
            
            # Define AI objectives to test
            ai_objectives = ['linear_equity', 'max_roi', 'min_drawdown', 'high_winrate']
            
            print(f"🧪 Testing {len(ai_objectives)} AI strategy objectives:")
            for obj in ai_objectives:
                print(f"   • {obj}")
            
            try:
                # Run comprehensive comparison
                print(f"\n📊 Running comprehensive performance analysis...")
                comparison_results = performance_comparator.comprehensive_comparison(
                    data=data,
                    objectives=ai_objectives
                )
                
                # AI makes recommendation
                ai_score = comparison_results.ai_recommendation_score
                best_strategy = comparison_results.best_overall_strategy
                recommended_allocation = comparison_results.recommended_allocation
                
                print(f"\n🎯 AI STRATEGY SELECTION RESULTS")
                print("=" * 50)
                print(f"AI Recommendation Score: {ai_score:.0f}/100")
                print(f"Best Overall Strategy:   {best_strategy}")
                print(f"Statistical Significance: {'YES' if comparison_results.return_difference_significant else 'NO'}")
                
                # Show performance comparison
                original_performance = comparison_results.original_metrics
                best_ai_performance = max(comparison_results.ai_metrics, key=lambda x: x.total_return_pct)
                
                print(f"\n📈 PERFORMANCE COMPARISON:")
                print(f"Original nGS:     {original_performance.total_return_pct:+7.2f}% return, {original_performance.max_drawdown_pct:7.2f}% drawdown")
                print(f"Best AI Strategy: {best_ai_performance.total_return_pct:+7.2f}% return, {best_ai_performance.max_drawdown_pct:7.2f}% drawdown")
                
                # Set operating mode based on AI recommendation
                print(f"\n🤖 AI DECISION:")
                
                if ai_score >= 70:
                    print("✅ AI RECOMMENDS: AI-Focused Strategy")
                    print(f"   Reason: AI shows significant improvements (score: {ai_score:.0f}/100)")
                    ai_integration_manager.set_operating_mode('ai_only')
                elif ai_score >= 40:
                    print("✅ AI RECOMMENDS: Hybrid Strategy")
                    print(f"   Reason: Balanced approach optimal (score: {ai_score:.0f}/100)")
                    ai_integration_manager.set_operating_mode('hybrid', {
                        'ai_allocation_pct': 60.0,
                        'original_allocation_pct': 40.0
                    })
                else:
                    print("✅ AI RECOMMENDS: Original nGS Strategy")
                    print(f"   Reason: Original strategy remains superior (score: {ai_score:.0f}/100)")
                    ai_integration_manager.set_operating_mode('original')
                
                print(f"\n💰 RECOMMENDED ALLOCATION:")
                for strategy_name, allocation_pct in recommended_allocation.items():
                    print(f"   {strategy_name}: {allocation_pct:.1f}%")
                
                # Execute AI-selected strategy
                print(f"\n🚀 Executing AI-selected strategy...")
                results = ai_integration_manager.run_integrated_strategy(data)
                
                # Show AI results
                print(f"\n📊 AI EXECUTION COMPLETED!")
                print(f"Mode: {results['mode'].upper()}")
                print(f"AI Recommendation Score: {ai_score:.0f}/100")
                
                if results.get('integration_summary', {}).get('recommendations'):
                    print(f"AI Recommendations:")
                    for rec in results['integration_summary']['recommendations']:
                        print(f"   • {rec}")
                
                print(f"✅ AI-powered strategy execution completed!")
                
            except Exception as e:
                print(f"❌ AI analysis failed: {e}")
                print("Falling back to original nGS strategy...")
                AI_AVAILABLE = False
        
        if not AI_AVAILABLE:
            # FALLBACK: Original nGS execution
            print(f"\n📊 Running Original nGS Strategy...")
            strategy = NGSStrategy(account_size=1000000)
            results = strategy.run(data)
            
            # Show original results
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
            
            if strategy.trades:
                print(f"Winning trades:       {winning_trades}/{len(strategy.trades)} ({winning_trades/len(strategy.trades)*100:.1f}%)")
                
            print(f"✅ Original nGS strategy execution completed!")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
