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
    Neural Grid Strategy (nGS) implementation with Sector Management, L/S Ratio VaR, and M/E Rebalancing.
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
        
        # Sector Management Configuration
        self.sector_allocation_enabled = False
        self.sector_targets = {}  # Will be populated from config or adaptive logic
        self.max_sector_weight = 0.35  # Max 35% in any single sector
        self.min_sector_weight = 0.02  # Min 2% in any single sector
        self.sector_rebalance_threshold = 0.02  # 2% deviation threshold
        
        # NEW: M/E Rebalancing Configuration
        self.me_target_min = 50.0  # 50% minimum M/E ratio
        self.me_target_max = 80.0  # 80% maximum M/E ratio
        self.min_positions_for_rebalance = 5  # Minimum positions to trigger upward rebalancing
        self.rebalancing_enabled = True  # Can be disabled for testing
        
        self.inputs = {
            'Length': 25,
            'NumDevs': 2,
            'MinPrice': 10,
            'MaxPrice': 500,
            'AfStep': 0.05,
            'AfLimit': 0.21,
            'PositionSize': 5000  # Base position size (will be adjusted by L/S ratio)
        }
        init_data_manager()
        self._load_positions()
        
        logger.info(f"nGS Strategy initialized with {self.retention_days}-day data retention")
        logger.info(f"Data cutoff date: {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"Sector management: {'ENABLED' if self.sector_allocation_enabled else 'DISABLED'}")
        logger.info(f"M/E Rebalancing: {'ENABLED' if self.rebalancing_enabled else 'DISABLED'} (Target: {self.me_target_min}-{self.me_target_max}%)")

    # --- NEW: L/S RATIO CALCULATION AND POSITION SIZING ---
    
    def calculate_ls_ratio(self) -> Optional[float]:
        """
        Calculate Long/Short ratio based on position counts.
        Returns:
            - Positive value if net long (longs/shorts)
            - Negative value if net short (-(shorts/longs))
            - None if only longs or only shorts (can't calculate ratio)
        """
        try:
            long_count = 0
            short_count = 0
            
            for symbol, pos in self.positions.items():
                if pos['shares'] > 0:
                    long_count += 1
                elif pos['shares'] < 0:
                    short_count += 1
            
            # Only calculate ratio if we have both longs and shorts
            if long_count > 0 and short_count > 0:
                if long_count >= short_count:
                    # Net long or equal: positive ratio
                    ls_ratio = long_count / short_count
                else:
                    # Net short: negative ratio
                    ls_ratio = -(short_count / long_count)
                
                logger.debug(f"L/S Ratio: {ls_ratio:.2f} (Longs: {long_count}, Shorts: {short_count})")
                return ls_ratio
            else:
                logger.debug(f"L/S Ratio: N/A (Longs: {long_count}, Shorts: {short_count})")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating L/S ratio: {e}")
            return None
    
    def get_adjusted_short_position_size(self, base_price: float) -> int:
        """
        Calculate adjusted position size for short entries based on L/S ratio.
        
        Args:
            base_price: Current price of the symbol
            
        Returns:
            Adjusted shares for short position
        """
        try:
            ls_ratio = self.calculate_ls_ratio()
            
            # If no L/S ratio available (all longs or all shorts), use base position size
            if ls_ratio is None:
                base_position_value = self.inputs['PositionSize']
                shares = int(round(base_position_value / base_price))
                logger.debug(f"No L/S ratio available, using base position size: {shares} shares")
                return shares
            
            # Apply L/S ratio-based margin adjustments for SHORT entries only
            if ls_ratio > 1.5:
                # Lower risk (heavily net long): 50% short margin
                # $10,000 position size
                position_value = 10000
                margin_type = "50% (Lower Risk)"
            elif ls_ratio > -1.5:
                # Higher risk (balanced to moderately net short): 75% short margin  
                # $3,750 position size
                position_value = 3750
                margin_type = "75% (Higher Risk)"
            else:
                # Very net short: use conservative sizing
                position_value = 2500
                margin_type = "Conservative (Very Net Short)"
            
            shares = int(round(position_value / base_price))
            logger.info(f"L/S Ratio: {ls_ratio:.2f} → Short margin: {margin_type} → {shares} shares (${position_value:,.0f})")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating adjusted short position size: {e}")
            # Fallback to base calculation
            base_position_value = self.inputs['PositionSize']
            return int(round(base_position_value / base_price))

    # --- NEW: M/E REBALANCING LOGIC ---
    
    def check_me_rebalancing_needed(self) -> Dict[str, Union[bool, float, str]]:
        """
        Check if M/E ratio rebalancing is needed.
        
        Returns:
            Dictionary with rebalancing status and details
        """
        try:
            current_me = self.calculate_current_me_ratio()
            total_positions = len(self.positions)
            
            rebalance_info = {
                'rebalance_needed': False,
                'current_me': current_me,
                'target_min': self.me_target_min,
                'target_max': self.me_target_max,
                'total_positions': total_positions,
                'action': 'none',
                'reason': 'M/E within target range'
            }
            
            if not self.rebalancing_enabled:
                rebalance_info['reason'] = 'Rebalancing disabled'
                return rebalance_info
            
            # Check if rebalancing needed
            if current_me < self.me_target_min:
                # Need to increase M/E (scale up positions)
                if total_positions >= self.min_positions_for_rebalance:
                    rebalance_info.update({
                        'rebalance_needed': True,
                        'action': 'scale_up',
                        'reason': f'M/E too low ({current_me:.1f}% < {self.me_target_min}%)'
                    })
                else:
                    rebalance_info['reason'] = f'M/E low but insufficient positions ({total_positions} < {self.min_positions_for_rebalance})'
                    
            elif current_me > self.me_target_max:
                # Need to decrease M/E (scale down positions)
                rebalance_info.update({
                    'rebalance_needed': True,
                    'action': 'scale_down',
                    'reason': f'M/E too high ({current_me:.1f}% > {self.me_target_max}%)'
                })
            
            return rebalance_info
            
        except Exception as e:
            logger.error(f"Error checking M/E rebalancing: {e}")
            return {
                'rebalance_needed': False,
                'current_me': 0.0,
                'action': 'error',
                'reason': f'Error: {e}'
            }
    
    def execute_me_rebalancing(self, action: str, current_me: float) -> bool:
        """
        Execute M/E ratio rebalancing by scaling all positions proportionally.
        
        Args:
            action: 'scale_up' or 'scale_down'
            current_me: Current M/E ratio
            
        Returns:
            True if rebalancing executed successfully
        """
        try:
            if not self.positions:
                logger.warning("No positions to rebalance")
                return False
            
            # Calculate scaling factor
            if action == 'scale_up':
                # Scale up to reach minimum M/E
                target_me = self.me_target_min
                scale_factor = target_me / current_me if current_me > 0 else 1.0
            elif action == 'scale_down':
                # Scale down to reach maximum M/E
                target_me = self.me_target_max
                scale_factor = target_me / current_me if current_me > 0 else 1.0
            else:
                logger.error(f"Invalid rebalancing action: {action}")
                return False
            
            logger.info(f"Executing M/E rebalancing: {action} with scale factor {scale_factor:.3f}")
            logger.info(f"Target: {current_me:.1f}% → {target_me:.1f}%")
            
            # Track rebalancing for reporting
            rebalanced_positions = []
            
            # Scale all positions proportionally
            for symbol, position in self.positions.items():
                try:
                    old_shares = position['shares']
                    
                    # Skip if no position
                    if old_shares == 0:
                        continue
                    
                    # Calculate new shares (maintain sign for long/short)
                    new_shares = int(round(old_shares * scale_factor))
                    
                    # Ensure we don't go to zero unless scale factor is very small
                    if new_shares == 0 and abs(old_shares) > 0:
                        new_shares = 1 if old_shares > 0 else -1
                    
                    # Update position
                    position['shares'] = new_shares
                    
                    # Update M/E calculator
                    trade_type = 'long' if new_shares > 0 else 'short'
                    current_price = position.get('current_price', position['entry_price'])
                    self.me_calculator.update_position(symbol, new_shares, position['entry_price'], current_price, trade_type)
                    
                    rebalanced_positions.append({
                        'symbol': symbol,
                        'old_shares': old_shares,
                        'new_shares': new_shares,
                        'change': new_shares - old_shares
                    })
                    
                    logger.debug(f"Rebalanced {symbol}: {old_shares} → {new_shares} shares")
                    
                except Exception as e:
                    logger.error(f"Error rebalancing position {symbol}: {e}")
            
            # Log rebalancing summary
            logger.info(f"M/E Rebalancing completed: {len(rebalanced_positions)} positions adjusted")
            
            # Show sample of changes
            if rebalanced_positions:
                logger.info("Sample position changes:")
                for pos in rebalanced_positions[:5]:
                    change_str = f"+{pos['change']}" if pos['change'] >= 0 else str(pos['change'])
                    logger.info(f"  {pos['symbol']}: {pos['old_shares']} → {pos['new_shares']} ({change_str})")
            
            # Verify new M/E ratio
            new_me = self.calculate_current_me_ratio()
            logger.info(f"M/E Rebalancing result: {current_me:.1f}% → {new_me:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing M/E rebalancing: {e}")
            return False
    
    def perform_eod_rebalancing(self):
        """
        Perform end-of-day M/E rebalancing check and execution.
        Called at the end of each trading day/run.
        """
        try:
            logger.info("Performing EOD M/E rebalancing check...")
            
            rebalance_info = self.check_me_rebalancing_needed()
            
            logger.info(f"M/E Status: {rebalance_info['current_me']:.1f}% "
                       f"(Target: {self.me_target_min}-{self.me_target_max}%)")
            logger.info(f"Positions: {rebalance_info['total_positions']}")
            logger.info(f"Status: {rebalance_info['reason']}")
            
            if rebalance_info['rebalance_needed']:
                logger.info(f"M/E Rebalancing triggered: {rebalance_info['action']}")
                success = self.execute_me_rebalancing(
                    rebalance_info['action'], 
                    rebalance_info['current_me']
                )
                
                if success:
                    logger.info("✅ M/E Rebalancing completed successfully")
                else:
                    logger.error("❌ M/E Rebalancing failed")
            else:
                logger.info("✅ No M/E rebalancing needed")
                
        except Exception as e:
            logger.error(f"Error in EOD rebalancing: {e}")

    # --- SECTOR MANAGEMENT METHODS (PRESERVED) ---
    
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

    # --- EXISTING M/E RATIO METHODS (PRESERVED) ---
    
    def calculate_current_me_ratio(self) -> float:
        return self.me_calculator.calculate_daily_me_ratio()['ME_Ratio']

    def calculate_historical_me_ratio(self, current_prices: Dict[str, float] = None) -> float:
        return self.me_calculator.calculate_daily_me_ratio()['ME_Ratio']  # Historical is now daily snapshot

    def record_historical_me_ratio(self, date_str: str, trade_occurred: bool = False, current_prices: Dict[str, float] = None):
        self.me_calculator.calculate_daily_me_ratio(date_str)  # Automatically appends to history

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

    # --- SIGNAL GENERATION (ENHANCED WITH L/S POSITION SIZING) ---
    
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
            # Use base position size for long entries
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
            # NEW: Use L/S ratio-adjusted position size for SHORT entries
            df.loc[df.index[i], 'Shares'] = self.get_adjusted_short_position_size(df['Close'].iloc[i])
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
            df.loc[df.index[i], 'Shares'] = self.get_adjusted_short_position_size(df['Close'].iloc[i])
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
            df.loc[df.index[i], 'Shares'] = self.get_adjusted_short_position_size(df['Close'].iloc[i])
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
            df.loc[df.index[i], 'Shares'] = self.get_adjusted_short_position_size(df['Close'].iloc[i])

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
                        # For reversal exits that become entries, use L/S adjusted sizing for shorts
                        if entry_sig == -1:  # Short entry
                            df.loc[df.index[i], 'Shares'] = self.get_adjusted_short_position_size(df['Close'].iloc[i]) * 2
                        else:  # Long entry
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
            
            # Check sector limits before entering position
            current_portfolio_value = self.cash  # Use cash as proxy for portfolio value
            if not self.check_sector_limits(symbol, abs(cost), current_portfolio_value):
                logger.warning(f"Position entry rejected for {symbol} due to sector limits")
                return
            
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
            
            # Log sector and L/S information
            sector = get_symbol_sector(symbol)
            ls_ratio = self.calculate_ls_ratio()
            ls_info = f"L/S: {ls_ratio:.2f}" if ls_ratio is not None else "L/S: N/A"
            logger.info(f"Entry {symbol} ({sector}) [{ls_info}]: {df['SignalType'].iloc[i]} with {shares} shares at {df['Close'].iloc[i]}")
            
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
            return result_df
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    # --- MAIN RUN METHOD (ENHANCED WITH L/S VaR AND M/E REBALANCING) ---
    
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # Initialize for this run
        self.trades = []
        self.me_calculator = DailyMERatioCalculator(self.account_size)  # Reset M/E tracking
        
        # Filter trades by retention period
        self._filter_trades_by_retention()
        
        # Calculate initial M/E ratio and L/S ratio for any existing positions
        if self.positions:
            initial_me_ratio = self.calculate_current_me_ratio()
            initial_ls_ratio = self.calculate_ls_ratio()
            logger.info(f"Initial M/E ratio with {len(self.positions)} existing positions: {initial_me_ratio:.2f}%")
            if initial_ls_ratio is not None:
                logger.info(f"Initial L/S ratio: {initial_ls_ratio:.2f}")
            else:
                logger.info("Initial L/S ratio: N/A (single-sided portfolio)")
        
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
        
        # NEW: Perform EOD M/E rebalancing
        self.perform_eod_rebalancing()
        
        # Generate and display sector report if enabled
        if self.sector_allocation_enabled:
            sector_report = self.generate_sector_report()
            self._display_sector_summary(sector_report)
        
        # Final status with enhanced reporting
        current_me = self.calculate_current_me_ratio()
        final_ls_ratio = self.calculate_ls_ratio()
        
        print(f"\n{'='*80}")
        print("FINAL PORTFOLIO STATUS")
        print(f"{'='*80}")
        print(f"M/E Ratio: {current_me:.2f}% (Target: {self.me_target_min}-{self.me_target_max}%)")
        
        if final_ls_ratio is not None:
            print(f"L/S Ratio: {final_ls_ratio:.2f}")
            if final_ls_ratio > 1.5:
                print(f"Portfolio Status: Heavily Net Long (Lower Risk for new shorts)")
            elif final_ls_ratio > -1.5:
                print(f"Portfolio Status: Balanced to Moderately Net Short (Higher Risk for new shorts)")
            else:
                print(f"Portfolio Status: Heavily Net Short (Conservative short sizing)")
        else:
            print(f"L/S Ratio: N/A (Single-sided portfolio)")
        
        # Debug: Show M/E calculation details
        total_equity = 0
        for symbol, position in self.positions.items():
            if position['shares'] != 0:
                equity = position['entry_price'] * abs(position['shares'])
                total_equity += equity
        
        print(f"\nM/E Calculation Details:")
        print(f"Total Open Trade Equity: ${total_equity:,.2f}")
        print(f"Account Value (Cash): ${self.cash:,.2f}")
        if self.cash > 0:
            print(f"Calculated M/E: {(total_equity/self.cash*100):.2f}%")
        
        # Risk assessment
        risk = self.me_calculator.get_risk_assessment()
        print(f"\nRisk Assessment:")
        print(f"M/E Risk Level: {risk['risk_level']}")
        print(f"M/E Realized P&L: ${self.me_calculator.realized_pnl:.2f}")
        print(f"Active Positions: {len(self.me_calculator.current_positions)}")
        
        print(f"{'='*80}")
        
        logger.info(f"Strategy run complete. Processed {len(data)} symbols, currently have {len(all_positions)} positions")
        logger.info(f"Data retention: {self.retention_days} days, cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}")
        logger.info(f"Sector management: {'ENABLED' if self.sector_allocation_enabled else 'DISABLED'}")
        logger.info(f"M/E Rebalancing: {'ENABLED' if self.rebalancing_enabled else 'DISABLED'}")
        
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

# --- MAIN EXECUTION (ENHANCED WITH L/S VaR AND M/E REBALANCING) ---

if __name__ == "__main__":
    print("nGS Trading Strategy - Neural Grid System with L/S VaR and M/E Rebalancing")
    print("=" * 80)
    print(f"Data Retention: {RETENTION_DAYS} days (6 months)")
    print("=" * 80)
    
    try:
        strategy = NGSStrategy(account_size=1000000)
        
        # Demo: Enable sector rebalancing (optional)
        print("\n🎯 SECTOR MANAGEMENT DEMO")
        print("Enabling sector-based rebalancing...")
        strategy.enable_sector_rebalancing()  # Use S&P 500 sector weights
        
        # Show configuration
        print(f"\n⚙️  CONFIGURATION")
        print(f"M/E Target Range: {strategy.me_target_min}% - {strategy.me_target_max}%")
        print(f"Min Positions for Rebalancing: {strategy.min_positions_for_rebalance}")
        print(f"L/S Ratio VaR: ENABLED")
        print(f"  - L/S > 1.5: 50% short margin ($10,000 positions)")
        print(f"  - L/S > -1.5: 75% short margin ($3,750 positions)")
        
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
        print(f"\nRunning enhanced nGS strategy on {len(data)} symbols...")
        print("Processing signals, L/S VaR position sizing, and M/E rebalancing...")
        
        results = strategy.run(data)
        
        # Results summary
        print(f"\n{'='*80}")
        print("ENHANCED STRATEGY BACKTEST RESULTS (Last 6 Months)")
        print(f"{'='*80}")
        
        total_profit = sum(trade['profit'] for trade in strategy.trades)
        winning_trades = sum(1 for trade in strategy.trades if trade['profit'] > 0)
        
        print(f"Starting capital:     ${strategy.account_size:,.2f}")
        print(f"Ending cash:          ${strategy.cash:,.2f}")
        print(f"Total P&L:            ${total_profit:,.2f}")
        print(f"Return:               {((strategy.cash - strategy.account_size) / strategy.account_size * 100):+.2f}%")
        print(f"Total trades:         {len(strategy.trades)}")
        print(f"Symbols processed:    {len(data)}")
        print(f"Data period:          {strategy.cutoff_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Features enabled:     L/S VaR, M/E Rebalancing, Sector Management")
        
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
            
            # L/S analysis
            long_trades = [t for t in strategy.trades if t['type'] == 'long']
            short_trades = [t for t in strategy.trades if t['type'] == 'short']
            
            print(f"\nL/S Trade Analysis:")
            print(f"Long trades:          {len(long_trades)} (Avg P&L: ${np.mean([t['profit'] for t in long_trades]):.2f})" if long_trades else "Long trades: 0")
            print(f"Short trades:         {len(short_trades)} (Avg P&L: ${np.mean([t['profit'] for t in short_trades]):.2f})" if short_trades else "Short trades: 0")
        
        # Current positions with L/S analysis
        long_pos, short_pos = strategy.get_current_positions()
        total_positions = len(long_pos) + len(short_pos)
        current_ls_ratio = strategy.calculate_ls_ratio()
        
        print(f"\nCurrent positions: {total_positions} total ({len(long_pos)} long, {len(short_pos)} short)")
        if current_ls_ratio is not None:
            print(f"Current L/S ratio: {current_ls_ratio:.2f}")
        
        if total_positions > 0:
            print(f"\nSample positions (first 10):")
            all_pos = long_pos[:5] + short_pos[:5]
            for pos in all_pos[:10]:
                side = "Long " if pos in long_pos else "Short"
                shares = pos['shares'] if pos in long_pos else pos['shares']
                sector = get_symbol_sector(pos['symbol'])
                print(f"  {side} {pos['symbol']:6s} ({sector:15s}): {shares:4d} shares @ ${pos['entry_price']:7.2f}")
        
        print(f"\n✅ Enhanced strategy backtest completed successfully!")
        print(f"✅ Processed all {len(data)} S&P 500 symbols")
        print(f"✅ Features: L/S VaR Position Sizing, M/E Rebalancing ({strategy.me_target_min}-{strategy.me_target_max}%), Sector Management")
        print(f"✅ Data retention enforced: {RETENTION_DAYS} days")
        
    except Exception as e:
        logger.error(f"Enhanced strategy backtest failed: {e}")
        import traceback
        traceback.print_exc()