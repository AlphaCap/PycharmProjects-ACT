import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Ensure the project root is added to PYTHONPATH
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Define the historical data path relative to the script's location
HISTORICAL_DATA_PATH = str(Path(PROJECT_ROOT, "signal_analysis.json"))

print("Looking for file at:", HISTORICAL_DATA_PATH)
if not os.path.exists(HISTORICAL_DATA_PATH):
    print("File does not exist! Check the path.")
    exit(1)
else:
    print("File exists! Proceeding to load data.")

from data_manager import (
    RETENTION_DAYS,
    calculate_sector_rebalance_needs,
    get_positions,
    get_positions_df,
    get_sector_symbols,
    get_sector_weights,
    get_symbol_sector,
)
from data_manager import initialize as init_data_manager
from data_manager import load_price_data, save_positions, save_trades
from shared_utils import load_polygon_data


class DailyMERatioCalculator:
    def __init__(self, initial_portfolio_value: float = 1000000) -> None:
        self.initial_portfolio_value: float = initial_portfolio_value
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.realized_pnl: float = 0.0
        self.daily_me_history: List[Dict[str, Any]] = []

    def update_position(
        self,
        symbol: str,
        shares: int,
        entry_price: float,
        current_price: float,
        trade_type: str = "long",
    ) -> None:
        if shares == 0:
            if symbol in self.current_positions:
                del self.current_positions[symbol]
        else:
            position_market_value = abs(shares) * current_price
            unrealized_pnl = self._calculate_unrealized_pnl(
                shares, entry_price, current_price, trade_type
            )
            self.current_positions[symbol] = {
                "shares": shares,
                "entry_price": entry_price,
                "current_price": current_price,
                "type": trade_type,
                "position_value": position_market_value,
                "unrealized_pnl": unrealized_pnl,
            }

    def _calculate_unrealized_pnl(
        self, shares: int, entry_price: float, current_price: float, trade_type: str
    ) -> float:
        if trade_type.lower() == "long":
            return (current_price - entry_price) * shares
        else:
            return (entry_price - current_price) * abs(shares)

    def update_all_positions_with_current_prices(
        self, current_prices: Dict[str, float]
    ) -> None:
        for symbol in list(self.current_positions.keys()):
            if symbol in current_prices:
                position = self.current_positions[symbol]
                new_price = current_prices[symbol]
                self.update_position(
                    symbol=symbol,
                    shares=position["shares"],
                    entry_price=position["entry_price"],
                    current_price=new_price,
                    trade_type=position["type"],
                )

    def add_realized_pnl(self, profit: float) -> None:
        self.realized_pnl += profit

    def calculate_daily_me_ratio(
        self,
        date: Optional[str] = None,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        if current_prices:
            self.update_all_positions_with_current_prices(current_prices)
        total_position_value: float = 0.0
        long_value: float = 0.0
        short_value: float = 0.0
        total_unrealized_pnl: float = 0.0

        for symbol, pos in self.current_positions.items():
            position_market_value = pos["position_value"]
            total_position_value += position_market_value
            if pos["type"].lower() == "long" and pos["shares"] > 0:
                long_value += position_market_value
            elif pos["type"].lower() == "short" and pos["shares"] < 0:
                short_value += position_market_value
            total_unrealized_pnl += pos["unrealized_pnl"]

        portfolio_equity = (
            self.initial_portfolio_value + self.realized_pnl + total_unrealized_pnl
        )
        me_ratio = (
            (total_position_value / portfolio_equity * 100)
            if portfolio_equity > 0
            else 0.0
        )

        daily_metrics: Dict[str, Any] = {
            "Date": date,
            "Portfolio_Equity": round(portfolio_equity, 2),
            "Long_Value": round(long_value, 2),
            "Short_Value": round(short_value, 2),
            "Total_Position_Value": round(total_position_value, 2),
            "ME_Ratio": round(me_ratio, 2),
            "Realized_PnL": round(self.realized_pnl, 2),
            "Unrealized_PnL": round(total_unrealized_pnl, 2),
            "Long_Positions": len(
                [
                    p
                    for p in self.current_positions.values()
                    if p["type"].lower() == "long" and p["shares"] > 0
                ]
            ),
            "Short_Positions": len(
                [
                    p
                    for p in self.current_positions.values()
                    if p["type"].lower() == "short" and p["shares"] < 0
                ]
            ),
        }

        self.daily_me_history.append(daily_metrics)
        return daily_metrics

    def get_me_history_df(self) -> pd.DataFrame:
        if not self.daily_me_history:
            return pd.DataFrame()
        return pd.DataFrame(self.daily_me_history)

    def save_daily_me_data(self, data_dir: str = "data/daily") -> str:
        import os

        os.makedirs(data_dir, exist_ok=True)
        current_metrics = self.calculate_daily_me_ratio()
        filename = os.path.join(data_dir, "portfolio_ME.csv")

        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            existing_df["Date"] = pd.to_datetime(existing_df["Date"], errors="coerce")
            today = datetime.now().strftime("%Y-%m-%d")
            existing_df = existing_df[
                existing_df["Date"].dt.strftime("%Y-%m-%d") != today
            ]
            new_row = pd.DataFrame([current_metrics])
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        else:
            updated_df = pd.DataFrame([current_metrics])
        updated_df.to_csv(filename, index=False)
        return filename


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class NGSStrategy:
    """
    Neural Grid Strategy (nGS) implementation with Active M/E Rebalancing.
    Handles both signal generation and position management with 6-month data retention.
    """

    def __init__(self, account_size: float = 1000000, data_dir: str = "data") -> None:
        self.account_size: float = round(float(account_size), 2)
        self.cash: float = round(float(account_size), 2)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._trades: List[Dict[str, Any]] = []
        self.data_dir: str = data_dir
        self.retention_days: int = RETENTION_DAYS
        self.cutoff_date: datetime = datetime.now() - timedelta(
            days=self.retention_days
        )

        self.me_calculator: DailyMERatioCalculator = DailyMERatioCalculator(
            initial_portfolio_value=account_size
        )
        self.sector_allocation_enabled: bool = False
        self.sector_targets: Dict[str, float] = {}
        self.max_sector_weight: float = 1.0
        self.min_sector_weight: float = 0.0
        self.sector_rebalance_threshold: float = 1.0

        self.me_rebalancing_enabled: bool = True
        self.me_target_min: float = 50.0
        self.me_target_max: float = 80.0
        self.min_positions_for_scaling_up: int = 5
        self.ls_ratio_enabled: bool = True

        self.inputs: Dict[str, Union[int, float]] = {
            "Length": 25,
            "NumDevs": 2,
            "MinPrice": 10,
            "MaxPrice": 500,
            "AfStep": 0.05,
            "AfLimit": 0.21,
            "PositionSize": 5000,
        }
        init_data_manager()
        self._load_positions()
        # ... (rest of your __init__ code/logging)

    # ... (Continue with further methods, fully type-annotated, in the next section) ...
    # FIXED: Added method to update M/E calculator with current prices
    def update_me_calculator_with_current_prices(
        self, current_prices: Dict[str, float]
    ) -> None:
        for symbol, position in self.positions.items():
            if symbol in current_prices and position["shares"] != 0:
                current_price = current_prices[symbol]
                trade_type = "long" if position["shares"] > 0 else "short"
                self.me_calculator.update_position(
                    symbol=symbol,
                    shares=position["shares"],
                    entry_price=position["entry_price"],
                    current_price=current_price,
                    trade_type=trade_type,
                )

    def enable_sector_rebalancing(
        self, custom_targets: Optional[Dict[str, float]] = None
    ) -> None:
        self.sector_allocation_enabled = True
        if custom_targets:
            self.sector_targets = custom_targets
        else:
            self.sector_targets = get_sector_weights()
        logger.info("Sector rebalancing enabled with targets:")
        for sector, weight in self.sector_targets.items():
            logger.info(f"  {sector}: {weight:.1%}")

    def disable_sector_rebalancing(self) -> None:
        self.sector_allocation_enabled = False
        self.sector_targets = {}
        logger.info("Sector rebalancing disabled")

    def get_rebalance_candidates(
        self, positions_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        if not self.sector_allocation_enabled:
            return {"buy": [], "sell": [], "hold": []}
        rebalance_needs = calculate_sector_rebalance_needs(
            positions_df, self.sector_targets
        )
        buy_sectors = [
            sector
            for sector, data in rebalance_needs.items()
            if data["action"] == "buy"
            and abs(data["difference"]) > self.sector_rebalance_threshold
        ]
        sell_sectors = [
            sector
            for sector, data in rebalance_needs.items()
            if data["action"] == "sell"
            and abs(data["difference"]) > self.sector_rebalance_threshold
        ]
        buy_candidates: List[str] = []
        sell_candidates: List[str] = []
        for sector in buy_sectors:
            sector_symbols = get_sector_symbols(sector)
            buy_candidates.extend(self.filter_buy_candidates(sector_symbols))
        current_positions = (
            positions_df["symbol"].tolist() if not positions_df.empty else []
        )
        for sector in sell_sectors:
            sector_positions = [
                symbol
                for symbol in current_positions
                if get_symbol_sector(symbol) == sector
            ]
            sell_candidates.extend(
                self.filter_sell_candidates(sector_positions, positions_df)
            )
        return {
            "buy": buy_candidates,
            "sell": sell_candidates,
            "hold": [
                symbol
                for symbol in current_positions
                if symbol not in buy_candidates + sell_candidates
            ],
        }

    def filter_buy_candidates(self, sector_symbols: List[str]) -> List[str]:
        candidates: List[str] = []
        for symbol in sector_symbols:
            if self.meets_buy_criteria(symbol):
                candidates.append(symbol)
        return sorted(
            candidates, key=lambda x: self.get_signal_strength(x), reverse=True
        )[:5]

    def filter_sell_candidates(
        self, sector_positions: List[str], positions_df: pd.DataFrame
    ) -> List[str]:
        candidates: List[str] = []
        for symbol in sector_positions:
            if self.meets_sell_criteria(symbol, positions_df):
                candidates.append(symbol)
        return candidates

    def meets_buy_criteria(self, symbol: str) -> bool:
        try:
            df = load_price_data(symbol)
            if df.empty or len(df) < 2:
                return False
            current_price = df["Close"].iloc[-1]
            return self.inputs["MinPrice"] <= current_price <= self.inputs["MaxPrice"]
        except Exception as e:
            logger.debug(f"Error checking buy criteria for {symbol}: {e}")
            return False

    def meets_sell_criteria(self, symbol: str, positions_df: pd.DataFrame) -> bool:
        try:
            if positions_df.empty:
                return False
            position = positions_df[positions_df["symbol"] == symbol]
            if position.empty:
                return False
            profit_pct = position["profit_pct"].iloc[0]
            days_held = position["days_held"].iloc[0]
            return profit_pct > 10 or profit_pct < -5 or days_held > 30
        except Exception as e:
            logger.debug(f"Error checking sell criteria for {symbol}: {e}")
            return False

    def get_signal_strength(self, symbol: str) -> float:
        try:
            df = load_price_data(symbol)
            if df.empty or len(df) < 5:
                return 0.0
            recent_returns = df["Close"].pct_change().tail(5)
            momentum = recent_returns.mean() * 100
            volatility = recent_returns.std() * 100
            return max(0, momentum - volatility)
        except Exception as e:
            logger.debug(f"Error calculating signal strength for {symbol}: {e}")
            return 0.0

    def check_sector_limits(
            self,
            symbol: str,
            proposed_position_value: float,
            current_portfolio_value: float,
    ) -> bool:
        if not self.sector_allocation_enabled:
            return True
        sector = get_symbol_sector(symbol)
        if sector == "Unknown":
            logger.warning(f"Unknown sector for {symbol} - allowing position")
            return True
        # Mock current exposure as an empty value
        current_sector_value = 0
        new_sector_value = current_sector_value + abs(proposed_position_value)
        new_sector_weight = (
            new_sector_value / current_portfolio_value
            if current_portfolio_value > 0
            else 0
        )
        if new_sector_weight > self.max_sector_weight:
            logger.warning(
                f"Sector limit exceeded: {sector} would be {new_sector_weight:.1%} > {self.max_sector_weight:.1%}"
            )
            return False
        logger.debug(f"Sector check passed: {sector} would be {new_sector_weight:.1%}")
        return True

    def generate_sector_report(self) -> Dict[str, Any]:
        positions_df = get_positions_df()

        # Mock current_exposure if needed
        current_exposure = {}

        if self.sector_allocation_enabled:
            rebalance_needs = calculate_sector_rebalance_needs(
                positions_df, self.sector_targets
            )
        else:
            rebalance_needs = {}

        return {
            "current_exposure": current_exposure,  # Mocked placeholder
            "target_weights": (
                self.sector_targets if self.sector_allocation_enabled else {}
            ),
            "rebalance_needs": rebalance_needs,
            "sector_allocation_enabled": self.sector_allocation_enabled,
            "max_sector_weight": self.max_sector_weight,
            "min_sector_weight": self.min_sector_weight,
            "rebalance_threshold": self.sector_rebalance_threshold,
        }
    def calculate_ls_ratio(self) -> Optional[float]:
        try:
            long_count = len(
                [pos for pos in self.positions.values() if pos["shares"] > 0]
            )
            short_count = len(
                [pos for pos in self.positions.values() if pos["shares"] < 0]
            )
            if long_count > 0 and short_count > 0:
                if long_count >= short_count:
                    ls_ratio = long_count / short_count
                    logger.debug(
                        f"L/S Ratio (net long): {ls_ratio:.2f} ({long_count}L/{short_count}S)"
                    )
                    return ls_ratio
                else:
                    ls_ratio = -(short_count / long_count)
                    logger.debug(
                        f"L/S Ratio (net short): {ls_ratio:.2f} ({long_count}L/{short_count}S)"
                    )
                    return ls_ratio
            else:
                logger.debug(f"L/S Ratio: N/A (only {long_count}L/{short_count}S)")
                return None
        except Exception as e:
            logger.error(f"Error calculating L/S ratio: {e}")
            return None

    def get_short_position_size(self, base_position_size: float) -> float:
        if not self.ls_ratio_enabled:
            return base_position_size
        ls_ratio = self.calculate_ls_ratio()
        if ls_ratio is None:
            return base_position_size
        try:
            if ls_ratio > 1.5:
                adjusted_size = base_position_size * 2.0
                logger.debug(
                    f"L/S {ls_ratio:.2f} > 1.5: Lower risk, 50% margin -> ${adjusted_size:,.0f}"
                )
                return adjusted_size
            elif ls_ratio > -1.5:
                adjusted_size = base_position_size * 0.75
                logger.debug(
                    f"L/S {ls_ratio:.2f} > -1.5: Higher risk, 75% margin -> ${adjusted_size:,.0f}"
                )
                return adjusted_size
            else:
                logger.debug(
                    f"L/S {ls_ratio:.2f} <= -1.5: Very high risk, standard size"
                )
                return base_position_size
        except Exception as e:
            logger.error(f"Error adjusting short position size: {e}")
            return base_position_size

    def check_me_rebalancing_needed(self) -> Optional[str]:
        if not self.me_rebalancing_enabled:
            print(f"M/E REBALANCING DISABLED - No action taken")
            return None
        try:
            current_me = self.calculate_current_me_ratio()
            position_count = len(
                [pos for pos in self.positions.values() if pos["shares"] != 0]
            )
            print(f"\nM/E REBALANCING CHECK:")
            print(f"   Current M/E:        {current_me:.1f}%")
            print(
                f"   Target Range:       {self.me_target_min:.1f}% - {self.me_target_max:.1f}%"
            )
            print(f"   Position Count:     {position_count}")
            print(f"   Min for Scale-Up:   {self.min_positions_for_scaling_up}")
            if current_me < self.me_target_min:
                if position_count >= self.min_positions_for_scaling_up:
                    print(
                        f"SCALE UP NEEDED: {current_me:.1f}% < {self.me_target_min}% with {position_count} positions"
                    )
                    logger.info(
                        f"M/E {current_me:.1f}% < {self.me_target_min}% with {position_count} positions: SCALE UP needed"
                    )
                    return "scale_up"
                else:
                    print(
                        f"SCALE UP BLOCKED: Only {position_count} positions (need min {self.min_positions_for_scaling_up})"
                    )
                    logger.info(
                        f"M/E {current_me:.1f}% < {self.me_target_min}% but only {position_count} positions (min {self.min_positions_for_scaling_up}): No scaling"
                    )
                    return None
            elif current_me > self.me_target_max:
                print(f"SCALE DOWN NEEDED: {current_me:.1f}% > {self.me_target_max}%")
                logger.info(
                    f"M/E {current_me:.1f}% > {self.me_target_max}%: SCALE DOWN needed"
                )
                return "scale_down"
            else:
                print(
                    f"M/E IN TARGET RANGE: {current_me:.1f}% within {self.me_target_min:.1f}%-{self.me_target_max:.1f}%"
                )
                logger.debug(
                    f"M/E {current_me:.1f}% within target range {self.me_target_min}-{self.me_target_max}%"
                )
                return None
        except Exception as e:
            logger.error(f"Error checking M/E rebalancing: {e}")
            return None

    def perform_me_rebalancing(self, action: str) -> bool:
        if not self.me_rebalancing_enabled or not action:
            print(
                f"M/E REBALANCING SKIPPED: {'Disabled' if not self.me_rebalancing_enabled else 'No action specified'}"
            )
            return False
        try:
            current_me = self.calculate_current_me_ratio()
            positions_to_rebalance = {
                symbol: pos
                for symbol, pos in self.positions.items()
                if pos["shares"] != 0
            }
            if not positions_to_rebalance:
                print(f"NO POSITIONS TO REBALANCE")
                logger.warning("No positions to rebalance")
                return False
            if action == "scale_up":
                target_me = (self.me_target_min + self.me_target_max) / 2
                target_scaling = target_me / current_me if current_me > 0 else 1.3
                target_scaling = min(target_scaling, 1.8)
            elif action == "scale_down":
                target_me = (self.me_target_min + self.me_target_max) / 2
                target_scaling = target_me / current_me if current_me > 0 else 0.75
                target_scaling = max(target_scaling, 0.6)
            else:
                logger.error(f"Invalid rebalancing action: {action}")
                return False
            print(f"\nPERFORMING M/E REBALANCING:")
            print(f"   Action:             {action.upper()}")
            print(f"   Target M/E:         {target_me:.1f}% (middle of range)")
            print(f"   Scaling Factor:     {target_scaling:.3f}x")
            print(f"   Before M/E:         {current_me:.1f}%")
            print(f"   Positions:          {len(positions_to_rebalance)}")
            logger.info(
                f"M/E Rebalancing: {action} targeting {target_me:.1f}% with {target_scaling:.2f}x scaling factor"
            )
            logger.info(
                f"Before: M/E {current_me:.1f}%, {len(positions_to_rebalance)} positions"
            )
            rebalanced_positions: List[Dict[str, Any]] = []
            total_value_change = 0
            for symbol, position in positions_to_rebalance.items():
                old_shares = position["shares"]
                new_shares = int(round(old_shares * target_scaling))
                if old_shares > 0:
                    new_shares = max(1, new_shares)
                elif old_shares < 0:
                    new_shares = min(-1, new_shares)
                shares_change = new_shares - old_shares
                value_change = shares_change * position["entry_price"]
                self.positions[symbol]["shares"] = new_shares
                current_price = position.get("current_price", position["entry_price"])
                trade_type = "long" if new_shares > 0 else "short"
                self.me_calculator.update_position(
                    symbol,
                    new_shares,
                    position["entry_price"],
                    current_price,
                    trade_type,
                )
                total_value_change += value_change
                rebalanced_positions.append(
                    {
                        "symbol": symbol,
                        "old_shares": old_shares,
                        "new_shares": new_shares,
                        "shares_change": shares_change,
                        "value_change": value_change,
                    }
                )
                print(
                    f"   {symbol}: {old_shares:+4d} -> {new_shares:+4d} shares ({shares_change:+3d}) | ${value_change:+8,.0f}"
                )
                logger.debug(
                    f"  {symbol}: {old_shares} -> {new_shares} shares ({shares_change:+d}), value change: ${value_change:+,.2f}"
                )
            self.cash = round(float(self.cash - total_value_change), 2)
            new_me = self.calculate_current_me_ratio()
            print(f"   After M/E:          {new_me:.1f}%")
            print(f"   Cash Change:        ${total_value_change:+,.0f}")
            if self.me_target_min <= new_me <= self.me_target_max:
                print(
                    f"M/E REBALANCING COMPLETED: {len(rebalanced_positions)} positions scaled"
                )
                print(
                    f"TARGET ACHIEVED: {new_me:.1f}% is within {self.me_target_min:.1f}%-{self.me_target_max:.1f}% range"
                )
            else:
                print(f"M/E REBALANCING WARNING: {new_me:.1f}% outside target range")
                if new_me > self.me_target_max:
                    print(
                        f"   Still above {self.me_target_max:.1f}% limit - may need further adjustment"
                    )
                elif new_me < self.me_target_min:
                    print(
                        f"   Still below {self.me_target_min:.1f}% limit - may need further adjustment"
                    )
            logger.info(
                f"After: M/E {new_me:.1f}%, cash change: ${total_value_change:+,.2f}"
            )
            logger.info(
                f"M/E Rebalancing completed: {len(rebalanced_positions)} positions scaled"
            )
            self.record_historical_me_ratio(
                datetime.now().strftime("%Y-%m-%d"), trade_occurred=True
            )
            return True
        except Exception as e:
            logger.error(f"Error performing M/E rebalancing: {e}")
            return False

    def end_of_day_rebalancing(self) -> None:
        try:
            print(f"\n{'=' * 70}")
            print("END OF DAY M/E REBALANCING CHECK")
            print(f"{'=' * 70}")
            print(
                f"M/E Rebalancing Status: {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}"
            )
            if not self.me_rebalancing_enabled:
                print(f"M/E REBALANCING DISABLED - Skipping EOD check")
                logger.info("M/E rebalancing disabled - skipping EOD check")
                return
            logger.info("=== END OF DAY M/E REBALANCING CHECK ===")
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
            final_me = self.calculate_current_me_ratio()
            position_count = len(
                [pos for pos in self.positions.values() if pos["shares"] != 0]
            )
            ls_ratio = self.calculate_ls_ratio()
            print(f"\nEOD STATUS:")
            print(f"   Final M/E:          {final_me:.1f}%")
            print(f"   Positions:          {position_count}")
            print(f"   L/S Ratio:          {ls_ratio if ls_ratio else 'N/A'}")
            print(f"{'=' * 70}")
            logger.info(
                f"EOD Status: M/E {final_me:.1f}%, {position_count} positions, L/S {ls_ratio if ls_ratio else 'N/A'}"
            )
            logger.info("=== END OF DAY REBALANCING COMPLETE ===")
        except Exception as e:
            logger.error(f"Error in end-of-day rebalancing: {e}")

    def calculate_current_me_ratio(self) -> float:
        return float(self.me_calculator.calculate_daily_me_ratio()["ME_Ratio"])

    def calculate_historical_me_ratio(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> float:
        return float(
            self.me_calculator.calculate_daily_me_ratio(current_prices=current_prices)[
                "ME_Ratio"
            ]
        )

    def record_historical_me_ratio(
        self,
        date_str: str,
        trade_occurred: bool = False,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> None:
        self.me_calculator.calculate_daily_me_ratio(
            date_str, current_prices=current_prices
        )

    def _filter_recent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to only include data from the last 6 months (retention period).
        """
        if df is None or df.empty:
            return df
        df = df.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[df["Date"] >= self.cutoff_date].copy()
            logger.debug(
                f"Filtered data to last {self.retention_days} days: {len(df)} rows remaining"
            )
        return df

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty or len(df) < 20:
            logger.warning(
                f"Insufficient data for indicator calculation: {len(df) if df is not None else 'None'} rows"
            )
            return None
        
        logger.debug(f"Starting indicator calculation on {len(df)} rows")
        
        df = self._filter_recent_data(df)
        if df is None or df.empty or len(df) < 20:
            logger.warning(
                f"Insufficient data after 6-month filtering: {len(df) if df is not None else 'None'} rows"
            )
            return None
        
        df = df.copy()
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None
        
        logger.debug(f"All required columns present: {required_cols}")
        
        indicator_columns = [
            "BBAvg",
            "BBSDev",
            "UpperBB",
            "LowerBB",
            "High_Low",
            "High_Close",
            "Low_Close",
            "TR",
            "ATR",
            "ATRma",
            "LongPSAR",
            "ShortPSAR",
            "PSAR_EP",
            "PSAR_AF",
            "PSAR_IsLong",
            "oLRSlope",
            "oLRAngle",
            "oLRIntercept",
            "TSF",
            "oLRSlope2",
            "oLRAngle2",
            "oLRIntercept2",
            "TSF5",
            "Value1",
            "ROC",
            "LRV",
            "LinReg",
            "oLRValue",
            "oLRValue2",
            "SwingLow",
            "SwingHigh",
        ]
        for col in indicator_columns:
            if col not in df.columns:
                df[col] = np.nan
        try:
            df["BBAvg"] = (
                df["Close"].rolling(window=self.inputs["Length"]).mean().round(2)
            )
            df["BBSDev"] = (
                df["Close"].rolling(window=self.inputs["Length"]).std().round(2)
            )
            df["UpperBB"] = (df["BBAvg"] + self.inputs["NumDevs"] * df["BBSDev"]).round(
                2
            )
            df["LowerBB"] = (df["BBAvg"] - self.inputs["NumDevs"] * df["BBSDev"]).round(
                2
            )
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation error: {e}")
        try:
            df["High_Low"] = (df["High"] - df["Low"]).round(2)
            df["High_Close"] = abs(df["High"] - df["Close"].shift(1)).round(2)
            df["Low_Close"] = abs(df["Low"] - df["Close"].shift(1)).round(2)
            df["TR"] = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1).round(2)
            df["ATR"] = df["TR"].rolling(window=5).mean().round(2)
            df["ATRma"] = df["ATR"].rolling(window=13).mean().round(2)
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
        try:
            self._calculate_psar(df)
        except Exception as e:
            logger.warning(f"PSAR calculation error: {e}")
        try:
            self._calculate_linear_regression(df)
        except Exception as e:
            logger.warning(f"Linear regression calculation error: {e}")
        try:
            # Calculate Value1 and ROC with debugging
            logger.debug("Calculating Value1 and ROC indicators")
            
            # Use shorter windows for testing/development to ensure signals can be generated
            short_window = min(5, len(df) // 4)  # Adaptive short window
            long_window = min(20, len(df) // 2)   # Adaptive long window (was 35)
            
            logger.debug(f"Using adaptive windows: short={short_window}, long={long_window} (data length={len(df)})")
            
            df["Value1"] = (
                df["Close"].rolling(window=short_window).mean()
                - df["Close"].rolling(window=long_window).mean()
            ).round(2)
            df["ROC"] = (df["Value1"] - df["Value1"].shift(3)).round(2)
            
            # Check validity of intermediate calculations
            value1_valid = df["Value1"].notna().sum()
            roc_valid = df["ROC"].notna().sum()
            logger.debug(f"Intermediate indicators: Value1={value1_valid}/{len(df)} valid, ROC={roc_valid}/{len(df)} valid")
            
            self._calculate_lrv(df)
        except Exception as e:
            logger.warning(f"Additional indicators calculation error: {e}")
        try:
            df["SwingLow"] = df["Close"].rolling(window=4).min().round(2)
            df["SwingHigh"] = df["Close"].rolling(window=4).max().round(2)
        except Exception as e:
            logger.warning(f"Swing High/Low calculation error: {e}")
        
        # Log indicator calculation summary
        critical_indicators = ["LowerBB", "UpperBB", "oLRValue", "oLRValue2", "ATR", "SwingLow", "SwingHigh"]
        indicator_status = {}
        for ind in critical_indicators:
            if ind in df.columns:
                valid_count = df[ind].notna().sum()
                indicator_status[ind] = f"{valid_count}/{len(df)} valid"
            else:
                indicator_status[ind] = "missing"
        
        logger.info(f"Indicator calculation complete: {indicator_status}")
        
        # Check for completely empty indicators
        empty_indicators = [ind for ind, status in indicator_status.items() if "0/" in status]
        if empty_indicators:
            logger.warning(f"WARNING: These indicators have no valid values: {empty_indicators}")
        
        return df

    def _calculate_psar(self, df: pd.DataFrame) -> None:
        df["LongPSAR"] = 0.0
        df["ShortPSAR"] = 0.0
        df["PSAR_EP"] = 0.0
        df["PSAR_AF"] = self.inputs["AfStep"]
        df["PSAR_IsLong"] = (df["Close"] > df["Open"]).astype("Int64")
        df.loc[df["PSAR_IsLong"] == 1, "PSAR_EP"] = df.loc[
            df["PSAR_IsLong"] == 1, "High"
        ]
        df.loc[df["PSAR_IsLong"] == 1, "LongPSAR"] = df.loc[
            df["PSAR_IsLong"] == 1, "Low"
        ]
        df.loc[df["PSAR_IsLong"] == 1, "ShortPSAR"] = df.loc[
            df["PSAR_IsLong"] == 1, "High"
        ]
        df.loc[df["PSAR_IsLong"] == 0, "PSAR_EP"] = df.loc[
            df["PSAR_IsLong"] == 0, "Low"
        ]
        df.loc[df["PSAR_IsLong"] == 0, "LongPSAR"] = df.loc[
            df["PSAR_IsLong"] == 0, "High"
        ]
        df.loc[df["PSAR_IsLong"] == 0, "ShortPSAR"] = df.loc[
            df["PSAR_IsLong"] == 0, "Low"
        ]
        for i in range(1, len(df)):
            if pd.isna(df["Close"].iloc[i]) or pd.isna(df["Open"].iloc[i]):
                continue
            try:
                prev_is_long = (
                    int(df["PSAR_IsLong"].iloc[i - 1])
                    if not pd.isna(df["PSAR_IsLong"].iloc[i - 1])
                    else 1
                )
                if prev_is_long == 1:
                    long_psar = df["LongPSAR"].iloc[i - 1] + df["PSAR_AF"].iloc[
                        i - 1
                    ] * (df["PSAR_EP"].iloc[i - 1] - df["LongPSAR"].iloc[i - 1])
                    long_psar = min(
                        long_psar,
                        df["Low"].iloc[i],
                        df["Low"].iloc[i - 1] if i > 1 else df["Low"].iloc[i],
                    )
                    df.loc[df.index[i], "LongPSAR"] = round(long_psar, 2)
                    if df["High"].iloc[i] > df["PSAR_EP"].iloc[i - 1]:
                        df.loc[df.index[i], "PSAR_EP"] = round(df["High"].iloc[i], 2)
                        df.loc[df.index[i], "PSAR_AF"] = round(
                            min(
                                df["PSAR_AF"].iloc[i - 1] + self.inputs["AfStep"],
                                self.inputs["AfLimit"],
                            ),
                            2,
                        )
                    else:
                        df.loc[df.index[i], "PSAR_EP"] = df["PSAR_EP"].iloc[i - 1]
                        df.loc[df.index[i], "PSAR_AF"] = df["PSAR_AF"].iloc[i - 1]
                    if df["Low"].iloc[i] <= long_psar:
                        df.loc[df.index[i], "PSAR_IsLong"] = 0
                        df.loc[df.index[i], "ShortPSAR"] = round(
                            df["PSAR_EP"].iloc[i - 1], 2
                        )
                        df.loc[df.index[i], "PSAR_EP"] = round(df["Low"].iloc[i], 2)
                        df.loc[df.index[i], "PSAR_AF"] = round(self.inputs["AfStep"], 2)
                    else:
                        df.loc[df.index[i], "PSAR_IsLong"] = 1
                        df.loc[df.index[i], "ShortPSAR"] = df["ShortPSAR"].iloc[i - 1]
                else:
                    short_psar = df["ShortPSAR"].iloc[i - 1] - df["PSAR_AF"].iloc[
                        i - 1
                    ] * (df["ShortPSAR"].iloc[i - 1] - df["PSAR_EP"].iloc[i - 1])
                    short_psar = max(
                        short_psar,
                        df["High"].iloc[i],
                        df["High"].iloc[i - 1] if i > 1 else df["High"].iloc[i],
                    )
                    df.loc[df.index[i], "ShortPSAR"] = round(short_psar, 2)
                    if df["Low"].iloc[i] < df["PSAR_EP"].iloc[i - 1]:
                        df.loc[df.index[i], "PSAR_EP"] = round(df["Low"].iloc[i], 2)
                        df.loc[df.index[i], "PSAR_AF"] = round(
                            min(
                                df["PSAR_AF"].iloc[i - 1] + self.inputs["AfStep"],
                                self.inputs["AfLimit"],
                            ),
                            2,
                        )
                    else:
                        df.loc[df.index[i], "PSAR_EP"] = df["PSAR_EP"].iloc[i - 1]
                        df.loc[df.index[i], "PSAR_AF"] = df["PSAR_AF"].iloc[i - 1]
                    if df["High"].iloc[i] >= short_psar:
                        df.loc[df.index[i], "PSAR_IsLong"] = 1
                        df.loc[df.index[i], "LongPSAR"] = round(
                            df["PSAR_EP"].iloc[i - 1], 2
                        )
                        df.loc[df.index[i], "PSAR_EP"] = round(df["High"].iloc[i], 2)
                        df.loc[df.index[i], "PSAR_AF"] = round(self.inputs["AfStep"], 2)
                    else:
                        df.loc[df.index[i], "PSAR_IsLong"] = 0
                        df.loc[df.index[i], "LongPSAR"] = df["LongPSAR"].iloc[i - 1]
            except Exception as e:
                logger.warning(f"PSAR calculation error at index {i}: {e}")

    def _calculate_linear_regression(self, df: pd.DataFrame) -> None:
        def linear_reg(
            series: pd.Series, period: int, shift: int
        ) -> Tuple[float, float, float, float]:
            if len(series) < period or series.isna().any():
                return np.nan, np.nan, np.nan, np.nan
            try:
                x = np.arange(period)
                slope, intercept = np.polyfit(x, series[-period:], 1)
                value = slope * (period + shift - 1) + intercept
                return (
                    round(slope, 2),
                    round(np.degrees(np.arctan(slope)), 2),
                    round(intercept, 2),
                    round(value, 2),
                )
            except Exception as e:
                logger.warning(f"Linear regression calculation error: {e}")
                return np.nan, np.nan, np.nan, np.nan

        for i in range(len(df)):
            try:
                if i >= 3 and not df["Close"].iloc[max(0, i - 3) : i + 1].isna().any():
                    slope, angle, intercept, value = linear_reg(
                        df["Close"].iloc[max(0, i - 3) : i + 1], 3, -2
                    )
                    df.loc[df.index[i], "oLRSlope"] = slope
                    df.loc[df.index[i], "oLRAngle"] = angle
                    df.loc[df.index[i], "oLRIntercept"] = intercept
                    df.loc[df.index[i], "TSF"] = value
                if i >= 5 and not df["Close"].iloc[max(0, i - 5) : i + 1].isna().any():
                    slope2, angle2, intercept2, value2 = linear_reg(
                        df["Close"].iloc[max(0, i - 5) : i + 1], 5, -3
                    )
                    df.loc[df.index[i], "oLRSlope2"] = slope2
                    df.loc[df.index[i], "oLRAngle2"] = angle2
                    df.loc[df.index[i], "oLRIntercept2"] = intercept2
                    df.loc[df.index[i], "TSF5"] = value2
            except Exception as e:
                logger.warning(f"Linear regression error at index {i}: {e}")

    def _calculate_lrv(self, df: pd.DataFrame) -> None:
        logger.debug("Starting LRV calculation")
        
        def linear_reg_value(series: pd.Series, period: int, shift: int) -> float:
            if len(series) < period or series.isna().any():
                return np.nan
            try:
                x = np.arange(period)
                slope, intercept = np.polyfit(x, series[-period:], 1)
                return round(slope * (period + shift - 1) + intercept, 2)
            except Exception as e:
                logger.warning(f"Linear regression value calculation error: {e}")
                return np.nan

        lrv_calculated = 0
        oLRValue_calculated = 0
        oLRValue2_calculated = 0
        
        for i in range(len(df)):
            try:
                # Calculate LRV (needs ROC to be valid)
                if i >= 8 and not df["ROC"].iloc[max(0, i - 8) : i + 1].isna().any():
                    df.loc[df.index[i], "LRV"] = linear_reg_value(
                        df["ROC"].iloc[max(0, i - 8) : i + 1], 8, 0
                    )
                    lrv_calculated += 1
                
                # Calculate LinReg
                if i >= 8 and not df["Close"].iloc[max(0, i - 8) : i + 1].isna().any():
                    df.loc[df.index[i], "LinReg"] = linear_reg_value(
                        df["Close"].iloc[max(0, i - 8) : i + 1], 8, 0
                    )
                
                # Calculate oLRValue (needs LRV to be valid)
                if i >= 3 and not df["LRV"].iloc[max(0, i - 3) : i + 1].isna().any():
                    df.loc[df.index[i], "oLRValue"] = linear_reg_value(
                        df["LRV"].iloc[max(0, i - 3) : i + 1], 3, -2
                    )
                    oLRValue_calculated += 1
                
                # Calculate oLRValue2 (needs LRV to be valid)  
                if i >= 5 and not df["LRV"].iloc[max(0, i - 5) : i + 1].isna().any():
                    df.loc[df.index[i], "oLRValue2"] = linear_reg_value(
                        df["LRV"].iloc[max(0, i - 5) : i + 1], 5, -3
                    )
                    oLRValue2_calculated += 1
                    
            except Exception as e:
                logger.warning(f"LRV calculation error at index {i}: {e}")
        
        logger.debug(f"LRV calculation results: LRV={lrv_calculated}, oLRValue={oLRValue_calculated}, oLRValue2={oLRValue2_calculated}")
        
        # Check the dependency chain
        roc_valid = df["ROC"].notna().sum()
        lrv_valid = df["LRV"].notna().sum()
        olrvalue_valid = df["oLRValue"].notna().sum()
        olrvalue2_valid = df["oLRValue2"].notna().sum()
        
        logger.debug(f"Dependency chain: ROC={roc_valid} → LRV={lrv_valid} → oLRValue={olrvalue_valid}, oLRValue2={olrvalue2_valid}")
        
        if roc_valid == 0:
            logger.warning("ROC has no valid values - check Value1 calculation")
        if lrv_valid == 0 and roc_valid > 8:
            logger.warning("LRV calculation failed despite valid ROC values")

    def _check_long_signals(self, df: pd.DataFrame, i: int) -> None:
        """Check for long signal conditions with comprehensive debugging."""
        debug_signal = i % 200 == 0  # Log detailed debug info every 200 rows
        
        if debug_signal:
            logger.debug(f"Checking long signals at row {i}")

        if (
            df["Open"].iloc[i] < df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] > df["Open"].iloc[i - 1]
            and df["Close"].iloc[i] > df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] < df["Open"].iloc[i - 1]
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and df["High"].iloc[i] - df["Close"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Open"].iloc[i - 1] - df["Close"].iloc[i - 1] > 0.05
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] >= df["oLRValue2"].iloc[i]
            and df["Low"].iloc[i] <= df["LowerBB"].iloc[i] * 1.02
            and df["Close"].iloc[i] <= df["UpperBB"].iloc[i] * 0.95
        ):
            df.loc[df.index[i], "Signal"] = 1
            df.loc[df.index[i], "SignalType"] = "Engf L"
            df.loc[df.index[i], "Shares"] = int(
                round(self.inputs["PositionSize"] / df["Close"].iloc[i])
            )
            logger.info(f"LONG SIGNAL GENERATED: Engf L at row {i}, price={df['Close'].iloc[i]:.2f}, shares={df['Shares'].iloc[i]}")
        elif (
            df["Open"].iloc[i] < df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] > df["Open"].iloc[i - 1]
            and df["Close"].iloc[i] > df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] < df["Open"].iloc[i - 1]
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and df["High"].iloc[i] - df["Close"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Open"].iloc[i - 1] - df["Close"].iloc[i - 1] > 0.05
            and df["Close"].iloc[i - 1] <= df["SwingLow"].iloc[i]
            and df["Low"].iloc[i] <= df["LowerBB"].iloc[i] * 1.02
            and df["Close"].iloc[i] <= df["UpperBB"].iloc[i] * 0.95
        ):
            df.loc[df.index[i], "Signal"] = 1
            df.loc[df.index[i], "SignalType"] = "Engf L NuLo"
            df.loc[df.index[i], "Shares"] = int(
                round(self.inputs["PositionSize"] / df["Close"].iloc[i])
            )
            logger.info(f"LONG SIGNAL GENERATED: Engf L NuLo at row {i}, price={df['Close'].iloc[i]:.2f}, shares={df['Shares'].iloc[i]}")
        elif (
            df["Open"].iloc[i] <= df["Close"].iloc[i - 1] * 1.001
            and df["Open"].iloc[i] > df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] > df["Open"].iloc[i - 1] + (df["ATR"].iloc[i] * 0.5)
            and abs(
                (df["Open"].iloc[i - 1] - df["Close"].iloc[i - 1])
                / df["Close"].iloc[i - 1]
            )
            >= 0.003
            and df["Close"].iloc[i] > df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] < df["Open"].iloc[i - 1]
            and df["High"].iloc[i] - df["Close"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Open"].iloc[i - 1] - df["Close"].iloc[i - 1] > 0.05
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] >= df["oLRValue2"].iloc[i]
            and df["Low"].iloc[i] <= df["LowerBB"].iloc[i] * 1.02
            and df["Close"].iloc[i] <= df["UpperBB"].iloc[i] * 0.95
        ):
            df.loc[df.index[i], "Signal"] = 1
            df.loc[df.index[i], "SignalType"] = "SemiEng L"
            df.loc[df.index[i], "Shares"] = int(
                round(self.inputs["PositionSize"] / df["Close"].iloc[i])
            )
            logger.info(f"LONG SIGNAL GENERATED: SemiEng L at row {i}, price={df['Close'].iloc[i]:.2f}, shares={df['Shares'].iloc[i]}")
        elif (
            df["Open"].iloc[i] <= df["Close"].iloc[i - 1] * 1.001
            and df["Open"].iloc[i] > df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] > df["Open"].iloc[i - 1] + (df["ATR"].iloc[i] * 0.5)
            and abs(
                (df["Open"].iloc[i - 1] - df["Close"].iloc[i - 1])
                / df["Close"].iloc[i - 1]
            )
            >= 0.003
            and df["Close"].iloc[i] > df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] < df["Open"].iloc[i - 1]
            and df["High"].iloc[i] - df["Close"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Open"].iloc[i - 1] - df["Close"].iloc[i - 1] > 0.05
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and df["Close"].iloc[i - 1] <= df["SwingLow"].iloc[i]
            and df["Low"].iloc[i] <= df["LowerBB"].iloc[i] * 1.02
            and df["Close"].iloc[i] <= df["UpperBB"].iloc[i] * 0.95
        ):
            df.loc[df.index[i], "Signal"] = 1
            df.loc[df.index[i], "SignalType"] = "SemiEng L NuLo"
            df.loc[df.index[i], "Shares"] = int(
                round(self.inputs["PositionSize"] / df["Close"].iloc[i])
            )
            logger.info(f"LONG SIGNAL GENERATED: SemiEng L NuLo at row {i}, price={df['Close'].iloc[i]:.2f}, shares={df['Shares'].iloc[i]}")

    def _check_short_signals(self, df: pd.DataFrame, i: int) -> None:
        if (
            df["Open"].iloc[i] > df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] < df["Open"].iloc[i - 1]
            and df["Close"].iloc[i] < df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] > df["Open"].iloc[i - 1]
            and df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Close"].iloc[i - 1] - df["Open"].iloc[i - 1] > 0.05
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] <= df["oLRValue2"].iloc[i]
            and df["High"].iloc[i] >= df["UpperBB"].iloc[i] * 0.98
            and df["Close"].iloc[i] >= df["LowerBB"].iloc[i] * 1.05
        ):
            df.loc[df.index[i], "Signal"] = -1
            df.loc[df.index[i], "SignalType"] = "Engf S"
            adjusted_position_size = self.get_short_position_size(
                self.inputs["PositionSize"]
            )
            df.loc[df.index[i], "Shares"] = int(
                round(adjusted_position_size / df["Close"].iloc[i])
            )
        elif (
            df["Open"].iloc[i] >= df["Close"].iloc[i - 1] * 0.999
            and df["Open"].iloc[i] < df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] < df["Open"].iloc[i - 1]
            and df["Close"].iloc[i] < df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] > df["Open"].iloc[i - 1]
            and df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Close"].iloc[i - 1] - df["Open"].iloc[i - 1] > 0.05
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] <= df["oLRValue2"].iloc[i]
            and df["High"].iloc[i] >= df["UpperBB"].iloc[i] * 0.98
            and df["Close"].iloc[i] >= df["LowerBB"].iloc[i] * 1.05
        ):
            df.loc[df.index[i], "Signal"] = -1
            df.loc[df.index[i], "SignalType"] = "SemiEng S"
            adjusted_position_size = self.get_short_position_size(
                self.inputs["PositionSize"]
            )
            df.loc[df.index[i], "Shares"] = int(
                round(adjusted_position_size / df["Close"].iloc[i])
            )
        elif (
            df["Open"].iloc[i] >= df["Close"].iloc[i - 1] * 0.999
            and df["Open"].iloc[i] < df["Close"].iloc[i - 1]
            and df["Close"].iloc[i] < df["Open"].iloc[i - 1]
            and df["Close"].iloc[i] < df["Open"].iloc[i]
            and df["Close"].iloc[i - 1] > df["Open"].iloc[i - 1]
            and df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1] <= df["ATR"].iloc[i] * 2
            and df["Close"].iloc[i - 1] - df["Open"].iloc[i - 1] > 0.05
            and df["Close"].iloc[i - 1] >= df["SwingHigh"].iloc[i]
            and abs(df["Close"].iloc[i - 1] - df["Low"].iloc[i - 1])
            / df["Close"].iloc[i - 1]
            < 0.05
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] <= df["oLRValue2"].iloc[i]
            and df["High"].iloc[i] >= df["UpperBB"].iloc[i] * 0.98
            and df["Close"].iloc[i] >= df["LowerBB"].iloc[i] * 1.05
        ):
            df.loc[df.index[i], "Signal"] = -1
            df.loc[df.index[i], "SignalType"] = "SemiEng S NuHi"
            adjusted_position_size = self.get_short_position_size(
                self.inputs["PositionSize"]
            )
            df.loc[df.index[i], "Shares"] = int(
                round(adjusted_position_size / df["Close"].iloc[i])
            )

    def _check_long_exits(
        self, df: pd.DataFrame, i: int, position: Dict[str, Any]
    ) -> None:
        possible_exits: List[Tuple[str, str, int, Optional[int]]] = []

        # Gap out / Target
        if (
            position["profit"] > 0
            and position["bars_since_entry"] > 1
            and (
                df["Open"].iloc[i] >= df["Close"].iloc[i - 1] * 1.05
                or (
                    not pd.isna(df["UpperBB"].iloc[i])
                    and df["Open"].iloc[i] >= df["UpperBB"].iloc[i]
                )
            )
        ):
            exit_type = (
                "Gap out L"
                if df["Open"].iloc[i] >= df["Close"].iloc[i - 1] * 1.05
                else "Target L"
            )
            possible_exits.append(("gap_target", exit_type, -1, None))
        # BE L
        if (
            not pd.isna(
                df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
            )
            and df["Close"].iloc[i]
            >= position["entry_price"]
            + df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
        ):
            possible_exits.append(("be", "BE L", -1, None))
        # L ATR X
        if (
            not pd.isna(
                df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
            )
            and df["Close"].iloc[i]
            >= position["entry_price"]
            + df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
            * 1.5
        ):
            possible_exits.append(("atr_x", "L ATR X", -1, None))
        # Reversal S 2 L
        if (
            position["bars_since_entry"] > 5
            and not pd.isna(df["LinReg"].iloc[i])
            and not pd.isna(df["LinReg"].iloc[i - 1])
            and df["LinReg"].iloc[i] < df["LinReg"].iloc[i - 1]
            and position["profit"] < 0
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] < df["oLRValue2"].iloc[i]
            and not pd.isna(df["ATR"].iloc[i])
            and not pd.isna(df["ATRma"].iloc[i])
            and df["ATR"].iloc[i] > df["ATRma"].iloc[i]
        ):
            possible_exits.append(("reversal", "S 2 L", -1, -1))
        # Hard Stop
        if df["Close"].iloc[i] < position["entry_price"] * 0.9:
            possible_exits.append(("hard_stop", "Hard Stop S", -1, None))

        priority_order = ["hard_stop", "reversal", "atr_x", "be", "gap_target"]
        for prio in priority_order:
            for exit_type, exit_label, exit_sig, entry_sig in possible_exits:
                if exit_type == prio:
                    df.loc[df.index[i], "ExitSignal"] = exit_sig
                    df.loc[df.index[i], "ExitType"] = exit_label
                    if entry_sig is not None:
                        df.loc[df.index[i], "Signal"] = entry_sig
                        df.loc[df.index[i], "SignalType"] = exit_label
                        df.loc[df.index[i], "Shares"] = int(
                            round(self.inputs["PositionSize"] * 2 / df["Close"].iloc[i])
                        )
                    return

    def _check_short_exits(
        self, df: pd.DataFrame, i: int, position: Dict[str, Any]
    ) -> None:
        possible_exits: List[Tuple[str, str, int, Optional[int]]] = []

        if (
            position["profit"] > 0
            and position["bars_since_entry"] > 1
            and (
                df["Open"].iloc[i] <= df["Close"].iloc[i - 1] * 0.95
                or (
                    not pd.isna(df["LowerBB"].iloc[i])
                    and df["Open"].iloc[i] <= df["LowerBB"].iloc[i]
                )
            )
        ):
            exit_type = (
                "Gap out S"
                if df["Open"].iloc[i] <= df["Close"].iloc[i - 1] * 0.95
                else "Target S"
            )
            possible_exits.append(("gap_target", exit_type, 1, None))
        # BE S
        if (
            not pd.isna(
                df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
            )
            and df["Close"].iloc[i]
            <= position["entry_price"]
            - df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
        ):
            possible_exits.append(("be", "BE S", 1, None))
        # S ATR X
        if (
            not pd.isna(
                df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
            )
            and df["Close"].iloc[i]
            <= position["entry_price"]
            - df["ATR"].iloc[max(0, i - position["bars_since_entry"]) : i + 1].max()
            * 1.5
        ):
            possible_exits.append(("atr_x", "S ATR X", 1, None))
        # Reversal L 2 S
        if (
            position["bars_since_entry"] > 5
            and not pd.isna(df["LinReg"].iloc[i])
            and not pd.isna(df["LinReg"].iloc[i - 1])
            and df["LinReg"].iloc[i] > df["LinReg"].iloc[i - 1]
            and position["profit"] < 0
            and not pd.isna(df["oLRValue"].iloc[i])
            and not pd.isna(df["oLRValue2"].iloc[i])
            and df["oLRValue"].iloc[i] > df["oLRValue2"].iloc[i]
            and not pd.isna(df["ATR"].iloc[i])
            and not pd.isna(df["ATRma"].iloc[i])
            and df["ATR"].iloc[i] > df["ATRma"].iloc[i]
        ):
            possible_exits.append(("reversal", "L 2 S", 1, 1))
        if df["Close"].iloc[i] > position["entry_price"] * 1.1:
            possible_exits.append(("hard_stop", "Hard Stop L", 1, None))

        priority_order = ["hard_stop", "reversal", "atr_x", "be", "gap_target"]
        for prio in priority_order:
            for exit_type, exit_label, exit_sig, entry_sig in possible_exits:
                if exit_type == prio:
                    df.loc[df.index[i], "ExitSignal"] = exit_sig
                    df.loc[df.index[i], "ExitType"] = exit_label
                    if entry_sig is not None:
                        df.loc[df.index[i], "Signal"] = entry_sig
                        df.loc[df.index[i], "SignalType"] = exit_label
                        df.loc[df.index[i], "Shares"] = int(
                            round(self.inputs["PositionSize"] * 2 / df["Close"].iloc[i])
                        )
                    return

    def _process_exit(
        self, df: pd.DataFrame, i: int, symbol: str, position: Dict[str, Any]
    ) -> None:
        exit_price = round(float(df["Close"].iloc[i]), 2)
        profit = round(
            float(
                (exit_price - position["entry_price"]) * position["shares"]
                if position["shares"] > 0
                else (position["entry_price"] - exit_price) * abs(position["shares"])
            ),
            2,
        )
        exit_date = str(df["Date"].iloc[i])[:10]
        exit_datetime = datetime.strptime(exit_date, "%Y-%m-%d")
        if exit_datetime >= self.cutoff_date:
            trade = {
                "symbol": symbol,
                "type": "long" if position["shares"] > 0 else "short",
                "entry_date": position["entry_date"],
                "exit_date": exit_date,
                "entry_price": round(float(position["entry_price"]), 2),
                "exit_price": exit_price,
                "shares": int(round(abs(position["shares"]))),
                "profit": profit,
                "exit_reason": df["ExitType"].iloc[i],
            }
            if all(
                k in trade
                for k in [
                    "symbol",
                    "type",
                    "entry_date",
                    "exit_date",
                    "entry_price",
                    "exit_price",
                    "shares",
                    "profit",
                    "exit_reason",
                ]
            ):
                self._trades.append(trade)
                save_trades([trade])
            else:
                logger.warning(f"Invalid trade skipped: {trade}")
        trade_type = "long" if position["shares"] > 0 else "short"
        self.me_calculator.update_position(symbol, 0, 0, 0, trade_type)
        self.me_calculator.add_realized_pnl(profit)
        self.cash = round(float(self.cash + position["shares"] * exit_price), 2)
        self.record_historical_me_ratio(exit_date, trade_occurred=True)
        logger.info(
            f"Exit {symbol}: {df['ExitType'].iloc[i]} at {exit_price}, profit: {profit}"
        )

    def _process_entry(
        self, df: pd.DataFrame, i: int, symbol: str, position: Dict[str, Any]
    ) -> None:
        shares = (
            int(round(df["Shares"].iloc[i]))
            if df["Signal"].iloc[i] > 0
            else -int(round(df["Shares"].iloc[i]))
        )
        cost = round(float(shares * df["Close"].iloc[i]), 2)
        entry_date = str(df["Date"].iloc[i])[:10]
        entry_datetime = datetime.strptime(entry_date, "%Y-%m-%d")
        if entry_datetime >= self.cutoff_date and abs(cost) <= self.cash:
            self.cash = round(float(self.cash - cost), 2)
            position = {
                "shares": shares,
                "entry_price": round(float(df["Close"].iloc[i]), 2),
                "entry_date": entry_date,
                "bars_since_entry": 0,
                "profit": 0,
            }
            self.positions[symbol] = position
            current_price = round(float(df["Close"].iloc[i]), 2)
            trade_type = "long" if shares > 0 else "short"
            self.me_calculator.update_position(
                symbol, shares, position["entry_price"], current_price, trade_type
            )
            self.record_historical_me_ratio(entry_date, trade_occurred=True)
            sector = get_symbol_sector(symbol)
            logger.info(
                f"Entry {symbol} ({sector}): {df['SignalType'].iloc[i]} with {shares} shares at {df['Close'].iloc[i]}"
            )
        elif entry_datetime < self.cutoff_date:
            logger.debug(f"Entry {symbol} skipped - outside retention period")
        else:
            logger.warning(
                f"Insufficient cash for {symbol}: {cost} required, {self.cash} available"
            )

    def manage_positions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        df["ExitSignal"] = 0
        df["ExitType"] = ""
        position = self.positions.get(
            symbol,
            {
                "shares": 0,
                "entry_price": 0,
                "entry_date": None,
                "bars_since_entry": 0,
                "profit": 0,
            },
        )
        for i in range(1, len(df)):
            if (
                pd.isna(df["Close"].iloc[i])
                or pd.isna(df["Open"].iloc[i])
                or pd.isna(df["ATR"].iloc[i])
            ):
                continue
            current_date = str(df["Date"].iloc[i])[:10]
            if position["shares"] != 0:
                position["bars_since_entry"] += 1
                position["profit"] = round(
                    (
                        (df["Close"].iloc[i] - position["entry_price"])
                        * position["shares"]
                        if position["shares"] > 0
                        else (position["entry_price"] - df["Close"].iloc[i])
                        * abs(position["shares"])
                    ),
                    2,
                )
                if position["shares"] > 0:
                    self._check_long_exits(df, i, position)
                elif position["shares"] < 0:
                    self._check_short_exits(df, i, position)
            if df["ExitSignal"].iloc[i] != 0:
                self._process_exit(df, i, symbol, position)
                position = {
                    "shares": 0,
                    "entry_price": 0,
                    "entry_date": None,
                    "bars_since_entry": 0,
                    "profit": 0,
                }
            if df["Signal"].iloc[i] != 0:
                self._process_entry(df, i, symbol, position)
                position = self.positions.get(
                    symbol,
                    {
                        "shares": 0,
                        "entry_price": 0,
                        "entry_date": None,
                        "bars_since_entry": 0,
                        "profit": 0,
                    },
                )
            else:
                self.record_historical_me_ratio(current_date, trade_occurred=False)
        self.positions[symbol] = position
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("Empty dataframe provided to generate_signals")
            return df
        
        logger.info(f"Starting signal generation on {len(df)} rows of data")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        logger.debug(f"Input parameters: MinPrice={self.inputs['MinPrice']}, MaxPrice={self.inputs['MaxPrice']}, PositionSize={self.inputs['PositionSize']}")
        
        df = df.copy()
        df["Signal"] = 0
        df["SignalType"] = ""
        df["Shares"] = 0
        
        processed_count = 0
        price_filtered_count = 0
        nan_filtered_count = 0
        
        for i in range(1, len(df)):
            # Check for NaN values in price data
            if (
                pd.isna(df["Open"].iloc[i])
                or pd.isna(df["Close"].iloc[i])
                or pd.isna(df["High"].iloc[i])
                or pd.isna(df["Low"].iloc[i])
            ):
                nan_filtered_count += 1
                continue
            
            # Check price range filter
            if not (
                df["Open"].iloc[i] > self.inputs["MinPrice"]
                and df["Open"].iloc[i] < self.inputs["MaxPrice"]
            ):
                price_filtered_count += 1
                continue
            
            processed_count += 1
            # Check for required indicators before signal generation
            required_indicators = ["LowerBB", "UpperBB", "oLRValue", "oLRValue2", "ATR", "SwingLow", "SwingHigh"]
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns or pd.isna(df[ind].iloc[i])]
            
            if missing_indicators:
                logger.debug(f"Row {i}: Missing/NaN indicators: {missing_indicators}")
                continue
                
            # Log key indicator values for debugging
            if i % 100 == 0:  # Log every 100th row to avoid spam
                logger.debug(f"Row {i}: Open={df['Open'].iloc[i]:.2f}, Close={df['Close'].iloc[i]:.2f}, "
                           f"LowerBB={df['LowerBB'].iloc[i]:.2f}, UpperBB={df['UpperBB'].iloc[i]:.2f}, "
                           f"oLRValue={df['oLRValue'].iloc[i]:.4f}, oLRValue2={df['oLRValue2'].iloc[i]:.4f}")
            
            self._check_long_signals(df, i)
            self._check_short_signals(df, i)
        
        signals_generated = (df["Signal"] != 0).sum()
        logger.info(f"Signal generation complete: processed {processed_count} rows, filtered {nan_filtered_count} NaN rows, {price_filtered_count} price-filtered rows")
        logger.info(f"Generated {signals_generated} signals total")
        
        if signals_generated > 0:
            signal_breakdown = df[df["Signal"] != 0]["SignalType"].value_counts()
            logger.info(f"Signal breakdown: {dict(signal_breakdown)}")
        else:
            logger.warning("NO SIGNALS GENERATED - investigating causes...")
            # Log a sample of rows to understand why no signals were generated
            sample_rows = min(5, len(df)-1)
            for i in range(1, sample_rows + 1):
                logger.debug(f"Sample row {i}: Open={df['Open'].iloc[i]:.2f} (valid price range: {self.inputs['MinPrice']} < x < {self.inputs['MaxPrice']})")
                if 'LowerBB' in df.columns:
                    logger.debug(f"  LowerBB={df['LowerBB'].iloc[i]:.2f}, UpperBB={df['UpperBB'].iloc[i]:.2f}")
                if 'oLRValue' in df.columns:
                    logger.debug(f"  oLRValue={df['oLRValue'].iloc[i]:.4f}, oLRValue2={df['oLRValue2'].iloc[i]:.4f}")
        
        return df

    def _load_positions(self) -> None:
        positions_list = get_positions()
        logger.info(
            f"Attempting to load {len(positions_list)} positions from data manager"
        )
        for pos in positions_list:
            symbol = pos.get("symbol")
            entry_date = pos.get("entry_date")
            logger.debug(
                f"Processing position: {symbol}, entry_date: {entry_date} (type: {type(entry_date)})"
            )
            if symbol and entry_date:
                try:
                    if isinstance(entry_date, str):
                        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
                            try:
                                entry_datetime = datetime.strptime(
                                    entry_date.split()[0], "%Y-%m-%d"
                                )
                                entry_date_str = entry_datetime.strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                continue
                        else:
                            logger.warning(
                                f"Could not parse date for position {symbol}: {entry_date}"
                            )
                            continue
                    else:
                        entry_datetime = (
                            entry_date.to_pydatetime()
                            if hasattr(entry_date, "to_pydatetime")
                            else entry_date
                        )
                        entry_date_str = entry_datetime.strftime("%Y-%m-%d")
                    if entry_datetime >= self.cutoff_date:
                        self.positions[symbol] = {
                            "shares": int(pos.get("shares", 0)),
                            "entry_price": float(pos.get("entry_price", 0)),
                            "entry_date": entry_date_str,
                            "bars_since_entry": int(pos.get("days_held", 0)),
                            "profit": float(pos.get("profit", 0)),
                        }
                        current_price = pos.get(
                            "current_price", pos.get("entry_price", 0)
                        )
                        trade_type = pos.get("side", "long")
                        self.me_calculator.update_position(
                            symbol,
                            self.positions[symbol]["shares"],
                            self.positions[symbol]["entry_price"],
                            current_price,
                            trade_type,
                        )
                        logger.debug(f"Loaded position for {symbol}")
                    else:
                        logger.debug(
                            f"Position {symbol} outside retention period: {entry_datetime} < {self.cutoff_date}"
                        )
                except (ValueError, AttributeError, TypeError) as e:
                    logger.warning(
                        f"Invalid date format for position {symbol}: {entry_date} - {e}"
                    )
        logger.info(
            f"Loaded {len(self.positions)} positions within {self.retention_days}-day retention period"
        )

    def _filter_trades_by_retention(self) -> None:
        logger.info(f"Starting trade retention filtering. Initial trades count: {len(self._trades)}")
        logger.debug(f"Cutoff date for retention: {self.cutoff_date.strftime('%Y-%m-%d')}")
        
        if self._trades:
            filtered_trades: List[Dict[str, Any]] = []
            invalid_trade_count = 0
            
            for trade in self._trades:
                try:
                    exit_date = datetime.strptime(trade["exit_date"], "%Y-%m-%d")
                    if exit_date >= self.cutoff_date:
                        filtered_trades.append(trade)
                        logger.debug(f"Trade kept: exit_date={exit_date.strftime('%Y-%m-%d')}, symbol={trade.get('symbol', 'unknown')}")
                    else:
                        logger.debug(f"Trade filtered out: exit_date={exit_date.strftime('%Y-%m-%d')}, symbol={trade.get('symbol', 'unknown')}")
                except (ValueError, KeyError) as e:
                    invalid_trade_count += 1
                    logger.warning(f"Invalid trade date format: {trade}, error: {e}")
            
            trades_removed = len(self._trades) - len(filtered_trades)
            self._trades = filtered_trades
            
            logger.info(
                f"Trade filtering complete: {len(self._trades)} trades kept, {trades_removed} removed, {invalid_trade_count} invalid"
            )
        else:
            logger.info("No trades to filter - starting with empty trade list")

    @property
    def trades(self) -> List[Dict[str, Any]]:
        return self._trades

    @trades.setter
    def trades(self, value: List[Dict[str, Any]]) -> None:
        self._trades = value

    def get_current_positions(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        long_positions: List[Dict[str, Any]] = []
        short_positions: List[Dict[str, Any]] = []
        for symbol, pos in self.positions.items():
            if pos["shares"] > 0:
                long_positions.append(
                    {
                        "symbol": symbol,
                        "shares": int(round(pos["shares"])),
                        "entry_price": round(float(pos["entry_price"]), 2),
                        "entry_date": pos["entry_date"],
                        "gap_target": round(float(pos["entry_price"] * 1.05), 2),
                        "bb_target": None,
                    }
                )
            elif pos["shares"] < 0:
                short_positions.append(
                    {
                        "symbol": symbol,
                        "shares": int(round(abs(pos["shares"]))),
                        "entry_price": round(float(pos["entry_price"]), 2),
                        "entry_date": pos["entry_date"],
                        "gap_target": round(float(pos["entry_price"] * 0.95), 2),
                        "bb_target": None,
                    }
                )
        return long_positions, short_positions

    def process_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.debug(f"Processing symbol {symbol} with {len(df)} rows of data")
        try:
            df_with_indicators = self.calculate_indicators(df)
            if df_with_indicators is None or df_with_indicators.empty:
                logger.warning(f"Failed to calculate indicators for {symbol}")
                return None
            
            logger.debug(f"Successfully calculated indicators for {symbol}")
            df_with_signals = self.generate_signals(df_with_indicators)
            
            # Log signal generation results for this symbol
            signals_count = (df_with_signals["Signal"] != 0).sum()
            if signals_count > 0:
                logger.info(f"Symbol {symbol}: Generated {signals_count} signals")
            else:
                logger.debug(f"Symbol {symbol}: No signals generated")
            
            result_df = self.manage_positions(df_with_signals, symbol)
            if not result_df.empty and symbol in self.positions:
                current_price = result_df["Close"].iloc[-1]
                if self.positions[symbol]["shares"] != 0:
                    trade_type = (
                        "long" if self.positions[symbol]["shares"] > 0 else "short"
                    )
                    self.me_calculator.update_position(
                        symbol=symbol,
                        shares=self.positions[symbol]["shares"],
                        entry_price=self.positions[symbol]["entry_price"],
                        current_price=current_price,
                        trade_type=trade_type,
                    )
                    logger.debug(f"Updated position for {symbol}: {self.positions[symbol]['shares']} shares at {current_price:.2f}")
            return result_df
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            logger.debug(f"Traceback for {symbol}: {traceback.format_exc()}")
            return None

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        self.trades = []
        self.me_calculator = DailyMERatioCalculator(self.account_size)
        self._filter_trades_by_retention()
        print(f"\n{'=' * 70}")
        print("STARTING STRATEGY RUN - M/E VERIFICATION")
        print(f"{'=' * 70}")
        print(
            f"M/E Rebalancing:      {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}"
        )
        print(
            f"M/E Target Range:     {self.me_target_min:.1f}% - {self.me_target_max:.1f}%"
        )
        print(f"Min Positions Scale:  {self.min_positions_for_scaling_up}")
        print(f"Initial Positions:    {len(self.positions)}")
        if self.positions:
            initial_me_ratio = self.calculate_current_me_ratio()
            print(f"Initial M/E:          {initial_me_ratio:.2f}%")
            logger.info(
                f"Initial M/E ratio with {len(self.positions)} existing positions: {initial_me_ratio:.2f}%"
            )
        else:
            print(f"Initial M/E:          0.00% (no positions)")
        print(f"{'=' * 70}")
        results: Dict[str, pd.DataFrame] = {}
        for i, (symbol, df) in enumerate(data.items()):
            result = self.process_symbol(symbol, df)
            if result is not None and not result.empty:
                results[symbol] = result
                if (i + 1) % 50 == 0:
                    current_me = self.calculate_current_me_ratio()
                    print(
                        f"Progress: {i + 1}/{len(data)} symbols | Current M/E: {current_me:.1f}%"
                    )
                logger.info(
                    f"Processed {symbol}: {len(result)} rows ({i + 1}/{len(data)})"
                )
        final_prices: Dict[str, float] = {}
        for symbol, df in results.items():
            if not df.empty:
                final_prices[symbol] = df["Close"].iloc[-1]
        self.update_me_calculator_with_current_prices(final_prices)
        self.me_calculator.save_daily_me_data()
        long_positions, short_positions = self.get_current_positions()
        all_positions: List[Dict[str, Any]] = []
        for pos in long_positions:
            if pos["symbol"] in results and not results[pos["symbol"]].empty:
                current_price = results[pos["symbol"]].iloc[-1]["Close"]
            else:
                current_price = pos["entry_price"]
            entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_dt).days
            all_positions.append(
                {
                    "symbol": pos["symbol"],
                    "shares": pos["shares"],
                    "entry_price": pos["entry_price"],
                    "entry_date": pos["entry_date"],
                    "current_price": round(float(current_price), 2),
                    "current_value": round(float(current_price * pos["shares"]), 2),
                    "profit": round(
                        float((current_price - pos["entry_price"]) * pos["shares"]), 2
                    ),
                    "profit_pct": round(
                        float((current_price / pos["entry_price"] - 1) * 100), 2
                    ),
                    "days_held": days_held,
                    "side": "long",
                    "strategy": "nGS",
                }
            )
        for pos in short_positions:
            if pos["symbol"] in results and not results[pos["symbol"]].empty:
                current_price = results[pos["symbol"]].iloc[-1]["Close"]
            else:
                current_price = pos["entry_price"]
            shares_abs = abs(pos["shares"])
            profit = round(float((pos["entry_price"] - current_price) * shares_abs), 2)
            entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
            days_held = (datetime.now() - entry_dt).days
            all_positions.append(
                {
                    "symbol": pos["symbol"],
                    "shares": -shares_abs,
                    "entry_price": pos["entry_price"],
                    "entry_date": pos["entry_date"],
                    "current_price": round(float(current_price), 2),
                    "current_value": round(float(current_price * shares_abs), 2),
                    "profit": profit,
                    "profit_pct": round(
                        float((pos["entry_price"] / current_price - 1) * 100), 2
                    ),
                    "days_held": days_held,
                    "side": "short",
                    "strategy": "nGS",
                }
            )
        save_positions(all_positions)
        print(f"\nCALLING END-OF-DAY M/E REBALANCING...")
        self.end_of_day_rebalancing()
        final_me_metrics = self.me_calculator.calculate_daily_me_ratio(
            date=datetime.now().strftime("%Y-%m-%d"), current_prices=final_prices
        )
        final_me = final_me_metrics["ME_Ratio"]
        print(f"\n{'=' * 70}")
        print("FINAL M/E STATUS")
        print(f"{'=' * 70}")
        print(f"Final M/E Ratio:      {final_me:.2f}%")
        print(
            f"Target Range:         {self.me_target_min:.1f}% - {self.me_target_max:.1f}%"
        )
        print(f"Final Positions:      {len(all_positions)}")
        print(
            f"Rebalancing System:   {'ACTIVE' if self.me_rebalancing_enabled else 'INACTIVE'}"
        )
        print(f"\nM/E Calculation Details:")
        print(f"Portfolio Equity:       ${final_me_metrics['Portfolio_Equity']:,.2f}")
        print(
            f"Total Position Value:   ${final_me_metrics['Total_Position_Value']:,.2f}"
        )
        print(f"Long Value:             ${final_me_metrics['Long_Value']:,.2f}")
        print(f"Short Value:            ${final_me_metrics['Short_Value']:,.2f}")
        print(f"Unrealized P&L:         ${final_me_metrics['Unrealized_PnL']:,.2f}")
        print(f"Realized P&L:           ${final_me_metrics['Realized_PnL']:,.2f}")
        print(f"Starting Account:       ${self.account_size:,.2f}")
        risk = self.me_calculator.get_risk_assessment()
        print(f"M/E Risk Status:        {risk['risk_level']}")
        print(f"M/E Active Positions:   {len(self.me_calculator.current_positions)}")
        print(f"{'=' * 70}")
        logger.info(
            f"Strategy run complete. Processed {len(data)} symbols, currently have {len(all_positions)} positions"
        )
        logger.info(
            f"Data retention: {self.retention_days} days, cutoff: {self.cutoff_date.strftime('%Y-%m-%d')}"
        )
        logger.info(
            f"M/E rebalancing: {'ENABLED' if self.me_rebalancing_enabled else 'DISABLED'}"
        )
        logger.info(f"Final M/E ratio: {final_me:.2f}%")
        logger.info(
            f"L/S ratio adjustments: {'ENABLED' if self.ls_ratio_enabled else 'DISABLED'}"
        )
        logger.info(
            f"Note: Sector management DISABLED - using M/E ratio control instead"
        )
        return results

    def backfill_symbol(self, symbol: str, data: pd.DataFrame) -> None:
        if data is not None and not data.empty:
            df = data.reset_index().rename(
                columns={
                    "timestamp": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            df_with_indicators = self.calculate_indicators(df)
            if df_with_indicators is not None:
                from data_manager import save_price_data

                save_price_data(symbol, df_with_indicators)
                logger.info(f"Backfilled and saved indicators for {symbol}")
        else:
            logger.warning(f"No data to backfill for {symbol}")


def load_polygon_data(
    symbols: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=RETENTION_DAYS + 30)).strftime(
            "%Y-%m-%d"
        )
    data: Dict[str, pd.DataFrame] = {}
    logger.info(f"Loading data for {len(symbols)} symbols")
    logger.info(f"Note: data_manager automatically filters to 6-month retention period")
    batch_size = 50
    total_batches = (len(symbols) + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(symbols))
        batch_symbols = symbols[start_idx:end_idx]
        print(
            f"\nLoading batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)..."
        )
        for i, symbol in enumerate(batch_symbols):
            try:
                df = load_price_data(symbol)
                if df is not None and not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    if len(df) >= 20:
                        data[symbol] = df
                        if (start_idx + i + 1) % 10 == 0:
                            logger.info(
                                f"Progress: {start_idx + i + 1}/{len(symbols)} symbols loaded"
                            )
                    else:
                        logger.warning(
                            f"Insufficient data for {symbol}: only {len(df)} rows"
                        )
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
    logger.info(
        f"\nCompleted loading data. Successfully loaded {len(data)} out of {len(symbols)} symbols"
    )
    return data


def run_ngs_automated_reporting(comparison: Optional[Any] = None) -> None:
    import os

    import pandas as pd

    from ngs_ai_integration_manager import NGSAIIntegrationManager

    HISTORICAL_DATA_PATH = r"c:\ACT\Python NEW 2025\signal_analysis.json"
    DATA_FORMAT = "json"

    def load_data(path: str, data_format: str) -> Dict[str, pd.DataFrame]:
        if data_format == "json":
            df = pd.read_json(path)
            if "symbol" in df.columns:
                data_dict = {
                    sym: df[df["symbol"] == sym].copy() for sym in df["symbol"].unique()
                }
                return data_dict
            return {"default": df}
        elif data_format == "csv":
            df = pd.read_csv(path)
            if "symbol" in df.columns:
                data_dict = {
                    sym: df[df["symbol"] == sym].copy() for sym in df["symbol"].unique()
                }
                return data_dict
            return {"default": df}
        else:
            raise ValueError("Unsupported data format")

    print("🚀 nGS Trading Strategy with AI SELECTION ENABLED")
    print("=" * 70)

    # 1. Load your historical/back data (change path if needed)
    data = load_data(HISTORICAL_DATA_PATH, DATA_FORMAT)

    # 2. Initialize integration manager
    manager = NGSAIIntegrationManager(account_size=1_000_000, data_dir="data")

    # 3. Run AI integration manager on your data
    results = manager.run_integrated_strategy(data)

    # 4. Save results for dashboard
    manager.save_integration_session(results, filename="latest_results.json")
    print("\n✅ AI integration complete. Results saved for dashboard.")

    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = load_polygon_data(symbols)
    strategy = NGSStrategy(account_size=1_000_000)
    results = strategy.run(data)
    trade_history_path = "data/trade_history.csv"
    if os.path.exists(trade_history_path):
        prior_trades = pd.read_csv(trade_history_path)
    else:
        prior_trades = pd.DataFrame()
    new_trades_df = pd.DataFrame(
        [
            {
                "symbol": trade["symbol"],
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "profit_loss": trade["profit"],
            }
            for trade in strategy.trades
        ]
    )
    all_trades_df = pd.concat([prior_trades, new_trades_df], ignore_index=True)
    all_trades_df = all_trades_df.drop_duplicates(
        subset=["symbol", "entry_date", "exit_date"]
    )
    all_trades_df.to_csv(trade_history_path, index=False)
    print(" Trades exported for Streamlit dashboard (no summary stats).")


if __name__ == "__main__":
    from ngs_ai_integration_manager import NGSAIIntegrationManager
    
    print(" nGS Trading Strategy with AI SELECTION ENABLED")
    print("=" * 70)
    print(f"Data Retention: {RETENTION_DAYS} days (6 months)")
    print("=" * 70)

    try:
        print("\n Initializing AI Strategy Selection System...")
        AI_AVAILABLE = True
        print(" AI modules imported successfully")
        ai_integration_manager = NGSAIIntegrationManager(
            account_size=1000000, data_dir="data"
        )

        sp500_file = os.path.join("data", "sp500_symbols.txt")
        try:
            with open(sp500_file, "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
            print(f" Loaded {len(symbols)} S&P 500 symbols")
        except FileNotFoundError:
            print(f"  {sp500_file} not found. Using sample symbols.")
            symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "TSLA",
                "AMZN",
                "META",
                "NVDA",
                "JPM",
                "JNJ",
                "PG",
                "UNH",
                "HD",
                "BAC",
                "XOM",
                "CVX",
                "PFE",
            ]

        print(f" Loading market data for {len(symbols)} symbols...")
        data = load_polygon_data(symbols)

        if not data:
            print(" No data loaded - check your data files")
            exit(1)

        print(f" Successfully loaded data for {len(data)} symbols")

        if AI_AVAILABLE:
            print(f"\n AI ANALYZING STRATEGY OPTIONS...")
            ai_objectives = ["linear_equity", "max_roi", "min_drawdown", "high_winrate"]

            print(f" Testing {len(ai_objectives)} AI strategy objectives:")
            for obj in ai_objectives:
                print(f"   • {obj}")

            try:
                print(f"\n Running comprehensive performance analysis...")
                performance_comparator = NGSAIPerformanceComparator()
                comparison_results = performance_comparator.comprehensive_comparison(
                    data=data, objectives=ai_objectives
                )

                ai_score = comparison_results.ai_recommendation_score
                best_strategy = comparison_results.best_overall_strategy
                recommended_allocation = comparison_results.recommended_allocation

                print(f"\n AI STRATEGY SELECTION RESULTS")
                print("=" * 50)
                print(f"AI Recommendation Score: {ai_score:.0f}/100")
                print(f"Best Overall Strategy:   {best_strategy}")
                print(
                    f"Statistical Significance: {'YES' if comparison_results.return_difference_significant else 'NO'}"
                )

                original_performance = comparison_results.original_metrics
                best_ai_performance = max(
                    comparison_results.ai_metrics, key=lambda x: x.total_return_pct
                )

                print(f"\n PERFORMANCE COMPARISON:")
                print(
                    f"Original nGS:     {original_performance.total_return_pct:+7.2f}% return, {original_performance.max_drawdown_pct:7.2f}% drawdown"
                )
                print(
                    f"Best AI Strategy: {best_ai_performance.total_return_pct:+7.2f}% return, {best_ai_performance.max_drawdown_pct:7.2f}% drawdown"
                )

                print(f"\n AI DECISION:")
                print(" AI RECOMMENDS: AI-Focused Strategy")
                print(f"   Reason: Default AI analysis engaged")
                ai_integration_manager.set_operating_mode("ai_only")

                print(f"\n RECOMMENDED ALLOCATION:")
                for strategy_name, allocation_pct in recommended_allocation.items():
                    print(f"   {strategy_name}: {allocation_pct:.1f}%")

                print(f"\n Executing AI-selected strategy...")
                results = ai_integration_manager.run_integrated_strategy(data)
                print(f" AI-powered strategy execution completed!")
                print(f"Mode: AI-ONLY")

            except Exception as e:
                print(f" AI analysis failed: {e}")
                exit(1)
            finally:
                print("Execution attempt completed.")
                print(f"\n{'=' * 70}")
                print("STRATEGY BACKTEST RESULTS (Last 6 Months)")
                print(f"{'=' * 70}")

                # Assume strategy is available in current scope
                # (if not, adapt to your main strategy object)
                strategy = NGSStrategy(account_size=1000000)
                total_profit = sum(trade["profit"] for trade in strategy.trades)
                winning_trades = sum(
                    1 for trade in strategy.trades if trade["profit"] > 0
                )

                print(f"Starting capital:     ${strategy.account_size:,.2f}")
                print(f"Ending cash:          ${strategy.cash:,.2f}")
                print(f"Total P&L:            ${total_profit:,.2f}")
                print(
                    f"Return:               {((strategy.cash - strategy.account_size) / strategy.account_size * 100):+.2f}%"
                )
                print(f"Total trades:         {len(strategy.trades)}")

                if strategy.trades:
                    print(
                        f"Winning trades:       {winning_trades}/{len(strategy.trades)} ({winning_trades / len(strategy.trades) * 100:.1f}%)"
                    )
                print(f" Original nGS strategy execution completed!")

    except Exception as e:
        print(f" Execution failed: {e}")
        import traceback

        traceback.print_exc()

    if __name__ == "__main__":
        # Example inputs for testing
        print("Testing `load_polygon_data`...")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        start_date = "2023-01-01"
        end_date = "2023-06-30"

        try:
            result = load_polygon_data(symbols, start_date, end_date)
            print(f"Loaded data for symbols: {result.keys()}")
        except Exception as e:
            print(f"Error in load_polygon_data: {e}")

        print("Testing `run_ngs_automated_reporting`...")
        comparison = {"example_key": "example_value"}  # Replace with actual inputs
        try:
            run_ngs_automated_reporting(
                comparison
            )  # Ensure comparison is structured correctly
            print("Successfully ran automated reporting.")
        except Exception as e:
            print(f"Error in run_ngs_automated_reporting: {e}")
