"""
Streamlined nGS AI Backtesting System
Focused on AI-generated strategy backtesting without comparisons or options.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# Import necessary components
from data_utils import load_polygon_data
from strategy_generator_ai import TradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for individual backtest results"""

    strategy_id: str
    objective_name: str
    start_date: str
    end_date: str
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    avg_trade_pct: float
    sharpe_ratio: float
    volatility_pct: float
    best_trade: float
    worst_trade: float
    avg_duration_days: float
    equity_curve: pd.Series
    trades: List[Dict]


class NGSAIBacktestingSystem:
    """
    Streamlined backtesting system for AI-generated strategies
    """

    def __init__(self, account_size: float = 1000000, data_dir: str = "data"):
        self.account_size = account_size
        self.data_dir = data_dir
        self.results_dir = os.path.join(data_dir, "backtest_results")
        os.makedirs(self.results_dir, exist_ok=True)

        print(" Streamlined nGS AI Backtesting System initialized")
        print(f"   Account Size:        ${account_size:,.0f}")
        print(f"   Results Directory:   {self.results_dir}")

    def backtest_ai_strategy(
        self,
        strategy: TradingStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None,
    ) -> BacktestResult:
        """
        Backtest a single AI strategy against historical data
        Returns comprehensive performance metrics
        """
        print(f"\n Backtesting AI Strategy: {strategy.objective_name}")
        print(f"   Strategy ID: {strategy.strategy_id}")

        # Filter data by date range if specified
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)

        # Execute strategy on historical data
        try:
            results = strategy.execute_on_data(data)

            # Apply trading costs
            adjusted_trades = self._apply_trading_costs(results["trades"])

            # Calculate comprehensive metrics
            backtest_result = self._calculate_comprehensive_metrics(
                strategy, adjusted_trades, results["equity_curve"], start_date, end_date
            )

            print(f"    Backtest completed: {backtest_result.total_trades} trades")
            print(f"    Total Return: {backtest_result.total_return_pct:.2f}%")
            print(f"    Max Drawdown: {backtest_result.max_drawdown_pct:.2f}%")
            print(f"    Win Rate: {backtest_result.win_rate:.1%}")

            return backtest_result

        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy.strategy_id}: {e}")
            raise

    def _apply_trading_costs(self, trades: List[Dict]) -> List[Dict]:
        """Apply commission and slippage to trades"""
        adjusted_trades = []

        for trade in trades:
            adjusted_trade = trade.copy()

            # Apply commission (reduces profit)
            commission = 1.0  # $1 per trade
            adjusted_trade["profit"] = trade["profit"] - commission

            # Apply slippage (reduces profit)
            entry_price = trade["entry_price"]
            slippage_cost = entry_price * (0.05 / 100)  # 0.05% slippage
            shares = trade.get("shares", 1)
            total_slippage = slippage_cost * shares

            adjusted_trade["profit"] -= total_slippage
            adjusted_trade["commission"] = commission
            adjusted_trade["slippage"] = total_slippage

            adjusted_trades.append(adjusted_trade)

        return adjusted_trades

    def _calculate_comprehensive_metrics(
        self,
        strategy: TradingStrategy,
        trades: List[Dict],
        equity_curve: pd.Series,
        start_date: str = None,
        end_date: str = None,
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics for a strategy"""

        if not trades:
            # Return empty result if no trades
            return BacktestResult(
                strategy_id=strategy.strategy_id,
                objective_name=strategy.objective_name,
                start_date=start_date or "N/A",
                end_date=end_date or "N/A",
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_pct=0.0,
                sharpe_ratio=0.0,
                volatility_pct=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                avg_duration_days=0.0,
                equity_curve=pd.Series([self.account_size]),
                trades=[],
            )

        # Basic trade statistics
        total_trades = len(trades)
        profits = [trade["profit"] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Return calculations
        total_profit = sum(profits)
        total_return_pct = (total_profit / self.account_size) * 100
        avg_trade_pct = (np.mean(profits) / self.account_size) * 100 if profits else 0

        # Risk calculations
        daily_returns = equity_curve.pct_change().dropna()
        max_drawdown_pct = self._calculate_max_drawdown(equity_curve)
        volatility_pct = (
            daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        )

        # Sharpe ratio
        excess_returns = daily_returns - (0.02 / 252)  # 2% annual risk-free rate
        sharpe_ratio = (
            (excess_returns.mean() / excess_returns.std() * np.sqrt(252))
            if excess_returns.std() > 0
            else 0
        )

        # Trade statistics
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0

        # Duration analysis
        durations = []
        for trade in trades:
            if "entry_date" in trade and "exit_date" in trade:
                try:
                    entry_dt = pd.to_datetime(trade["entry_date"])
                    exit_dt = pd.to_datetime(trade["exit_date"])
                    duration = (exit_dt - entry_dt).days
                    durations.append(duration)
                except:
                    pass

        avg_duration_days = np.mean(durations) if durations else 0

        return BacktestResult(
            strategy_id=strategy.strategy_id,
            objective_name=strategy.objective_name,
            start_date=(
                start_date or equity_curve.index[0].strftime("%Y-%m-%d")
                if not equity_curve.empty
                else "N/A"
            ),
            end_date=(
                end_date or equity_curve.index[-1].strftime("%Y-%m-%d")
                if not equity_curve.empty
                else "N/A"
            ),
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_pct=avg_trade_pct,
            sharpe_ratio=sharpe_ratio,
            volatility_pct=volatility_pct,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_duration_days=avg_duration_days,
            equity_curve=equity_curve,
            trades=trades,
        )

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        if equity_curve.empty:
            return 0.0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100

        return max_drawdown

    def _filter_data_by_date(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """Filter data dictionary by date range"""
        if not start_date and not end_date:
            return data

        filtered_data = {}

        for symbol, df in data.items():
            if df.empty:
                continue

            df_copy = df.copy()
            df_copy["Date"] = pd.to_datetime(df_copy["Date"])

            if start_date:
                df_copy = df_copy[df_copy["Date"] >= pd.to_datetime(start_date)]

            if end_date:
                df_copy = df_copy[df_copy["Date"] <= pd.to_datetime(end_date)]

            if not df_copy.empty:
                filtered_data[symbol] = df_copy
<<<<<<< HEAD
=======

>>>>>>> c108ef4 (Bypass pre-commit for now)
        return filtered_data


