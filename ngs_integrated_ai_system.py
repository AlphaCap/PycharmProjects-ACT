"""
nGS AI Backtesting System (FIXED IMPORTS)
Comprehensive backtesting framework for AI-generated strategies using YOUR nGS parameters
Tests multiple strategies, timeframes, and objectives with detailed performance analysis
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore

warnings.filterwarnings("ignore")

from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from ngs_ai_integration_manager import NGSAIIntegrationManager
from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
from performance_objectives import ObjectiveManager

# Import your existing components
from shared_utils import load_polygon_data
from strategy_generator_ai import ObjectiveAwareStrategyGenerator, TradingStrategy

# Create aliases to match expected names
NGSAwareStrategyGenerator = ObjectiveAwareStrategyGenerator
NGSIndicatorLibrary = ComprehensiveIndicatorLibrary

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
    fitness_score: float
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    daily_returns: pd.Series

@dataclass
class BacktestComparison:
    """Container for comparing multiple backtest results"""
    original_ngs_result: BacktestResult
    ai_results: List[BacktestResult]
    comparison_metrics: Dict[str, Any]
    recommendation: str
    summary_stats: Dict[str, Any]

class NGSAIBacktestingSystem:
    """
    Comprehensive backtesting system for AI-generated strategies
    Tests strategies against historical data with YOUR proven nGS parameters
    """
    def __init__(self, account_size: float = 1000000, data_dir: str = "data", production_mode: bool = True) -> None:
        self.account_size: float = account_size
        self.data_dir: str = data_dir
        self.production_mode: bool = production_mode  # Production mode bypasses brokerage costs
        self.results_dir: str = os.path.join(data_dir, "backtest_results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.integration_manager: NGSAIIntegrationManager = NGSAIIntegrationManager(account_size, data_dir, production_mode)
        self.ngs_indicator_lib: ComprehensiveIndicatorLibrary = ComprehensiveIndicatorLibrary()
        self.objective_manager: ObjectiveManager = ObjectiveManager()
        self.ai_generator: ObjectiveAwareStrategyGenerator = ObjectiveAwareStrategyGenerator(
            self.ngs_indicator_lib, self.objective_manager
        )
        self.backtest_config: Dict[str, Any] = {
            "commission_per_trade": 0.0 if production_mode else 1.0,
            "slippage_pct": 0.0 if production_mode else 0.05,
            "min_trade_spacing": 1,
            "max_concurrent_positions": 20,
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02,
            "lookback_periods": [30, 60, 90, 180, 252],
            "production_mode": production_mode,
        }
        self.backtest_history: List[BacktestResult] = []
        self.strategy_rankings: Dict[str, Any] = {}
        
        # Enhanced logging setup
        logger.info("nGS AI Backtesting System initialized")
        logger.info(f"Account Size: ${account_size:,.0f}")
        logger.info(f"Results Directory: {self.results_dir}")
        logger.info(f"Production Mode: {production_mode}")
        
        print(" nGS AI Backtesting System initialized")
        print(f"   Account Size:        ${account_size:,.0f}")
        print(f"   Results Directory:   {self.results_dir}")
        print(f"   Production Mode:     {'ENABLED' if production_mode else 'DISABLED'}")
        
        if not production_mode:
            print(f"   Commission:          ${self.backtest_config['commission_per_trade']:.2f} per trade")
            print(f"   Slippage:            {self.backtest_config['slippage_pct']:.2f}%")
        else:
            print(f"   Trading Costs:       BYPASSED (Production Mode)")
            logger.info("Trading costs (commission/slippage) bypassed for production mode")
    # =============================================================================
    # SINGLE STRATEGY BACKTESTING
    # =============================================================================

    def backtest_ai_strategy(
        self,
        strategy: TradingStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Backtest a single AI strategy against historical data
        Returns comprehensive performance metrics
        """
        # Validate input strategy
        if not strategy:
            logger.error("No strategy provided for backtesting")
            raise ValueError("Strategy cannot be None")
        
        logger.info(f"Starting backtest for AI strategy: {strategy.objective_name}")
        logger.info(f"Strategy ID: {strategy.strategy_id}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        print(f"\n Backtesting AI Strategy: {strategy.objective_name}")
        print(f"   Strategy ID: {strategy.strategy_id}")

        # Validate input data
        if not data:
            logger.error("No data provided for backtesting")
            raise ValueError("Data dictionary cannot be empty")

        # Filter data by date range if specified
        if start_date or end_date:
            logger.debug(f"Filtering data by date range: {start_date} to {end_date}")
            data = self._filter_data_by_date(data, start_date, end_date)
            
            # Validate filtered data
            if not data:
                logger.error("No data remaining after date filtering")
                raise ValueError(f"No data available for date range {start_date} to {end_date}")

        # Execute strategy on historical data
        try:
            logger.debug("Executing strategy on historical data")
            
            # Some implementations expect a DataFrame, others a dict; handle both
            try:
                # Try full dict (preferred)
                logger.debug("Attempting strategy execution with full data dictionary")
                results = strategy.execute_on_data(data, self.ngs_indicator_lib)
            except TypeError:
                # Try per-symbol (legacy)
                logger.debug("Falling back to per-symbol execution")
                first_df = next(iter(data.values()))
                if first_df.empty:
                    logger.error("First dataframe is empty")
                    raise ValueError("No valid data in first symbol dataframe")
                results = strategy.execute_on_data(first_df, self.ngs_indicator_lib)

            # Validate strategy results
            if not isinstance(results, dict):
                logger.error(f"Strategy returned invalid results type: {type(results)}")
                raise ValueError("Strategy must return a dictionary with 'trades' and 'equity_curve'")
            
            if "trades" not in results or "equity_curve" not in results:
                logger.error(f"Strategy results missing required keys: {list(results.keys())}")
                raise ValueError("Strategy results must contain 'trades' and 'equity_curve' keys")

            trades = results["trades"]
            equity_curve = results["equity_curve"]
            
            logger.info(f"Strategy execution completed: {len(trades)} trades generated")

            # Apply trading costs (bypassed in production mode)
            adjusted_trades = self._apply_trading_costs(trades)

            # Calculate comprehensive metrics
            logger.debug("Calculating comprehensive performance metrics")
            backtest_result = self._calculate_comprehensive_metrics(
                strategy, adjusted_trades, equity_curve, start_date, end_date
            )

            # Log results
            logger.info(f"Backtest completed successfully")
            logger.info(f"Total trades: {backtest_result.total_trades}")
            logger.info(f"Total return: {backtest_result.total_return_pct:.2f}%")
            logger.info(f"Max drawdown: {backtest_result.max_drawdown_pct:.2f}%")
            logger.info(f"Win rate: {backtest_result.win_rate:.1%}")
            logger.info(f"Sharpe ratio: {backtest_result.sharpe_ratio:.3f}")

            print(f"    Backtest completed: {backtest_result.total_trades} trades")
            print(f"    Total Return: {backtest_result.total_return_pct:.2f}%")
            print(f"    Max Drawdown: {backtest_result.max_drawdown_pct:.2f}%")
            print(f"    Win Rate: {backtest_result.win_rate:.1%}")

            return backtest_result

        except Exception as e:
            logger.error(f"Error backtesting strategy {strategy.strategy_id}: {e}", exc_info=True)
            raise

    def backtest_original_ngs(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Backtest your original nGS strategy for comparison
        """
        from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
        
        logger.info(f"Starting backtest for original nGS strategy")
        logger.info(f"Date range: {start_date} to {end_date}")
        print(f"\n Backtesting Original nGS Strategy")

        # Validate input data
        if not data:
            logger.error("No data provided for original nGS backtesting")
            raise ValueError("Data dictionary cannot be empty")

        # Filter data by date range if specified
        if start_date or end_date:
            logger.debug(f"Filtering data by date range: {start_date} to {end_date}")
            data = self._filter_data_by_date(data, start_date, end_date)
            
            if not data:
                logger.error("No data remaining after date filtering for original nGS")
                raise ValueError(f"No data available for date range {start_date} to {end_date}")

        # Create fresh nGS instance for backtesting
        try:
            logger.debug("Initializing original nGS strategy")
            ngs_strategy = NGSStrategy(
                account_size=self.account_size, data_dir=self.data_dir
            )
        except Exception as e:
            logger.error(f"Failed to initialize original nGS strategy: {e}")
            raise

        try:
            # Run original strategy
            logger.debug("Running original nGS strategy")
            ngs_strategy.run(data)

            # Validate strategy results
            if not hasattr(ngs_strategy, 'trades'):
                logger.error("Original nGS strategy did not generate trades attribute")
                raise ValueError("Original nGS strategy must have 'trades' attribute after execution")

            trades = ngs_strategy.trades
            logger.info(f"Original nGS execution completed: {len(trades)} trades generated")

            # Apply trading costs (bypassed in production mode)
            adjusted_trades = self._apply_trading_costs(trades)

            # Create equity curve from trades
            logger.debug("Creating equity curve from trades")
            equity_curve = self._create_equity_curve_from_trades(
                adjusted_trades, self.account_size
            )

            # Calculate metrics
            logger.debug("Calculating comprehensive metrics for original nGS")
            backtest_result = self._calculate_comprehensive_metrics_from_trades(
                "original_ngs",
                "original",
                adjusted_trades,
                equity_curve,
                start_date,
                end_date,
            )

            # Log results
            logger.info(f"Original nGS backtest completed successfully")
            logger.info(f"Total trades: {backtest_result.total_trades}")
            logger.info(f"Total return: {backtest_result.total_return_pct:.2f}%")
            logger.info(f"Max drawdown: {backtest_result.max_drawdown_pct:.2f}%")
            logger.info(f"Win rate: {backtest_result.win_rate:.1%}")

            print(
                f"    Original nGS backtest completed: {backtest_result.total_trades} trades"
            )
            print(f"    Total Return: {backtest_result.total_return_pct:.2f}%")
            print(f"    Max Drawdown: {backtest_result.max_drawdown_pct:.2f}%")
            print(f"    Win Rate: {backtest_result.win_rate:.1%}")

            return backtest_result

        except Exception as e:
            logger.error(f"Error backtesting original nGS: {e}", exc_info=True)
            raise

    # =============================================================================
    # MULTI-STRATEGY BACKTESTING
    # =============================================================================

    def backtest_multiple_objectives(
        self,
        objectives: List[str],
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[BacktestResult]:
        """
        Backtest AI strategies for multiple objectives and compare performance
        """
        print(f"\n Multi-Objective Backtesting: {len(objectives)} objectives")
        print(f"   Objectives: {', '.join(objectives)}")

        results: List[BacktestResult] = []

        for objective in objectives:
            try:
                print(f"\n Creating and testing {objective} strategy...")

                # Generate AI strategy for objective
                strategy = self.ai_generator.generate_strategy_for_objective(objective)

                # Backtest the strategy
                backtest_result = self.backtest_ai_strategy(
                    strategy, data, start_date, end_date
                )
                results.append(backtest_result)

            except Exception as e:
                print(f"    Failed to backtest {objective}: {e}")
                logger.error(f"Failed to backtest objective {objective}: {e}")

        # Rank strategies by multiple metrics
        self._rank_strategies(results)

        print(
            f"\n Multi-objective backtesting completed: {len(results)} strategies tested"
        )
        return results

    def backtest_comprehensive_comparison(
        self,
        objectives: List[str],
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestComparison:
        """
        Comprehensive comparison between original nGS and multiple AI strategies
        """
        print(f"\n COMPREHENSIVE STRATEGY COMPARISON")
        print(f"   Original nGS vs {len(objectives)} AI objectives")

        # Backtest original nGS
        original_result: BacktestResult = self.backtest_original_ngs(data, start_date, end_date)

        # Backtest AI strategies
        ai_results: List[BacktestResult] = self.backtest_multiple_objectives(
            objectives, data, start_date, end_date
        )

        # Generate comparison analysis
        comparison: BacktestComparison = self._generate_comprehensive_comparison(
            original_result, ai_results
        )

        # Save comparison results
        self._save_comparison_results(comparison)

        return comparison
    # =============================================================================
    # WALK-FORWARD ANALYSIS
    # =============================================================================

    def walk_forward_analysis(
        self,
        strategy: TradingStrategy,
        data: Dict[str, pd.DataFrame],
        training_months: int = 6,
        testing_months: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis to test strategy robustness over time
        """
        print(f"\n Walk-Forward Analysis")
        print(f"   Training Period: {training_months} months")
        print(f"   Testing Period: {testing_months} months")

        # Get date range from data
        all_dates: List[datetime] = []
        for symbol_data in data.values():
            if not symbol_data.empty:
                all_dates.extend(pd.to_datetime(symbol_data["Date"]).tolist())

        start_date: datetime = min(all_dates)
        end_date: datetime = max(all_dates)

        # Create walk-forward windows
        windows: List[Tuple[datetime, datetime, datetime, datetime]] = self._create_walk_forward_windows(
            start_date, end_date, training_months, testing_months
        )

        walk_forward_results: List[Dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(
                f"\n   Window {i+1}/{len(windows)}: Train {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}, "
                f"Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"
            )

            try:
                # Filter data for training period (could be used for parameter optimization)
                train_data = self._filter_data_by_date(
                    data,
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                )

                # Filter data for testing period
                test_data = self._filter_data_by_date(
                    data, test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")
                )

                if not test_data:
                    print(f"      No data for testing period, skipping window")
                    continue

                # Backtest on testing data
                window_result = self.backtest_ai_strategy(
                    strategy,
                    test_data,
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                )

                walk_forward_results.append(
                    {
                        "window": i + 1,
                        "train_period": f"{train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}",
                        "test_period": f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
                        "result": window_result,
                    }
                )

                print(
                    f"      Return: {window_result.total_return_pct:.2f}%, Trades: {window_result.total_trades}"
                )

            except Exception as e:
                print(f"      Error in window {i+1}: {e}")
                logger.error(f"Walk-forward window {i+1} error: {e}")

        # Analyze walk-forward results
        wf_analysis = self._analyze_walk_forward_results(walk_forward_results)

        print(
            f"\n Walk-Forward Analysis Complete: {len(walk_forward_results)} windows"
        )
        print(f"   Average Return: {wf_analysis['avg_return']:.2f}%")
        print(f"   Return Std Dev: {wf_analysis['return_std']:.2f}%")
        print(
            f"   Positive Windows: {wf_analysis['positive_windows']}/{len(walk_forward_results)}"
        )

        return {
            "windows": walk_forward_results,
            "analysis": wf_analysis,
            "strategy_id": strategy.strategy_id,
            "parameters": {
                "training_months": training_months,
                "testing_months": testing_months,
            },
        }

    # =============================================================================
    # PERFORMANCE ANALYSIS METHODS
    # =============================================================================

    def _apply_trading_costs(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply commission and slippage to trades (bypassed in production mode)"""
        logger.debug(f"Processing {len(trades)} trades for cost application")
        
        # Validate input
        if not trades:
            logger.warning("No trades provided to _apply_trading_costs")
            return []
        
        # Production mode bypasses all trading costs
        if self.production_mode:
            logger.debug("Production mode: bypassing all trading costs")
            adjusted_trades = []
            for trade in trades:
                adjusted_trade = trade.copy()
                adjusted_trade["commission"] = 0.0
                adjusted_trade["slippage"] = 0.0
                # Profit remains unchanged in production mode
                adjusted_trades.append(adjusted_trade)
            return adjusted_trades
        
        # Non-production mode: apply trading costs
        logger.debug("Non-production mode: applying commission and slippage")
        adjusted_trades: List[Dict[str, Any]] = []

        for trade in trades:
            try:
                adjusted_trade = trade.copy()

                # Apply commission (reduces profit)
                commission = self.backtest_config['commission_per_trade']
                adjusted_trade['profit'] = trade['profit'] - commission

                # Apply slippage (reduces profit)
                entry_price = trade.get("entry_price", 0)
                if entry_price <= 0:
                    logger.warning(f"Invalid entry_price {entry_price} in trade, skipping slippage calculation")
                    total_slippage = 0.0
                else:
                    slippage_cost = entry_price * (self.backtest_config["slippage_pct"] / 100)
                    shares = trade.get("shares", 1)
                    total_slippage = slippage_cost * shares

                adjusted_trade["profit"] -= total_slippage
                adjusted_trade["commission"] = commission
                adjusted_trade["slippage"] = total_slippage

                adjusted_trades.append(adjusted_trade)
                
            except (KeyError, TypeError, ValueError) as e:
                logger.error(f"Error processing trade {trade}: {e}")
                # Include the trade with zero costs if there's an error
                adjusted_trade = trade.copy()
                adjusted_trade["commission"] = 0.0
                adjusted_trade["slippage"] = 0.0
                adjusted_trades.append(adjusted_trade)

        logger.debug(f"Applied trading costs to {len(adjusted_trades)} trades")
        return adjusted_trades

    def _calculate_comprehensive_metrics(
        self,
        strategy: TradingStrategy,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics for a strategy"""
        logger.debug(f"Calculating metrics for {len(trades)} trades")
        
        # Validate inputs
        if not isinstance(trades, list):
            logger.error(f"Invalid trades type: {type(trades)}")
            trades = []
        
        if not isinstance(equity_curve, pd.Series):
            logger.warning(f"Invalid equity_curve type: {type(equity_curve)}, creating empty series")
            equity_curve = pd.Series([self.account_size])
        
        if equity_curve.empty:
            logger.warning("Empty equity curve, using account size as single value")
            equity_curve = pd.Series([self.account_size])
        
        if not trades:
            # Return empty result if no trades
            logger.warning("No trades available for metrics calculation")
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
                fitness_score=0.0,
                equity_curve=equity_curve,
                trades=[],
                daily_returns=pd.Series([]),
            )

        try:
            # Basic trade statistics
            total_trades = len(trades)
            profits = []
            
            # Safely extract profits with validation
            for trade in trades:
                try:
                    profit = float(trade.get("profit", 0))
                    profits.append(profit)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid profit value in trade: {trade.get('profit')} - {e}")
                    profits.append(0.0)
            
            winning_trades = [p for p in profits if p > 0]
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

            # Return calculations
            total_profit = float(sum(profits))
            total_return_pct = (total_profit / float(self.account_size)) * 100
            avg_trade_pct = (np.mean(profits) / float(self.account_size)) * 100 if profits else 0.0

            # Risk calculations
            try:
                daily_returns = equity_curve.pct_change().dropna()
                if daily_returns.empty or len(daily_returns) <= 1:
                    logger.warning("Insufficient data for volatility calculation")
                    volatility_pct = 0.0
                    sharpe_ratio = 0.0
                else:
                    volatility_pct = float(daily_returns.std()) * np.sqrt(252) * 100
                    
                    # Sharpe ratio
                    excess_returns = daily_returns - (self.backtest_config["risk_free_rate"] / 252)
                    sharpe_ratio = (
                        (float(excess_returns.mean()) / float(excess_returns.std()) * np.sqrt(252))
                        if excess_returns.std() > 0 else 0.0
                    )
            except Exception as e:
                logger.warning(f"Error calculating risk metrics: {e}")
                daily_returns = pd.Series([])
                volatility_pct = 0.0
                sharpe_ratio = 0.0
            
            max_drawdown_pct = self._calculate_max_drawdown(equity_curve)

            # Trade statistics
            best_trade = max(profits) if profits else 0.0
            worst_trade = min(profits) if profits else 0.0

            # Duration analysis
            durations = []
            for trade in trades:
                if "entry_date" in trade and "exit_date" in trade:
                    try:
                        entry_dt = pd.to_datetime(trade["entry_date"])
                        exit_dt = pd.to_datetime(trade["exit_date"])
                        duration = (exit_dt - entry_dt).days
                        durations.append(max(duration, 0))  # Ensure non-negative
                    except Exception as e:
                        logger.debug(f"Could not calculate duration for trade: {e}")

            avg_duration_days = float(np.mean(durations)) if durations else 0.0

            # Calculate fitness score using strategy's objective
            try:
                objective = self.objective_manager.get_objective(strategy.objective_name)
                fitness_score = float(objective.calculate_fitness(trades, equity_curve))
            except Exception as e:
                logger.warning(f"Could not calculate fitness score: {e}")
                fitness_score = total_return_pct - max_drawdown_pct  # Fallback calculation

            logger.debug("Metrics calculation completed successfully")
            
            return BacktestResult(
                strategy_id=strategy.strategy_id,
                objective_name=strategy.objective_name,
                start_date=(
                    start_date or (
                        equity_curve.index[0].strftime("%Y-%m-%d")
                        if not equity_curve.empty and hasattr(equity_curve.index[0], "strftime")
                        else "N/A"
                    )
                ),
                end_date=(
                    end_date or (
                        equity_curve.index[-1].strftime("%Y-%m-%d")
                        if not equity_curve.empty and hasattr(equity_curve.index[-1], "strftime")
                        else "N/A"
                    )
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
                fitness_score=fitness_score,
                equity_curve=equity_curve,
                trades=trades,
                daily_returns=daily_returns if 'daily_returns' in locals() else pd.Series([]),
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive metrics calculation: {e}", exc_info=True)
            # Return basic result on error
            return BacktestResult(
                strategy_id=strategy.strategy_id,
                objective_name=strategy.objective_name,
                start_date=start_date or "N/A",
                end_date=end_date or "N/A",
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                total_trades=len(trades),
                avg_trade_pct=0.0,
                sharpe_ratio=0.0,
                volatility_pct=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                avg_duration_days=0.0,
                fitness_score=0.0,
                equity_curve=equity_curve,
                trades=trades,
                daily_returns=pd.Series([]),
            )
    def _calculate_comprehensive_metrics_from_trades(
        self,
        strategy_id: str,
        objective_name: str,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """Calculate metrics when we only have trades and equity curve (for original nGS)"""

        if not trades:
            return BacktestResult(
                strategy_id=strategy_id,
                objective_name=objective_name,
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
                fitness_score=0.0,
                equity_curve=pd.Series([self.account_size]),
                trades=[],
                daily_returns=pd.Series([]),
            )

        total_trades = len(trades)
        profits = [float(trade["profit"]) for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        total_profit = float(sum(profits))
        total_return_pct = (total_profit / float(self.account_size)) * 100
        avg_trade_pct = (np.mean(profits) / float(self.account_size)) * 100 if profits else 0.0

        daily_returns = equity_curve.pct_change().dropna()
        max_drawdown_pct = self._calculate_max_drawdown(equity_curve)
        volatility_pct = (
            float(daily_returns.std()) * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0.0
        )

        excess_returns = daily_returns - (self.backtest_config["risk_free_rate"] / 252)
        sharpe_ratio = (
            (float(excess_returns.mean()) / float(excess_returns.std()) * np.sqrt(252))
            if excess_returns.std() > 0
            else 0.0
        )

        best_trade = max(profits) if profits else 0.0
        worst_trade = min(profits) if profits else 0.0

        durations = []
        for trade in trades:
            if "entry_date" in trade and "exit_date" in trade:
                try:
                    entry_dt = pd.to_datetime(trade["entry_date"])
                    exit_dt = pd.to_datetime(trade["exit_date"])
                    duration = (exit_dt - entry_dt).days
                    durations.append(duration)
                except Exception:
                    pass

        avg_duration_days = float(np.mean(durations)) if durations else 0.0

        fitness_score = total_return_pct - max_drawdown_pct

        return BacktestResult(
            strategy_id=strategy_id,
            objective_name=objective_name,
            start_date=(
                start_date or (
                    equity_curve.index[0].strftime("%Y-%m-%d")
                    if not equity_curve.empty and hasattr(equity_curve.index[0], "strftime")
                    else "N/A"
                )
            ),
            end_date=(
                end_date or (
                    equity_curve.index[-1].strftime("%Y-%m-%d")
                    if not equity_curve.empty and hasattr(equity_curve.index[-1], "strftime")
                    else "N/A"
                )
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
            fitness_score=fitness_score,
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns,
        )

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        if equity_curve.empty:
            return 0.0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min()) * 100

        return float(max_drawdown)

    def _create_equity_curve_from_trades(
        self, trades: List[Dict[str, Any]], starting_capital: float
    ) -> pd.Series:
        """Create equity curve from trade list"""
        if not trades:
            return pd.Series([starting_capital])

        # Sort trades by exit date
        sorted_trades = sorted(trades, key=lambda x: x.get("exit_date", ""))

        equity_values = [starting_capital]
        dates = [
            pd.to_datetime(
                sorted_trades[0].get("entry_date", datetime.now().strftime("%Y-%m-%d"))
            )
        ]

        current_equity = starting_capital

        for trade in sorted_trades:
            current_equity += float(trade["profit"])
            equity_values.append(current_equity)
            dates.append(
                pd.to_datetime(
                    trade.get("exit_date", datetime.now().strftime("%Y-%m-%d"))
                )
            )

        return pd.Series(equity_values, index=dates)

    # =============================================================================
    # COMPARISON AND RANKING
    # =============================================================================

    def _generate_comprehensive_comparison(
        self, original_result: BacktestResult, ai_results: List[BacktestResult]
    ) -> BacktestComparison:
        """Generate comprehensive comparison between original and AI strategies"""

        # Combine all results for analysis
        all_results = [original_result] + ai_results

        # Performance comparison matrix
        comparison_metrics: Dict[str, Any] = {
            "returns_comparison": {},
            "risk_comparison": {},
            "efficiency_comparison": {},
            "consistency_comparison": {},
        }

        # Returns comparison
        returns = [r.total_return_pct for r in all_results]
        comparison_metrics["returns_comparison"] = {
            "original_ngs": original_result.total_return_pct,
            "ai_best": (
                max([r.total_return_pct for r in ai_results]) if ai_results else 0.0
            ),
            "ai_worst": (
                min([r.total_return_pct for r in ai_results]) if ai_results else 0.0
            ),
            "ai_average": (
                float(np.mean([r.total_return_pct for r in ai_results])) if ai_results else 0.0
            ),
            "original_rank": sorted(returns, reverse=True).index(
                original_result.total_return_pct
            )
            + 1,
        }

        # Risk comparison
        drawdowns = [r.max_drawdown_pct for r in all_results]
        comparison_metrics["risk_comparison"] = {
            "original_drawdown": original_result.max_drawdown_pct,
            "ai_best_drawdown": (
                min([r.max_drawdown_pct for r in ai_results]) if ai_results else 0.0
            ),
            "ai_worst_drawdown": (
                max([r.max_drawdown_pct for r in ai_results]) if ai_results else 0.0
            ),
            "original_risk_rank": sorted(drawdowns).index(
                original_result.max_drawdown_pct
            )
            + 1,
        }

        # Efficiency comparison (Sharpe ratios)
        sharpes = [r.sharpe_ratio for r in all_results]
        comparison_metrics["efficiency_comparison"] = {
            "original_sharpe": original_result.sharpe_ratio,
            "ai_best_sharpe": (
                max([r.sharpe_ratio for r in ai_results]) if ai_results else 0.0
            ),
            "original_sharpe_rank": sorted(sharpes, reverse=True).index(
                original_result.sharpe_ratio
            )
            + 1,
        }

        # Generate recommendation
        recommendation = self._generate_strategy_recommendation(
            original_result, ai_results, comparison_metrics
        )

        if ai_results and len(ai_results) > 0:
            ai_mean_return = float(np.mean([r.total_return_pct for r in ai_results]))
        else:
            ai_mean_return = 0.0

        summary_stats = {
            "total_strategies_tested": len(all_results),
            "original_vs_ai_winner": (
                "original"
                if original_result.total_return_pct >= ai_mean_return
                else "ai"
            ),
        }
        output = {
            "best_performing_strategy": max(
                strategies, key=lambda x: x.get("performance_metric", 0)
            ),
            "additional_info": some_other_value,
            "lowest_risk_strategy": min(
                all_results, key=lambda x: x.max_drawdown_pct
            ).strategy_id,
            "highest_sharpe_strategy": max(
                all_results, key=lambda x: x.sharpe_ratio
            ).strategy_id,
        }
        return BacktestComparison(
            original_ngs_result=original_result,
            ai_results=ai_results,
            comparison_metrics=comparison_metrics,
            recommendation=recommendation,
            summary_stats=summary_stats,
        )

    def _generate_strategy_recommendation(
        self,
        original: BacktestResult,
        ai_results: List[BacktestResult],
        comparison: Dict[str, Any],
    ) -> str:
        """Generate recommendation based on backtest comparison"""

        if not ai_results:
            return "No AI strategies to compare. Continue using original nGS strategy."

        original_return = original.total_return_pct
        original_drawdown = original.max_drawdown_pct
        original_sharpe = original.sharpe_ratio

        ai_returns = [r.total_return_pct for r in ai_results]
        ai_drawdowns = [r.max_drawdown_pct for r in ai_results]
        ai_sharpes = [r.sharpe_ratio for r in ai_results]

        best_ai_return = max(ai_returns)
        best_ai_sharpe = max(ai_sharpes)
        best_ai_drawdown = min(ai_drawdowns)

        recommendations = []

        # Return analysis
        if best_ai_return > original_return * 1.2:  # 20% better
            best_ai_strategy = ai_results[ai_returns.index(best_ai_return)]
            recommendations.append(
                f"AI strategy '{best_ai_strategy.objective_name}' shows {best_ai_return:.1f}% return vs {original_return:.1f}% original - consider allocation"
            )

        # Risk analysis
        if best_ai_drawdown < original_drawdown * 0.8:  # 20% less risk
            safest_ai_strategy = ai_results[ai_drawdowns.index(best_ai_drawdown)]
            recommendations.append(
                f"AI strategy '{safest_ai_strategy.objective_name}' shows lower risk ({best_ai_drawdown:.1f}% vs {original_drawdown:.1f}% drawdown)"
            )

        # Efficiency analysis
        if best_ai_sharpe > original_sharpe * 1.1:  # 10% better risk-adjusted returns
            best_sharpe_strategy = ai_results[ai_sharpes.index(best_ai_sharpe)]
            recommendations.append(
                f"AI strategy '{best_sharpe_strategy.objective_name}' shows superior risk-adjusted returns"
            )

        if not recommendations:
            return "Original nGS strategy performs competitively. Consider hybrid approach for diversification."

        return " | ".join(recommendations)

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    def _filter_data_by_date(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Filter data dictionary by date range"""
        if not start_date and not end_date:
            return data

        filtered_data: Dict[str, pd.DataFrame] = {}

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

        return filtered_data

    def _rank_strategies(self, results: List[BacktestResult]) -> None:
        """Rank strategies by multiple criteria"""

        if not results:
            return

        # Rank by different metrics
        by_return = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
        by_sharpe = sorted(results, key=lambda x: x.sharpe_ratio, reverse=True)
        by_drawdown = sorted(
            results, key=lambda x: x.max_drawdown_pct
        )  # Lower is better
        by_winrate = sorted(results, key=lambda x: x.win_rate, reverse=True)
        by_fitness = sorted(results, key=lambda x: x.fitness_score, reverse=True)

        self.strategy_rankings = {
            "by_return": [(r.strategy_id, r.total_return_pct) for r in by_return],
            "by_sharpe": [(r.strategy_id, r.sharpe_ratio) for r in by_sharpe],
            "by_drawdown": [(r.strategy_id, r.max_drawdown_pct) for r in by_drawdown],
            "by_winrate": [(r.strategy_id, r.win_rate) for r in by_winrate],
            "by_fitness": [(r.strategy_id, r.fitness_score) for r in by_fitness],
        }

        print(f"\n Strategy Rankings:")
        print(
            f"   Best Return:    {by_return[0].strategy_id} ({by_return[0].total_return_pct:.2f}%)"
        )
        print(
            f"   Best Sharpe:    {by_sharpe[0].strategy_id} ({by_sharpe[0].sharpe_ratio:.2f})"
        )
        print(
            f"   Lowest Risk:    {by_drawdown[0].strategy_id} ({by_drawdown[0].max_drawdown_pct:.2f}%)"
        )
        print(
            f"   Highest Fitness: {by_fitness[0].strategy_id} ({by_fitness[0].fitness_score:.3f})"
        )

    def _create_walk_forward_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        training_months: int,
        testing_months: int,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Create walk-forward analysis windows"""

        windows: List[Tuple[datetime, datetime, datetime, datetime]] = []
        current_date = start_date

        while current_date < end_date:
            # Training period
            train_start = current_date
            train_end = train_start + timedelta(days=training_months * 30)

            # Testing period
            test_start = train_end
            test_end = test_start + timedelta(days=testing_months * 30)

            if test_end > end_date:
                test_end = end_date

            if test_start < end_date:
                windows.append((train_start, train_end, test_start, test_end))

            # Move to next window (slide by testing period)
            current_date = test_start

        return windows

    def _analyze_walk_forward_results(self, wf_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward results for consistency"""

        if not wf_results:
            return {}

        returns = [r["result"].total_return_pct for r in wf_results]
        sharpes = [r["result"].sharpe_ratio for r in wf_results]
        drawdowns = [r["result"].max_drawdown_pct for r in wf_results]

        analysis = {
            "avg_return": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "avg_sharpe": float(np.mean(sharpes)),
            "avg_drawdown": float(np.mean(drawdowns)),
            "positive_windows": len([r for r in returns if r > 0]),
            "negative_windows": len([r for r in returns if r <= 0]),
            "best_window_return": max(returns),
            "worst_window_return": min(returns),
            "consistency_score": 1
            - (
                float(np.std(returns)) / max(abs(float(np.mean(returns))), 1)
            ),  # Higher is more consistent
        }

        return analysis

    def _save_comparison_results(self, comparison: BacktestComparison) -> None:
        """Save comprehensive comparison results"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_comparison_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        # Prepare data for JSON serialization
        comparison_data = {
            "timestamp": timestamp,
            "original_ngs": {
                "strategy_id": comparison.original_ngs_result.strategy_id,
                "total_return_pct": comparison.original_ngs_result.total_return_pct,
                "max_drawdown_pct": comparison.original_ngs_result.max_drawdown_pct,
                "win_rate": comparison.original_ngs_result.win_rate,
                "sharpe_ratio": comparison.original_ngs_result.sharpe_ratio,
                "total_trades": comparison.original_ngs_result.total_trades,
            },
            "ai_strategies": [
                {
                    "strategy_id": result.strategy_id,
                    "objective": result.objective_name,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate": result.win_rate,
                    "sharpe_ratio": result.sharpe_ratio,
                    "total_trades": result.total_trades,
                    "fitness_score": result.fitness_score,
                }
                for result in comparison.ai_results
            ],
            "comparison_metrics": comparison.comparison_metrics,
            "recommendation": comparison.recommendation,
            "summary_stats": comparison.summary_stats,
        }

        with open(filepath, "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f" Comparison results saved: {filepath}")


def demonstrate_backtesting_system() -> None:
    """Demonstrate the backtesting system capabilities"""
    print("\n nGS AI BACKTESTING SYSTEM DEMONSTRATION")
    print("=" * 60)

    # Initialize backtesting system in production mode
    backtester = NGSAIBacktestingSystem(account_size=1000000, production_mode=True)

    print(f"\n Backtesting Configuration:")
    print(f"   Production Mode:     {'ENABLED' if backtester.production_mode else 'DISABLED'}")
    
    if not backtester.production_mode:
        print(
            f"   Commission:          ${backtester.backtest_config['commission_per_trade']:.2f} per trade"
        )
        print(f"   Slippage:            {backtester.backtest_config['slippage_pct']:.2f}%")
    else:
        print(f"   Trading Costs:       BYPASSED (Production Mode)")
        
    print(
        f"   Max Positions:       {backtester.backtest_config['max_concurrent_positions']}"
    )
    print(f"   Risk-free Rate:      {backtester.backtest_config['risk_free_rate']:.1%}")

    print(f"\n Available Analysis Types:")
    print(f"    Single Strategy Backtesting")
    print(f"    Multi-Objective Comparison")
    print(f"    Original vs AI Comprehensive Comparison")
    print(f"    Walk-Forward Analysis")
    print(f"    Strategy Ranking & Selection")

    print(f"\n Key Features:")
    print(f"    Uses YOUR nGS parameters and patterns")
    if backtester.production_mode:
        print(f"    Production-ready: No brokerage costs applied")
        print(f"    Enhanced debug logging for track record development")
    else:
        print(f"    Realistic trading costs (commission + slippage)")
    print(f"    Comprehensive performance metrics")
    print(f"    Risk-adjusted analysis (Sharpe, drawdown)")
    print(f"    Strategy robustness testing")
    print(f"    Automated strategy ranking")
    print(f"    Enhanced edge case validation")


if __name__ == "__main__":
    print(" nGS AI BACKTESTING SYSTEM")
    print("=" * 60)
    print(" Comprehensive backtesting for AI-generated strategies")
    print(" Test, compare, and validate strategy performance")

    # Run demonstration
    demonstrate_backtesting_system()

    print(f"\n BACKTESTING SYSTEM READY!")
    print("\n Usage Examples:")
    print("   backtester.backtest_original_ngs(data)")
    print(
        "   backtester.backtest_multiple_objectives(['linear_equity', 'max_roi'], data)"
    )
    print("   backtester.backtest_comprehensive_comparison(objectives, data)")
    print("   backtester.walk_forward_analysis(strategy, data)")


class NGSProvenParameters:
    """
    Container for proven nGS parameters based on your original strategy
    Used by AI system to generate strategy variants using your proven base parameters
    """

    def __init__(self) -> None:
        # Your proven parameters from the original nGS strategy
        self.base_parameters: Dict[str, Any] = {
            "Length": 25,
            "NumDevs": 2,
            "MinPrice": 10,
            "MaxPrice": 500,
            "AfStep": 0.05,
            "AfLimit": 0.21,
            "PositionSize": 5000,
            "me_target_min": 50.0,
            "me_target_max": 80.0,
            "min_positions_for_scaling_up": 5,
            "profit_target_pct": 1.05,
            "stop_loss_pct": 0.90,
        }

        # Pattern-specific parameters that work well
        self.pattern_parameters: Dict[str, Any] = {
            "engulfing_threshold": 0.05,
            "atr_multiplier": 2.0,
            "bb_tolerance": 0.02,
            "reversal_bars": 5,
            "gap_out_threshold": 1.05,
            "hard_stop_threshold": 0.90,
        }

        # Risk management parameters
        self.risk_parameters: Dict[str, Any] = {
            "max_position_pct": 0.05,
            "max_sector_weight": 0.20,
            "ls_ratio_adjustment": True,
            "me_rebalancing": True,
            "sector_allocation_enabled": False,
        }

        # Market conditions parameters
        self.market_parameters: Dict[str, Any] = {
            "min_volume": 100000,
            "min_market_cap": 1000000000,  # $1B
            "excluded_sectors": [],
            "trading_hours_only": True,
        }

    def get_base_parameters(self) -> Dict[str, Any]:
        """Get base trading parameters"""
        return self.base_parameters.copy()

    def get_pattern_parameters(self) -> Dict[str, Any]:
        """Get pattern recognition parameters"""
        return self.pattern_parameters.copy()

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get risk management parameters"""
        return self.risk_parameters.copy()

    def get_market_parameters(self) -> Dict[str, Any]:
        """Get market condition parameters"""
        return self.market_parameters.copy()

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameters combined"""
        all_params: Dict[str, Any] = {}
        all_params.update(self.base_parameters)
        all_params.update(self.pattern_parameters)
        all_params.update(self.risk_parameters)
        all_params.update(self.market_parameters)
        return all_params

    def get_parameter_ranges_for_optimization(self) -> Dict[str, List[Any]]:
        """Get parameter ranges for optimization testing"""
        return {
            "PositionSize": [3000, 4000, 5000, 6000, 7000],
            "me_target_min": [40, 45, 50, 55, 60],
            "me_target_max": [70, 75, 80, 85, 90],
            "Length": [15, 20, 25, 30, 35],
            "NumDevs": [1.5, 2.0, 2.5, 3.0],
            "profit_target_pct": [1.03, 1.04, 1.05, 1.06, 1.08, 1.10],
            "stop_loss_pct": [0.88, 0.90, 0.92, 0.95, 0.97],
        }

    def create_variant_parameters(self, variant_name: str) -> Dict[str, Any]:
        """Create parameter variants for different strategies"""
        base = self.get_base_parameters()

        if variant_name == "aggressive":
            base.update(
                {
                    "PositionSize": 7000,
                    "me_target_min": 60.0,
                    "me_target_max": 90.0,
                    "NumDevs": 2.5,
                }
            )
        elif variant_name == "conservative":
            base.update(
                {
                    "PositionSize": 3000,
                    "me_target_min": 40.0,
                    "me_target_max": 70.0,
                    "NumDevs": 1.5,
                }
            )
        elif variant_name == "high_frequency":
            base.update(
                {
                    "PositionSize": 4000,
                    "Length": 15,
                    "profit_target_pct": 1.03,
                    "stop_loss_pct": 0.97,
                }
            )
        elif variant_name == "trend_following":
            base.update(
                {
                    "PositionSize": 6000,
                    "Length": 35,
                    "profit_target_pct": 1.08,
                    "NumDevs": 3.0,
                }
            )

        return base
