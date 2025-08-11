"""
nGS Sector Adapter for PyBroker
Integrates sector-optimized parameters with PyBroker backtesting framework.
Uses optimized parameters from SectorParameterManager for enhanced performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import warnings

warnings.filterwarnings("ignore")

# PyBroker imports
try:
    import pybroker as pb
    from pybroker import Strategy, StrategyConfig, ExecContext
    from pybroker.data import AKShare

    PYBROKER_AVAILABLE = True
    print(" PyBroker available")
except ImportError:
    PYBROKER_AVAILABLE = False
    print("  PyBroker not available. Install with: pip install lib-pybroker")

# Import our optimization framework
from sector_parameter_manager import SectorParameterManager

# Import existing data functions if available
try:
    from data_manager import load_price_data, get_symbol_sector

    DATA_MANAGER_AVAILABLE = True
    print(" Data manager integration available")
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    print("  Data manager not found - using fallback data loading")


class NGSSectorAdapter:
    """
    Adapter class that integrates sector-optimized nGS strategy with PyBroker.

    Features:
    - Uses sector-specific optimized parameters
    - Implements full nGS strategy logic in PyBroker
    - Handles data loading and preprocessing
    - Supports multiple symbols with different sector parameters
    - Comprehensive performance analysis
    """

    def __init__(self, config_dir: str = "optimization_framework/config"):
        self.param_manager = SectorParameterManager(config_dir)
        self.strategy_config = None
        self.strategy = None

        # Performance tracking
        self.last_backtest_results = None
        self.symbol_parameters = {}

        print(" NGS Sector Adapter initialized")
        print(
            f" Optimized sectors available: {len(self.param_manager.get_all_optimized_sectors())}"
        )

        if not PYBROKER_AVAILABLE:
            print(" PyBroker is required. Install with: pip install lib-pybroker")

    def load_symbol_data(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Load price data for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        try:
            if DATA_MANAGER_AVAILABLE:
                # Use existing data manager
                df = load_price_data(symbol)
                if df is not None and not df.empty:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").reset_index(drop=True)

                    # Filter by date range if provided
                    if start_date:
                        df = df[df["Date"] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df["Date"] <= pd.to_datetime(end_date)]

                    return df
            else:
                print(f"  No data loading mechanism available for {symbol}")
                return None

        except Exception as e:
            print(f" Error loading data for {symbol}: {e}")
            return None

    def prepare_pybroker_data(
        self, symbols: List[str], start_date: str = None, end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        Prepare data in PyBroker format for multiple symbols

        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, volume
        """
        if not symbols:
            print(" No symbols provided")
            return None

        all_data = []

        for symbol in symbols:
            print(f" Loading data for {symbol}...")
            df = self.load_symbol_data(symbol, start_date, end_date)

            if df is not None and not df.empty:
                # Convert to PyBroker format
                pb_df = df.copy()
                pb_df["symbol"] = symbol
                pb_df = pb_df.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                # Ensure required columns exist
                required_cols = [
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
                if all(col in pb_df.columns for col in required_cols):
                    all_data.append(pb_df[required_cols])
                    print(f" {symbol}: {len(pb_df)} records loaded")
                else:
                    print(f" {symbol}: Missing required columns")
            else:
                print(f" {symbol}: No data available")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(["symbol", "date"]).reset_index(
                drop=True
            )

            print(
                f" Combined data: {len(combined_df)} total records for {len(symbols)} symbols"
            )
            return combined_df
        else:
            print(" No data loaded for any symbols")
            return None

    def create_ngs_indicators(
        self, df: pd.DataFrame, symbol: str, params: Dict
    ) -> pd.DataFrame:
        """
        Create nGS strategy indicators using sector-specific parameters

        Args:
            df: Price data DataFrame
            symbol: Stock symbol for parameter lookup
            params: Strategy parameters for this symbol
        """
        df = df.copy()

        # Get parameters
        bb_length = int(params.get("Length", 25))
        bb_devs = params.get("NumDevs", 2.0)
        me_min = params.get("me_target_min", 50.0)
        me_max = params.get("me_target_max", 80.0)

        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(window=bb_length).mean()
        df["bb_std"] = df["close"].rolling(window=bb_length).std()
        df["bb_upper"] = df["bb_mid"] + (bb_devs * df["bb_std"])
        df["bb_lower"] = df["bb_mid"] - (bb_devs * df["bb_std"])

        # Bollinger Band position (where price is relative to bands)
        df["bb_position"] = (
            (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]) * 100
        ).fillna(50)

        # Market Efficiency (simplified calculation)
        # This is a placeholder - replace with your actual M/E calculation
        price_change = df["close"].pct_change(20)  # 20-day price change
        volatility = df["close"].rolling(20).std() / df["close"].rolling(20).mean()
        df["market_efficiency"] = abs(price_change) / volatility * 100
        df["market_efficiency"] = df["market_efficiency"].fillna(50).clip(0, 100)

        # nGS Entry Signals
        df["ngs_long_signal"] = (
            (df["bb_position"] <= 20)  # Price near lower band
            & (df["market_efficiency"] >= me_min)  # Sufficient market efficiency
            & (
                df["market_efficiency"] <= me_max
            )  # Not too efficient (avoiding whipsaws)
            & (df["close"] > params.get("MinPrice", 10))  # Price filters
            & (df["close"] < params.get("MaxPrice", 500))
            & (
                df["volume"] > df["volume"].rolling(20).mean() * 0.8
            )  # Volume confirmation
        )

        # Exit conditions
        df["profit_target_price"] = df["close"] * params.get("profit_target_pct", 1.05)
        df["stop_loss_price"] = df["close"] * params.get("stop_loss_pct", 0.90)

        return df

    def ngs_entry_signal(self, ctx: ExecContext):
        """
        PyBroker entry signal function for nGS strategy
        """
        if not PYBROKER_AVAILABLE:
            return

        symbol = ctx.symbol

        # Get symbol-specific parameters
        if symbol not in self.symbol_parameters:
            self.symbol_parameters[symbol] = (
                self.param_manager.get_parameters_for_symbol(symbol)
            )

        params = self.symbol_parameters[symbol]

        # Skip if we don't have enough data
        if ctx.bars < max(30, int(params.get("Length", 25)) + 5):
            return

        # Get current and recent data
        current_price = ctx.close[-1]
        bb_lower = ctx.indicator("bb_lower")[-1]
        bb_position = ctx.indicator("bb_position")[-1]
        market_efficiency = ctx.indicator("market_efficiency")[-1]

        # Check entry conditions
        me_min = params.get("me_target_min", 50.0)
        me_max = params.get("me_target_max", 80.0)
        min_price = params.get("MinPrice", 10)
        max_price = params.get("MaxPrice", 500)

        entry_conditions = (
            bb_position <= 20  # Near lower Bollinger Band
            and me_min <= market_efficiency <= me_max  # Market efficiency in range
            and min_price <= current_price <= max_price  # Price filters
            and not ctx.long_pos()  # Not already in position
        )

        if entry_conditions:
            position_size = params.get("PositionSize", 5000)
            shares = int(position_size / current_price)

            if shares > 0:
                ctx.buy_shares = shares
                ctx.score = market_efficiency  # Use M/E as position scoring

    def ngs_exit_signal(self, ctx: ExecContext):
        """
        PyBroker exit signal function for nGS strategy
        """
        if not PYBROKER_AVAILABLE:
            return

        if not ctx.long_pos():
            return

        symbol = ctx.symbol
        params = self.symbol_parameters.get(
            symbol, self.param_manager.default_parameters
        )

        current_price = ctx.close[-1]
        entry_price = ctx.long_pos().entry_price
        bars_held = ctx.bars - ctx.long_pos().entry_bar

        # Exit conditions
        profit_target = entry_price * params.get("profit_target_pct", 1.05)
        stop_loss = entry_price * params.get("stop_loss_pct", 0.90)
        max_hold_days = params.get("max_hold_days", 30)

        exit_conditions = (
            current_price >= profit_target  # Profit target hit
            or current_price <= stop_loss  # Stop loss hit
            or bars_held >= max_hold_days  # Time-based exit
        )

        if exit_conditions:
            ctx.sell_all()

    def create_pybroker_strategy(
        self, symbols: List[str], start_date: str = None, end_date: str = None
    ) -> Optional[pb.Strategy]:
        """
        Create PyBroker strategy with nGS logic and sector-optimized parameters

        Args:
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
        """
        if not PYBROKER_AVAILABLE:
            print(" PyBroker not available")
            return None

        print(f"  Creating PyBroker strategy for {len(symbols)} symbols")

        # Load and prepare data
        data = self.prepare_pybroker_data(symbols, start_date, end_date)
        if data is None:
            print(" No data available for strategy creation")
            return None

        # Load symbol parameters
        for symbol in symbols:
            self.symbol_parameters[symbol] = (
                self.param_manager.get_parameters_for_symbol(symbol)
            )
            params = self.symbol_parameters[symbol]
            sector = params.get("assigned_sector", "Unknown")
            status = params.get("optimization_status", "default")
            print(f" {symbol} ({sector}): {status} parameters")

        # Create indicators for each symbol
        def add_ngs_indicators(symbol_data):
            symbol = symbol_data["symbol"].iloc[0]
            params = self.symbol_parameters[symbol]
            return self.create_ngs_indicators(symbol_data, symbol, params)

        # Apply indicators to each symbol's data
        indicator_data = []
        for symbol in symbols:
            symbol_data = data[data["symbol"] == symbol].copy()
            if len(symbol_data) > 30:  # Ensure enough data for indicators
                with_indicators = add_ngs_indicators(symbol_data)
                indicator_data.append(with_indicators)

        if not indicator_data:
            print(" No symbols have sufficient data for indicators")
            return None

        final_data = pd.concat(indicator_data, ignore_index=True)

        # Register indicators with PyBroker
        bb_lower = pb.indicator("bb_lower", lambda data: data["bb_lower"])
        bb_position = pb.indicator("bb_position", lambda data: data["bb_position"])
        market_efficiency = pb.indicator(
            "market_efficiency", lambda data: data["market_efficiency"]
        )

        # Create strategy configuration
        config = pb.StrategyConfig(
            initial_cash=100000,
            max_long_positions=10,
            portfolio_max_long_positions=None,
            buy_delay=1,
            sell_delay=1,
        )

        # Create strategy
        strategy = pb.Strategy(
            data_source=final_data,
            start_date=start_date,
            end_date=end_date,
            config=config,
        )

        # Add indicators
        strategy.add_indicator([bb_lower, bb_position, market_efficiency])

        # Add entry and exit rules
        strategy.add_rule("ngs_entry", self.ngs_entry_signal)
        strategy.add_rule("ngs_exit", self.ngs_exit_signal)

        self.strategy = strategy
        print(f" Strategy created with {len(symbols)} symbols")

        return strategy

    def run_backtest(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        show_summary: bool = True,
    ) -> Optional[Dict]:
        """
        Run backtest using sector-optimized parameters

        Args:
            symbols: List of symbols to trade
            start_date: Backtest start date ('YYYY-MM-DD')
            end_date: Backtest end date ('YYYY-MM-DD')
            show_summary: Whether to print summary results

        Returns:
            Dict with backtest results and analysis
        """
        if not PYBROKER_AVAILABLE:
            print(" PyBroker not available")
            return None

        print(f"\n Running nGS backtest on {len(symbols)} symbols")
        if start_date:
            print(f" Period: {start_date} to {end_date or 'latest'}")

        # Create strategy
        strategy = self.create_pybroker_strategy(symbols, start_date, end_date)
        if strategy is None:
            return None

        # Run backtest
        try:
            print(" Executing backtest...")
            result = strategy.backtest()

            if result is None:
                print(" Backtest returned no results")
                return None

            self.last_backtest_results = result

            # Calculate performance metrics
            trades = result.trades if hasattr(result, "trades") else pd.DataFrame()
            portfolio = (
                result.portfolio if hasattr(result, "portfolio") else pd.DataFrame()
            )

            analysis = self._analyze_backtest_results(trades, portfolio, symbols)

            if show_summary:
                self._print_backtest_summary(analysis)

            return {
                "pybroker_result": result,
                "analysis": analysis,
                "trades": trades,
                "portfolio": portfolio,
                "symbols": symbols,
                "symbol_parameters": self.symbol_parameters,
            }

        except Exception as e:
            print(f" Error running backtest: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _analyze_backtest_results(
        self, trades: pd.DataFrame, portfolio: pd.DataFrame, symbols: List[str]
    ) -> Dict:
        """Analyze backtest results and calculate performance metrics"""
        analysis = {}

        if not trades.empty:
            # Trade statistics
            analysis["total_trades"] = len(trades)
            analysis["winning_trades"] = len(trades[trades["pnl"] > 0])
            analysis["losing_trades"] = len(trades[trades["pnl"] <= 0])
            analysis["win_rate"] = (
                (analysis["winning_trades"] / analysis["total_trades"] * 100)
                if analysis["total_trades"] > 0
                else 0
            )

            # PnL statistics
            analysis["total_pnl"] = trades["pnl"].sum()
            analysis["avg_trade_pnl"] = trades["pnl"].mean()
            analysis["best_trade"] = trades["pnl"].max()
            analysis["worst_trade"] = trades["pnl"].min()

            # Return statistics
            if "return_pct" in trades.columns:
                analysis["avg_return_pct"] = trades["return_pct"].mean()
                analysis["win_avg_return"] = (
                    trades[trades["pnl"] > 0]["return_pct"].mean()
                    if analysis["winning_trades"] > 0
                    else 0
                )
                analysis["loss_avg_return"] = (
                    trades[trades["pnl"] <= 0]["return_pct"].mean()
                    if analysis["losing_trades"] > 0
                    else 0
                )

            # By symbol analysis
            symbol_stats = {}
            for symbol in symbols:
                symbol_trades = (
                    trades[trades["symbol"] == symbol]
                    if "symbol" in trades.columns
                    else pd.DataFrame()
                )
                if not symbol_trades.empty:
                    symbol_stats[symbol] = {
                        "trades": len(symbol_trades),
                        "win_rate": len(symbol_trades[symbol_trades["pnl"] > 0])
                        / len(symbol_trades)
                        * 100,
                        "total_pnl": symbol_trades["pnl"].sum(),
                        "avg_pnl": symbol_trades["pnl"].mean(),
                    }
            analysis["by_symbol"] = symbol_stats

        else:
            analysis["total_trades"] = 0
            analysis["win_rate"] = 0
            analysis["total_pnl"] = 0

        # Portfolio analysis
        if not portfolio.empty and "equity" in portfolio.columns:
            initial_equity = portfolio["equity"].iloc[0]
            final_equity = portfolio["equity"].iloc[-1]
            analysis["total_return_pct"] = (
                ((final_equity - initial_equity) / initial_equity * 100)
                if initial_equity > 0
                else 0
            )
            analysis["max_equity"] = portfolio["equity"].max()
            analysis["min_equity"] = portfolio["equity"].min()

            # Drawdown calculation
            rolling_max = portfolio["equity"].expanding().max()
            drawdown = (portfolio["equity"] - rolling_max) / rolling_max * 100
            analysis["max_drawdown_pct"] = drawdown.min()

        return analysis

    def _print_backtest_summary(self, analysis: Dict):
        """Print formatted backtest summary"""
        print(f"\n BACKTEST RESULTS SUMMARY")
        print("=" * 50)

        # Overall performance
        print(f" Total PnL: ${analysis.get('total_pnl', 0):.2f}")
        if "total_return_pct" in analysis:
            print(f" Total Return: {analysis['total_return_pct']:.2f}%")
        if "max_drawdown_pct" in analysis:
            print(f" Max Drawdown: {analysis['max_drawdown_pct']:.2f}%")

        # Trade statistics
        print(f"\n TRADE STATISTICS")
        print(f"Total Trades: {analysis.get('total_trades', 0)}")
        print(f"Win Rate: {analysis.get('win_rate', 0):.1f}%")
        print(f"Avg Trade PnL: ${analysis.get('avg_trade_pnl', 0):.2f}")
        print(f"Best Trade: ${analysis.get('best_trade', 0):.2f}")
        print(f"Worst Trade: ${analysis.get('worst_trade', 0):.2f}")

        # By symbol breakdown
        if "by_symbol" in analysis and analysis["by_symbol"]:
            print(f"\n BY SYMBOL PERFORMANCE")
            for symbol, stats in analysis["by_symbol"].items():
                print(
                    f"{symbol}: {stats['trades']} trades, {stats['win_rate']:.1f}% win rate, ${stats['total_pnl']:.2f} PnL"
                )

    def get_optimization_status(self) -> pd.DataFrame:
        """Get status of parameter optimization for all symbols"""
        return self.param_manager.get_optimization_summary()

    def update_sector_parameters(self, sector: str, new_parameters: Dict):
        """Update parameters for a specific sector"""
        self.param_manager.update_sector_parameters(sector, new_parameters)

        # Clear cached parameters to force reload
        symbols_to_update = []
        for symbol, params in self.symbol_parameters.items():
            if params.get("assigned_sector") == sector:
                symbols_to_update.append(symbol)

        for symbol in symbols_to_update:
            self.symbol_parameters[symbol] = (
                self.param_manager.get_parameters_for_symbol(symbol)
            )

        print(
            f" Updated {sector} parameters and refreshed {len(symbols_to_update)} symbol caches"
        )


# Example usage and testing
if __name__ == "__main__":
    print(" nGS Sector Adapter Testing")
    print("=" * 50)

    if not PYBROKER_AVAILABLE:
        print(" PyBroker not installed. Run: pip install lib-pybroker")
        exit(1)

    # Initialize adapter
    adapter = NGSSectorAdapter()

    # Test symbols (mix of sectors for demonstration)
    test_symbols = ["AAPL", "MSFT", "JPM", "XOM"]  # Tech, Tech, Finance, Energy

    print(f"\n Testing with symbols: {test_symbols}")

    # Show parameter status
    print("\n Current optimization status:")
    status = adapter.get_optimization_status()
    print(status.to_string(index=False))

    # Test data loading
    print(f"\n Testing data loading...")
    try:
        data = adapter.prepare_pybroker_data(
            test_symbols, start_date="2023-01-01", end_date="2023-12-31"
        )
        if data is not None:
            print(f" Successfully loaded data: {len(data)} records")
            print(f"   Symbols: {data['symbol'].unique()}")
            print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        else:
            print(" No data loaded")
    except Exception as e:
        print(f" Error loading data: {e}")

    # Test strategy creation (without full backtest)
    print(f"\n  Testing strategy creation...")
    try:
        strategy = adapter.create_pybroker_strategy(
            test_symbols[:2], start_date="2023-01-01", end_date="2023-06-30"
        )
        if strategy:
            print(" Strategy creation successful")
        else:
            print(" Strategy creation failed")
    except Exception as e:
        print(f" Error creating strategy: {e}")

    print(f"\n To run a full backtest, use:")
    print(f"   adapter = NGSSectorAdapter()")
    print(
        f"   results = adapter.run_backtest(['AAPL', 'MSFT'], start_date='2023-01-01', end_date='2023-12-31')"
    )


