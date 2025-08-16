# Python
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # type: ignore

from performance_objectives import ObjectiveManager

# Safe import: prefer combined module, fallback to underlying modules if needed
try:
    from ngs_integrated_ai_system import NGSAwareStrategyGenerator, NGSIndicatorLibrary
except Exception:
    from comprehensive_indicator_library import (
        ComprehensiveIndicatorLibrary as NGSIndicatorLibrary,
    )
    from strategy_generator_ai import (
        ObjectiveAwareStrategyGenerator as NGSAwareStrategyGenerator,
    )

# Persist using data_manager
from data_manager import save_positions as dm_save_positions
from data_manager import save_trades as dm_save_trades

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NGSAIIntegrationManager:
    """
    AI-only integration manager.
    - Uses ObjectiveManager + NGSAwareStrategyGenerator to create symbol strategies
    - Executes per-symbol strategies, generating closed trades and positions
    - Saves trades and positions via data_manager
    - Provides evaluate_linear_equity (used by Streamlit page)
    """

    def __init__(self, account_size: float = 1_000_000, data_dir: str = "data") -> None:
        self.account_size = float(account_size)
        self.data_dir = data_dir

        # AI components
        self.ngs_indicator_lib = NGSIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        self.ai_generator = NGSAwareStrategyGenerator(
            self.ngs_indicator_lib, self.objective_manager
        )

        # State
        self.active_strategies: Dict[str, Any] = {}
        self.strategy_performance: Dict[str, Any] = {}

        # AI-only configuration
        self.operating_mode = "ai_only"
        self.integration_config = {
            "ai_allocation_pct": 100.0,
            "max_ai_strategies": 3,
            "rebalance_frequency": "weekly",
            "performance_tracking": True,
            "risk_sync": False,
        }

        # Execution config
        self.execution_config = {
            "commission_per_trade": 1.0,  # $1 commission per closed trade
            "slippage_pct": 0.05,  # 0.05% slippage
        }

        # Output
        self.results_dir = os.path.join(self.data_dir, "integration_results")
        os.makedirs(self.results_dir, exist_ok=True)

        logger.info("NGS AI Integration Manager initialized (AI-only)")

    # --------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------

    def set_operating_mode(self, mode: str) -> None:
        """
        Retained for compatibility. AI-only enforced.
        """
        valid_modes = ["ai_only"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.operating_mode = mode
        logger.info(f"Operating mode set to: {mode}")

    def evaluate_linear_equity(
        self, equity_curve: Optional[pd.Series] | Optional[list]
    ) -> float:
        """
        R-squared of a linear fit to the equity curve (0..1). Returns 0 if not enough data.
        """
        try:
            if equity_curve is None:
                return 0.0
            y = pd.Series(equity_curve).dropna().astype(float)
            if y.empty or len(y) < 3:
                return 0.0
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y.values)
            y_hat = model.predict(x)
            ss_res = float(np.sum((y.values - y_hat) ** 2))
            ss_tot = float(np.sum((y.values - np.mean(y.values)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            return max(0.0, min(1.0, float(r2)))
        except Exception as e:
            logger.warning(f"evaluate_linear_equity failed: {e}")
            return 0.0

    def create_ai_strategy_set(
        self, data: Dict[str, pd.DataFrame], objective: str
    ) -> Dict[str, Any]:
        """
        Create per-symbol AI strategies for the selected objective.
        """
        strategies: Dict[str, Any] = {}
        for symbol, df in data.items():
            try:
                strat = self.ai_generator.generate_strategy_for_objective(objective)
                # Ensure the strategy object has a 'data' attribute; otherwise, skip or raise
                setattr(strat, "data", df)
                strategies[symbol] = strat
                logger.debug(f"Generated AI strategy for {symbol}")
            except Exception as e:
                logger.error(f"Failed to generate strategy for {symbol}: {e}")
        return strategies

    def run_integrated_strategy(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Executes AI strategies only, per symbol.
        Returns:
            {
              "<SYMBOL>": {
                 "trades": [closed-trade dicts],
                 "positions": [closed position dicts],
                 "performance": {"sharpe": float, "total_return": float},
                 "equity_curve": pd.Series
              },
              ...
            }
        """
        logger.info("Running integrated AI strategy (AI-only)")
        objective = self._get_primary_objective_safe()
        self.active_strategies = self.create_ai_strategy_set(data, objective)

        results: Dict[str, Dict[str, Any]] = {}
        for symbol, strategy in self.active_strategies.items():
            try:
                sym_res = self._execute_symbol_ai(symbol, strategy, data)
                results[symbol] = sym_res
            except Exception as e:
                logger.error(f"Execution failed for {symbol}: {e}")
                results[symbol] = {
                    "trades": [],
                    "positions": [],
                    "performance": {"sharpe": 0.0, "total_return": 0.0},
                    "equity_curve": pd.Series(dtype=float),
                }

        self.strategy_performance = results
        return results

    def rebalance_portfolio(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Retained for compatibility. Uses current strategy_performance to refresh strategies.
        """
        if self.integration_config.get("rebalance_frequency") == "weekly":
            now = datetime.now()
            if now.weekday() != 0:  # Monday
                logger.debug("Skipping rebalance (not Monday)")
                return

        # Refresh strategies using the same primary objective
        objective = self._get_primary_objective_safe()
        self.active_strategies = self.create_ai_strategy_set(data, objective)
        logger.info("Portfolio rebalanced based on AI strategy configuration")

    def save_integration_session(
        self, results: Dict[str, Any], filename: str = "integration_results.json"
    ) -> None:
        """
        Save results JSON under data/integration_results/
        """
        try:
            output_path = os.path.join(self.results_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved integration results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save integration results: {e}")

    # --------------------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------------------

    def _get_primary_objective_safe(self) -> str:
        try:
            if hasattr(self.objective_manager, "get_primary_objective"):
                return self.objective_manager.get_primary_objective()
            return "linear_equity"
        except Exception:
            return "linear_equity"

    def _execute_symbol_ai(
        self, symbol: str, strategy: Any, all_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Execute an AI strategy on the symbol's data. Builds closed trades and positions,
        computes performance, and saves via data_manager.
        """
        df = getattr(strategy, "data", None)
        if df is None or df.empty:
            return {
                "trades": [],
                "positions": [],
                "performance": {"sharpe": 0.0, "total_return": 0.0},
                "equity_curve": pd.Series(dtype=float),
            }

        # Normalize index/time
        idx = (
            pd.to_datetime(df["Date"])
            if "Date" in df.columns
            else pd.to_datetime(df.index)
        )

        # Position sizing (divide capital across symbols)
        per_symbol_budget = max(self.account_size / max(len(all_data), 1), 1.0)

        # Execution state
        open_shares = 0
        entry_price = 0.0
        entry_time: Optional[datetime] = None

        closed_trades: List[Dict] = []
        equity_vals: List[float] = []
        equity = per_symbol_budget

        for i in range(len(df)):
            row = df.iloc[i]
            ts = idx[i]
            price = (
                float(row["Close"])
                if "Close" in df.columns
                else float(row.get("close", np.nan))
            )
            if np.isnan(price):
                equity_vals.append(equity if equity_vals else per_symbol_budget)
                continue

            # Generate signal
            try:
                signal = strategy.generate_signal(row)
            except Exception as e:
                logger.debug(f"{symbol}: signal generation failed at {i}: {e}")
                equity_vals.append(equity if equity_vals else per_symbol_budget)
                continue

            if signal == 1 and open_shares == 0:
                # Open long
                buy_px = self._apply_slippage(price, side="buy")
                sh = int(per_symbol_budget // buy_px)
                if sh > 0:
                    open_shares = sh
                    entry_price = buy_px
                    entry_time = ts

            elif signal == -1 and open_shares > 0:
                # Close long
                sell_px = self._apply_slippage(price, side="sell")
                gross = (sell_px - entry_price) * open_shares
                pnl = gross - self.execution_config["commission_per_trade"]

                # Handle entry_time None for strftime
                entry_date_str = (
                    entry_time.strftime("%Y-%m-%d") if entry_time is not None else ""
                )
                exit_date_str = ts.strftime("%Y-%m-%d") if ts is not None else ""

                closed_trades.append(
                    {
                        "symbol": symbol,
                        "type": "long",
                        "entry_date": entry_date_str,
                        "exit_date": exit_date_str,
                        "entry_price": round(entry_price, 6),
                        "exit_price": round(sell_px, 6),
                        "shares": int(open_shares),
                        "profit": float(pnl),
                        "exit_reason": "signal",
                        "side": "long",
                        "strategy": getattr(strategy, "strategy_id", "ai_strategy"),
                    }
                )

                equity += pnl
                open_shares = 0
                entry_price = 0.0
                entry_time = None

            # Append equity snapshot for this bar
            equity_vals.append(equity if equity_vals else per_symbol_budget)

        # Persist closed trades
        if closed_trades:
            try:
                dm_save_trades(closed_trades)
            except Exception as e:
                logger.warning(f"Failed to save trades for {symbol}: {e}")

        # Build positions from closed trades (dashboard compatibility)
        positions = self._calculate_positions_from_closed_trades(symbol, closed_trades)

        if positions:
            try:
                dm_save_positions(positions)
            except Exception as e:
                logger.warning(f"Failed to save positions for {symbol}: {e}")

        equity_curve = pd.Series(equity_vals, index=idx[: len(equity_vals)]).dropna()
        perf = self._calculate_performance(closed_trades, df)  # retain original metrics

        return {
            "trades": closed_trades,
            "positions": positions,
            "performance": perf,
            "equity_curve": equity_curve,
        }

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.execution_config["slippage_pct"] / 100.0
        if side == "buy":
            return price * (1.0 + slip)
        return price * (1.0 - slip)

    def _calculate_positions_from_closed_trades(
        self, symbol: str, trades: List[Dict]
    ) -> List[Dict]:
        """
        Convert closed trades to positions records expected by data_manager.save_positions.
        """
        positions: List[Dict] = []
        for t in trades:
            try:
                entry_dt = t["entry_date"]
                exit_dt = t["exit_date"]
                entry_px = float(t["entry_price"])
                exit_px = float(t["exit_price"])
                shares = int(t["shares"])
                pnl = float(t["profit"])
                positions.append(
                    {
                        "symbol": symbol,
                        "shares": shares,
                        "entry_price": entry_px,
                        "entry_date": entry_dt,
                        "current_price": exit_px,
                        "current_value": exit_px * shares,
                        "profit": pnl,
                        "profit_pct": (
                            ((exit_px / entry_px) - 1.0) * 100.0
                            if entry_px > 0
                            else 0.0
                        ),
                        "days_held": int(
                            max(
                                (
                                    pd.to_datetime(exit_dt) - pd.to_datetime(entry_dt)
                                ).days,
                                0,
                            )
                        ),
                        "side": "long",
                        "strategy": t.get("strategy", "ai_strategy"),
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to create position from trade for {symbol}: {e}")
        return positions

    def _calculate_performance(
        self, trades: List[Dict], data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Retains original performance output structure: Sharpe and total_return.
        """
        returns = []
        for trade in trades:
            if trade.get("type") == "long":
                entry_price = trade.get("entry_price")
                exit_price = trade.get("exit_price")
                if entry_price and exit_price and entry_price > 0:
                    returns.append((exit_price - entry_price) / entry_price)

        if not returns:
            return {"sharpe": 0.0, "total_return": 0.0}

        returns = np.array(returns, dtype=float)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns)) if len(returns) > 1 else 0.0
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0
        total_return = float(np.prod(1 + returns) - 1.0)

        return {"sharpe": sharpe, "total_return": total_return}
