"""
Performance Objectives for Objective-Aware AI
Defines different performance goals that AI can optimize trading strategies for
Each objective generates completely different trading logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod


class PerformanceObjective(ABC):
    """Base class for all performance objectives"""

    @abstractmethod
    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Calculate how well results meet this objective"""
        pass

    @abstractmethod
    def get_objective_description(self) -> str:
        """Human-readable description of this objective"""
        pass

    @abstractmethod
    def get_strategy_preferences(self) -> Dict[str, Any]:
        """Return strategy preferences for this objective"""
        pass


class LinearEquityObjective(PerformanceObjective):
    """
    Optimize for smooth, linear equity curve
    Prioritizes consistency and steady growth over maximum returns
    """

    def __init__(self, smoothness_weight: float = 0.7, growth_weight: float = 0.3) -> None:
        self.smoothness_weight = smoothness_weight
        self.growth_weight = growth_weight

    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Fitness based on how linear and smooth the equity curve is
        Higher fitness = more linear growth
        """
        if len(equity_curve) < 10:
            return 0.0

        try:
            # Calculate linearity (R-squared of linear regression)
            x = np.arange(len(equity_curve))
            y = equity_curve.values

            # Fit linear regression
            coeffs = np.polyfit(x, y, 1)
            predicted = np.polyval(coeffs, x)

            # Calculate R-squared (linearity measure)
            ss_res: float = float(np.sum((y - predicted) ** 2))
            ss_tot: float = float(np.sum((y - np.mean(y)) ** 2))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            r_squared = max(0, r_squared)

            # Ensure positive slope (growing equity)
            slope_factor = max(0, min(1, coeffs[0] / 100))  # Normalize slope

            # Calculate smoothness (inverse of volatility)
            daily_returns = equity_curve.pct_change().dropna()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                smoothness = 1 / (1 + daily_returns.std())
            else:
                smoothness = 1.0

            # Combined fitness
            linearity_score = r_squared * slope_factor
            smoothness_score = smoothness

            fitness = (
                linearity_score * self.smoothness_weight
                + smoothness_score * self.growth_weight
            )

            return max(0, float(fitness))

        except Exception:
            return 0.0

    def get_objective_description(self) -> str:
        return "Linear Equity: Optimize for smooth, consistent equity curve growth"

    def get_strategy_preferences(self) -> Dict[str, Any]:
        """Strategy characteristics preferred for linear equity"""
        return {
            "entry_selectivity": "very_high",  # Be very selective
            "profit_targets": "small_consistent",  # Small but consistent profits
            "stop_losses": "tight",  # Tight stops to prevent large losses
            "position_sizing": "conservative",  # Smaller positions
            "hold_duration": "short",  # Quick turnaround
            "risk_tolerance": "low",  # Low risk approach
            "preferred_indicators": ["bb_position", "rsi", "market_efficiency"],
            "avoid_conditions": ["high_volatility", "trending_strongly"],
        }


class MaxROIObjective(PerformanceObjective):
    """
    Optimize for maximum return on investment
    Prioritizes total returns over consistency
    """

    def __init__(self, drawdown_penalty: float = 0.1) -> None:
        self.drawdown_penalty = drawdown_penalty

    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Fitness based on total return with slight drawdown penalty
        """
        if not trades or len(equity_curve) < 2:
            return 0.0

        try:
            # Calculate total return
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

            # Calculate maximum drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(float(drawdown.min())) * 100

            # Apply drawdown penalty (reduce fitness for excessive drawdowns)
            drawdown_factor = 1 - min(self.drawdown_penalty * max_drawdown / 100, 0.5)

            # Raw ROI with drawdown adjustment
            fitness = total_return * drawdown_factor

            return max(0, float(fitness))

        except Exception:
            return 0.0

    def get_objective_description(self) -> str:
        return (
            "Maximum ROI: Optimize for highest total returns, accept higher volatility"
        )

    def get_strategy_preferences(self) -> Dict[str, Any]:
        """Strategy characteristics preferred for maximum ROI"""
        return {
            "entry_selectivity": "moderate",  # Balance opportunity vs selectivity
            "profit_targets": "large",  # Let winners run
            "stop_losses": "wide",  # Wide stops for big moves
            "position_sizing": "aggressive",  # Larger positions
            "hold_duration": "long",  # Hold for big moves
            "risk_tolerance": "high",  # Accept higher risk for higher returns
            "preferred_indicators": ["tsf", "linreg_slope", "bb_position"],
            "favor_conditions": ["trending_strongly", "momentum_building"],
        }


class MinDrawdownObjective(PerformanceObjective):
    """
    Optimize for minimum drawdown
    Prioritizes capital preservation over returns
    """

    def __init__(self, return_bonus: float = 0.2) -> None:
        self.return_bonus = return_bonus

    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Fitness based on minimizing maximum drawdown
        """
        if len(equity_curve) < 2:
            return 0.0

        try:
            # Calculate maximum drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = abs(float(drawdown.min()))

            # Fitness is inverse of drawdown (lower drawdown = higher fitness)
            drawdown_fitness = 1 / (1 + max_drawdown)

            # Small bonus for positive returns (but drawdown is priority)
            total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
            return_factor = 1 + max(0, float(total_return) * self.return_bonus)

            fitness = drawdown_fitness * return_factor

            return float(fitness)

        except Exception:
            return 0.0

    def get_objective_description(self) -> str:
        return "Minimum Drawdown: Prioritize capital preservation and minimal losses"

    def get_strategy_preferences(self) -> Dict[str, Any]:
        """Strategy characteristics preferred for minimal drawdown"""
        return {
            "entry_selectivity": "extremely_high",  # Only perfect setups
            "profit_targets": "very_small",  # Take profits quickly
            "stop_losses": "extremely_tight",  # Cut losses immediately
            "position_sizing": "very_conservative",  # Tiny positions
            "hold_duration": "very_short",  # Quick in and out
            "risk_tolerance": "minimal",  # Extreme risk aversion
            "preferred_indicators": ["bb_position", "rsi", "support_resistance"],
            "avoid_conditions": ["high_volatility", "uncertain_signals"],
        }


class HighWinRateObjective(PerformanceObjective):
    """
    Optimize for maximum percentage of winning trades
    Prioritizes being right often over profit size
    """

    def __init__(self, min_trade_threshold: int = 20) -> None:
        self.min_trade_threshold = min_trade_threshold

    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Fitness based on win rate percentage
        """
        if not trades or len(trades) < self.min_trade_threshold:
            return 0.0  # Need minimum trades for valid win rate

        try:
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade.get("pnl_pct", 0) > 0)
            win_rate = winning_trades / len(trades)

            # Slight bonus for having more trades (more opportunities)
            trade_volume_factor = min(1.2, len(trades) / 50)

            fitness = win_rate * trade_volume_factor

            return float(fitness)

        except Exception:
            return 0.0

    def get_objective_description(self) -> str:
        return "High Win Rate: Optimize for maximum percentage of winning trades"

    def get_strategy_preferences(self) -> Dict[str, Any]:
        """Strategy characteristics preferred for high win rate"""
        return {
            "entry_selectivity": "very_high",  # Only high-probability setups
            "profit_targets": "quick_small",  # Take profits quickly
            "stop_losses": "tight",  # Prevent losses from growing
            "position_sizing": "moderate",  # Balanced sizing
            "hold_duration": "short",  # Quick profits
            "risk_tolerance": "low_moderate",  # Conservative but not extreme
            "preferred_indicators": ["bb_position", "rsi", "stochastic"],
            "favor_conditions": ["mean_reversion", "oversold_bounces"],
        }


class SharpeRatioObjective(PerformanceObjective):
    """
    Optimize for maximum Sharpe ratio
    Balances returns against volatility (risk-adjusted returns)
    """

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate

    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Fitness based on Sharpe ratio of returns
        """
        if not trades or len(trades) < 10:
            return 0.0

        try:
            # Calculate daily returns
            daily_returns = equity_curve.pct_change().dropna()

            if len(daily_returns) < 2:
                return 0.0

            # Calculate excess returns (above risk-free rate)
            excess_returns = daily_returns - self.risk_free_rate

            # Calculate Sharpe ratio
            if excess_returns.std() > 0:
                sharpe_ratio = excess_returns.mean() / excess_returns.std()
                # Annualize the Sharpe ratio
                annualized_sharpe = sharpe_ratio * np.sqrt(252)
            else:
                annualized_sharpe = 0.0

            # Only return positive Sharpe ratios
            return max(0, float(annualized_sharpe))

        except Exception:
            return 0.0

    def get_objective_description(self) -> str:
        return "Sharpe Ratio: Optimize risk-adjusted returns (return per unit of risk)"

    def get_strategy_preferences(self) -> Dict[str, Any]:
        """Strategy characteristics preferred for good Sharpe ratio"""
        return {
            "entry_selectivity": "high",  # Good selectivity
            "profit_targets": "moderate",  # Balanced profit taking
            "stop_losses": "moderate",  # Balanced risk management
            "position_sizing": "volatility_adjusted",  # Size based on volatility
            "hold_duration": "moderate",  # Balanced holding periods
            "risk_tolerance": "moderate",  # Balanced risk approach
            "preferred_indicators": ["bb_position", "atr", "volatility_ratio"],
            "favor_conditions": ["stable_volatility", "clear_signals"],
        }


class CustomObjective(PerformanceObjective):
    """
    Template for creating custom objectives
    Users can define their own fitness functions and strategy preferences
    """

    def __init__(
        self,
        name: str,
        fitness_function: Callable[[List[Dict[str, Any]], pd.Series, Optional[Dict[str, Any]]], float],
        strategy_preferences: Dict[str, Any],
        description: str,
    ) -> None:
        self.name = name
        self.fitness_function = fitness_function
        self.strategy_prefs = strategy_preferences
        self.description = description

    def calculate_fitness(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: pd.Series,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Use custom fitness function"""
        try:
            return float(self.fitness_function(trades, equity_curve, additional_metrics))
        except Exception:
            return 0.0

    def get_objective_description(self) -> str:
        return f"Custom Objective: {self.description}"

    def get_strategy_preferences(self) -> Dict[str, Any]:
        return self.strategy_prefs


class ObjectiveManager:
    """
    Manages all available performance objectives
    Provides easy access and comparison capabilities
    """

    def __init__(self) -> None:
        self.objectives: Dict[str, PerformanceObjective] = {}
        self._initialize_default_objectives()

    def _initialize_default_objectives(self) -> None:
        """Initialize all default objectives"""
        self.objectives = {
            "linear_equity": LinearEquityObjective(),
            "max_roi": MaxROIObjective(),
            "min_drawdown": MinDrawdownObjective(),
            "high_winrate": HighWinRateObjective(),
            "sharpe_ratio": SharpeRatioObjective(),
        }

        print(
            f" Objective Manager initialized with {len(self.objectives)} objectives"
        )

    def get_objective(self, name: str) -> PerformanceObjective:
        """Get objective by name"""
        if name not in self.objectives:
            raise ValueError(
                f"Objective '{name}' not found. Available: {list(self.objectives.keys())}"
            )
        return self.objectives[name]

    def list_objectives(self) -> Dict[str, str]:
        """List all available objectives with descriptions"""
        return {name: obj.get_objective_description()
                for name, obj in self.objectives.items()}

    def get_primary_objective(self) -> str:
        """Return the name of the first objective as the primary objective."""
        if not hasattr(self, 'objectives') or not self.objectives:
            raise ValueError("No objectives available in ObjectiveManager")
        return next(iter(self.objectives))

    def add_custom_objective(
        self,
        name: str,
        fitness_function: Callable[[List[Dict[str, Any]], pd.Series, Optional[Dict[str, Any]]], float],
        strategy_preferences: Dict[str, Any],
        description: str,
    ) -> None:
        """Add a custom objective"""
        self.objectives[name] = CustomObjective(
            name, fitness_function, strategy_preferences, description
        )
        print(f" Added custom objective: {name}")

    def compare_objectives(
        self, trades: List[Dict[str, Any]], equity_curve: pd.Series
    ) -> pd.DataFrame:
        """Compare how all objectives rate the same results"""
        results: List[Dict[str, Any]] = []

        for name, objective in self.objectives.items():
            fitness = objective.calculate_fitness(trades, equity_curve)
            results.append(
                {
                    "Objective": name.replace("_", " ").title(),
                    "Fitness Score": f"{fitness:.3f}",
                    "Description": objective.get_objective_description(),
                }
            )

        return pd.DataFrame(results)


def demo_objectives() -> None:
    """Demonstrate how objectives work with sample data"""
    print("\n OBJECTIVES DEMONSTRATION")
    print("=" * 50)

    # Create sample equity curves for demonstration
    np.random.seed(42)

    # Smooth linear growth
    linear_equity = pd.Series(10000 + np.arange(100) * 50 + np.random.randn(100) * 10)

    # High return but volatile
    volatile_returns = np.random.randn(100) * 0.02
    volatile_returns[::10] += 0.05  # Occasional big wins
    volatile_equity = pd.Series(10000 * np.cumprod(1 + volatile_returns))

    # Conservative growth
    conservative_returns = np.random.randn(100) * 0.005 + 0.001
    conservative_equity = pd.Series(10000 * np.cumprod(1 + conservative_returns))

    # Sample trades (simplified)
    sample_trades = [{"pnl_pct": np.random.randn() * 2 + 1} for _ in range(50)]

    # Initialize objectives
    manager = ObjectiveManager()

    print("\n Testing different equity curves against all objectives:")

    equity_types = {
        "Linear Growth": linear_equity,
        "High Volatility": volatile_equity,
        "Conservative": conservative_equity,
    }

    for eq_type, equity in equity_types.items():
        print(f"\n{eq_type} Equity Curve:")
        comparison = manager.compare_objectives(sample_trades, equity)
        for _, row in comparison.iterrows():
            print(f"  {row['Objective']}: {row['Fitness Score']}")


if __name__ == "__main__":
    print(" PERFORMANCE OBJECTIVES SYSTEM")
    print("=" * 50)

    # Initialize objective manager
    manager = ObjectiveManager()

    # Show available objectives
    print("\n Available Objectives:")
    objectives_list = manager.list_objectives()
    for name, desc in objectives_list.items():
        print(f"  {name}: {desc}")

    # Show strategy preferences example
    print(f"\n Strategy Preferences Example (Linear Equity):")
    linear_obj = manager.get_objective("linear_equity")
    prefs = linear_obj.get_strategy_preferences()
    for key, value in prefs.items():
        print(f"  {key}: {value}")

    # Run demonstration
    demo_objectives()

    print(f"\n STEP 2 COMPLETE!")
    print("Next: Objective-Aware AI that generates strategy logic")
