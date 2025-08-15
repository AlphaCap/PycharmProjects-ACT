from dataclasses import dataclass


def calculate_advanced_metrics(daily_returns, benchmark_returns, equity_curve):
    """Stub for advanced metric calculation."""
    # Example: Calculate additional metrics
    return {
        "excess_return": (daily_returns.mean() - benchmark_returns.mean()) * 100,
        "return_to_volatility": daily_returns.mean() / daily_returns.std()
        if daily_returns.std() > 0 else 0,
    }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container"""

    strategy_name: str
    strategy_type: str
    objective: str

    # Return Metrics
    total_return_pct: float
    annualized_return_pct: float
    monthly_return_avg: float
    monthly_return_std: float

    # Risk Metrics
    max_drawdown_pct: float
    avg_drawdown_pct: float
    drawdown_duration_avg: float
    volatility_annual_pct: float
    downside_deviation_pct: float

    # Risk-Adjusted Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    jensen_alpha: float
    omega_ratio: float

    # Trading Metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win: float
    largest_loss: float
    avg_trade_duration_days: float

    # Consistency Metrics
    monthly_win_rate: float
    consecutive_wins_max: int
    consecutive_losses_max: int
    recovery_factor: float

    # Growth Metrics
    cagr: float  # Compound Annual Growth Rate

    # Market Metrics
    market_correlation: float
    beta: float
    alpha_annualized: float

    # Custom Metrics
    fitness_score: float
    me_ratio_efficiency: float
    pattern_success_rate: float

    @staticmethod
    def calculate_detailed_metrics(backtest_result, strategy_name: str, objective: str):
        """
        Calculate detailed performance metrics based on the backtest result.

        backtest_result: an object with all the necessary attributes:
            - strategy_type
            - daily_returns
            - benchmark_returns
            - equity_curve
            - and other raw stats as needed
        """
        daily_returns = backtest_result.daily_returns
        benchmark_returns = backtest_result.benchmark_returns
        equity_curve = backtest_result.equity_curve

        # Collecting metrics from the backtest result
        metrics = {
            "strategy_name": strategy_name,
            "strategy_type": getattr(backtest_result, "strategy_type", ""),
            "objective": objective,
            "total_return_pct": getattr(backtest_result, "total_return_pct", 0.0),
            "annualized_return_pct": getattr(
                backtest_result, "annualized_return_pct", 0.0
            ),
            "monthly_return_avg": getattr(backtest_result, "monthly_return_avg", 0.0),
            "monthly_return_std": getattr(backtest_result, "monthly_return_std", 0.0),
            "max_drawdown_pct": getattr(backtest_result, "max_drawdown_pct", 0.0),
            "avg_drawdown_pct": getattr(backtest_result, "avg_drawdown_pct", 0.0),
            "drawdown_duration_avg": getattr(
                backtest_result, "drawdown_duration_avg", 0.0
            ),
            "volatility_annual_pct": getattr(
                backtest_result, "volatility_annual_pct", 0.0
            ),
            "downside_deviation_pct": getattr(
                backtest_result, "downside_deviation_pct", 0.0
            ),
            "sharpe_ratio": getattr(backtest_result, "sharpe_ratio", 0.0),
            "sortino_ratio": getattr(backtest_result, "sortino_ratio", 0.0),
            "calmar_ratio": getattr(backtest_result, "calmar_ratio", 0.0),
            "jensen_alpha": getattr(backtest_result, "jensen_alpha", 0.0),
            "omega_ratio": getattr(backtest_result, "omega_ratio", 0.0),
            "total_trades": getattr(backtest_result, "total_trades", 0),
            "win_rate": getattr(backtest_result, "win_rate", 0.0),
            "profit_factor": getattr(backtest_result, "profit_factor", 0.0),
            "avg_win_pct": getattr(backtest_result, "avg_win_pct", 0.0),
            "avg_loss_pct": getattr(backtest_result, "avg_loss_pct", 0.0),
            "largest_win": getattr(backtest_result, "largest_win", 0.0),
            "largest_loss": getattr(backtest_result, "largest_loss", 0.0),
            "avg_trade_duration_days": getattr(
                backtest_result, "avg_trade_duration_days", 0.0
            ),
            "monthly_win_rate": getattr(backtest_result, "monthly_win_rate", 0.0),
            "consecutive_wins_max": getattr(backtest_result, "consecutive_wins_max", 0),
            "consecutive_losses_max": getattr(
                backtest_result, "consecutive_losses_max", 0
            ),
            "recovery_factor": getattr(backtest_result, "recovery_factor", 0.0),
            "cagr": getattr(backtest_result, "cagr", 0.0),
            "market_correlation": getattr(backtest_result, "market_correlation", 0.0),
            "beta": getattr(backtest_result, "beta", 0.0),
            "alpha_annualized": getattr(backtest_result, "alpha_annualized", 0.0),
            "fitness_score": getattr(backtest_result, "fitness_score", 0.0),
            "me_ratio_efficiency": getattr(backtest_result, "me_ratio_efficiency", 0.0),
            "pattern_success_rate": getattr(
                backtest_result, "pattern_success_rate", 0.0
            ),
        }

        # Compute advanced metrics and merge them
        if "calculate_advanced_metrics" in globals():
            advanced_metrics = calculate_advanced_metrics(
                daily_returns, benchmark_returns, equity_curve
            )
            metrics.update(advanced_metrics)

        return PerformanceMetrics(**metrics)


class NGSAIPerformanceComparator:
    """
    A comparator class for analyzing and comparing the performance of various AI strategies.
    """

    def comprehensive_comparison(self, data: dict, objectives: list) -> dict:
        """
        Perform a comprehensive comparison of strategies based on objectives.

        Args:
            data (dict): Backtest or strategy data for each strategy.
            objectives (list): Optimization objectives like 'max_roi', 'min_drawdown'.

        Returns:
            dict: Results of the comparison including scores, best strategy, etc.
        """
        print("Performing comprehensive performance comparison across strategies...")

        if not data or not objectives:
            raise ValueError("Invalid data or objectives for comparison.")

        comparison_results = []
        for strategy_name, backtest_result in data.items():
            # Collect detailed metrics for each strategy
            metrics = PerformanceMetrics.calculate_detailed_metrics(
                backtest_result, strategy_name=strategy_name, objective="general"
            )
            overall_score = self._calculate_score(metrics, objectives)
            comparison_results.append((strategy_name, metrics, overall_score))

        # Determine the best strategy based on score
        best_strategy = max(comparison_results, key=lambda x: x[2])
        best_strategy_name, best_metrics, best_score = best_strategy

        return {
            "ai_recommendation_score": round(best_score, 2),
            "best_overall_strategy": best_strategy_name,
            "recommended_allocation": {best_strategy_name: 100.0},
            "original_metrics": comparison_results[0][1],  # Assuming first is the base
            "ai_metrics": [result[1] for result in comparison_results],
            "return_difference_significant": best_score > comparison_results[0][2] * 1.2,
        }

    def _calculate_score(self, metrics: PerformanceMetrics, objectives: list) -> float:
        """
        Calculate an overall score for a strategy's performance based on objectives.

        Args:
            metrics (PerformanceMetrics): The performance metrics of a strategy.
            objectives (list): List of target objectives like 'max_roi', 'sharpe_ratio'.

        Returns:
            float: Calculated score.
        """
        score = 0.0

        scoring_weights = {
            "max_roi": 0.4,
            "sharpe_ratio": 0.3,
            "min_drawdown": 0.2,
            "high_winrate": 0.1,
        }

        for objective in objectives:
            if objective == "max_roi":
                score += metrics.total_return_pct * scoring_weights.get(objective, 0)
            elif objective == "sharpe_ratio":
                score += metrics.sharpe_ratio * scoring_weights.get(objective, 0)
            elif objective == "min_drawdown":
                score -= metrics.max_drawdown_pct * scoring_weights.get(objective, 0)
            elif objective == "high_winrate":
                score += metrics.win_rate * scoring_weights.get(objective, 0)

        return score
