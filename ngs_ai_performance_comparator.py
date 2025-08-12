from dataclasses import dataclass

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
    def calculate_detailed_metrics(
        backtest_result,
        strategy_name: str,
        objective: str
    ):
        """
        Calculate detailed performance metrics based on the backtest result.

        backtest_result: an object with all the necessary attributes:
            - strategy_type
            - daily_returns
            - benchmark_returns
            - equity_curve
            - and other raw stats as needed
        """
        # Example extraction (replace as needed for your backtest_result structure)
        daily_returns = backtest_result.daily_returns
        benchmark_returns = backtest_result.benchmark_returns
        equity_curve = backtest_result.equity_curve

        # Fill out the required metrics from your backtest_result data
        metrics = {
            "strategy_name": strategy_name,
            "strategy_type": getattr(backtest_result, "strategy_type", ""),
            "objective": objective,
            "total_return_pct": getattr(backtest_result, "total_return_pct", 0.0),
            "annualized_return_pct": getattr(backtest_result, "annualized_return_pct", 0.0),
            "monthly_return_avg": getattr(backtest_result, "monthly_return_avg", 0.0),
            "monthly_return_std": getattr(backtest_result, "monthly_return_std", 0.0),
            "max_drawdown_pct": getattr(backtest_result, "max_drawdown_pct", 0.0),
            "avg_drawdown_pct": getattr(backtest_result, "avg_drawdown_pct", 0.0),
            "drawdown_duration_avg": getattr(backtest_result, "drawdown_duration_avg", 0.0),
            "volatility_annual_pct": getattr(backtest_result, "volatility_annual_pct", 0.0),
            "downside_deviation_pct": getattr(backtest_result, "downside_deviation_pct", 0.0),
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
            "avg_trade_duration_days": getattr(backtest_result, "avg_trade_duration_days", 0.0),
            "monthly_win_rate": getattr(backtest_result, "monthly_win_rate", 0.0),
            "consecutive_wins_max": getattr(backtest_result, "consecutive_wins_max", 0),
            "consecutive_losses_max": getattr(backtest_result, "consecutive_losses_max", 0),
            "recovery_factor": getattr(backtest_result, "recovery_factor", 0.0),
            "cagr": getattr(backtest_result, "cagr", 0.0),
            "market_correlation": getattr(backtest_result, "market_correlation", 0.0),
            "beta": getattr(backtest_result, "beta", 0.0),
            "alpha_annualized": getattr(backtest_result, "alpha_annualized", 0.0),
            "fitness_score": getattr(backtest_result, "fitness_score", 0.0),
            "me_ratio_efficiency": getattr(backtest_result, "me_ratio_efficiency", 0.0),
            "pattern_success_rate": getattr(backtest_result, "pattern_success_rate", 0.0),
        }

        # Optionally compute advanced metrics and merge (function should return dict)
        if "calculate_advanced_metrics" in globals():
            advanced_metrics = calculate_advanced_metrics(daily_returns, benchmark_returns, equity_curve)
            metrics.update(advanced_metrics)

        return PerformanceMetrics(**metrics)
