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
    sortino_ratio: float  # Added
    calmar_ratio: float  # Added
    jensen_alpha: float  # Added
    omega_ratio: float  # Added

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
    cagr: float  # Compound Annual Growth Rate; Added

    # Market Metrics
    market_correlation: float
    beta: float
    alpha_annualized: float

    # Custom Metrics
    fitness_score: float
    me_ratio_efficiency: float
    pattern_success_rate: float

    def _calculate_detailed_metrics(
            self, backtest_result: BacktestResult, strategy_name: str, objective: str
    ) -> PerformanceMetrics:
        """Calculate detailed performance metrics."""
        # Existing calculations remain here...
        advanced_metrics = calculate_advanced_metrics(daily_returns, benchmark_returns, equity_curve)
        metrics.update(advanced_metrics)
        return PerformanceMetrics(**metrics)
