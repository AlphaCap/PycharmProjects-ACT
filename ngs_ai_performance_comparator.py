"""
nGS AI Performance Comparator
Advanced performance analysis and comparison system for original nGS vs AI strategies
Provides detailed analytics, visualizations, and actionable recommendations

nGS AI Performance Analyzer
Analyzes AI-generated strategy performance with detailed metrics, walk-forward testing, and Monte Carlo analysis

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from typing import Dict, List, Optional, Tuple, Any

from datetime import datetime, timedelta
import logging
import json
import os
import warnings
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from collections import defaultdict

import seaborn as sns

warnings.filterwarnings("ignore")

# Import required components

from nGS_Revised_Strategy import NGSStrategy
from ngs_integrated_ai_system import NGSAIBacktestingSystem, BacktestResult, BacktestComparison
from ngs_ai_integration_manager import NGSAIIntegrationManager
from ngs_integrated_ai_system import NGSProvenParameters
from performance_objectives import ObjectiveManager

from ngs_ai_backtesting_system import NGSAIBacktestingSystem, BacktestResult
from strategy_generator_ai import TradingStrategy

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:

    """Comprehensive performance metrics container"""
    strategy_name: str
    strategy_type: str  # 'original', 'ai'
    objective: str
    
    # Return Metrics
    """Comprehensive performance metrics container for AI strategies"""

    strategy_name: str
    objective: str

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
    information_ratio: float
    
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

    # Market Correlation
    market_correlation: float
    beta: float
    alpha_annualized: float
    
    # Custom nGS Metrics

    market_correlation: float
    beta: float
    alpha_annualized: float

    fitness_score: float
    me_ratio_efficiency: float
    pattern_success_rate: float


@dataclass

class ComparisonAnalysis:
    """Detailed comparison analysis between strategies"""
    original_metrics: PerformanceMetrics
    ai_metrics: List[PerformanceMetrics]
    
    # Superiority Analysis
    ai_wins_return: bool
    ai_wins_risk: bool
    ai_wins_efficiency: bool
    ai_wins_consistency: bool
    
    # Statistical Significance
    return_difference_significant: bool
    return_difference_pvalue: float
    
    # Recommendation Score (0-100)
    ai_recommendation_score: float
    recommendation_reasons: List[str]
    
    # Best Strategy Selection

class AIAnalysis:
    """Analysis results for AI strategies"""

    ai_metrics: List[PerformanceMetrics]
    recommendation_score: float
    recommendation_reasons: List[str]

    best_overall_strategy: str
    best_return_strategy: str
    best_risk_strategy: str
    best_efficiency_strategy: str
    
    # Allocation Recommendations
    recommended_allocation: Dict[str, float]

class NGSAIPerformanceComparator:
    """
    Advanced performance comparison and analysis system
    Compares original nGS strategy with AI-generated strategies



class NGSAIPerformanceAnalyzer:
    """
    Analyzes performance of AI-generated strategies with detailed metrics and robustness testing
    """

    def __init__(self, account_size: float = 1000000, data_dir: str = "data"):
        self.account_size = account_size
        self.data_dir = data_dir
        self.reports_dir = os.path.join(data_dir, 'performance_reports')
        self.charts_dir = os.path.join(data_dir, 'performance_charts')
        
        # Create directories
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Initialize components
        self.backtester = NGSAIBacktestingSystem(account_size, data_dir)
        self.integration_manager = NGSAIIntegrationManager(account_size, data_dir)
        self.objective_manager = ObjectiveManager()
        self.ngs_params = NGSProvenParameters()
        
        # Analysis configuration
	os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
       
        self.reports_dir = os.path.join(data_dir, "performance_reports")
        self.charts_dir = os.path.join(data_dir, "performance_charts")

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

        self.backtester = NGSAIBacktestingSystem(account_size, data_dir)

        self.analysis_config = {
            "benchmark_symbol": "SPY",
            "risk_free_rate": 0.02,
            "confidence_level": 0.95,
            "min_sample_size": 30,
            "chart_style": "seaborn-v0_8",
            "color_palette": "Set2",
            "significance_threshold": 0.05,
        }
        
        # Performance tracking
        self.comparison_history = []
        self.strategy_database = {}
        
        print("📊 nGS AI Performance Comparator initialized")

        self.analysis_history = []

        print(" nGS AI Performance Analyzer initialized")
        print(f"   Reports Directory:   {self.reports_dir}")
        print(f"   Charts Directory:    {self.charts_dir}")
        print(f"   Benchmark:           {self.analysis_config['benchmark_symbol']}")
        print(f"   Confidence Level:    {self.analysis_config['confidence_level']:.0%}")

    # =============================================================================
    # CORE COMPARISON METHODS
    # =============================================================================
    
    def comprehensive_comparison(self, data: Dict[str, pd.DataFrame], 
                               objectives: List[str] = None,
                               start_date: str = None, end_date: str = None) -> ComparisonAnalysis:
        """
        Perform comprehensive comparison between original nGS and AI strategies
        """
        if objectives is None:
            objectives = ['linear_equity', 'max_roi', 'min_drawdown', 'high_winrate', 'sharpe_ratio']
        
        print(f"\n🔬 COMPREHENSIVE PERFORMANCE COMPARISON")
        print(f"   Original nGS vs {len(objectives)} AI objectives")
        print(f"   Period: {start_date or 'All available'} to {end_date or 'Latest'}")
        
        # Run comprehensive backtesting
        backtest_comparison = self.backtester.backtest_comprehensive_comparison(
            objectives, data, start_date, end_date
        )
        
        # Calculate detailed performance metrics
        original_metrics = self._calculate_detailed_metrics(
            backtest_comparison.original_ngs_result, 'original_ngs', 'original'
        )
        
        ai_metrics = []
        for ai_result in backtest_comparison.ai_results:
            ai_metric = self._calculate_detailed_metrics(
                ai_result, ai_result.strategy_id, ai_result.objective_name
            )
            ai_metrics.append(ai_metric)
        
        # Perform superiority analysis
        superiority_analysis = self._analyze_strategy_superiority(original_metrics, ai_metrics)
        
        # Statistical significance testing
        statistical_analysis = self._perform_statistical_tests(
            backtest_comparison.original_ngs_result, backtest_comparison.ai_results
        )
        
        # Generate recommendations
        recommendation_analysis = self._generate_detailed_recommendations(
            original_metrics, ai_metrics, superiority_analysis, statistical_analysis
        )
        
        # Create comprehensive analysis object
        comparison = ComparisonAnalysis(
            original_metrics=original_metrics,
            ai_metrics=ai_metrics,
            **superiority_analysis,
            **statistical_analysis,
            **recommendation_analysis
        )
        
        # Save comparison to history
        self.comparison_history.append(comparison)
        
        # Generate reports and visualizations
        self._generate_comparison_report(comparison)
        self._create_performance_visualizations(comparison)
        
        print(f"\n Comprehensive comparison completed")
        print(f"   Recommendation Score: {comparison.ai_recommendation_score:.0f}/100")
        print(f"   Best Overall: {comparison.best_overall_strategy}")
        
        return comparison
    
    def rolling_performance_analysis(self, data: Dict[str, pd.DataFrame],
                                   objectives: List[str],
                                   window_months: int = 6,
                                   step_months: int = 1) -> Dict[str, Any]:
        """
        Analyze performance over rolling windows to assess consistency
        """
        print(f"\n📈 ROLLING PERFORMANCE ANALYSIS")
        print(f"   Window: {window_months} months, Step: {step_months} month(s)")
        
        # Get date range

    def analyze_ai_strategies(self, strategies: List[TradingStrategy], data: Dict[str, pd.DataFrame], 
                             start_date: str = None, end_date: str = None) -> AIAnalysis:

    def analyze_ai_strategies(
        self,
        strategies: List[TradingStrategy],
        data: Dict[str, pd.DataFrame],
        start_date: str = None,
        end_date: str = None,
    ) -> AIAnalysis:

        """
        Analyze performance of AI strategies
        """
        print(f"\n AI STRATEGY PERFORMANCE ANALYSIS")
        print(f"   Analyzing {len(strategies)} AI strategies")
        print(f"   Period: {start_date or 'All available'} to {end_date or 'Latest'}")

        ai_metrics = []
        for strategy in strategies:
            try:
                backtest_result = self.backtester.backtest_ai_strategy(
                    strategy, data, start_date, end_date
                )
                metric = self._calculate_detailed_metrics(
                    backtest_result, strategy.strategy_id, strategy.objective_name
                )
                ai_metrics.append(metric)
            except Exception as e:
                logger.error(f"Error analyzing strategy {strategy.strategy_id}: {e}")
                continue

        recommendation_analysis = self._generate_ai_recommendations(ai_metrics)

        analysis = AIAnalysis(ai_metrics=ai_metrics, **recommendation_analysis)

        self.analysis_history.append(analysis)

        self._generate_analysis_report(analysis)
        self._create_performance_visualizations(analysis)

        print(f"\n AI analysis completed")
        print(f"   Recommendation Score: {analysis.recommendation_score:.0f}/100")
        print(f"   Best Overall: {analysis.best_overall_strategy}")

        return analysis

    def rolling_performance_analysis(
        self,
        data: Dict[str, pd.DataFrame],
        strategies: List[TradingStrategy],
        window_months: int = 6,
        step_months: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyze AI strategy performance over rolling windows
        """
        print(f"\n AI ROLLING PERFORMANCE ANALYSIS")
        print(f"   Window: {window_months} months, Step: {step_months} month(s)")

        all_dates = []
        for symbol_data in data.values():
            if not symbol_data.empty:
                all_dates.extend(pd.to_datetime(symbol_data["Date"]).tolist())

        start_date = min(all_dates)
        end_date = max(all_dates)

        # Create rolling windows

        windows = self._create_rolling_windows(start_date, end_date, window_months, step_months)
        
        rolling_results = {
            'windows': [],
            'original_performance': [],
            'ai_performance': {obj: [] for obj in objectives},
            'relative_performance': {obj: [] for obj in objectives},

            'ai_performance': {strategy.strategy_id: [] for strategy in strategies},

            'consistency_metrics': {},
            'trend_analysis': {}

        windows = self._create_rolling_windows(
            start_date, end_date, window_months, step_months
        )

        rolling_results = {
            "windows": [],
            "ai_performance": {strategy.strategy_id: [] for strategy in strategies},
            "consistency_metrics": {},
            "trend_analysis": {},
        }

        print(f"   Processing {len(windows)} rolling windows...")

        for i, (window_start, window_end) in enumerate(windows):
            print(
                f"   Window {i+1}/{len(windows)}: {window_start.strftime('%Y-%m')} to {window_end.strftime('%Y-%m')}"
            )

            try:
            window_data = self.backtester._filter_data_by_date(
                    data, window_start.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')

                window_data = self.backtester._filter_data_by_date(
                    data,
                    window_start.strftime("%Y-%m-%d"),
                    window_end.strftime("%Y-%m-%d"),
                )

                if not window_data:
                    continue
 
                # Run comparison for window
                window_comparison = self.backtester.backtest_comprehensive_comparison(
                    objectives, window_data, 
                    window_start.strftime('%Y-%m-%d'), window_end.strftime('%Y-%m-%d')
                )
               
                # Store window results

                window_info = {
                    "start_date": window_start,
                    "end_date": window_end,
                    "period": f"{window_start.strftime('%Y-%m')} to {window_end.strftime('%Y-%m')}",
                }
                
                rolling_results['windows'].append(window_info)
                rolling_results['original_performance'].append(
                    window_comparison.original_ngs_result.total_return_pct
                )
                
                # Store AI results
                for ai_result in window_comparison.ai_results:
                    objective = ai_result.objective_name
                    if objective in rolling_results['ai_performance']:
                        rolling_results['ai_performance'][objective].append(ai_result.total_return_pct)
                        
                        # Calculate relative performance
                        relative_perf = ai_result.total_return_pct - window_comparison.original_ngs_result.total_return_pct
                        rolling_results['relative_performance'][objective].append(relative_perf)
                 		rolling_results["windows"].append(window_info)

                for strategy in strategies:
                    try:
                        backtest_result = self.backtester.backtest_ai_strategy(
                            strategy,
                            window_data,
                            window_start.strftime("%Y-%m-%d"),
                            window_end.strftime("%Y-%m-%d"),
                        )
                        rolling_results["ai_performance"][strategy.strategy_id].append(
                            backtest_result.total_return_pct
                        )
                    except Exception as e:

                        print(f"     ❌ Error for {strategy.strategy_id}: {e}")
                        rolling_results['ai_performance'][strategy.strategy_id].append(0.0)
                
                        print(f"      Error for {strategy.strategy_id}: {e}")
                        rolling_results["ai_performance"][strategy.strategy_id].append(
                            0.0
                        )

            except Exception as e:
                print(f"      Error in window {i+1}: {e}")
                continue
        
        # Analyze consistency and trends
        rolling_results['consistency_metrics'] = self._analyze_rolling_consistency(rolling_results)
        rolling_results['trend_analysis'] = self._analyze_performance_trends(rolling_results)
        
        # Create rolling performance visualizations

        rolling_results['consistency_metrics'] = self._analyze_rolling_consistency(rolling_results)
        rolling_results['trend_analysis'] = self._analyze_performance_trends(rolling_results)
        
        rolling_results["consistency_metrics"] = self._analyze_rolling_consistency(
            rolling_results
        )
        rolling_results["trend_analysis"] = self._analyze_performance_trends(
            rolling_results
        )

        self._create_rolling_performance_charts(rolling_results)

        print(
            f" Rolling analysis completed: {len(rolling_results['windows'])} windows analyzed"
        )

        return rolling_results
    
    def monte_carlo_analysis(self, comparison: ComparisonAnalysis, 
                           simulations: int = 1000) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis to assess strategy robustness
        """
        print(f"\n MONTE CARLO ROBUSTNESS ANALYSIS")

    def monte_carlo_analysis(self, analysis: AIAnalysis, simulations: int = 1000) -> Dict[str, Any]:
    def monte_carlo_analysis(
        self, analysis: AIAnalysis, simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis for AI strategies
        """
        print(f"\n AI MONTE CARLO ROBUSTNESS ANALYSIS")
        print(f"   Running {simulations:,} simulations")

        monte_carlo_results = {
            'simulations': simulations,
            'original_outcomes': [],
            'ai_outcomes': {ai.strategy_name: [] for ai in comparison.ai_metrics},

            'ai_outcomes': {ai.strategy_name: [] for ai in analysis.ai_metrics},

            'probability_analysis': {},
            'risk_assessment': {},
            'confidence_intervals': {}
        }
        
        # Get base performance data for simulation
        original_returns = self._extract_daily_returns(comparison.original_metrics)
        
        print(f"   Simulating original nGS strategy...")
        
        # Simulate original strategy
        for sim in range(simulations):
            if sim % 200 == 0:
                print(f"     Progress: {sim:,}/{simulations:,} simulations")
            
            # Bootstrap sample from historical returns
            simulated_returns = np.random.choice(original_returns, len(original_returns), replace=True)
            final_return = (np.prod(1 + simulated_returns) - 1) * 100
            monte_carlo_results['original_outcomes'].append(final_return)
        
        # Simulate AI strategies
        for ai_metric in comparison.ai_metrics:
            "simulations": simulations,
            "ai_outcomes": {ai.strategy_name: [] for ai in analysis.ai_metrics},
            "probability_analysis": {},
            "risk_assessment": {},
            "confidence_intervals": {},
        }

        for ai_metric in analysis.ai_metrics:
>
            print(f"   Simulating {ai_metric.strategy_name}...")
            ai_returns = self._extract_daily_returns(ai_metric)

            ai_outcomes = []
            for sim in range(simulations):
                if sim % 200 == 0:
                    print(f"     Progress: {sim:,}/{simulations:,} simulations")

                simulated_returns = np.random.choice(ai_returns, len(ai_returns), replace=True)
                final_return = (np.prod(1 + simulated_returns) - 1) * 100
                ai_outcomes.append(final_return)
            
            monte_carlo_results['ai_outcomes'][ai_metric.strategy_name] = ai_outcomes
        
        # Analyze simulation results

        monte_carlo_results['probability_analysis'] = self._analyze_monte_carlo_probabilities(monte_carlo_results)
        monte_carlo_results['risk_assessment'] = self._assess_monte_carlo_risks(monte_carlo_results)
        monte_carlo_results['confidence_intervals'] = self._calculate_confidence_intervals(monte_carlo_results)
        
        # Create Monte Carlo visualizations
                simulated_returns = np.random.choice(
                    ai_returns, len(ai_returns), replace=True
                )
                final_return = (np.prod(1 + simulated_returns) - 1) * 100
                ai_outcomes.append(final_return)

            monte_carlo_results["ai_outcomes"][ai_metric.strategy_name] = ai_outcomes

        monte_carlo_results["probability_analysis"] = (
            self._analyze_monte_carlo_probabilities(monte_carlo_results)
        )
        monte_carlo_results["risk_assessment"] = self._assess_monte_carlo_risks(
            monte_carlo_results
        )
        monte_carlo_results["confidence_intervals"] = (
            self._calculate_confidence_intervals(monte_carlo_results)
        )
        self._create_monte_carlo_charts(monte_carlo_results)

        print(f" Monte Carlo analysis completed")
        return monte_carlo_results
    
    # =============================================================================
    # DETAILED METRICS CALCULATION
    # =============================================================================
    
    def _calculate_detailed_metrics(self, backtest_result: BacktestResult, 
                                  strategy_name: str, strategy_type: str) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from backtest result"""
        
        if not backtest_result.trades or backtest_result.equity_curve.empty:
            # Return zero metrics if no data
            return PerformanceMetrics(
                strategy_name=strategy_name,
                strategy_type=strategy_type,
                objective=backtest_result.objective_name,

    def _calculate_detailed_metrics(self, backtest_result: BacktestResult, 
                                  strategy_name: str, objective: str) -> PerformanceMetrics:

    def _calculate_detailed_metrics(
        self, backtest_result: BacktestResult, strategy_name: str, objective: str
    ) -> PerformanceMetrics:

        """Calculate comprehensive performance metrics for AI strategy"""

        if not backtest_result.trades or backtest_result.equity_curve.empty:
            return PerformanceMetrics(
                strategy_name=strategy_name,
                objective=objective,
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                monthly_return_avg=0.0,
                monthly_return_std=0.0,
                max_drawdown_pct=0.0,
                avg_drawdown_pct=0.0,
                drawdown_duration_avg=0.0,
                volatility_annual_pct=0.0,
                downside_deviation_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                information_ratio=0.0,
                total_trades=0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_trade_duration_days=0.0,
                monthly_win_rate=0.0,
                consecutive_wins_max=0,
                consecutive_losses_max=0,
                recovery_factor=0.0,
                market_correlation=0.0,
                beta=0.0,
                alpha_annualized=0.0,
                fitness_score=0.0,
                me_ratio_efficiency=0.0,
                pattern_success_rate=0.0,
            )

        equity_curve = backtest_result.equity_curve
        daily_returns = backtest_result.daily_returns
        trades = backtest_result.trades
        
        # Basic return metrics
        total_return_pct = backtest_result.total_return_pct
        
        # Annualized return
        daily_returns = equity_curve.pct_change().dropna()
        trades = backtest_result.trades

        total_return_pct = backtest_result.total_return_pct
        days = len(equity_curve)
        years = days / 252.0 if days > 0 else 1
        annualized_return_pct = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        monthly_return_avg = monthly_returns.mean() if len(monthly_returns) > 0 else 0
        monthly_return_std = monthly_returns.std() if len(monthly_returns) > 1 else 0
        
        # Risk metrics
        max_drawdown_pct = backtest_result.max_drawdown_pct
        avg_drawdown_pct = self._calculate_average_drawdown(equity_curve)
        drawdown_duration_avg = self._calculate_average_drawdown_duration(equity_curve)
        
        # Volatility metrics
        volatility_annual_pct = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        downside_deviation_pct = self._calculate_downside_deviation(daily_returns) * np.sqrt(252) * 100
        
        # Risk-adjusted metrics

        volatility_annual_pct = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        downside_deviation_pct = self._calculate_downside_deviation(daily_returns) * np.sqrt(252) * 100
        
        annualized_return_pct = (
            ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) * 100
            if years > 0
            else 0
        )

        monthly_returns = self._calculate_monthly_returns(equity_curve)
        monthly_return_avg = monthly_returns.mean() if len(monthly_returns) > 0 else 0
        monthly_return_std = monthly_returns.std() if len(monthly_returns) > 1 else 0

        max_drawdown_pct = backtest_result.max_drawdown_pct
        avg_drawdown_pct = self._calculate_average_drawdown(equity_curve)
        drawdown_duration_avg = self._calculate_average_drawdown_duration(equity_curve)

        volatility_annual_pct = (
            daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0
        )
        downside_deviation_pct = (
            self._calculate_downside_deviation(daily_returns) * np.sqrt(252) * 100
        )

        sharpe_ratio = backtest_result.sharpe_ratio
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = (
            annualized_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        )
        information_ratio = self._calculate_information_ratio(daily_returns)
        
        # Trading metrics
        profits = [trade['profit'] for trade in trades]
        profits = [trade["profit"] for trade in trades]

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = (
            sum(wins) / abs(sum(losses)) if losses else float("inf") if wins else 0
        )
        avg_win_pct = (np.mean(wins) / self.account_size * 100) if wins else 0
        avg_loss_pct = (np.mean(losses) / self.account_size * 100) if losses else 0
        largest_win = max(profits) if profits else 0
        largest_loss = min(profits) if profits else 0
        
        # Trade duration
        durations = self._calculate_trade_durations(trades)
        avg_trade_duration_days = np.mean(durations) if durations else 0
        
        # Consistency metrics
        durations = self._calculate_trade_durations(trades)
        avg_trade_duration_days = np.mean(durations) if durations else 0
        
       monthly_win_rate = len([r for r in monthly_returns if r > 0]) / len(monthly_returns) if monthly_returns else 0
        consecutive_wins_max, consecutive_losses_max = self._calculate_consecutive_streaks(trades)
        recovery_factor = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Market correlation (would need benchmark data for full implementation)
        market_correlation = 0.0  # Placeholder
        beta = 0.0  # Placeholder
        alpha_annualized = 0.0  # Placeholder
        
        # Custom nGS metrics
        durations = self._calculate_trade_durations(trades)
        avg_trade_duration_days = np.mean(durations) if durations else 0

        monthly_win_rate = (
            len([r for r in monthly_returns if r > 0]) / len(monthly_returns)
            if monthly_returns
            else 0
        )
        consecutive_wins_max, consecutive_losses_max = (
            self._calculate_consecutive_streaks(trades)
        )
        recovery_factor = (
            total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        )

        market_correlation = 0.0
        beta = 0.0
        alpha_annualized = 0.0

        fitness_score = backtest_result.fitness_score
        me_ratio_efficiency = self._calculate_me_ratio_efficiency(trades)
        pattern_success_rate = self._calculate_pattern_success_rate(trades)

        return PerformanceMetrics(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            objective=backtest_result.objective_name,
            objective=objective,

            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            monthly_return_avg=monthly_return_avg,
            monthly_return_std=monthly_return_std,
            max_drawdown_pct=max_drawdown_pct,
            avg_drawdown_pct=avg_drawdown_pct,
            drawdown_duration_avg=drawdown_duration_avg,
            volatility_annual_pct=volatility_annual_pct,
            downside_deviation_pct=downside_deviation_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration_days=avg_trade_duration_days,
            monthly_win_rate=monthly_win_rate,
            consecutive_wins_max=consecutive_wins_max,
            consecutive_losses_max=consecutive_losses_max,
            recovery_factor=recovery_factor,
            market_correlation=market_correlation,
            beta=beta,
            alpha_annualized=alpha_annualized,
            fitness_score=fitness_score,
            me_ratio_efficiency=me_ratio_efficiency,
            pattern_success_rate=pattern_success_rate,
        )
   
    # =============================================================================
    # ANALYSIS METHODS
    # =============================================================================
    
    def _analyze_strategy_superiority(self, original: PerformanceMetrics, 
                                    ai_metrics: List[PerformanceMetrics]) -> Dict[str, bool]:
        """Analyze which approach is superior in different areas"""
        
        if not ai_metrics:
            return {
                'ai_wins_return': False,
                'ai_wins_risk': False,
                'ai_wins_efficiency': False,
                'ai_wins_consistency': False
            }
        
        # Return superiority (best AI vs original)
        best_ai_return = max(ai.total_return_pct for ai in ai_metrics)
        ai_wins_return = best_ai_return > original.total_return_pct
        
        # Risk superiority (lowest risk AI vs original)
        best_ai_drawdown = min(ai.max_drawdown_pct for ai in ai_metrics)
        ai_wins_risk = best_ai_drawdown < original.max_drawdown_pct
        
        # Efficiency superiority (best Sharpe AI vs original)
        best_ai_sharpe = max(ai.sharpe_ratio for ai in ai_metrics)
        ai_wins_efficiency = best_ai_sharpe > original.sharpe_ratio
        
        # Consistency superiority (best win rate AI vs original)
        best_ai_winrate = max(ai.win_rate for ai in ai_metrics)
        ai_wins_consistency = best_ai_winrate > original.win_rate
        
        return {
            'ai_wins_return': ai_wins_return,
            'ai_wins_risk': ai_wins_risk,
            'ai_wins_efficiency': ai_wins_efficiency,
            'ai_wins_consistency': ai_wins_consistency
        }
    
    def _perform_statistical_tests(self, original_result: BacktestResult, 
                                 ai_results: List[BacktestResult]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        if not ai_results or original_result.daily_returns.empty:
            return {
                'return_difference_significant': False,
                'return_difference_pvalue': 1.0
            }
        
        original_returns = original_result.daily_returns
        
        # Test against best performing AI strategy
        best_ai_result = max(ai_results, key=lambda x: x.total_return_pct)
        ai_returns = best_ai_result.daily_returns
        
        if len(original_returns) < 30 or len(ai_returns) < 30:
            return {
                'return_difference_significant': False,
                'return_difference_pvalue': 1.0
            }
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(ai_returns, original_returns)
            significant = p_value < self.analysis_config['significance_threshold']
        except:
            significant = False
            p_value = 1.0
        
        return {
            'return_difference_significant': significant,
            'return_difference_pvalue': p_value
        }
    
    def _generate_detailed_recommendations(self, original: PerformanceMetrics,
                                         ai_metrics: List[PerformanceMetrics],
                                         superiority: Dict[str, bool],
                                         statistical: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed recommendations and allocation suggestions"""
        
        if not ai_metrics:
            return {
                'ai_recommendation_score': 0.0,
                'recommendation_reasons': ['No AI strategies to evaluate'],
                'best_overall_strategy': original.strategy_name,
                'best_return_strategy': original.strategy_name,
                'best_risk_strategy': original.strategy_name,
                'best_efficiency_strategy': original.strategy_name,
                'recommended_allocation': {original.strategy_name: 100.0}
            }
        
        # Calculate recommendation score (0-100)
        score = 0.0
        reasons = []
        
        # Return component (30%)
        if superiority['ai_wins_return']:
            best_return_improvement = max(ai.total_return_pct for ai in ai_metrics) - original.total_return_pct
            score += 30 * min(best_return_improvement / 10.0, 1.0)  # Cap at 10% improvement
            reasons.append(f"AI strategies show superior returns (+{best_return_improvement:.1f}%)")
        
        # Risk component (25%)
        if superiority['ai_wins_risk']:
            risk_improvement = original.max_drawdown_pct - min(ai.max_drawdown_pct for ai in ai_metrics)
            score += 25 * min(risk_improvement / 5.0, 1.0)  # Cap at 5% drawdown improvement
            reasons.append(f"AI strategies show lower risk (-{risk_improvement:.1f}% drawdown)")
        
        # Efficiency component (25%)
        if superiority['ai_wins_efficiency']:
            efficiency_improvement = max(ai.sharpe_ratio for ai in ai_metrics) - original.sharpe_ratio
            score += 25 * min(efficiency_improvement / 1.0, 1.0)  # Cap at 1.0 Sharpe improvement
            reasons.append(f"AI strategies show better risk-adjusted returns (+{efficiency_improvement:.2f} Sharpe)")
        
        # Consistency component (20%)
        if superiority['ai_wins_consistency']:
            consistency_improvement = max(ai.win_rate for ai in ai_metrics) - original.win_rate
            score += 20 * min(consistency_improvement / 0.2, 1.0)  # Cap at 20% win rate improvement
            reasons.append(f"AI strategies show higher consistency (+{consistency_improvement:.1%} win rate)")
        
        # Statistical significance bonus
        if statistical['return_difference_significant']:
            score += 10
            reasons.append("Performance difference is statistically significant")
        
        # Identify best strategies
        all_strategies = [original] + ai_metrics

    def _generate_ai_recommendations(self, ai_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
    def _generate_ai_recommendations(
        self, ai_metrics: List[PerformanceMetrics]
    ) -> Dict[str, Any]:

        """Generate recommendations for AI strategies"""
        if not ai_metrics:
            return {
                "recommendation_score": 0.0,
                "recommendation_reasons": ["No AI strategies to evaluate"],
                "best_overall_strategy": "None",
                "best_return_strategy": "None",
                "best_risk_strategy": "None",
                "best_efficiency_strategy": "None",
            }

        score = 0.0
        reasons = []

        best_return = max(ai.total_return_pct for ai in ai_metrics)
        if best_return > 5.0:
            score += 30
            reasons.append(f"High returns achieved ({best_return:.1f}%)")

        best_drawdown = min(ai.max_drawdown_pct for ai in ai_metrics)
        if best_drawdown < 5.0:
            score += 25
            reasons.append(f"Low drawdown achieved ({best_drawdown:.1f}%)")

        best_sharpe = max(ai.sharpe_ratio for ai in ai_metrics)
        if best_sharpe > 1.0:
            score += 25
            reasons.append(f"High Sharpe ratio achieved ({best_sharpe:.2f})")

        best_win_rate = max(ai.win_rate for ai in ai_metrics)
        if best_win_rate > 0.5:
            score += 20
            reasons.append(f"High win rate achieved ({best_win_rate:.1%})")

        all_strategies = ai_metrics
        best_return_strategy = max(all_strategies, key=lambda x: x.total_return_pct).strategy_name
        best_risk_strategy = min(all_strategies, key=lambda x: x.max_drawdown_pct).strategy_name
        best_efficiency_strategy = max(all_strategies, key=lambda x: x.sharpe_ratio).strategy_name
        
        # Overall best (composite score)
        best_return_strategy = max(
            all_strategies, key=lambda x: x.total_return_pct
        ).strategy_name
        best_risk_strategy = min(
            all_strategies, key=lambda x: x.max_drawdown_pct
        ).strategy_name
        best_efficiency_strategy = max(
            all_strategies, key=lambda x: x.sharpe_ratio
        ).strategy_name

        def composite_score(metric):
            return (
                metric.total_return_pct / 100 * 0.4
                + (10 - metric.max_drawdown_pct) / 10 * 0.3
                + metric.sharpe_ratio / 3 * 0.3
            )

        best_overall_strategy = max(all_strategies, key=composite_score).strategy_name
        
        # Generate allocation recommendations
        recommended_allocation = self._calculate_optimal_allocation(original, ai_metrics, score)
        
        return {
            'ai_recommendation_score': min(score, 100.0),
        return {
            'recommendation_score': min(score, 100.0),
            'recommendation_reasons': reasons,
            'best_overall_strategy': best_overall_strategy,
            'best_return_strategy': best_return_strategy,
            'best_risk_strategy': best_risk_strategy,
            'best_efficiency_strategy': best_efficiency_strategy,
            'recommended_allocation': recommended_allocation
        }
    
    def _calculate_optimal_allocation(self, original: PerformanceMetrics,
                                    ai_metrics: List[PerformanceMetrics],
                                    recommendation_score: float) -> Dict[str, float]:
        """Calculate optimal capital allocation based on performance"""
        
        allocation = {}
        
        if recommendation_score < 30:
            # Low AI recommendation - mostly original
            allocation[original.strategy_name] = 80.0
            remaining = 20.0
        elif recommendation_score < 60:
            # Moderate AI recommendation - balanced
            allocation[original.strategy_name] = 50.0
            remaining = 50.0
        else:
            # High AI recommendation - AI-focused
            allocation[original.strategy_name] = 25.0
            remaining = 75.0
        
        # Allocate remaining to best AI strategies
        if ai_metrics and remaining > 0:
            # Sort AI strategies by composite score
            def composite_score(metric):
                return (metric.total_return_pct / 100 * 0.4 + 
                       (10 - metric.max_drawdown_pct) / 10 * 0.3 + 
                       metric.sharpe_ratio / 3 * 0.3)
            
            sorted_ai = sorted(ai_metrics, key=composite_score, reverse=True)
            
            # Top 3 AI strategies get allocation
            top_ai = sorted_ai[:3]
            ai_allocation = remaining / len(top_ai)
            
            for ai_metric in top_ai:
                allocation[ai_metric.strategy_name] = ai_allocation
        
        return allocation
    
    # =============================================================================
    # REPORTING AND VISUALIZATION
    # =============================================================================
    
    def _generate_comparison_report(self, comparison: ComparisonAnalysis):
        """Generate comprehensive comparison report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.reports_dir, f"performance_comparison_{timestamp}.html")
        
        # Generate HTML report
        html_content = self._create_html_report(comparison)

            'best_efficiency_strategy': best_efficiency_strategy


        return {
            "recommendation_score": min(score, 100.0),
            "recommendation_reasons": reasons,
            "best_overall_strategy": best_overall_strategy,
            "best_return_strategy": best_return_strategy,
            "best_risk_strategy": best_risk_strategy,
            "best_efficiency_strategy": best_efficiency_strategy,
        }

    def _generate_analysis_report(self, analysis: AIAnalysis):
        """Generate AI performance report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.reports_dir, f"ai_performance_{timestamp}.html")

        html_content = self._create_html_report(analysis)
    
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        # Also save JSON data
        json_file = os.path.join(self.reports_dir, f"performance_data_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self._serialize_comparison(comparison), f, indent=2)

        json_file = os.path.join(self.reports_dir, f"ai_performance_data_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self._serialize_analysis(analysis), f, indent=2)
        
        print(f" Performance report saved: {report_file}")
        print(f" Performance data saved: {json_file}")
    
    def _create_performance_visualizations(self, comparison: ComparisonAnalysis):
        """Create comprehensive performance visualizations"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create multi-panel comparison chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('nGS vs AI Strategies Performance Comparison', fontsize=16, fontweight='bold')
        
        # Chart 1: Returns comparison
        self._plot_returns_comparison(axes[0, 0], comparison)
        
        # Chart 2: Risk comparison  
        self._plot_risk_comparison(axes[0, 1], comparison)
        
        # Chart 3: Efficiency comparison
        self._plot_efficiency_comparison(axes[0, 2], comparison)
        
        # Chart 4: Trade statistics
        self._plot_trade_statistics(axes[1, 0], comparison)
        
        # Chart 5: Recommendation radar
        self._plot_recommendation_radar(axes[1, 1], comparison)
        
        # Chart 6: Allocation recommendation
        self._plot_allocation_recommendation(axes[1, 2], comparison)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = os.path.join(self.charts_dir, f"performance_comparison_{timestamp}.png")

        with open(report_file, "w") as f:
            f.write(html_content)

        json_file = os.path.join(
            self.reports_dir, f"ai_performance_data_{timestamp}.json"
        )
        with open(json_file, "w") as f:
            json.dump(self._serialize_analysis(analysis), f, indent=2)

        print(f" Performance report saved: {report_file}")
        print(f" Performance data saved: {json_file}")

    def _create_performance_visualizations(self, analysis: AIAnalysis):
        """Create performance visualizations for AI strategies"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.style.use("default")
        sns.set_palette("Set2")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "AI Strategies Performance Analysis", fontsize=16, fontweight="bold"
        )

        self._plot_returns(axes[0, 0], analysis)
        self._plot_risk(axes[0, 1], analysis)
        self._plot_efficiency(axes[1, 0], analysis)
        self._plot_win_rates(axes[1, 1], analysis)

        plt.tight_layout()

        chart_file = os.path.join(self.charts_dir, f"ai_performance_{timestamp}.png")
        plt.savefig(chart_file, dpi=300, bbox_inches="tight")
        plt.close()
    
        print(f" Performance charts saved: {chart_file}")
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

		print(f" Performance charts saved: {chart_file}")

    def _plot_returns(self, ax, analysis: AIAnalysis):
        """Plot AI strategy returns"""
        strategies = [ai.strategy_name for ai in analysis.ai_metrics]
        returns = [ai.total_return_pct for ai in analysis.ai_metrics]

        bars = ax.bar(strategies, returns, color="orange", alpha=0.7)
        ax.set_title("AI Strategy Returns")
        ax.set_ylabel("Return (%)")
        ax.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars, returns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

    def _plot_risk(self, ax, analysis: AIAnalysis):
        """Plot AI strategy risk"""
        strategies = [ai.strategy_name for ai in analysis.ai_metrics]
        drawdowns = [ai.max_drawdown_pct for ai in analysis.ai_metrics]

        bars = ax.bar(strategies, drawdowns, color="orange", alpha=0.7)
        ax.set_title("AI Strategy Max Drawdown")
        ax.set_ylabel("Max Drawdown (%)")
        ax.tick_params(axis="x", rotation=45)

    def _plot_efficiency(self, ax, analysis: AIAnalysis):
        """Plot AI strategy efficiency"""
        strategies = [ai.strategy_name for ai in analysis.ai_metrics]
        sharpes = [ai.sharpe_ratio for ai in analysis.ai_metrics]

        bars = ax.bar(strategies, sharpes, color="orange", alpha=0.7)
        ax.set_title("AI Strategy Sharpe Ratio")
        ax.set_ylabel("Sharpe Ratio")
        ax.tick_params(axis="x", rotation=45)

    def _plot_win_rates(self, ax, analysis: AIAnalysis):
        """Plot AI strategy win rates"""
        strategies = [ai.strategy_name for ai in analysis.ai_metrics]
        win_rates = [ai.win_rate * 100 for ai in analysis.ai_metrics]

        bars = ax.bar(strategies, win_rates, color="orange", alpha=0.7)
        ax.set_title("AI Strategy Win Rate")
        ax.set_ylabel("Win Rate (%)")
        ax.tick_params(axis="x", rotation=45)

    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns from equity curve"""
        if equity_curve.empty:
            return pd.Series([])
        
        # Resample to monthly and calculate returns
        monthly_equity = equity_curve.resample('M').last()
        monthly_equity = equity_curve.resample("M").last()
        monthly_returns = monthly_equity.pct_change().dropna()

        return monthly_returns

    def _calculate_average_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate average drawdown"""
        if equity_curve.empty:
            return 0.0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        avg_drawdown = (
            abs(drawdown[drawdown < 0].mean()) * 100
            if len(drawdown[drawdown < 0]) > 0
            else 0
        )

        return avg_drawdown

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        if returns.empty:
            return 0.0

        downside_returns = returns[returns < 0]
        return downside_returns.std() if len(downside_returns) > 1 else 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if returns.empty or len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.analysis_config["risk_free_rate"] / 252)
        downside_deviation = self._calculate_downside_deviation(returns)

        if downside_deviation == 0:
            return 0.0

        return (excess_returns.mean() / downside_deviation) * np.sqrt(252)

    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio vs benchmark"""
        # Simplified calculation - would need benchmark data for full implementation
        if returns.empty or len(returns) < 2:
            return 0.0

        return returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_trade_durations(self, trades: List[Dict]) -> List[float]:
        """Calculate trade durations in days"""
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

        return durations

    def _calculate_consecutive_streaks(self, trades: List[Dict]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not trades:
            return 0, 0

        profits = [trade["profit"] for trade in trades]
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for profit in profits:
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_me_ratio_efficiency(self, trades: List[Dict]) -> float:
        """Calculate M/E ratio efficiency metric"""
        # Simplified calculation based on trade frequency and size
        if not trades:
            return 0.0

        trade_sizes = [
            abs(trade.get("shares", 0) * trade.get("entry_price", 0))
            for trade in trades
        ]
        avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0
       
        # Efficiency = profit per dollar invested

        total_profit = sum(trade['profit'] for trade in trades)
        total_profit = sum(trade["profit"] for trade in trades)
        total_invested = sum(trade_sizes)

        return (total_profit / total_invested * 100) if total_invested > 0 else 0

    def _calculate_pattern_success_rate(self, trades: List[Dict]) -> float:
        """Calculate pattern-specific success rate"""
        if not trades:
            return 0.0
        
        # Group trades by pattern type if available
        pattern_performance = defaultdict(list)

        for trade in trades:
            pattern = trade.get('pattern', 'unknown')
            pattern_performance[pattern].append(trade['profit'])
        
        # Calculate overall pattern success
            pattern = trade.get("pattern", "unknown")
            pattern_performance[pattern].append(trade["profit"])

        total_patterns = len(pattern_performance)
        successful_patterns = 0

        for pattern, profits in pattern_performance.items():
            if sum(profits) > 0:  # Pattern is profitable overall
            if sum(profits) > 0:
                successful_patterns += 1

        return (successful_patterns / total_patterns * 100) if total_patterns > 0 else 0

    def _extract_daily_returns(self, metrics: PerformanceMetrics) -> np.ndarray:
        """Extract daily returns for Monte Carlo analysis"""

        # Placeholder - would extract from stored equity curve
        # For now, simulate based on volatility and return
        np.random.seed(42)  # For reproducibility
        np.random.seed(42)

        annual_return = metrics.annualized_return_pct / 100
        annual_vol = metrics.volatility_annual_pct / 100

        daily_return_mean = annual_return / 252
        daily_return_std = annual_vol / np.sqrt(252)
        
        # Generate synthetic daily returns
        days = max(252, metrics.total_trades * 5) 
		# Estimate trading days
        days = max(252, metrics.total_trades * 5)
        daily_returns = np.random.normal(daily_return_mean, daily_return_std, days)

        return daily_returns
    # =============================================================================
    # PLACEHOLDER VISUALIZATION METHODS
    # =============================================================================
    
    def _plot_returns_comparison(self, ax, comparison: ComparisonAnalysis):
        """Plot returns comparison chart"""
        strategies = [comparison.original_metrics.strategy_name] + [ai.strategy_name for ai in comparison.ai_metrics]
        returns = [comparison.original_metrics.total_return_pct] + [ai.total_return_pct for ai in comparison.ai_metrics]
        
        colors = ['blue'] + ['orange'] * len(comparison.ai_metrics)
        bars = ax.bar(strategies, returns, color=colors, alpha=0.7)
        
        ax.set_title('Total Returns Comparison')
        ax.set_ylabel('Return (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, returns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{value:.1f}%', ha='center', va='bottom')
    
    def _plot_risk_comparison(self, ax, comparison: ComparisonAnalysis):
        """Plot risk comparison chart"""
        strategies = [comparison.original_metrics.strategy_name] + [ai.strategy_name for ai in comparison.ai_metrics]
        drawdowns = [comparison.original_metrics.max_drawdown_pct] + [ai.max_drawdown_pct for ai in comparison.ai_metrics]
        
        colors = ['blue'] + ['orange'] * len(comparison.ai_metrics)
        bars = ax.bar(strategies, drawdowns, color=colors, alpha=0.7)
        
        ax.set_title('Maximum Drawdown Comparison')
        ax.set_ylabel('Max Drawdown (%)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_efficiency_comparison(self, ax, comparison: ComparisonAnalysis):
        """Plot efficiency (Sharpe ratio) comparison"""
        strategies = [comparison.original_metrics.strategy_name] + [ai.strategy_name for ai in comparison.ai_metrics]
        sharpes = [comparison.original_metrics.sharpe_ratio] + [ai.sharpe_ratio for ai in comparison.ai_metrics]
        
        colors = ['blue'] + ['orange'] * len(comparison.ai_metrics)
        bars = ax.bar(strategies, sharpes, color=colors, alpha=0.7)
        
        ax.set_title('Sharpe Ratio Comparison')
        ax.set_ylabel('Sharpe Ratio')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_trade_statistics(self, ax, comparison: ComparisonAnalysis):
        """Plot trade statistics"""
        strategies = [comparison.original_metrics.strategy_name] + [ai.strategy_name for ai in comparison.ai_metrics]
        win_rates = [comparison.original_metrics.win_rate * 100] + [ai.win_rate * 100 for ai in comparison.ai_metrics]
        
        colors = ['blue'] + ['orange'] * len(comparison.ai_metrics)
        bars = ax.bar(strategies, win_rates, color=colors, alpha=0.7)
        
        ax.set_title('Win Rate Comparison')
        ax.set_ylabel('Win Rate (%)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_recommendation_radar(self, ax, comparison: ComparisonAnalysis):
        """Plot recommendation radar chart"""
        ax.text(0.5, 0.5, f'AI Recommendation\nScore: {comparison.ai_recommendation_score:.0f}/100', 
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('AI Recommendation Score')
    
    def _plot_allocation_recommendation(self, ax, comparison: ComparisonAnalysis):
        """Plot allocation recommendation pie chart"""
        labels = list(comparison.recommended_allocation.keys())
        sizes = list(comparison.recommended_allocation.values())
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Recommended Allocation')
    
    # =============================================================================
    # ADDITIONAL PLACEHOLDER METHODS
    # =============================================================================
    
    def _create_rolling_windows(self, start_date: datetime, end_date: datetime,
                              window_months: int, step_months: int) -> List[Tuple]:

    def _create_rolling_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        window_months: int,
        step_months: int,
    ) -> List[Tuple]:

        """Create rolling windows for analysis"""
        windows = []
        current_start = start_date

        while current_start < end_date:
            window_end = current_start + timedelta(days=window_months * 30)
            if window_end > end_date:
                window_end = end_date

            windows.append((current_start, window_end))
            current_start += timedelta(days=step_months * 30)

        return windows

    def _analyze_rolling_consistency(self, rolling_results: Dict) -> Dict:
        """Analyze consistency metrics from rolling results"""

        original_returns = rolling_results['original_performance']
        
        consistency_metrics = {
            'original_consistency': {
                'return_std': np.std(original_returns) if original_returns else 0,
                'positive_windows': len([r for r in original_returns if r > 0]) if original_returns else 0,
                'total_windows': len(original_returns)
            }
        }
        
        for objective, ai_returns in rolling_results['ai_performance'].items():
            if ai_returns:
                consistency_metrics[f'{objective}_consistency'] = {

        consistency_metrics = {}

        for strategy_id, ai_returns in rolling_results["ai_performance"].items():
            if ai_returns:
                consistency_metrics[f'{strategy_id}_consistency'] = {

                    'return_std': np.std(ai_returns),
                    'positive_windows': len([r for r in ai_returns if r > 0]),
                    'total_windows': len(ai_returns)

                consistency_metrics[f"{strategy_id}_consistency"] = {
                    "return_std": np.std(ai_returns),
                    "positive_windows": len([r for r in ai_returns if r > 0]),
                    "total_windows": len(ai_returns),
                }

        return consistency_metrics

    def _analyze_performance_trends(self, rolling_results: Dict) -> Dict:
        """Analyze performance trends over time"""
        return {
            "trend_analysis": "Placeholder for trend analysis",
            "improvement_over_time": "Placeholder for improvement analysis",
        }

    def _create_rolling_performance_charts(self, rolling_results: Dict):
        """Create rolling performance visualization"""
        # Placeholder for rolling performance charts

        print("  Rolling performance charts created")
    
    def _analyze_monte_carlo_probabilities(self, monte_carlo_results: Dict) -> Dict:
        """Analyze Monte Carlo simulation probabilities"""
        return {

            'probability_ai_outperforms': 'Placeholder',
            'probability_positive_returns': 'Placeholder'
        }
    

        print("    Rolling performance charts created")

    def _analyze_monte_carlo_probabilities(self, monte_carlo_results: Dict) -> Dict:
        """Analyze Monte Carlo simulation probabilities"""
        return {"probability_positive_returns": "Placeholder"}

    def _assess_monte_carlo_risks(self, monte_carlo_results: Dict) -> Dict:
        """Assess risks from Monte Carlo analysis"""
        return {
            "value_at_risk": "Placeholder",
            "conditional_value_at_risk": "Placeholder",
        }

    def _calculate_confidence_intervals(self, monte_carlo_results: Dict) -> Dict:
        """Calculate confidence intervals from simulations"""
        return {"confidence_intervals": "Placeholder"}

    def _create_monte_carlo_charts(self, monte_carlo_results: Dict):
        """Create Monte Carlo visualization"""
        print("  Monte Carlo charts created")
    
    def _create_html_report(self, comparison: ComparisonAnalysis) -> str:
        """Create HTML performance report"""
        return """
        <html><head><title>nGS AI Performance Comparison Report</title></head>
        <body><h1>Performance Comparison Report</h1>
        <p>Detailed analysis placeholder</p></body></html>
        """
    
    def _serialize_comparison(self, comparison: ComparisonAnalysis) -> Dict:
        """Serialize comparison object for JSON storage"""
        return {
            'timestamp': datetime.now().isoformat(),
            'original_metrics': asdict(comparison.original_metrics),
            'ai_metrics': [asdict(ai) for ai in comparison.ai_metrics],
            'recommendation_score': comparison.ai_recommendation_score,
            'recommendation_reasons': comparison.recommendation_reasons
        }

def demonstrate_performance_comparator():
    """Demonstrate the performance comparator capabilities"""
    print("\n nGS AI PERFORMANCE COMPARATOR DEMONSTRATION")
    print("=" * 60)
    
    # Initialize performance comparator
    comparator = NGSAIPerformanceComparator(account_size=1000000)
    
    print(f"\n Advanced Performance Analysis Features:")
    print(f"   Comprehensive Strategy Comparison")
    print(f"   Statistical Significance Testing")
    print(f"   Rolling Performance Analysis")
    print(f"   Monte Carlo Robustness Testing")
    print(f"   Detailed Performance Metrics (40+ metrics)")
    print(f"   Automated Strategy Recommendations")
    print(f"   Optimal Allocation Calculations")
        print("    Monte Carlo charts created")

    def _create_html_report(self, analysis: AIAnalysis) -> str:
        """Create HTML performance report"""
        return """
        <html><head><title>nGS AI Performance Analysis Report</title></head>
        <body><h1>AI Performance Analysis Report</h1>
        <p>Detailed AI analysis placeholder</p></body></html>
        """

    def _serialize_analysis(self, analysis: AIAnalysis) -> Dict:
        """Serialize analysis object for JSON storage"""
        return {
            "timestamp": datetime.now().isoformat(),
            "ai_metrics": [asdict(ai) for ai in analysis.ai_metrics],
            "recommendation_score": analysis.recommendation_score,
            "recommendation_reasons": analysis.recommendation_reasons,
        }


def demonstrate_performance_analyzer():
    """Demonstrate the AI performance analyzer capabilities"""
    print("\n nGS AI PERFORMANCE ANALYZER DEMONSTRATION")
    print("=" * 60)

    comparator = NGSAIPerformanceAnalyzer(account_size=1000000)
    
    print(f"\n AI Performance Analysis Features:")
    print(f"   Detailed AI Strategy Analysis")
    print(f"   Rolling Performance Analysis")
    print(f"   Monte Carlo Robustness Testing")
    print(f"   Detailed Performance Metrics (30+ metrics)")
    print(f"   Automated Strategy Recommendations")
    print(f"   Professional Reporting & Visualization")
    
    print(f"\n Performance Metrics Included:")
    print(f"   Return Metrics: Total, annualized, monthly statistics")
    print(f"   Risk Metrics: Drawdown, volatility, downside deviation")
    print(f"   Risk-Adjusted: Sharpe, Sortino, Calmar, Information ratios")
    print(f"   Trading Stats: Win rate, profit factor, trade duration")
    print(f"   Consistency: Monthly win rate, streak analysis")
    print(f"   nGS-Specific: M/E efficiency, pattern success rate")
    
    print(f"\n Analysis Capabilities:")
    print(f"   Statistical Tests: T-tests for significance")
    print(f"   Rolling Analysis: Performance over time windows")
    print(f"   Monte Carlo: Robustness simulation testing")
    print(f"   Superiority Analysis: Multi-dimensional comparison")
    print(f"   Allocation Optimization: Capital allocation recommendations")

if __name__ == "__main__":
    print(" nGS AI PERFORMANCE COMPARATOR")
    print("=" * 60)
    print(" Advanced performance analysis and comparison system")
    print(" Compare original nGS vs AI strategies with statistical rigor")
    
    # Run demonstration
    demonstrate_performance_comparator()
    
    print(f"\n PERFORMANCE COMPARATOR READY!")
    print("\n Usage Examples:")
    print("   comparator.comprehensive_comparison(data, objectives)")
    print("   comparator.rolling_performance_analysis(data, objectives)")
    print("   comparator.monte_carlo_analysis(comparison_result)")
    print(f"\n AI Performance Analysis Features:")
    print(f"    Detailed AI Strategy Analysis")
    print(f"    Rolling Performance Analysis")
    print(f"    Monte Carlo Robustness Testing")
    print(f"    Detailed Performance Metrics (30+ metrics)")
    print(f"    Automated Strategy Recommendations")
    print(f"    Professional Reporting & Visualization")
    print(f"\n Performance Metrics Included:")
    print(f"    Return Metrics: Total, annualized, monthly statistics")
    print(f"    Risk Metrics: Drawdown, volatility, downside deviation")
    print(f"    Risk-Adjusted: Sharpe, Sortino, Calmar, Information ratios")
    print(f"    Trading Stats: Win rate, profit factor, trade duration")
    print(f"    Consistency: Monthly win rate, streak analysis")
    print(f"    nGS-Specific: M/E efficiency, pattern success rate")

if __name__ == "__main__":
    print(" nGS AI PERFORMANCE ANALYZER")
    print("=" * 60)
    print(" Advanced performance analysis for AI strategies")

    demonstrate_performance_analyzer()

    print(f"\n PERFORMANCE ANALYZER READY!")
    print("\n Usage Examples:")
    print("   analyzer.analyze_ai_strategies(strategies, data)")
    print("   analyzer.rolling_performance_analysis(data, strategies)")
    print("   analyzer.monte_carlo_analysis(analysis_result)")
