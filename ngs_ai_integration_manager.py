import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
import json
import os

import nGS_Revised_Strategy

# Import AI components
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager
from strategy_generator_ai import ObjectiveAwareStrategyGenerator, TradingStrategy
from ngs_integrated_ai_system import NGSAwareStrategyGenerator, NGSIndicatorLibrary, NGSProvenParameters

# Import data management
from data_manager import save_trades, save_positions, load_price_data

# For linear regression (equity curve smoothness) and plotting
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class NGSAIIntegrationManager:
    """
    Manages integration with AI-generated strategies
    Provides AI-only and comparison operating modes
    """
    
    def __init__(self, account_size: float = 1000000, data_dir: str = 'data'):
        self.account_size = account_size
        self.data_dir = data_dir
        
        # Initialize your original nGS strategy (retained for comparison mode)
        self.original_ngs = NGSStrategy(account_size=account_size, data_dir=data_dir)
        
        # Initialize AI components with your parameters
        self.ngs_indicator_lib = NGSIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        self.ai_generator = NGSAwareStrategyGenerator(self.ngs_indicator_lib, self.objective_manager)
        
        # Strategy management
        self.active_strategies = {}  # strategy_id -> TradingStrategy
        self.strategy_performance = {}  # strategy_id -> performance_metrics
        self.operating_mode = 'ai_only'  # 'ai_only', 'comparison'
        
        # Integration settings (updated to remove original/hybrid allocations)
        self.integration_config = {
            'ai_allocation_pct': 100.0,     # % of capital for AI strategies
            'max_ai_strategies': 3,         # Max concurrent AI strategies
            'rebalance_frequency': 'weekly', # How often to rebalance allocations
            'performance_tracking': True,    # Track performance
            'risk_sync': False,             # No risk sync needed for AI-only
        }
        
        # Results directory for charts
        self.results_dir = os.path.join(self.data_dir, "integration_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("🎯 nGS AI Integration Manager initialized")
        print(f"   AI Generator:        Ready with YOUR parameters")
        print(f"   Operating Mode:      {self.operating_mode}")
        print(f"   Integration Config:  {self.integration_config['ai_allocation_pct']:.0f}% AI")
    
    # =============================================================================
    # AUTOMATED PERFORMANCE HIERARCHY/SELECTION
    # =============================================================================
    def evaluate_linear_equity(self, equity_curve: pd.Series) -> float:
        """Return R² of linear regression for equity curve (closer to 1 = more linear)."""
        if len(equity_curve) < 2:
            return 0.0
        x = np.arange(len(equity_curve)).reshape(-1, 1)
        y = equity_curve.values.reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        r2 = model.score(x, y)
        return r2

    def auto_select_mode(self, original_metrics: Dict, ai_metrics: Dict, verbose: bool = True) -> str:
        """
        Validates AI strategy performance based on hierarchy and selects AI-only mode.
        Hierarchy:
        1. Linear equity curve (R²)
        2. Minimum drawdown
        3. ROI
        4. Sharpe ratio
        """
        if not ai_metrics or ai_metrics.get('total_return_pct', 0) == 0:
            print("[ERROR] AI strategy metrics missing or invalid. Cannot proceed with AI-only mode.")
            raise ValueError("Invalid AI metrics provided.")
        
        ai_r2 = self.evaluate_linear_equity(ai_metrics.get('equity_curve', pd.Series(dtype=float)))
        ai_dd = ai_metrics.get('max_drawdown_pct', 0)
        ai_roi = ai_metrics.get('total_return_pct', 0)
        ai_sharpe = ai_metrics.get('sharpe_ratio', 0)
        
        if verbose:
            print(f"[HIERARCHY] AI Metrics - R²: {ai_r2:.4f}, Drawdown: {ai_dd:.2f}%, ROI: {ai_roi:.2f}%, Sharpe: {ai_sharpe:.3f}")
        if ai_r2 < 0.02:
            if verbose: print("[HIERARCHY] AI strategy warning: Low R² score.")
        if ai_dd > 2:
            if verbose: print("[HIERARCHY] AI strategy warning: High drawdown.")
        if ai_roi < 2:
            if verbose: print("[HIERARCHY] AI strategy warning: Low ROI.")
        if ai_sharpe < 0.1:
            if verbose: print("[HIERARCHY] AI strategy warning: Low Sharpe ratio.")
        if verbose:
            print("[HIERARCHY] Selecting AI-only mode.")
        return 'ai_only'

    def auto_update_mode(self, original_metrics: Dict, ai_metrics: Dict, verbose: bool=True):
        new_mode = self.auto_select_mode(original_metrics, ai_metrics, verbose=verbose)
        if new_mode != self.operating_mode:
            self.set_operating_mode(new_mode)
            print(f"[AUTO] Switched to mode: {new_mode}")

    # =============================================================================
    # OPERATING MODES
    # =============================================================================
    
    def set_operating_mode(self, mode: str, config: Dict = None):
        valid_modes = ['ai_only', 'comparison']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
        self.operating_mode = mode
        if config:
            self.integration_config.update(config)
        print(f"\n🎯 Operating Mode Set: {mode.upper()}")
        if mode == 'ai_only':
            print("   Using ONLY AI-generated strategies")
            print(f"   Max AI strategies: {self.integration_config['max_ai_strategies']}")
        elif mode == 'comparison':
            print("   Running both strategies in PARALLEL")
            print("   Performance comparison mode")
    
    def create_ai_strategy_set(self, objectives: List[str], allocation_per_strategy: float = None) -> Dict[str, TradingStrategy]:
        if allocation_per_strategy is None:
            allocation_per_strategy = self.integration_config['ai_allocation_pct'] / len(objectives)
        strategies = {}
        print(f"\n🧠 Creating AI strategy set for {len(objectives)} objectives:")
        for objective in objectives:
            try:
                print(f"\n🎯 Generating strategy for objective: {objective.upper()}")
                strategy = self.ai_generator.generate_strategy_for_objective(objective)
                allocated_capital = self.account_size * (allocation_per_strategy / 100)
                strategy.allocated_capital = allocated_capital
                strategies[strategy.strategy_id] = strategy
                self.active_strategies[strategy.strategy_id] = strategy
                print(f"   ✅ {objective}: ${allocated_capital:,.0f} allocated")
                # Print indicators and conditions if available
                if hasattr(strategy, 'indicators'):
                    print(f"✅ Generated strategy '{strategy.strategy_id}' with {len(strategy.indicators)} indicators")
                if hasattr(strategy, 'entry_conditions'):
                    print(f"📊 Entry conditions: {len(strategy.entry_conditions)}")
                if hasattr(strategy, 'exit_conditions'):
                    print(f"🎯 Exit conditions: {len(strategy.exit_conditions)}")
            except Exception as e:
                print(f"   ❌ {objective}: Failed to create strategy - {e}")
                logger.error(f"Failed to create AI strategy for {objective}: {e}")
        print(f"✅ Created {len(strategies)} AI strategies")
        return strategies
    
    # =============================================================================
    # STRATEGY EXECUTION
    # =============================================================================
    
    def run_integrated_strategy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        results = {
            'mode': self.operating_mode,
            'timestamp': datetime.now().isoformat(),
            'original_ngs': None,
            'ai_strategies': {},
            'comparison_metrics': None,
            'integration_summary': None
        }
        print(f"\n🚀 Running Integrated Strategy - Mode: {self.operating_mode.upper()}")
        print(f"   Processing {len(data)} symbols")
        if self.operating_mode == 'ai_only':
            results['ai_strategies'] = self._run_ai_strategies_only(data)
        elif self.operating_mode == 'comparison':
            results.update(self._run_comparison_mode(data))
        results['integration_summary'] = self._generate_integration_summary(results)
        return results
    
    def _run_original_ngs(self, data: Dict[str, pd.DataFrame]) -> Dict:
        print("📊 Running Original nGS Strategy for comparison...")
        original_results = self.original_ngs.run(data)
        performance = {
            'trades': len(self.original_ngs.trades),
            'total_pnl': sum(trade['profit'] for trade in self.original_ngs.trades),
            'cash': self.original_ngs.cash,
            'positions': len([pos for pos in self.original_ngs.positions.values() if pos['shares'] != 0]),
            'me_ratio': self.original_ngs.calculate_current_me_ratio(),
            'win_rate': len([t for t in self.original_ngs.trades if t['profit'] > 0]) / max(1, len(self.original_ngs.trades))
        }
        return {
            'signals': original_results,
            'performance': performance,
            'strategy_instance': self.original_ngs
        }
    
    def _run_ai_strategies_only(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        print("🧠 Running AI Strategies Only...")
        if not self.active_strategies:
            print("   No AI strategies active - creating default set")
            self.create_ai_strategy_set(['linear_equity', 'max_roi', 'high_winrate'])
        ai_results = {}
        for strategy_id, strategy in self.active_strategies.items():
            print(f"   Executing: {strategy.objective_name}")
            try:
                # If data is a dict of DataFrames, process each symbol
                if isinstance(data, dict):
                    combined_equity_curve = pd.Series(dtype=float)
                    symbol_results = {}
                    for symbol, df in data.items():
                        print(f"🔍 Executing strategy {strategy_id} on {len(df)} bars of {symbol} data")
                        if not isinstance(df, pd.DataFrame):
                            print(f"❌ Data for symbol {symbol} is not a DataFrame! Skipping.")
                            continue
                        # Validate data format
                        if hasattr(strategy, '_validate_data_format') and not strategy._validate_data_format(df):
                            print(f"❌ Data for symbol {symbol} did not pass format validation. Skipping.")
                            continue
                        result = strategy.execute_on_data(df, self.ngs_indicator_lib)
                        symbol_results[symbol] = result
                        # Combine equity curves if available
                        if 'equity_curve' in result and isinstance(result['equity_curve'], pd.Series):
                            if combined_equity_curve.empty:
                                combined_equity_curve = result['equity_curve']
                            else:
                                combined_equity_curve = combined_equity_curve.add(result['equity_curve'], fill_value=0)
                    # Plot combined equity curve if available
                    if not combined_equity_curve.empty:
                        chart_path = self._plot_equity_curve(combined_equity_curve, strategy_id)
                    else:
                        chart_path = None
                    self.strategy_performance[strategy_id] = {
                        "indicator_values": getattr(strategy, "indicators", []),
                        "entry_conditions": getattr(strategy, "entry_conditions", []),
                        "exit_conditions": getattr(strategy, "exit_conditions", []),
                        "combined_equity_curve": combined_equity_curve,
                        "equity_curve_chart": chart_path
                    }
                    ai_results[strategy_id] = {
                        'strategy': strategy,
                        'symbol_results': symbol_results,
                        'performance': self.strategy_performance[strategy_id]
                    }
                else:
                    # Single DataFrame
                    print(f"🔍 Executing strategy {strategy_id} on {len(data)} bars of data")
                    if hasattr(strategy, '_validate_data_format') and not strategy._validate_data_format(data):
                        print(f"❌ Data did not pass format validation.")
                        continue
                    result = strategy.execute_on_data(data, self.ngs_indicator_lib)
                    eq_curve = result.get('equity_curve', pd.Series(dtype=float))
                    chart_path = self._plot_equity_curve(eq_curve, strategy_id) if not eq_curve.empty else None
                    self.strategy_performance[strategy_id] = {
                        "indicator_values": getattr(strategy, "indicators", []),
                        "entry_conditions": getattr(strategy, "entry_conditions", []),
                        "exit_conditions": getattr(strategy, "exit_conditions", []),
                        "equity_curve": eq_curve,
                        "equity_curve_chart": chart_path
                    }
                    ai_results[strategy_id] = {
                        'strategy': strategy,
                        'results': result,
                        'performance': self.strategy_performance[strategy_id]
                    }
            except Exception as e:
                print(f"   ❌ Strategy execution failed: {e}")
                logger.error(f"Error executing AI strategy {strategy_id}: {e}", exc_info=True)
        return ai_results

    def _plot_equity_curve(self, equity_curve: pd.Series, strategy_id: str) -> Optional[str]:
        if equity_curve is None or equity_curve.empty:
            return None
        plt.figure(figsize=(10,4))
        plt.plot(equity_curve.index, equity_curve.values, label='Equity Curve')
        plt.title(f'Equity Curve: {strategy_id}')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        chart_path = os.path.join(self.results_dir, f"{strategy_id}_equity_curve.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"   📈 Equity curve chart saved: {chart_path}")
        return chart_path
    
    def _run_comparison_mode(self, data: Dict[str, pd.DataFrame]) -> Dict:
        print("📊 Running Comparison Mode...")
        print("   Both strategies run with full capital for comparison")
        original_results = self._run_original_ngs(data)
        if not self.active_strategies:
            self.create_ai_strategy_set(['linear_equity', 'max_roi', 'high_winrate'])
        ai_results = {}
        for strategy_id, strategy in self.active_strategies.items():
            try:
                result = strategy.execute_on_data(data, self.ngs_indicator_lib)
                ai_results[strategy_id] = {
                    'strategy': strategy,
                    'results': result,
                    'performance': strategy.performance_metrics
                }
            except Exception as e:
                print(f"   ❌ AI Strategy {strategy_id} comparison failed: {e}")
        comparison_metrics = self._generate_comparison_metrics(original_results, ai_results)
        return {
            'original_ngs': original_results,
            'ai_strategies': ai_results,
            'comparison_metrics': comparison_metrics
        }
    
    # =============================================================================
    # RISK MANAGEMENT & SYNCHRONIZATION
    # =============================================================================
    def _sync_risk_management(self, original_strategy: NGSStrategy, ai_results: Dict):
        print("   🔄 Risk sync disabled for AI-only mode.")
    
    def _calculate_strategy_performance(self, strategy_instance) -> Dict:
        if hasattr(strategy_instance, 'trades') and strategy_instance.trades:
            trades = strategy_instance.trades
            total_pnl = sum(trade['profit'] for trade in trades)
            winning_trades = len([t for t in trades if t['profit'] > 0])
            win_rate = winning_trades / len(trades)
        else:
            trades = []
            total_pnl = 0
            win_rate = 0
        return {
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'cash': getattr(strategy_instance, 'cash', 0),
            'positions': len(getattr(strategy_instance, 'positions', {})),
            'me_ratio': getattr(strategy_instance, 'calculate_current_me_ratio', lambda: 0)()
        }
    
    # =============================================================================
    # PERFORMANCE ANALYSIS & REPORTING
    # =============================================================================
    def _generate_comparison_metrics(self, original_results: Dict, ai_results: Dict) -> Dict:
        comparison = {
            'performance_comparison': {},
            'strategy_rankings': {},
            'risk_analysis': {},
            'trade_analysis': {}
        }
        original_perf = original_results['performance']
        comparison['performance_comparison']['original_ngs'] = {
            'total_pnl': original_perf['total_pnl'],
            'win_rate': original_perf['win_rate'],
            'total_trades': original_perf['trades'],
            'me_ratio': original_perf['me_ratio']
        }
        for strategy_id, ai_result in ai_results.items():
            ai_perf = ai_result['performance']
            comparison['performance_comparison'][strategy_id] = {
                'objective': ai_result['strategy'].objective_name,
                'total_pnl': ai_perf.get('total_return_pct', 0),
                'win_rate': ai_perf.get('win_rate', 0),
                'total_trades': ai_perf.get('total_trades', 0),
                'max_drawdown': ai_perf.get('max_drawdown_pct', 0)
            }
        all_strategies = {'original_ngs': original_perf['total_pnl']}
        for strategy_id, ai_result in ai_results.items():
            all_strategies[strategy_id] = ai_result['performance'].get('total_return_pct', 0)
        comparison['strategy_rankings']['by_pnl'] = sorted(
            all_strategies.items(), key=lambda x: x[1], reverse=True
        )
        return comparison
    
    def _generate_integration_summary(self, results: Dict) -> Dict:
        summary = {
            'session_timestamp': results['timestamp'],
            'operating_mode': results['mode'],
            'strategies_executed': 0,
            'total_capital_deployed': 0,
            'overall_performance': {},
            'recommendations': []
        }
        if results['ai_strategies']:
            summary['strategies_executed'] += len(results['ai_strategies'])
            if self.operating_mode == 'ai_only':
                summary['total_capital_deployed'] += sum(
                    s['strategy'].allocated_capital for s in results['ai_strategies'].values()
                )
        if self.operating_mode == 'comparison' and results['comparison_metrics']:
            rankings = results['comparison_metrics']['strategy_rankings']['by_pnl']
            best_strategy = rankings[0][0] if rankings else None
            if best_strategy != 'original_ngs':
                summary['recommendations'].append(f"AI strategy {best_strategy} performed strongly")
                summary['recommendations'].append("Consider continuing with AI-only mode")
        return summary
    
    # =============================================================================
    # STRATEGY MANAGEMENT
    # =============================================================================
    def add_custom_ai_strategy(self, objective_name: str, custom_config: Dict = None) -> str:
        strategy = self.ai_generator.generate_strategy_for_objective(objective_name)
        if custom_config:
            strategy.config.update(custom_config)
        self.active_strategies[strategy.strategy_id] = strategy
        print(f"✅ Added custom AI strategy: {strategy.strategy_id}")
        return strategy.strategy_id
    
    def remove_ai_strategy(self, strategy_id: str):
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            if strategy_id in self.strategy_performance:
                del self.strategy_performance[strategy_id]
            print(f"✅ Removed AI strategy: {strategy_id}")
        else:
            print(f"❌ Strategy not found: {strategy_id}")
    
    def list_active_strategies(self) -> Dict:
        active_list = {}
        for strategy_id, strategy in self.active_strategies.items():
            active_list[strategy_id] = {
                'status': 'active' if self.operating_mode in ['ai_only', 'comparison'] else 'inactive',
                'type': 'ai_generated',
                'objective': strategy.objective_name,
                'capital_allocation': getattr(strategy, 'allocated_capital', 0)
            }
        return active_list
    
    def save_integration_session(self, results: Dict, filename: str = None):
        if filename is None:
            filename = f"integration_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, 'integration_sessions', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        serializable_results = self._prepare_results_for_json(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✅ Integration session saved: {filepath}")
        return filepath
    
    def _prepare_results_for_json(self, results: Dict) -> Dict:
        json_results = {
            'mode': results['mode'],
            'timestamp': results['timestamp'],
            'integration_summary': results['integration_summary']
        }
        if results['ai_strategies']:
            json_results['ai_strategies_summary'] = {
                strategy_id: ai_result['performance'] 
                for strategy_id, ai_result in results['ai_strategies'].items()
            }
        return json_results

def demonstrate_integration_manager():
    print("\n🎯 nGS AI INTEGRATION MANAGER DEMONSTRATION")
    print("=" * 60)
    manager = NGSAIIntegrationManager(account_size=1000000)
    print("\n📋 Available Operating Modes:")
    modes = ['ai_only', 'comparison']
    for mode in modes:
        print(f"   {mode.upper()}: Use this mode for different integration approaches")
    print(f"\n🔄 Demonstrating Mode Switching:")
    manager.set_operating_mode('ai_only', {'max_ai_strategies': 2})
    manager.set_operating_mode('comparison')
    print(f"\n📊 Active Strategies:")
    active_strategies = manager.list_active_strategies()
    for strategy_id, info in active_strategies.items():
        print(f"   {strategy_id}: {info['status']} ({info['type']})")
    print(f"\n✅ Integration Manager Ready!")
    print("Use manager.run_integrated_strategy(data) to execute strategies")

if __name__ == "__main__":
    print("🎯 nGS AI INTEGRATION MANAGER")
    print("=" * 60)
    print("🔄 Seamlessly integrate AI strategies with your system")
    print("📊 Multiple operating modes for different use cases")
    demonstrate_integration_manager()
    print(f"\n✅ INTEGRATION MANAGER READY!")
    print("\n🚀 Key Features:")
    print("   ✅ AI-only and comparison modes")
    print("   ✅ Capital allocation management")
    print("   ✅ Performance tracking and comparison")
    print("   ✅ Session saving and analysis")
    print("   ✅ Uses YOUR proven nGS parameters")
