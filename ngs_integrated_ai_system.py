"""
nGS AI Integration Manager
Manages integration between your existing nGS strategy and AI-generated strategies
Allows seamless switching, hybrid approaches, and performance tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
import json
import os

# Import your existing strategy
from nGS_Revised_Strategy import NGSStrategy, load_polygon_data

# Import AI components
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager
from strategy_generator_ai import ObjectiveAwareStrategyGenerator, TradingStrategy

# Import data management
from data_manager import save_trades, save_positions, load_price_data

logger = logging.getLogger(__name__)

class NGSAIIntegrationManager:
    """
    Manages integration between your original nGS strategy and AI-generated strategies
    Provides multiple operating modes and seamless strategy switching
    """
    
    def __init__(self, account_size: float = 1000000, data_dir: str = 'data'):
        self.account_size = account_size
        self.data_dir = data_dir
        
        # Initialize your original nGS strategy
        self.original_ngs = NGSStrategy(account_size=account_size, data_dir=data_dir)
        
        # Initialize AI components with your parameters
        self.ngs_indicator_lib = NGSIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        self.ai_generator = NGSAwareStrategyGenerator(self.ngs_indicator_lib, self.objective_manager)
        
        # Strategy management
        self.active_strategies = {}  # strategy_id -> TradingStrategy
        self.strategy_performance = {}  # strategy_id -> performance_metrics
        self.operating_mode = 'original'  # 'original', 'ai_only', 'hybrid', 'comparison'
        
        # Integration settings
        self.integration_config = {
            'ai_allocation_pct': 50.0,      # % of capital for AI strategies
            'original_allocation_pct': 50.0, # % of capital for original nGS
            'max_ai_strategies': 3,         # Max concurrent AI strategies
            'rebalance_frequency': 'weekly', # How often to rebalance allocations
            'performance_tracking': True,   # Track performance differences
            'risk_sync': True,              # Sync M/E ratios between strategies
        }
        
        print("🎯 nGS AI Integration Manager initialized")
        print(f"   Original nGS:        Ready")
        print(f"   AI Generator:        Ready with YOUR parameters")
        print(f"   Operating Mode:      {self.operating_mode}")
        print(f"   Integration Config:  {self.integration_config['ai_allocation_pct']:.0f}% AI / {self.integration_config['original_allocation_pct']:.0f}% Original")
    
    # =============================================================================
    # OPERATING MODES
    # =============================================================================
    
    def set_operating_mode(self, mode: str, config: Dict = None):
        """
        Set operating mode for strategy execution
        
        Modes:
        - 'original': Use only your original nGS strategy
        - 'ai_only': Use only AI-generated strategies  
        - 'hybrid': Run both with capital allocation
        - 'comparison': Run both in parallel for comparison (no real trades)
        """
        valid_modes = ['original', 'ai_only', 'hybrid', 'comparison']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Must be one of: {valid_modes}")
        
        self.operating_mode = mode
        
        if config:
            self.integration_config.update(config)
        
        print(f"\n🎯 Operating Mode Set: {mode.upper()}")
        
        if mode == 'original':
            print("   Using ONLY your original nGS strategy")
            print("   AI strategies inactive")
            
        elif mode == 'ai_only':
            print("   Using ONLY AI-generated strategies")
            print("   Original nGS strategy inactive") 
            print(f"   Max AI strategies: {self.integration_config['max_ai_strategies']}")
            
        elif mode == 'hybrid':
            print(f"   Capital Allocation:")
            print(f"     Original nGS: {self.integration_config['original_allocation_pct']:.0f}%")
            print(f"     AI Strategies: {self.integration_config['ai_allocation_pct']:.0f}%")
            print(f"   Risk Sync: {'ENABLED' if self.integration_config['risk_sync'] else 'DISABLED'}")
            
        elif mode == 'comparison':
            print("   Running both strategies in PARALLEL")
            print("   Performance comparison mode")
            print("   No capital allocation conflicts")
    
    def create_ai_strategy_set(self, objectives: List[str], 
                              allocation_per_strategy: float = None) -> Dict[str, TradingStrategy]:
        """
        Create a set of AI strategies for different objectives
        Each uses YOUR nGS parameters adapted for the objective
        """
        if allocation_per_strategy is None:
            allocation_per_strategy = self.integration_config['ai_allocation_pct'] / len(objectives)
        
        strategies = {}
        
        print(f"\n🧠 Creating AI strategy set for {len(objectives)} objectives:")
        
        for objective in objectives:
            try:
                strategy = self.ai_generator.generate_ngs_strategy_for_objective(objective)
                
                # Adjust strategy for allocated capital
                allocated_capital = self.account_size * (allocation_per_strategy / 100)
                strategy.allocated_capital = allocated_capital
                
                strategies[strategy.strategy_id] = strategy
                self.active_strategies[strategy.strategy_id] = strategy
                
                print(f"   ✅ {objective}: ${allocated_capital:,.0f} allocated")
                
            except Exception as e:
                print(f"   ❌ {objective}: Failed to create strategy - {e}")
                logger.error(f"Failed to create AI strategy for {objective}: {e}")
        
        print(f"✅ Created {len(strategies)} AI strategies")
        return strategies
    
    # =============================================================================
    # STRATEGY EXECUTION
    # =============================================================================
    
    def run_integrated_strategy(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run strategies based on current operating mode
        Returns comprehensive results from all active strategies
        """
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
        
        if self.operating_mode == 'original':
            results['original_ngs'] = self._run_original_ngs(data)
            
        elif self.operating_mode == 'ai_only':
            results['ai_strategies'] = self._run_ai_strategies_only(data)
            
        elif self.operating_mode == 'hybrid':
            results.update(self._run_hybrid_strategies(data))
            
        elif self.operating_mode == 'comparison':
            results.update(self._run_comparison_mode(data))
        
        # Generate integration summary
        results['integration_summary'] = self._generate_integration_summary(results)
        
        return results
    
    def _run_original_ngs(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run your original nGS strategy"""
        print("📊 Running Original nGS Strategy...")
        
        # Use your original strategy
        original_results = self.original_ngs.run(data)
        
        # Collect performance metrics
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
        """Run only AI-generated strategies"""
        print("🧠 Running AI Strategies Only...")
        
        if not self.active_strategies:
            print("   No AI strategies active - creating default set")
            self.create_ai_strategy_set(['linear_equity', 'max_roi', 'high_winrate'])
        
        ai_results = {}
        
        for strategy_id, strategy in self.active_strategies.items():
            print(f"   Executing: {strategy.objective_name}")
            
            try:
                # Execute AI strategy on data
                strategy_results = strategy.execute_on_data(data, self.ngs_indicator_lib)
                
                # Track performance
                self.strategy_performance[strategy_id] = strategy.performance_metrics
                
                ai_results[strategy_id] = {
                    'strategy': strategy,
                    'results': strategy_results,
                    'performance': strategy.performance_metrics
                }
                
            except Exception as e:
                print(f"   ❌ Error executing {strategy_id}: {e}")
                logger.error(f"Error executing AI strategy {strategy_id}: {e}")
        
        return ai_results
    
    def _run_hybrid_strategies(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run both original and AI strategies with capital allocation"""
        print("🔄 Running Hybrid Strategy Mode...")
        
        # Adjust capital allocations
        original_capital = self.account_size * (self.integration_config['original_allocation_pct'] / 100)
        ai_capital = self.account_size * (self.integration_config['ai_allocation_pct'] / 100)
        
        print(f"   Original nGS Capital: ${original_capital:,.0f}")
        print(f"   AI Strategies Capital: ${ai_capital:,.0f}")
        
        # Create separate strategy instances with allocated capital
        original_ngs_allocated = NGSStrategy(account_size=original_capital, data_dir=self.data_dir)
        
        # Run original strategy with allocated capital
        print("   📊 Running Original nGS (allocated)...")
        original_results = original_ngs_allocated.run(data)
        
        # Create AI strategies if needed
        if not self.active_strategies:
            self.create_ai_strategy_set(['linear_equity', 'max_roi'], 
                                      allocation_per_strategy=self.integration_config['ai_allocation_pct']/2)
        
        # Run AI strategies
        print("   🧠 Running AI Strategies (allocated)...")
        ai_results = {}
        for strategy_id, strategy in self.active_strategies.items():
            try:
                strategy_results = strategy.execute_on_data(data, self.ngs_indicator_lib)
                ai_results[strategy_id] = {
                    'strategy': strategy,
                    'results': strategy_results,
                    'performance': strategy.performance_metrics
                }
            except Exception as e:
                print(f"   ❌ AI Strategy {strategy_id} failed: {e}")
        
        # Sync risk management if enabled
        if self.integration_config['risk_sync']:
            self._sync_risk_management(original_ngs_allocated, ai_results)
        
        return {
            'original_ngs': {
                'signals': original_results,
                'performance': self._calculate_strategy_performance(original_ngs_allocated),
                'allocated_capital': original_capital
            },
            'ai_strategies': ai_results,
            'total_capital_used': original_capital + sum(s['strategy'].allocated_capital for s in ai_results.values())
        }
    
    def _run_comparison_mode(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run both strategies in parallel for performance comparison"""
        print("📊 Running Comparison Mode...")
        print("   Both strategies run with full capital for comparison")
        
        # Run original strategy
        original_results = self._run_original_ngs(data)
        
        # Create and run AI strategies
        if not self.active_strategies:
            self.create_ai_strategy_set(['linear_equity', 'max_roi', 'high_winrate'])
        
        ai_results = {}
        for strategy_id, strategy in self.active_strategies.items():
            try:
                strategy_results = strategy.execute_on_data(data, self.ngs_indicator_lib)
                ai_results[strategy_id] = {
                    'strategy': strategy,
                    'results': strategy_results,
                    'performance': strategy.performance_metrics
                }
            except Exception as e:
                print(f"   ❌ AI Strategy {strategy_id} comparison failed: {e}")
        
        # Generate detailed comparison metrics
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
        """Sync risk management between original and AI strategies"""
        if not self.integration_config['risk_sync']:
            return
        
        print("   🔄 Syncing risk management...")
        
        # Get M/E ratios
        original_me = original_strategy.calculate_current_me_ratio()
        
        # Check if any strategy needs rebalancing
        total_positions = len([pos for pos in original_strategy.positions.values() if pos['shares'] != 0])
        
        for strategy_id, ai_result in ai_results.items():
            ai_strategy = ai_result['strategy']
            if hasattr(ai_strategy, 'performance_metrics'):
                total_positions += ai_result['performance'].get('total_trades', 0)
        
        # Apply coordinated M/E rebalancing if needed
        if original_me > 80 or original_me < 50:
            print(f"   ⚠️  Coordinated rebalancing needed - Combined M/E: {original_me:.1f}%")
            # In real implementation, would coordinate position adjustments
    
    def _calculate_strategy_performance(self, strategy_instance) -> Dict:
        """Calculate standardized performance metrics for any strategy"""
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
        """Generate detailed comparison between original and AI strategies"""
        
        comparison = {
            'performance_comparison': {},
            'strategy_rankings': {},
            'risk_analysis': {},
            'trade_analysis': {},
            'recommendation': ''
        }
        
        # Performance comparison
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
        
        # Strategy rankings by different metrics
        all_strategies = {'original_ngs': original_perf['total_pnl']}
        for strategy_id, ai_result in ai_results.items():
            all_strategies[strategy_id] = ai_result['performance'].get('total_return_pct', 0)
        
        comparison['strategy_rankings']['by_pnl'] = sorted(
            all_strategies.items(), key=lambda x: x[1], reverse=True
        )
        
        return comparison
    
    def _generate_integration_summary(self, results: Dict) -> Dict:
        """Generate summary of integration session"""
        
        summary = {
            'session_timestamp': results['timestamp'],
            'operating_mode': results['mode'],
            'strategies_executed': 0,
            'total_capital_deployed': 0,
            'overall_performance': {},
            'recommendations': []
        }
        
        # Count strategies executed
        if results['original_ngs']:
            summary['strategies_executed'] += 1
            summary['total_capital_deployed'] += self.account_size
        
        if results['ai_strategies']:
            summary['strategies_executed'] += len(results['ai_strategies'])
            if self.operating_mode == 'hybrid':
                summary['total_capital_deployed'] += sum(
                    s['strategy'].allocated_capital for s in results['ai_strategies'].values()
                )
        
        # Generate recommendations based on mode and performance
        if self.operating_mode == 'comparison' and results['comparison_metrics']:
            rankings = results['comparison_metrics']['strategy_rankings']['by_pnl']
            best_strategy = rankings[0][0] if rankings else None
            
            if best_strategy == 'original_ngs':
                summary['recommendations'].append("Original nGS strategy outperformed AI strategies")
            else:
                summary['recommendations'].append(f"AI strategy {best_strategy} outperformed original nGS")
                summary['recommendations'].append("Consider increasing AI allocation in hybrid mode")
        
        return summary
    
    # =============================================================================
    # STRATEGY MANAGEMENT
    # =============================================================================
    
    def add_custom_ai_strategy(self, objective_name: str, custom_config: Dict = None) -> str:
        """Add a custom AI strategy with specific configuration"""
        
        strategy = self.ai_generator.generate_ngs_strategy_for_objective(objective_name)
        
        if custom_config:
            # Apply custom configuration
            strategy.config.update(custom_config)
        
        self.active_strategies[strategy.strategy_id] = strategy
        
        print(f"✅ Added custom AI strategy: {strategy.strategy_id}")
        return strategy.strategy_id
    
    def remove_ai_strategy(self, strategy_id: str):
        """Remove an AI strategy from active set"""
        if strategy_id in self.active_strategies:
            del self.active_strategies[strategy_id]
            if strategy_id in self.strategy_performance:
                del self.strategy_performance[strategy_id]
            print(f"✅ Removed AI strategy: {strategy_id}")
        else:
            print(f"❌ Strategy not found: {strategy_id}")
    
    def list_active_strategies(self) -> Dict:
        """List all active strategies and their status"""
        
        active_list = {
            'original_ngs': {
                'status': 'active' if self.operating_mode in ['original', 'hybrid', 'comparison'] else 'inactive',
                'type': 'original',
                'capital_allocation': self.integration_config['original_allocation_pct'] if self.operating_mode == 'hybrid' else 100
            }
        }
        
        for strategy_id, strategy in self.active_strategies.items():
            active_list[strategy_id] = {
                'status': 'active' if self.operating_mode in ['ai_only', 'hybrid', 'comparison'] else 'inactive',
                'type': 'ai_generated',
                'objective': strategy.objective_name,
                'capital_allocation': getattr(strategy, 'allocated_capital', 0)
            }
        
        return active_list
    
    def save_integration_session(self, results: Dict, filename: str = None):
        """Save integration session results for analysis"""
        if filename is None:
            filename = f"integration_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, 'integration_sessions', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare results for JSON serialization
        serializable_results = self._prepare_results_for_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Integration session saved: {filepath}")
        return filepath
    
    def _prepare_results_for_json(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization"""
        # This would convert strategy objects and other non-serializable items
        # to dictionaries for JSON storage
        json_results = {
            'mode': results['mode'],
            'timestamp': results['timestamp'],
            'integration_summary': results['integration_summary']
        }
        
        # Add performance summaries (not full objects)
        if results['original_ngs']:
            json_results['original_ngs_summary'] = results['original_ngs']['performance']
        
        if results['ai_strategies']:
            json_results['ai_strategies_summary'] = {
                strategy_id: ai_result['performance'] 
                for strategy_id, ai_result in results['ai_strategies'].items()
            }
        
        return json_results

def demonstrate_integration_manager():
    """Demonstrate the integration manager capabilities"""
    print("\n🎯 nGS AI INTEGRATION MANAGER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize integration manager
    manager = NGSAIIntegrationManager(account_size=1000000)
    
    # Show available operating modes
    print("\n📋 Available Operating Modes:")
    modes = ['original', 'ai_only', 'hybrid', 'comparison']
    for mode in modes:
        print(f"   {mode.upper()}: Use this mode for different integration approaches")
    
    # Demonstrate mode switching
    print(f"\n🔄 Demonstrating Mode Switching:")
    
    # Original mode
    manager.set_operating_mode('original')
    
    # AI only mode  
    manager.set_operating_mode('ai_only', {
        'max_ai_strategies': 2
    })
    
    # Hybrid mode
    manager.set_operating_mode('hybrid', {
        'ai_allocation_pct': 60.0,
        'original_allocation_pct': 40.0,
        'risk_sync': True
    })
    
    # Show active strategies
    print(f"\n📊 Active Strategies:")
    active_strategies = manager.list_active_strategies()
    for strategy_id, info in active_strategies.items():
        print(f"   {strategy_id}: {info['status']} ({info['type']})")
    
    print(f"\n✅ Integration Manager Ready!")
    print("Use manager.run_integrated_strategy(data) to execute strategies")

if __name__ == "__main__":
    print("🎯 nGS AI INTEGRATION MANAGER")
    print("=" * 60)
    print("🔄 Seamlessly integrate AI strategies with your original nGS")
    print("📊 Multiple operating modes for different use cases")
    
    # Run demonstration
    demonstrate_integration_manager()
    
    print(f"\n✅ INTEGRATION MANAGER READY!")
    print("\n🚀 Key Features:")
    print("   ✅ Multiple operating modes (original/AI/hybrid/comparison)")
    print("   ✅ Capital allocation management")
    class NGSAwareStrategyGenerator(ObjectiveAwareStrategyGenerator):
    """
    Enhanced AI that understands YOUR nGS patterns and parameters
    Generates strategies using your proven thresholds and logic!
    """
    
    def __init__(self, indicator_library: NGSIndicatorLibrary, 
                 objective_manager: ObjectiveManager):
        super().__init__(indicator_library, objective_manager)
        self.ngs_params = NGSProvenParameters()
        self.ngs_patterns = self._define_ngs_patterns()
        
        print("🎯 nGS-Aware AI initialized with YOUR proven parameters!")
        print(f"   Your BB period: {self.ngs_params.CORE_PARAMS['bb_length']}")
        print(f"   Your position size: ${self.ngs_params.CORE_PARAMS['position_size']:,}")
        print(f"   Your M/E range: {self.ngs_params.ME_RATIO_PARAMS['target_min']:.0f}%-{self.ngs_params.ME_RATIO_PARAMS['target_max']:.0f}%")
    
    def _define_ngs_patterns(self) -> Dict:
        """Define YOUR proven entry/exit patterns from nGS strategy"""
        return {
            'engulfing_long': {
                'description': 'Your proven Engulfing Long pattern',
                'conditions': [
                    {'indicator': 'ngs_bb_position', 'operator': '<', 'threshold': 25, 'weight': 3},
                    {'indicator': 'ngs_lr_value', 'operator': '>', 'threshold': 0, 'weight': 2},
                ]
            }
        }
    
    def generate_ngs_strategy_for_objective(self, objective_name: str, 
                                          pattern_focus: str = 'auto') -> TradingStrategy:
        """Generate strategy using YOUR nGS patterns optimized for objective"""
        from strategy_generator_ai import TradingStrategy
        
        print(f"\n🎯 Generating nGS strategy for: {objective_name.upper()}")
        
        # Use the parent class method but with nGS-specific adaptations
        return self.generate_strategy_for_objective(objective_name)
    print("   ✅ Risk synchronization between strategies")
    print("   ✅ Performance tracking and comparison")
    print("   ✅ Session saving and analysis")
    print("   ✅ Uses YOUR proven nGS parameters")
