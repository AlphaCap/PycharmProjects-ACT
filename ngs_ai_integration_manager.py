"""
nGS AI Integration Manager
Manages integration between your existing nGS strategy and AI-generated strategies
Now runs exclusively in AI-only mode: all comparison/original logic removed for simplicity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import os

# Import AI components
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager
from strategy_generator_ai import ObjectiveAwareStrategyGenerator, TradingStrategy
from ngs_integrated_ai_system import NGSAwareStrategyGenerator, NGSIndicatorLibrary, NGSProvenParameters

# Import data management
from data_manager import save_trades, save_positions, load_price_data

# For linear regression (equity curve smoothness)
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class NGSAIIntegrationManager:
    """
    Manages integration with AI-generated strategies
    Exclusively provides AI-only operating mode
    """
    
    def __init__(self, account_size: float = 1000000, data_dir: str = 'data'):
        self.account_size = account_size
        self.data_dir = data_dir
        
        # Initialize AI components with your parameters
        self.ngs_indicator_lib = NGSIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        self.ai_generator = NGSAwareStrategyGenerator(self.ngs_indicator_lib, self.objective_manager)
        
        # Strategy management
        self.active_strategies = {}  # strategy_id -> TradingStrategy
        self.strategy_performance = {}  # strategy_id -> performance_metrics
        self.operating_mode = 'ai_only'  # Only AI mode
        
        # Integration settings
        self.integration_config = {
            'ai_allocation_pct': 100.0,     # % of capital for AI strategies
            'max_ai_strategies': 3,         # Max concurrent AI strategies
            'rebalance_frequency': 'weekly', # How often to rebalance allocations
            'performance_tracking': True,    # Track performance
            'risk_sync': False,             # No risk sync needed for AI-only
        }
        
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

    def auto_select_mode(self, ai_metrics: Dict, verbose: bool = True) -> str:
        """
        Validates AI strategy performance based on hierarchy and always selects AI-only mode.
        Hierarchy:
        1. Linear equity curve (R²)
        2. Minimum drawdown
        3. ROI
        4. Sharpe ratio
        """
        # Validate AI metrics before proceeding
        if not ai_metrics or ai_metrics.get('total_return_pct', 0) == 0:
            print("[ERROR] AI strategy metrics missing or invalid. Cannot proceed with AI-only mode.")
            raise ValueError("Invalid AI metrics provided.")
        
        # Evaluate AI metrics using performance hierarchy
        ai_r2 = self.evaluate_linear_equity(ai_metrics.get('equity_curve', pd.Series(dtype=float)))
        ai_dd = ai_metrics.get('max_drawdown_pct', 0)
        ai_roi = ai_metrics.get('total_return_pct', 0)
        ai_sharpe = ai_metrics.get('sharpe_ratio', 0)
        
        if verbose:
            print(f"[HIERARCHY] AI Metrics - R²: {ai_r2:.4f}, Drawdown: {ai_dd:.2f}%, ROI: {ai_roi:.2f}%, Sharpe: {ai_sharpe:.3f}")
        
        # Performance hierarchy checks (informational only)
        if ai_r2 < 0.02 and verbose:
            print("[HIERARCHY] AI strategy warning: Low R² score.")
        if ai_dd > 2 and verbose:
            print("[HIERARCHY] AI strategy warning: High drawdown.")
        if ai_roi < 2 and verbose:
            print("[HIERARCHY] AI strategy warning: Low ROI.")
        if ai_sharpe < 0.1 and verbose:
            print("[HIERARCHY] AI strategy warning: Low Sharpe ratio.")
        
        if verbose:
            print("[HIERARCHY] Selecting AI-only mode.")
        return 'ai_only'

    def auto_update_mode(self, ai_metrics: Dict, verbose: bool=True):
        """
        Performs auto-selection and sets the operating mode to AI-only.
        """
        new_mode = self.auto_select_mode(ai_metrics, verbose=verbose)
        if new_mode != self.operating_mode:
            self.set_operating_mode(new_mode)
            print(f"[AUTO] Switched to mode: {new_mode}")

    # =============================================================================
    # OPERATING MODE
    # =============================================================================
    def set_operating_mode(self, mode: str, config: Dict = None):
        """
        Set operating mode for strategy execution
        
        Mode:
        - 'ai_only': Use only AI-generated strategies
        """
        valid_modes = ['ai_only']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode. Only 'ai_only' is supported.")
        
        self.operating_mode = mode
        
        if config:
            self.integration_config.update(config)
        
        print(f"\n🎯 Operating Mode Set: {mode.upper()}")
        print("   Using ONLY AI-generated strategies")
        print(f"   Max AI strategies: {self.integration_config['max_ai_strategies']}")

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
        Run strategies in AI-only mode.
        Returns comprehensive results from all active strategies.
        """
        results = {
            'mode': self.operating_mode,
            'timestamp': datetime.now().isoformat(),
            'ai_strategies': {},
            'integration_summary': None
        }
        
        print(f"\n🚀 Running Integrated Strategy - Mode: {self.operating_mode.upper()}")
        print(f"   Processing {len(data)} symbols")
        
        results['ai_strategies'] = self._run_ai_strategies_only(data)
        results['integration_summary'] = self._generate_integration_summary(results)
        return results

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

    # =============================================================================
    # PERFORMANCE ANALYSIS & REPORTING
    # =============================================================================
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
        if results['ai_strategies']:
            summary['strategies_executed'] += len(results['ai_strategies'])
            summary['total_capital_deployed'] += sum(
                s['strategy'].allocated_capital for s in results['ai_strategies'].values()
            )
        
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
        active_list = {}
        for strategy_id, strategy in self.active_strategies.items():
            active_list[strategy_id] = {
                'status': 'active',
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
        serializable_results = self._prepare_results_for_json(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✅ Integration session saved: {filepath}")
        return filepath
    
    def _prepare_results_for_json(self, results: Dict) -> Dict:
        """Prepare results for JSON serialization"""
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
    """Demonstrate the integration manager capabilities"""
    print("\n🎯 nGS AI INTEGRATION MANAGER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize integration manager
    manager = NGSAIIntegrationManager(account_size=1000000)
    
    # Show available operating mode
    print("\n📋 Available Operating Mode:")
    print("   AI_ONLY: Use this mode for all integration approaches")
    
    # Demonstrate mode switching (AI only)
    print(f"\n🔄 Demonstrating Mode Setting:")
    manager.set_operating_mode('ai_only', {'max_ai_strategies': 2})
    
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
    print("🔄 Seamlessly integrate AI strategies with your system")
    print("📊 AI-only mode for all use cases")
    
    # Run demonstration
    demonstrate_integration_manager()
    
    print(f"\n✅ INTEGRATION MANAGER READY!")
    print("\n🚀 Key Features:")
    print("   ✅ AI-only mode")
    print("   ✅ Capital allocation management")
    print("   ✅ Performance tracking")
    print("   ✅ Session saving and analysis")
    print("   ✅ Uses YOUR proven nGS parameters")
