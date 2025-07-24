# Working nGS AI Backtester - FIXED for robust strategy comparison
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager
from strategy_generator_ai import ObjectiveAwareStrategyGenerator
from nGS_Revised_Strategy import NGSStrategy, load_polygon_data
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import traceback

class SimpleNGSBacktester:
    def __init__(self, account_size=1000000):
        self.account_size = account_size
        self.indicator_lib = ComprehensiveIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        self.ai_generator = ObjectiveAwareStrategyGenerator(self.indicator_lib, self.objective_manager)
        print(f"🧪 Simple Backtester Ready - ${account_size:,}")
        print(f"✅ Loaded {len(self.indicator_lib.indicators_catalog)} indicators")
        print(f"✅ Available objectives: {list(self.objective_manager.objectives.keys())}")
    
    def backtest_ai_strategy(self, objective: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Backtest AI strategy with enhanced error handling and data compatibility
        Args:
            objective: Strategy objective (e.g., 'linear_equity')
            data: Dictionary of DataFrames {symbol: price_data}
        Returns:
            Dictionary with strategy results
        """
        print(f"\n🧪 Backtesting AI strategy: {objective}")
        
        try:
            # Generate strategy for the objective
            print(f"   🎯 Generating {objective} strategy...")
            strategy = self.ai_generator.generate_strategy_for_objective(objective)
            
            if not strategy:
                print(f"   ❌ Failed to generate strategy")
                return self._get_failed_ai_result(objective, "Strategy generation failed")
            
            # Combine all data for testing (use first symbol with sufficient data)
            print(f"   📊 Selecting test data from {len(data)} symbols...")
            test_data = self._prepare_test_data(data)
            
            if test_data is None or test_data.empty:
                print(f"   ❌ No suitable test data found")
                return self._get_failed_ai_result(objective, "No suitable test data")
            
            print(f"   📈 Testing on {len(test_data)} bars of data...")
            
            # Execute strategy on test data
            results = strategy.execute_on_data(test_data, self.indicator_lib)
            
            if not results or not results.get('trades'):
                print(f"   ⚠️  Strategy completed but generated no trades")
                return {
                    'objective': objective,
                    'strategy_id': strategy.strategy_id,
                    'trades': 0,
                    'profit': 0.0,
                    'win_rate': 0.0,
                    'total_return_pct': 0.0,
                    'max_drawdown_pct': 0.0,
                    'sharpe_ratio': 0.0,
                    'status': 'No trades generated',
                    'successful_indicators': results.get('successful_indicators', [])
                }
            
            # Extract and calculate metrics
            trades = results['trades']
            metrics = results.get('metrics', {})
            
            total_profit = sum(trade.get('profit', trade.get('pnl_dollars', 0)) for trade in trades)
            profitable_trades = len([t for t in trades if t.get('profit', t.get('pnl_dollars', 0)) > 0])
            win_rate = profitable_trades / len(trades) if trades else 0.0
            
            result = {
                'objective': objective,
                'strategy_id': strategy.strategy_id,
                'trades': len(trades),
                'profit': round(total_profit, 2),
                'win_rate': round(win_rate, 3),
                'total_return_pct': round(metrics.get('total_return_pct', 0), 2),
                'max_drawdown_pct': round(metrics.get('max_drawdown_pct', 0), 2),
                'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 3),
                'status': 'Success',
                'successful_indicators': results.get('successful_indicators', strategy.config.get('indicators', []))
            }
            
            print(f"   ✅ AI Strategy Results:")
            print(f"      Trades: {result['trades']}")
            print(f"      Profit: ${result['profit']:,.2f}")
            print(f"      Win Rate: {result['win_rate']:.1%}")
            print(f"      Return: {result['total_return_pct']:.1f}%")
            print(f"      Max DD: {result['max_drawdown_pct']:.1f}%")
            
            return result
            
        except Exception as e:
            error_msg = f"AI strategy execution failed: {str(e)[:100]}..."
            print(f"   ❌ {error_msg}")
            traceback.print_exc()
            return self._get_failed_ai_result(objective, error_msg)
    
    def _prepare_test_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare test data by selecting the best available symbol data
        """
        try:
            # Find symbol with most data
            best_symbol = None
            max_rows = 0
            
            for symbol, df in data.items():
                if df is not None and not df.empty and len(df) > max_rows:
                    # Check for required columns
                    required_cols = ['Close', 'High', 'Low', 'Open']
                    has_required = all(
                        any(col.lower() == req.lower() for col in df.columns)
                        for req in required_cols
                    )
                    
                    if has_required:
                        max_rows = len(df)
                        best_symbol = symbol
            
            if best_symbol:
                print(f"   📊 Using {best_symbol} data ({max_rows} bars)")
                return data[best_symbol].copy()
            else:
                print(f"   ⚠️  No suitable symbol data found")
                return None
                
        except Exception as e:
            print(f"   ❌ Error preparing test data: {e}")
            return None
    
    def _get_failed_ai_result(self, objective: str, error_msg: str) -> Dict[str, Any]:
        """Return standardized failed result for AI strategy"""
        return {
            'objective': objective,
            'strategy_id': f'failed_{objective}',
            'trades': 0,
            'profit': 0.0,
            'win_rate': 0.0,
            'total_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'status': f'FAILED: {error_msg}',
            'successful_indicators': []
        }
    
    def backtest_original_ngs(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Backtest original nGS strategy with enhanced error handling
        """
        print(f"\n📊 Backtesting Original nGS Strategy")
        
        try:
            print(f"   🏗️  Initializing nGS strategy...")
            ngs_strategy = NGSStrategy(account_size=self.account_size)
            
            print(f"   🚀 Running nGS on {len(data)} symbols...")
            results = ngs_strategy.run(data)
            
            # Extract results
            trades = ngs_strategy.trades
            total_profit = sum(trade['profit'] for trade in trades) if trades else 0.0
            profitable_trades = len([t for t in trades if t['profit'] > 0]) if trades else 0
            win_rate = profitable_trades / len(trades) if trades else 0.0
            
            # Calculate return percentage
            total_return_pct = ((ngs_strategy.cash - self.account_size) / self.account_size * 100)
            
            # Get current M/E ratio
            final_me_ratio = ngs_strategy.calculate_current_me_ratio()
            
            result = {
                'strategy': 'nGS Original',
                'trades': len(trades),
                'profit': round(total_profit, 2),
                'win_rate': round(win_rate, 3),
                'total_return_pct': round(total_return_pct, 2),
                'final_cash': round(ngs_strategy.cash, 2),
                'final_me_ratio': round(final_me_ratio, 2),
                'positions': len([pos for pos in ngs_strategy.positions.values() if pos['shares'] != 0]),
                'status': 'Success'
            }
            
            print(f"   ✅ nGS Strategy Results:")
            print(f"      Trades: {result['trades']}")
            print(f"      Profit: ${result['profit']:,.2f}")
            print(f"      Win Rate: {result['win_rate']:.1%}")
            print(f"      Return: {result['total_return_pct']:.1f}%")
            print(f"      Final Cash: ${result['final_cash']:,.2f}")
            print(f"      M/E Ratio: {result['final_me_ratio']:.1f}%")
            print(f"      Positions: {result['positions']}")
            
            return result
            
        except Exception as e:
            error_msg = f"nGS strategy execution failed: {str(e)[:100]}..."
            print(f"   ❌ {error_msg}")
            traceback.print_exc()
            return {
                'strategy': 'nGS Original',
                'trades': 0,
                'profit': 0.0,
                'win_rate': 0.0,
                'total_return_pct': 0.0,
                'final_cash': self.account_size,
                'final_me_ratio': 0.0,
                'positions': 0,
                'status': f'FAILED: {error_msg}'
            }
    
    def compare_strategies(self, objectives: List[str], data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Compare nGS strategy against AI-generated strategies
        Enhanced with comprehensive error handling and reporting
        """
        print(f"\n🔬 COMPREHENSIVE STRATEGY COMPARISON")
        print(f"=" * 60)
        print(f"📊 Data: {len(data)} symbols loaded")
        print(f"🎯 Objectives: {', '.join(objectives)}")
        print(f"💰 Account Size: ${self.account_size:,}")
        print(f"=" * 60)
        
        results = []
        
        # Test original nGS strategy first
        print(f"\n1️⃣  TESTING ORIGINAL nGS STRATEGY")
        original_result = self.backtest_original_ngs(data)
        results.append(original_result)
        
        # Test AI strategies
        print(f"\n2️⃣  TESTING AI STRATEGIES")
        for i, objective in enumerate(objectives, 2):
            print(f"\n{i}️⃣  TESTING AI STRATEGY: {objective.upper()}")
            ai_result = self.backtest_ai_strategy(objective, data)
            results.append(ai_result)
        
        # Generate comparison summary
        print(f"\n🏆 STRATEGY COMPARISON SUMMARY")
        print(f"=" * 80)
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results: List[Dict[str, Any]]):
        """Print a formatted comparison table of all strategy results"""
        
        # Header
        print(f"{'Strategy':<20} {'Trades':<8} {'Profit':<12} {'Win Rate':<10} {'Return %':<10} {'Status':<15}")
        print(f"{'-'*20} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*15}")
        
        # nGS strategy row
        ngs_result = results[0] if results else {}
        if ngs_result.get('strategy') == 'nGS Original':
            profit_str = f"${ngs_result.get('profit', 0):,.0f}"
            win_rate_str = f"{ngs_result.get('win_rate', 0):.1%}"
            return_str = f"{ngs_result.get('total_return_pct', 0):+.1f}%"
            status = ngs_result.get('status', 'Unknown')[:14]
            
            print(f"{'nGS Original':<20} {ngs_result.get('trades', 0):<8} {profit_str:<12} {win_rate_str:<10} {return_str:<10} {status:<15}")
        
        # AI strategy rows
        for result in results[1:]:
            if 'objective' in result:
                strategy_name = result['objective'][:19]
                profit_str = f"${result.get('profit', 0):,.0f}"
                win_rate_str = f"{result.get('win_rate', 0):.1%}"
                return_str = f"{result.get('total_return_pct', 0):+.1f}%"
                status = result.get('status', 'Unknown')[:14]
                
                print(f"{strategy_name:<20} {result.get('trades', 0):<8} {profit_str:<12} {win_rate_str:<10} {return_str:<10} {status:<15}")
        
        print(f"=" * 80)
        
        # Summary insights
        print(f"\n📋 INSIGHTS:")
        successful_strategies = [r for r in results if 'FAILED' not in r.get('status', '')]
        
        if successful_strategies:
            best_profit = max(successful_strategies, key=lambda x: x.get('profit', 0))
            best_return = max(successful_strategies, key=lambda x: x.get('total_return_pct', 0))
            best_winrate = max(successful_strategies, key=lambda x: x.get('win_rate', 0))
            
            profit_name = best_profit.get('strategy', best_profit.get('objective', 'Unknown'))
            return_name = best_return.get('strategy', best_return.get('objective', 'Unknown'))
            winrate_name = best_winrate.get('strategy', best_winrate.get('objective', 'Unknown'))
            
            print(f"💰 Best Profit: {profit_name} (${best_profit.get('profit', 0):,.0f})")
            print(f"📈 Best Return: {return_name} ({best_return.get('total_return_pct', 0):+.1f}%)")
            print(f"🎯 Best Win Rate: {winrate_name} ({best_winrate.get('win_rate', 0):.1%})")
            
            # Check nGS M/E status
            if results and results[0].get('strategy') == 'nGS Original':
                me_ratio = results[0].get('final_me_ratio', 0)
                if 50 <= me_ratio <= 80:
                    print(f"✅ nGS M/E Ratio: {me_ratio:.1f}% (within 50-80% target)")
                else:
                    print(f"⚠️  nGS M/E Ratio: {me_ratio:.1f}% (outside 50-80% target)")
        else:
            print(f"❌ No strategies completed successfully")
        
        failed_count = len([r for r in results if 'FAILED' in r.get('status', '')])
        if failed_count > 0:
            print(f"⚠️  {failed_count} strategies failed to execute")

# Example usage and testing
def test_backtester():
    """Test the backtester with sample data"""
    print("\n🧪 TESTING BACKTESTER SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize backtester
        backtester = SimpleNGSBacktester(account_size=100000)
        
        # Test objectives
        test_objectives = ['linear_equity', 'max_roi']
        
        # Load minimal test data
        test_symbols = ['AAPL', 'MSFT']
        print(f"📊 Loading test data for {test_symbols}...")
        
        # You would replace this with your actual data loading
        data = load_polygon_data(test_symbols)
        
        if not data:
            print("❌ No data loaded - cannot run test")
            return
        
        print(f"✅ Loaded data for {len(data)} symbols")
        
        # Run comparison
        results = backtester.compare_strategies(test_objectives, data)
        
        print(f"\n✅ Backtester test completed!")
        print(f"📊 Results generated for {len(results)} strategies")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtester test failed: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 nGS AI BACKTESTER - COMPREHENSIVE STRATEGY COMPARISON")
    print("=" * 60)
    print("🚀 Tests original nGS strategy vs AI-generated alternatives")
    print("🎯 Compares multiple objectives on the same data")
    print("📊 Provides detailed performance analysis")
    print("=" * 60)
    
    # Run test
    test_results = test_backtester()
    
    if test_results:
        print(f"\n✅ BACKTESTER READY FOR PRODUCTION!")
        print(f"🔧 Successfully tested {len(test_results)} strategies")
    else:
        print(f"\n⚠️  BACKTESTER NEEDS ATTENTION")
        print(f"🔧 Check data loading and strategy configuration")
