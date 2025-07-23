# Working nGS AI Backtester
from comprehensive_indicator_library import ComprehensiveIndicatorLibrary
from performance_objectives import ObjectiveManager
from strategy_generator_ai import ObjectiveAwareStrategyGenerator
from nGS_Revised_Strategy import NGSStrategy, load_polygon_data

class SimpleNGSBacktester:
    def __init__(self, account_size=1000000):
        self.account_size = account_size
        self.indicator_lib = ComprehensiveIndicatorLibrary()
        self.objective_manager = ObjectiveManager()
        self.ai_generator = ObjectiveAwareStrategyGenerator(self.indicator_lib, self.objective_manager)
        print(f"🧪 Simple Backtester Ready - ${account_size:,}")
    
    def backtest_ai_strategy(self, objective, data):
        print(f"🧪 Backtesting AI strategy: {objective}")
        strategy = self.ai_generator.generate_strategy_for_objective(objective)
        results = strategy.execute_on_data(data, self.indicator_lib)
        trades = results['trades']
        total_profit = sum(trade['profit'] for trade in trades) if trades else 0
        win_rate = len([t for t in trades if t['profit'] > 0]) / max(1, len(trades))
        return {'objective': objective, 'trades': len(trades), 'profit': total_profit, 'win_rate': win_rate}
    
    def backtest_original_ngs(self, data):
        print(f"📊 Backtesting Original nGS Strategy")
        ngs_strategy = NGSStrategy(account_size=self.account_size)
        ngs_strategy.run(data)
        trades = ngs_strategy.trades
        total_profit = sum(trade['profit'] for trade in trades) if trades else 0
        win_rate = len([t for t in trades if t['profit'] > 0]) / max(1, len(trades))
        return {'strategy': 'original_ngs', 'trades': len(trades), 'profit': total_profit, 'win_rate': win_rate}
    
    def compare_strategies(self, objectives, data):
        print(f"🔬 COMPREHENSIVE COMPARISON")
        results = []
        original = self.backtest_original_ngs(data)
        results.append(original)
        for objective in objectives:
            ai_result = self.backtest_ai_strategy(objective, data)
            results.append(ai_result)
        return results
