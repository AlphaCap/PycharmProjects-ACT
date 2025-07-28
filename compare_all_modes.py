from ngs_ai_integration_manager import NGSAIIntegrationManager
from nGS_Revised_Strategy import load_polygon_data
from ngs_integrated_ai_system import NGSAIBacktestingSystem
import os

# Load symbols
symbols = []
sp500_file = os.path.join('data', 'sp500_symbols.txt')
if os.path.exists(sp500_file):
    with open(sp500_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
else:
    symbols = ["AAPL", "MSFT", "GOOGL"]

# Load historical data
data = load_polygon_data(symbols)

# Initialize manager and backtesting system
manager = NGSAIIntegrationManager(account_size=1_000_000)
comparison = NGSAIBacktestingSystem(account_size=1_000_000)

# Run Original
manager.set_operating_mode('original')
results_original = manager.run_integrated_strategy(data)

# Run AI-Only
manager.set_operating_mode('ai_only')
results_ai = manager.run_integrated_strategy(data)

# Run Hybrid
manager.set_operating_mode('hybrid')
results_hybrid = manager.run_integrated_strategy(data)

# Run automated reporting
from nGS_Revised_Strategy import run_ngs_automated_reporting
run_ngs_automated_reporting(comparison=comparison)

# Print summary table
def print_mode_performance(results_original, results_ai, results_hybrid):
    def summary(res, mode_name):
        if res.get('original_ngs'):
            perf = res['original_ngs']['performance']
        elif res.get('ai_strategies'):
            # Take best/first AI strategy
            perf = next(iter(res['ai_strategies'].values()))['performance']
        else:
            return [mode_name, '-', '-', '-', '-']
        return [
            mode_name,
            f"{perf.get('total_pnl', 0):,.2f}",
            f"{perf.get('win_rate', 0) * 100:.2f}%",
            f"{perf.get('total_trades', 0)}",
            f"{perf.get('me_ratio', 0):.2f}",
        ]

    print("\n=== Strategy Mode Performance Comparison ===")
    print("{:<12} {:>12} {:>12} {:>12} {:>12}".format(
        "Mode", "Total PnL", "Win Rate", "Trades", "M/E Ratio"))
    for summary_row in [
        summary(results_original, "Original"),
        summary(results_ai, "AI-Only"),
        summary(results_hybrid, "Hybrid")
    ]:
        print("{:<12} {:>12} {:>12} {:>12} {:>12}".format(*summary_row))

print_mode_performance(results_original, results_ai, results_hybrid)
