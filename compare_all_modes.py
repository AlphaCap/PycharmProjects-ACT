from ngs_ai_integration_manager import NGSAIIntegrationManager
from nGS_Revised_Strategy import load_polygon_data, run_ngs_automated_reporting
from ngs_ai_backtesting_system import NGSAIBacktestingSystem
import os
import pandas as pd

# Load symbols
symbols = []
sp500_file = os.path.join('data', 'sp500_symbols.txt')
if os.path.exists(sp500_file):
    with open(sp500_file, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
else:
    symbols = ["AAPL", "MSFT", "GOOGL"]

# Load historical data
try:
    data = load_polygon_data(symbols)
    print(f"Data type: {type(data)}, Data keys: {list(data.keys()) if isinstance(data, dict) else data}")
    if data is ... or data is None or (isinstance(data, dict) and not data):
        raise ValueError("load_polygon_data returned invalid data (Ellipsis, None, or empty)")
except Exception as e:
    print(f"Error loading data: {e}")
    data = {symbol: pd.DataFrame() for symbol in symbols}  # Fallback empty data
    print(f"Fallback to empty DataFrames for symbols: {symbols}")

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
