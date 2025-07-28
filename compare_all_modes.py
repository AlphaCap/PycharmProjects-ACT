from ngs_ai_integration_manager import NGSAIIntegrationManager
from nGS_Revised_Strategy import run_ngs_automated_reporting
from ngs_ai_backtesting_system import NGSAIBacktestingSystem
import os
import pandas as pd

# --------- Load ALL stocks from data/daily/ ----------
stock_data_dir = "data/daily"
data = {}
for fname in os.listdir(stock_data_dir):
    if fname.endswith(".csv"):
        symbol = fname.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(stock_data_dir, fname))
            data[symbol] = df
        except Exception as e:
            print(f"Error loading {fname}: {e}")

# --------- Load ALL sector ETFs from data/etf_historical/ ----------
etf_data_dir = "data/etf_historical"
for fname in os.listdir(etf_data_dir):
    if fname.endswith("_historical.csv"):
        symbol = fname.replace("_historical.csv", "")
        try:
            df = pd.read_csv(os.path.join(etf_data_dir, fname))
            data[symbol] = df
        except Exception as e:
            print(f"Error loading {fname}: {e}")

print(f"Loaded {len(data)} symbols (stocks + ETFs) from local files")
print("Sample symbols:", list(data.keys())[:10])

# Initialize manager and backtesting system
manager = NGSAIIntegrationManager(account_size=1_000_000)

# Run backtest for comparison (if needed)
try:
    comparison.run(data)
except Exception as e:
    print(f"Warning: Failed to run comparison backtest: {e}")

# Run Original
try:
    manager.set_operating_mode('original')
    results_original = manager.run_integrated_strategy(data)
except Exception as e:
    print(f"Error running Original mode: {e}")
    results_original = {}

# Run AI-Only
try:
    manager.set_operating_mode('ai_only')
    results_ai = manager.run_integrated_strategy(data)
except Exception as e:
    print(f"Error running AI-Only mode: {e}")
    results_ai = {}

# Run Hybrid
try:
    manager.set_operating_mode('hybrid')
    results_hybrid = manager.run_integrated_strategy(data)
except Exception as e:
    print(f"Error running Hybrid mode: {e}")
    results_hybrid = {}

# Run automated reporting
try:
    run_ngs_automated_reporting(comparison=comparison)
except Exception as e:
    print(f"Error running automated reporting: {e}")

# Print summary table
def print_mode_performance(results_original, results_ai, results_hybrid):
    def summary(res, mode_name):
        if not res:
            return [mode_name, '-', '-', '-', '-']
        if res.get('original_ngs'):
            perf = res['original_ngs']['performance']
        elif res.get('ai_strategies'):
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
