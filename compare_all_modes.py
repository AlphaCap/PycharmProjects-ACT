from ngs_ai_integration_manager import NGSAIIntegrationManager

# Load your data as needed
# Example: data = load_polygon_data(...)

data = ...  # Load your historical data here

manager = NGSAIIntegrationManager(account_size=1_000_000)

# Run Original
manager.set_operating_mode('original')
results_original = manager.run_integrated_strategy(data)

# Run AI-Only
manager.set_operating_mode('ai_only')
results_ai = manager.run_integrated_strategy(data)

# Run Hybrid
manager.set_operating_mode('hybrid')
results_hybrid = manager.run_integrated_strategy(data)

# Print summary table
def print_mode_performance(results_original, results_ai, results_hybrid):
    def summary(res, mode_name):
        if res['original_ngs']:
            perf = res['original_ngs']['performance']
        elif res['ai_strategies']:
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
