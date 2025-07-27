import pandas as pd
import os
import json
from nGS_Revised_Strategy import NGSStrategy
from data_utils import load_polygon_data
from ngs_ai_backtesting_system import NGSAIBacktestingSystem

def run_ngs_automated_reporting():
    """Run comprehensive backtesting with AI integration and save results."""
    print("🚀 Running nGS Automated Backtesting")
    print("=" * 70)
    
    # Load symbols
    sp500_file = os.path.join('data', 'sp500_symbols.txt')
    if os.path.exists(sp500_file):
        with open(sp500_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"📊 Loaded {len(symbols)} S&P 500 symbols")
    else:
        print(f"⚠️ {sp500_file} not found. Using sample symbols.")
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"]

    # Load price data
    print(f"🔄 Loading market data for {len(symbols)} symbols...")
    data = load_polygon_data(symbols)
    
    if not data:
        print("❌ No data loaded - check your data files")
        return

    print(f"✅ Successfully loaded data for {len(data)} symbols")
    
    # Initialize backtesting system
    backtester = NGSAIBacktestingSystem(account_size=1_000_000, data_dir='data')
    
    # Define objectives for AI strategies
    objectives = ['linear_equity', 'max_roi', 'min_drawdown', 'high_winrate', 'sharpe_ratio']
    
    # Run comprehensive backtest
    print(f"\n🔬 Running comprehensive comparison with {len(objectives)} AI strategies...")
    comparison = backtester.backtest_comprehensive_comparison(objectives, data)
    
    # Save trades to CSV
    all_trades = comparison.original_ngs_result.trades
    for ai_result in comparison.ai_results:
        all_trades.extend(ai_result.trades)
    
    trade_history_path = os.path.join('data', 'trade_history.csv')
    new_trades_df = pd.DataFrame([{
        'symbol': trade['symbol'],
        'entry_date': trade['entry_date'],
        'exit_date': trade['exit_date'],
        'entry_price': trade['entry_price'],
        'exit_price': trade['exit_price'],
        'profit_loss': trade['profit']
    } for trade in all_trades])
    
    if os.path.exists(trade_history_path):
        prior_trades = pd.read_csv(trade_history_path)
    else:
        prior_trades = pd.DataFrame()
    
    all_trades_df = pd.concat([prior_trades, new_trades_df], ignore_index=True)
    all_trades_df = all_trades_df.drop_duplicates(subset=['symbol', 'entry_date', 'exit_date'])
    all_trades_df.to_csv(trade_history_path, index=False)
    print(f"✅ Trades saved to {trade_history_path}")
    
    # Save summary stats
    summary_stats_path = os.path.join('data', 'summary_stats.json')
    with open(summary_stats_path, 'w') as f:
        json.dump(comparison.summary_stats, f, indent=2)
    print(f"✅ Summary stats saved to {summary_stats_path}")
    
    print("\n✅ Automated backtesting completed!")
    print(f"   Recommendation: {comparison.recommendation}")
    print(f"   Best Performing Strategy: {comparison.summary_stats['best_performing_strategy']}")
    print(f"   Lowest Risk Strategy: {comparison.summary_stats['lowest_risk_strategy']}")
    print(f"   Highest Sharpe Strategy: {comparison.summary_stats['highest_sharpe_strategy']}")

if __name__ == "__main__":
    try:
        run_ngs_automated_reporting()
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
