import pandas as pd
import os
import traceback
from nGS_Revised_Strategy import NGSStrategy
from data_utils import load_polygon_data
from typing import Dict, List, Any


class SimpleNGSBacktester:
    def __init__(self, account_size=1_000_000, data_dir="data"):
        self.account_size = account_size
        self.data_dir = data_dir
        print(f" Simple Backtester Initialized - Account Size: ${account_size:,}")

    def load_symbols(self) -> List[str]:
        """Load symbols from the S&P 500 file or use defaults."""
        sp500_file = os.path.join(self.data_dir, "sp500_symbols.txt")
        if os.path.exists(sp500_file):
            with open(sp500_file, "r") as f:
                symbols = [line.strip() for line in f if line.strip()]
            print(f" Loaded {len(symbols)} S&P 500 symbols")
        else:
            print(f" {sp500_file} not found. Using sample symbols.")
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA"]
        return symbols

    def load_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load market data for the given symbols."""
        print(f" Loading market data for {len(symbols)} symbols...")
        data = load_polygon_data(symbols)
        if not data:
            print(" No data loaded - check your data files")
            return {}
        print(f" Successfully loaded data for {len(data)} symbols")
        return data

    def backtest_original_ngs(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run the original nGS strategy."""
        print(f" Backtesting Original nGS Strategy")
        try:
            ngs_strategy = NGSStrategy(account_size=self.account_size)
            results = ngs_strategy.run(data)
            trades = ngs_strategy.trades
            profit = sum(trade["profit"] for trade in trades) if trades else 0.0
            win_rate = (
                len([t for t in trades if t["profit"] > 0]) / len(trades)
                if trades
                else 0.0
            )
            return {
                "strategy": "nGS Original",
                "trades": len(trades),
                "profit": profit,
                "win_rate": win_rate,
                "status": "Success",
            }
        except Exception as e:
            print(f" Error during nGS backtesting: {e}")
            traceback.print_exc()
            return {"strategy": "nGS Original", "status": "Failed"}

    def backtest_ai_strategy(
        self, objective: str, data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Backtest an AI-generated strategy."""
        print(f" Backtesting AI strategy: {objective}")
        try:
            # Placeholder for AI strategy backtesting logic
            # Replace with actual strategy generation and execution
            print(f"    Generating strategy for {objective}...")
            # Simulate results
            return {
                "objective": objective,
                "trades": 10,
                "profit": 1000.0,
                "win_rate": 0.6,
                "status": "Success",
            }
        except Exception as e:
            print(f" Error during AI strategy backtesting: {e}")
            traceback.print_exc()
            return {"objective": objective, "status": "Failed"}

    def save_trades(self, trades: List[Dict]):
        """Save trades to a CSV file."""
        trade_history_path = os.path.join(self.data_dir, "trade_history.csv")
        new_trades_df = pd.DataFrame(trades)
        if os.path.exists(trade_history_path):
            prior_trades = pd.read_csv(trade_history_path)
        else:
            prior_trades = pd.DataFrame()
        all_trades_df = pd.concat([prior_trades, new_trades_df], ignore_index=True)
        all_trades_df = all_trades_df.drop_duplicates(
            subset=["symbol", "entry_date", "exit_date"]
        )
        all_trades_df.to_csv(trade_history_path, index=False)
        print(f" Trades saved to {trade_history_path}")

    def run_comprehensive_backtest(self):
        """Run comprehensive backtesting and save results."""
        symbols = self.load_symbols()
        data = self.load_data(symbols)
        if not data:
            return

        # Backtest original nGS strategy
        original_result = self.backtest_original_ngs(data)

        # Backtest AI strategies
        objectives = [
            "linear_equity",
            "max_roi",
            "min_drawdown",
            "high_winrate",
            "sharpe_ratio",
        ]
        ai_results = [self.backtest_ai_strategy(obj, data) for obj in objectives]

        # Combine trades and save
        all_trades = [original_result] + ai_results
        self.save_trades(all_trades)

        print(" Comprehensive backtesting completed!")


if __name__ == "__main__":
    backtester = SimpleNGSBacktester()
    backtester.run_comprehensive_backtest()


