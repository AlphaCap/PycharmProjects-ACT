# reporting.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import json

from data_manager import get_trades_history, get_positions, init_metadata, update_metadata

def generate_performance_report(days: int = 30) -> pd.DataFrame:
    """Generate a performance report for the recent trading activity."""
    # Get trade history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    trades_df = get_trades_history(start_date=start_date.strftime("%Y-%m-%d"))
    
    if trades_df.empty:
        print("No trades in the specified period.")
        return pd.DataFrame()
    
    # Calculate performance metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = trades_df['profit'].sum()
    avg_profit = trades_df['profit'].mean()
    max_profit = trades_df['profit'].max()
    max_loss = trades_df['profit'].min()
    
    # Group by symbols
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': ['sum', 'mean', 'count'],
        'exit_reason': lambda x: x.value_counts().index[0] if not x.empty else None
    })
    symbol_performance.columns = ['total_profit', 'avg_profit', 'num_trades', 'most_common_exit']
    symbol_performance = symbol_performance.sort_values('total_profit', ascending=False)
    
    # Create a summary dataframe
    summary = pd.DataFrame({
        'Metric': ['Total Trades', 'Winning Trades', 'Win Rate', 'Total Profit', 
                  'Average Profit', 'Max Profit', 'Max Loss', 'Period (days)'],
        'Value': [total_trades, winning_trades, f"{win_rate:.2%}", f"${total_profit:.2f}", 
                 f"${avg_profit:.2f}", f"${max_profit:.2f}", f"${max_loss:.2f}", days]
    })
    
    # Print summary
    print("\n=== Performance Summary ===")
    print(summary.to_string(index=False))
    
    print("\n=== Top Performing Symbols ===")
    print(symbol_performance.head(10).to_string())
    
    # Create visualizations
    os.makedirs("reports", exist_ok=True)
    
    # Profit distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(trades_df['profit'], kde=True)
    plt.title('Profit Distribution')
    plt.xlabel('Profit ($)')
    plt.ylabel('Count')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig(os.path.join("reports", "profit_distribution.png"))
    
    # Profit by symbol
    plt.figure(figsize=(12, 8))
    top_symbols = symbol_performance.head(15).index
    symbol_trades = trades_df[trades_df['symbol'].isin(top_symbols)]
    sns.barplot(x='symbol', y='profit', data=symbol_trades)
    plt.title('Profit by Symbol')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "profit_by_symbol.png"))
    
    # Update metadata with performance
    update_metadata("performance.last_update", datetime.now().strftime("%Y-%m-%d"))
    update_metadata("performance.win_rate", round(win_rate * 100, 2))
    update_metadata("performance.total_profit_30d", round(float(total_profit), 2))
    
    return symbol_performance

def current_positions_report():
    """Generate a report on current positions."""
    positions = get_positions()
    
    if not positions:
        print("No active positions.")
        return
    
    # Calculate position metrics
    pos_df = pd.DataFrame(positions)
    total_value = pos_df['current_value'].sum()
    total_profit = pos_df['profit'].sum()
    avg_days_held = pos_df['days_held'].mean()
    
    # Sort by profit
    pos_df = pos_df.sort_values('profit', ascending=False)
    
    print("\n=== Current Positions ===")
    print(f"Total Positions: {len(pos_df)}")
    print(f"Total Position Value: ${total_value:.2f}")
    print(f"Unrealized Profit: ${total_profit:.2f}")
    print(f"Average Days Held: {avg_days_held:.1f}")
    
    print("\n=== Top Profitable Positions ===")
    print(pos_df.head(10)[['symbol', 'shares', 'entry_price', 'current_price', 'profit', 'profit_pct', 'days_held']].to_string())
    
    print("\n=== Worst Performing Positions ===")
    print(pos_df.tail(5)[['symbol', 'shares', 'entry_price', 'current_price', 'profit', 'profit_pct', 'days_held']].to_string())
    
    # Create visualizations
    os.makedirs("reports", exist_ok=True)
    
    # Position profit distribution
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='days_held', y='profit_pct', size='current_value', hue='profit_pct', 
                   palette='RdYlGn', data=pos_df)
    plt.title('Position Profit % vs Days Held')
    plt.xlabel('Days Held')
    plt.ylabel('Profit %')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "positions_profit.png"))

if __name__ == "__main__":
    print("=" * 50)
    print("nGS Trading System - Performance Report")
    print("=" * 50)
    
    # Generate performance reports
    generate_performance_report(days=30)
    current_positions_report()
