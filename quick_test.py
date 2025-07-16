import data_manager as dm

print('=== QUICK DATA TEST ===')
trades = dm.get_trades_history()
print(f'Trades: {len(trades)}')

if len(trades) > 0:
    print(f'Total profit: ${trades["profit"].sum():,.2f}')
    print(f'Sample trade: {trades.iloc[0]["symbol"]} - ${trades.iloc[0]["profit"]:.2f}')

metrics = dm.get_portfolio_metrics(1000000)
print(f'Total return %: {metrics["total_return_pct"]}')
print(f'Daily P&L: {metrics["daily_pnl"]}')
print(f'M/E ratio: {metrics["me_ratio"]}')
print(f'MTD return: {metrics["mtd_return"]}')
