# reporting.py - Section 1: Imports and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

from data_manager import get_trades_history, get_positions, init_metadata, update_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reporting.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set plot styles for better visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Create reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/charts", exist_ok=True)

# Constants for reporting
DEFAULT_REPORT_DAYS = 30
CHART_SIZE = (12, 8)
CHART_DPI = 100
# reporting.py - Section 2: Trade Performance Analysis
def generate_performance_report(days: int = DEFAULT_REPORT_DAYS) -> pd.DataFrame:
    """
    Generate a comprehensive performance report for the recent trading activity.
    
    Args:
        days: Number of days to include in the report
        
    Returns:
        DataFrame with symbol performance metrics
    """
    # Get trade history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    trades_df = get_trades_history(start_date=start_date_str)
    
    if trades_df.empty:
        logger.warning(f"No trades found in the last {days} days")
        print(f"No trades in the last {days} days.")
        return pd.DataFrame()
    
    # Calculate performance metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['profit'] > 0])
    losing_trades = len(trades_df[trades_df['profit'] <= 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = trades_df['profit'].sum()
    avg_profit = trades_df['profit'].mean()
    avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['profit'] <= 0]['profit'].mean() if losing_trades > 0 else 0
    max_profit = trades_df['profit'].max()
    max_loss = trades_df['profit'].min()
    profit_factor = abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / 
                        trades_df[trades_df['profit'] < 0]['profit'].sum()) if losing_trades > 0 else float('inf')
    
    # Group by signal types
    signal_performance = trades_df.groupby('type').agg({
        'profit': ['sum', 'mean', 'count'],
        'exit_reason': lambda x: x.value_counts().index[0] if not x.empty else None
    })
    signal_performance.columns = ['total_profit', 'avg_profit', 'num_trades', 'most_common_exit']
    
    # Group by symbols
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': ['sum', 'mean', 'count'],
        'exit_reason': lambda x: x.value_counts().index[0] if not x.empty else None
    })
    symbol_performance.columns = ['total_profit', 'avg_profit', 'num_trades', 'most_common_exit']
    symbol_performance = symbol_performance.sort_values('total_profit', ascending=False)
    
    # Group by exit reasons
    exit_performance = trades_df.groupby('exit_reason').agg({
        'profit': ['sum', 'mean', 'count']
    })
    exit_performance.columns = ['total_profit', 'avg_profit', 'num_trades']
    exit_performance = exit_performance.sort_values('num_trades', ascending=False)
    
    # Create a summary dataframe
    summary = pd.DataFrame({
        'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate', 
                  'Total Profit', 'Average Profit', 'Average Win', 'Average Loss',
                  'Max Profit', 'Max Loss', 'Profit Factor', 'Period (days)'],
        'Value': [total_trades, winning_trades, losing_trades, f"{win_rate:.2%}", 
                 f"${total_profit:.2f}", f"${avg_profit:.2f}", f"${avg_win:.2f}", f"${avg_loss:.2f}",
                 f"${max_profit:.2f}", f"${max_loss:.2f}", f"{profit_factor:.2f}", days]
    })
    
    # Print summary report
    print("\n" + "="*50)
    print(f"PERFORMANCE SUMMARY - LAST {days} DAYS")
    print("="*50)
    print(summary.to_string(index=False))
    
    print("\n" + "="*50)
    print("SIGNAL TYPE PERFORMANCE")
    print("="*50)
    if not signal_performance.empty:
        print(signal_performance.to_string())
    else:
        print("No signal type data available")
    
    print("\n" + "="*50)
    print("EXIT REASON PERFORMANCE")
    print("="*50)
    if not exit_performance.empty:
        print(exit_performance.to_string())
    else:
        print("No exit reason data available")
    
    print("\n" + "="*50)
    print("TOP PERFORMING SYMBOLS")
    print("="*50)
    if not symbol_performance.empty:
        print(symbol_performance.head(10).to_string())
    else:
        print("No symbol performance data available")
# reporting.py - Section 3: Visualization Functions
def create_performance_charts(days: int = DEFAULT_REPORT_DAYS) -> bool:
    """
    Create charts visualizing trading performance.
    
    Args:
        days: Number of days to include in the charts
        
    Returns:
        Boolean indicating success/failure
    """
    # Get trade history
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime("%Y-%m-%d")
    trades_df = get_trades_history(start_date=start_date_str)
    
    if trades_df.empty:
        logger.warning(f"No trades found for charting in the last {days} days")
        return False
    
    try:
        # Ensure we have datetime for exit_date
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        # Sort by date for time series
        trades_df = trades_df.sort_values('exit_date')
        
        # 1. Profit Distribution Histogram
        plt.figure(figsize=CHART_SIZE)
        sns.histplot(trades_df['profit'], kde=True, bins=20)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.title(f'Profit Distribution - Last {days} Days', fontsize=16)
        plt.xlabel('Profit ($)', fontsize=12)
        plt.ylabel('Number of Trades', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "charts", "profit_distribution.png"), dpi=CHART_DPI)
        plt.close()
        
        # 2. Cumulative Profit Over Time
        plt.figure(figsize=CHART_SIZE)
        cumulative_profit = trades_df.sort_values('exit_date').set_index('exit_date')['profit'].cumsum()
        plt.plot(cumulative_profit.index, cumulative_profit.values, linewidth=2)
        plt.title(f'Cumulative Profit Over Time - Last {days} Days', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Profit ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.fill_between(cumulative_profit.index, cumulative_profit.values, 
                         where=(cumulative_profit.values > 0), alpha=0.2, color='green')
        plt.fill_between(cumulative_profit.index, cumulative_profit.values, 
                         where=(cumulative_profit.values <= 0), alpha=0.2, color='red')
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "charts", "cumulative_profit.png"), dpi=CHART_DPI)
        plt.close()
        
        # 3. Win Rate by Signal Type
        plt.figure(figsize=CHART_SIZE)
        signal_win_rates = trades_df.groupby('type')['profit'].apply(
            lambda x: (x > 0).mean() if len(x) > 0 else 0
        ).reset_index()
        signal_win_rates.columns = ['Signal Type', 'Win Rate']
        
        # Only plot if we have signal types
        if not signal_win_rates.empty:
            sns.barplot(x='Signal Type', y='Win Rate', data=signal_win_rates)
            plt.title('Win Rate by Signal Type', fontsize=16)
            plt.xlabel('Signal Type', fontsize=12)
            plt.ylabel('Win Rate', fontsize=12)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join("reports", "charts", "win_rate_by_signal.png"), dpi=CHART_DPI)
            plt.close()
        
        # 4. Top Performing Symbols
        plt.figure(figsize=CHART_SIZE)
        top_symbols = trades_df.groupby('symbol')['profit'].sum().nlargest(10).reset_index()
        sns.barplot(x='symbol', y='profit', data=top_symbols)
        plt.title('Top 10 Performing Symbols', fontsize=16)
        plt.xlabel('Symbol', fontsize=12)
        plt.ylabel('Total Profit ($)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "charts", "top_symbols.png"), dpi=CHART_DPI)
        plt.close()
        
        # 5. Exit Reason Performance
        plt.figure(figsize=CHART_SIZE)
        exit_performance = trades_df.groupby('exit_reason')['profit'].mean().reset_index()
        exit_performance = exit_performance.sort_values('profit', ascending=False)
        sns.barplot(x='exit_reason', y='profit', data=exit_performance)
        plt.title('Average Profit by Exit Reason', fontsize=16)
        plt.xlabel('Exit Reason', fontsize=12)
        plt.ylabel('Average Profit ($)', fontsize=12)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join("reports", "charts", "exit_reason_performance.png"), dpi=CHART_DPI)
        plt.close()
        
        logger.info(f"Successfully created performance charts for last {days} days")
        return True
    
    except Exception as e:
        logger.error(f"Error creating performance charts: {e}")
        return False
# reporting.py - Section 4: Current Positions and Main Function
def current_positions_report() -> pd.DataFrame:
    """
    Generate a report on current positions.
    
    Returns:
        DataFrame with current positions
    """
    positions = get_positions()
    
    if not positions:
        logger.warning("No active positions found")
        print("\nNo active positions.")
        return pd.DataFrame()
    
    # Convert to DataFrame for easier analysis
    pos_df = pd.DataFrame(positions)
    
    # Calculate position metrics
    total_value = pos_df['current_value'].sum()
    total_profit = pos_df['profit'].sum()
    profit_pct = 100 * total_profit / (total_value - total_profit) if total_value > total_profit else 0
    avg_days_held = pos_df['days_held'].mean()
    
    # Calculate directional exposure
    long_value = pos_df[pos_df['shares'] > 0]['current_value'].sum()
    short_value = abs(pos_df[pos_df['shares'] < 0]['current_value'].sum())
    net_exposure = long_value - short_value
    gross_exposure = long_value + short_value
    
    # Sort by profit
    pos_df = pos_df.sort_values('profit', ascending=False)
    
    print("\n" + "="*50)
    print("CURRENT POSITIONS SUMMARY")
    print("="*50)
    print(f"Total Positions: {len(pos_df)}")
    print(f"Long Positions: {len(pos_df[pos_df['shares'] > 0])}")
    print(f"Short Positions: {len(pos_df[pos_df['shares'] < 0])}")
    print(f"Total Position Value: ${total_value:.2f}")
    print(f"Unrealized Profit/Loss: ${total_profit:.2f} ({profit_pct:.2f}%)")
    print(f"Average Days Held: {avg_days_held:.1f}")
    print(f"Net Market Exposure: ${net_exposure:.2f}")
    print(f"Gross Market Exposure: ${gross_exposure:.2f}")
    
    print("\n" + "="*50)
    print("TOP PROFITABLE POSITIONS")
    print("="*50)
    columns_to_show = ['symbol', 'shares', 'entry_price', 'current_price', 'profit', 'profit_pct', 'days_held']
    if not pos_df.empty:
        print(pos_df.head(10)[columns_to_show].to_string())
    
        if len(pos_df) > 10:
            print("\n" + "="*50)
            print("WORST PERFORMING POSITIONS")
            print("="*50)
            print(pos_df.tail(5)[columns_to_show].to_string())
    
    # Create visualization
    try:
        if not pos_df.empty:
            plt.figure(figsize=CHART_SIZE)
            scatter = plt.scatter(pos_df['days_held'], pos_df['profit_pct'], 
                       s=pos_df['current_value'].abs() / 100, 
                       c=pos_df['profit'], cmap='RdYlGn',
                       alpha=0.7)
            
            plt.colorbar(scatter, label='Profit ($)')
            plt.title('Current Positions - Profit % vs Days Held', fontsize=16)
            plt.xlabel('Days Held', fontsize=12)
            plt.ylabel('Profit %', fontsize=12)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join("reports", "charts", "current_positions.png"), dpi=CHART_DPI)
            plt.close()
            
            # Position distribution chart
            plt.figure(figsize=CHART_SIZE)
            labels = pos_df['symbol'].head(15)
            sizes = pos_df['current_value'].abs().head(15)
            colors = ['green' if p > 0 else 'red' for p in pos_df['profit'].head(15)]
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Position Size Distribution (Top 15)', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join("reports", "charts", "position_distribution.png"), dpi=CHART_DPI)
            plt.close()
    
    except Exception as e:
        logger.error(f"Error creating position charts: {e}")
    
    # Update metadata with position stats
    update_metadata("positions.count", len(pos_df))
    update_metadata("positions.total_value", round(float(total_value), 2))
    update_metadata("positions.profit", round(float(total_profit), 2))
    
    return pos_df

def export_report_to_html(days: int = DEFAULT_REPORT_DAYS) -> bool:
    """
    Export performance report to HTML file.
    
    Args:
        days: Number of days to include in the report
        
    Returns:
        Boolean indicating success/failure
    """
    try:
        # Get trade history
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%d")
        trades_df = get_trades_history(start_date=start_date_str)
        positions = get_positions()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>nGS Trading System Report - {datetime.now().strftime('%Y-%m-%d')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                .chart {{ max-width: 100%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>nGS Trading System Performance Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Performance Summary - Last {days} Days</h2>
        """
        
        # Add trade performance summary
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_profit = trades_df['profit'].sum()
            
            html_content += f"""
                <p><strong>Total Trades:</strong> {total_trades}</p>
                <p><strong>Win Rate:</strong> {win_rate:.2%}</p>
                <p><strong>Total Profit:</strong> ${total_profit:.2f}</p>
            """
        else:
            html_content += "<p>No trades in the specified period.</p>"
        
        # Add position summary
        if positions:
            pos_df = pd.DataFrame(positions)
            total_value = pos_df['current_value'].sum()
            total_profit = pos_df['profit'].sum()
            
            html_content += f"""
                <h3>Current Positions Summary</h3>
                <p><strong>Total Positions:</strong> {len(pos_df)}</p>
                <p><strong>Total Value:</strong> ${total_value:.2f}</p>
                <p><strong>Unrealized Profit:</strong> ${total_profit:.2f}</p>
            """
        
        # Close summary div
        html_content += "</div>"
        
        # Add charts
        html_content += """
            <h2>Performance Charts</h2>
            <div class="chart-container">
                <h3>Profit Distribution</h3>
                <img class="chart" src="charts/profit_distribution.png" alt="Profit Distribution">
            </div>
            
            <div class="chart-container">
                <h3>Cumulative Profit Over Time</h3>
                <img class="chart" src="charts/cumulative_profit.png" alt="Cumulative Profit">
            </div>
            
            <div class="chart-container">
                <h3>Top Performing Symbols</h3>
                <img class="chart" src="charts/top_symbols.png" alt="Top Symbols">
            </div>
            
            <div class="chart-container">
                <h3>Current Positions</h3>
                <img class="chart" src="charts/current_positions.png" alt="Current Positions">
            </div>
        """
        
        # Add position details
        if positions:
            pos_df = pd.DataFrame(positions)
            pos_df = pos_df.sort_values('profit', ascending=False)
            
            html_content += """
                <h2>Current Positions</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Shares</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>Profit</th>
                        <th>Profit %</th>
                        <th>Days Held</th>
                    </tr>
            """
            
            for _, row in pos_df.iterrows():
                profit_color = "green" if row['profit'] > 0 else "red"
                html_content += f"""
                    <tr>
                        <td>{row['symbol']}</td>
                        <td>{int(row['shares'])}</td>
                        <td>${row['entry_price']:.2f}</td>
                        <td>${row['current_price']:.2f}</td>
                        <td style="color: {profit_color}">${row['profit']:.2f}</td>
                        <td style="color: {profit_color}">{row['profit_pct']:.2f}%</td>
                        <td>{int(row['days_held'])}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add recent trades
        if not trades_df.empty:
            trades_df = trades_df.sort_values('exit_date', ascending=False).head(20)
            
            html_content += """
                <h2>Recent Trades (Last 20)</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Entry Date</th>
                        <th>Exit Date</th>
                        <th>Profit</th>
                        <th>Exit Reason</th>
                    </tr>
            """
            
            for _, row in trades_df.iterrows():
                profit_color = "green" if row['profit'] > 0 else "red"
                html_content += f"""
                    <tr>
                        <td>{row['symbol']}</td>
                        <td>{row['type']}</td>
                        <td>{row['entry_date']}</td>
                        <td>{row['exit_date']}</td>
                        <td style="color: {profit_color}">${row['profit']:.2f}</td>
                        <td>{row['exit_reason']}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Close HTML
        html_content += """
            <footer>
                <p>Generated by nGS Trading System</p>
            </footer>
        </body>
        </html>
        """
        
        # Write to file
        report_path = os.path.join("reports", f"performance_report_{datetime.now().strftime('%Y%m%d')}.html")
        with open(report_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported to {report_path}")
        print(f"\nHTML report saved to: {report_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting HTML report: {e}")
        return False

def main(days: int = DEFAULT_REPORT_DAYS, export_html: bool = True):
    """
    Run all reporting functions.
    
    Args:
        days: Number of days to include in report
        export_html: Whether to export report to HTML
    """
    print("\n" + "="*80)
    print(f" nGS TRADING SYSTEM PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d')}")
    print("="*80)
    
    # Generate trade performance report
    generate_performance_report(days=days)
    
    # Generate current positions report
    current_positions_report()
    
    # Create charts
    create_performance_charts(days=days)
    
    # Export to HTML if requested
    if export_html:
        export_report_to_html(days=days)
    
    print("\n" + "="*50)
    print("Report generation complete")
    print("="*50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate nGS Trading System performance reports")
    parser.add_argument("--days", type=int, default=DEFAULT_REPORT_DAYS, help="Days to include in report")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report export")
    
    args = parser.parse_args()
    
    main(days=args.days, export_html=not args.no_html)
