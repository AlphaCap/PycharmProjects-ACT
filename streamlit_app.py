import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import your trading system
from nGS_Strategy import NGSStrategy
from data_manager import get_positions, get_trades_history
from main import process_single_symbol, run_daily_update, display_system_status

# Set page config
st.set_page_config(
    page_title="nGS Trading System",
    page_icon="📈",
    layout="wide"
)

# Constants
VERSION = "1.0.0"

# Set API key from environment or use hardcoded
polygon_api_key = os.getenv('POLYGON_API_KEY')
if not polygon_api_key:
    polygon_api_key = "yTZVrttxzFCK58_gOUGGATWxQzytgAxy"  # Hardcoded API key
    os.environ['POLYGON_API_KEY'] = polygon_api_key
    st.sidebar.info("Using hardcoded Polygon API key.")

# Sidebar
st.sidebar.title("nGS Trading System")
st.sidebar.caption(f"Version {VERSION}")
option = st.sidebar.selectbox(
    "Choose an option",
    ["System Status", "Process Symbol", "Daily Update", "Performance Report"]
)

# Main content area
if option == "System Status":
    st.title("System Status")
    
    # Display basic info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Date", datetime.now().strftime("%Y-%m-%d"))
    with col2:
        st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))
    
    # Load positions
    positions = get_positions()
    
    st.subheader("Active Positions")
    if positions:
        positions_data = []
        for symbol, pos in positions.items():
            positions_data.append({
                "Symbol": symbol,
                "Shares": pos['shares'],
                "Entry Price": f"${pos['entry_price']:.2f}",
                "Entry Date": pos['entry_date'],
                "Days Held": pos['bars_since_entry']
            })
        st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
    else:
        st.info("No active positions")
    
    # Add system info
    with st.expander("System Information"):
        st.text(f"Python Version: {sys.version.split()[0]}")
        st.text(f"System Directory: {os.getcwd()}")

elif option == "Process Symbol":
    st.title("Process Symbol")
    
    symbol = st.text_input("Enter Symbol (e.g., AAPL)", max_chars=5).upper()
    
    if st.button("Process Symbol"):
        if symbol:
            with st.spinner(f"Processing {symbol}..."):
                success = process_single_symbol(symbol)
                
                if success:
                    st.success(f"Successfully processed {symbol}")
                    
                    # Get the strategy instance to show chart
                    strategy = NGSStrategy()
                    df = strategy.get_symbol_data(symbol)
                    
                    if df is not None and not df.empty:
                        # Create plot with matplotlib
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot price
                        ax.plot(df['Date'], df['Close'], label='Close Price')
                        
                        # Plot signals
                        buy_signals = df[df['Signal'] > 0]
                        sell_signals = df[df['ExitSignal'] > 0]
                        
                        ax.scatter(buy_signals['Date'], buy_signals['Close'], 
                                   color='green', label='Buy', marker='^', s=100)
                        ax.scatter(sell_signals['Date'], sell_signals['Close'], 
                                  color='red', label='Sell', marker='v', s=100)
                        
                        ax.set_title(f"{symbol} Price Chart with Signals")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price ($)")
                        ax.legend()
                        ax.grid(True)
                        
                        st.pyplot(fig)
                        
                        # Show recent signals
                        signal_rows = df[(df['Signal'] != 0) | (df['ExitSignal'] != 0)]
                        if not signal_rows.empty:
                            st.subheader("Recent Signal Activity")
                            st.dataframe(signal_rows[['Date', 'Close', 'Signal', 'SignalType', 
                                                     'ExitSignal', 'ExitType']].tail(10))
                else:
                    st.error(f"Failed to process {symbol}")
        else:
            st.warning("Please enter a symbol")

elif option == "Daily Update":
    st.title("Daily Update")
    
    st.warning("This will update data for all S&P 500 stocks. The process may take 1-2 hours due to API rate limits.")
    
    if st.button("Run Daily Update"):
        with st.spinner("Running daily update..."):
            # Use a placeholder to show progress
            progress_text = st.empty()
            
            # Monkey patch print to display in Streamlit
            original_print = print
            def streamlit_print(*args, **kwargs):
                progress_text.write(" ".join(map(str, args)))
                original_print(*args, **kwargs)
            
            # Replace print with our version temporarily
            import builtins
            builtins.print = streamlit_print
            
            # Run the update
            success = run_daily_update()
            
            # Restore print
            builtins.print = original_print
            
            if success:
                st.success("Daily update completed successfully!")
            else:
                st.error("Error occurred during daily update")

elif option == "Performance Report":
    st.title("Performance Report")
    
    days = st.slider("Days to include in report", 7, 90, 30)
    
    if st.button("Generate Report"):
        with st.spinner(f"Generating report for last {days} days..."):
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_date_str = start_date.strftime("%Y-%m-%d")
            trades_df = get_trades_history(start_date=start_date_str)
            
            if trades_df.empty:
                st.info(f"No trades found in the last {days} days")
            else:
                # Calculate metrics
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                losing_trades = len(trades_df[trades_df['profit'] <= 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                total_profit = trades_df['profit'].sum()
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Trades", total_trades)
                col2.metric("Win Rate", f"{win_rate:.2%}")
                col3.metric("Total Profit", f"${total_profit:.2f}")
                
                # Chart the equity curve
                trades_df['cumulative_profit'] = trades_df['profit'].cumsum()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(trades_df['exit_date'], trades_df['cumulative_profit'])
                ax.set_title("Equity Curve")
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Profit ($)")
                ax.grid(True)
                st.pyplot(fig)
                
                # Show trade list
                st.subheader("Trade List")
                st.dataframe(trades_df[['symbol', 'entry_date', 'exit_date', 
                                       'entry_price', 'exit_price', 'shares', 
                                       'profit']], use_container_width=True)

# Footer
st.sidebar.divider()
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
