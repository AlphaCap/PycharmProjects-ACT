import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from data_manager import (
    get_portfolio_metrics,
    get_trades_history,
    get_trades_history_formatted,
    get_me_ratio_history,
    format_dollars
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="nGS Historical Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HIDE STREAMLIT ELEMENTS ---
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    .stDecoration {display:none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stHeader"] {display: none;}
    .stApp > header {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
    
    .stAppViewContainer > .main .block-container {
        padding-top: 1rem;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("Trading Systems")
   if st.button("HOME", use_container_width=True):
        st.switch_page("app.py")
    
    st.markdown("---")
    st.caption("Historical Analysis")
    st.caption(f"Last updated: {datetime.now().strftime('%m/%d/%Y %H:%M')}")

# --- PAGE HEADER ---
st.title("nGulfStream - Performance")
st.caption("Historical analysis and performance metrics")

# --- INITIAL VALUE SETTING ---
initial_value = st.number_input(
    "Historical initial portfolio value:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- HISTORICAL METRICS ---
st.markdown("## Historical Performance Summary")
historical_metrics = get_portfolio_metrics(initial_portfolio_value=initial_value, is_historical=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Return", value=historical_metrics['total_return_pct'])
with col2:
    st.metric(label="Final Portfolio Value", value=historical_metrics['total_value'])
with col3:
    st.metric(label="Historical M/E Ratio", value=historical_metrics['historical_me_ratio'])
with col4:
    st.metric(label="YTD Return", value=historical_metrics['ytd_return'])

# --- CHARTS SECTION ---
st.markdown("## Performance Charts")

# Get data for charts
trades_df = get_trades_history()
me_ratio_df = get_me_ratio_history()

# Create two columns for charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Equity Curve")
    
    if not trades_df.empty:
        try:
            # Create equity curve
            trades_sorted = trades_df.copy()
            trades_sorted['exit_date'] = pd.to_datetime(trades_sorted['exit_date'])
            trades_sorted = trades_sorted.sort_values('exit_date')
            trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
            trades_sorted['portfolio_value'] = initial_value + trades_sorted['cumulative_profit']
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot equity curve
            ax.plot(trades_sorted['exit_date'], trades_sorted['portfolio_value'], 
                   linewidth=2, color='#1f77b4', label='Portfolio Value')
            
            # Add horizontal line for initial value
            ax.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, 
                      label=f'Initial Value (${initial_value:,})')
            
            # Fill areas
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['portfolio_value'], initial_value,
                           where=(trades_sorted['portfolio_value'] > initial_value), 
                           alpha=0.3, color='green', label='Profit')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['portfolio_value'], initial_value,
                           where=(trades_sorted['portfolio_value'] <= initial_value), 
                           alpha=0.3, color='red', label='Loss')
            
            # Formatting
            ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show key statistics
            final_value = trades_sorted['portfolio_value'].iloc[-1]
            total_return = final_value - initial_value
            total_return_pct = (total_return / initial_value) * 100
            
            st.caption(f"**Final Value:** ${final_value:,.0f} | **Total Return:** {format_dollars(total_return)} ({total_return_pct:.1f}%)")
            
        except Exception as e:
            st.error(f"Error creating equity curve: {e}")
    else:
        st.info("No trade history available for equity curve.")

with chart_col2:
    st.subheader("📊 M/E Ratio History")
    
    if not me_ratio_df.empty:
        try:
            # Create M/E ratio chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot M/E ratio
            ax.plot(me_ratio_df['Date'], me_ratio_df['ME_Ratio'], 
                   linewidth=2, color='#ff7f0e', label='M/E Ratio')
            
            # Add average line
            avg_me = me_ratio_df['ME_Ratio'].mean()
            ax.axhline(y=avg_me, color='red', linestyle='--', alpha=0.7, 
                      label=f'Average ({avg_me:.1f}%)')
            
            # Add target zones
            ax.axhspan(0, 50, alpha=0.1, color='green', label='Conservative (0-50%)')
            ax.axhspan(50, 100, alpha=0.1, color='yellow', label='Moderate (50-100%)')
            ax.axhspan(100, me_ratio_df['ME_Ratio'].max(), alpha=0.1, color='red', label='Aggressive (>100%)')
            
            # Formatting
            ax.set_title('Margin-to-Equity Ratio Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('M/E Ratio (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show M/E statistics
            max_me = me_ratio_df['ME_Ratio'].max()
            min_me = me_ratio_df['ME_Ratio'].min()
            
            st.caption(f"**Average M/E:** {avg_me:.1f}% | **Max:** {max_me:.1f}% | **Min:** {min_me:.1f}%")
            
        except Exception as e:
            st.error(f"Error creating M/E ratio chart: {e}")
    else:
        st.info("No M/E ratio history available. Run calculate_daily_me_ratio.py to generate.")

# --- DETAILED TRADE HISTORY ---
st.markdown("## Complete Trade History")

trades_formatted = get_trades_history_formatted()
if not trades_formatted.empty:
    # Show summary statistics first
    col1, col2, col3, col4 = st.columns(4)
    
    total_trades = len(trades_formatted)
    if not trades_df.empty:
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_profit = trades_df['profit'].sum()
        avg_profit = trades_df['profit'].mean()
    else:
        win_rate = 0
        total_profit = 0
        avg_profit = 0
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total Profit", format_dollars(total_profit))
    with col4:
        st.metric("Avg Profit/Trade", format_dollars(avg_profit))
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        # Trade type filter
        trade_types = ['All'] + list(trades_formatted['Type'].unique()) if 'Type' in trades_formatted.columns else ['All']
        selected_type = st.selectbox("Filter by Trade Type:", trade_types)
    
    with col2:
        # Number of trades to show
        num_trades = st.selectbox("Number of trades to display:", [20, 50, 100, "All"], index=0)
    
    # Apply filters
    filtered_trades = trades_formatted.copy()
    if selected_type != 'All':
        filtered_trades = filtered_trades[filtered_trades['Type'] == selected_type]
    
    if num_trades != "All":
        filtered_trades = filtered_trades.head(num_trades)
    
    # Display the trades table
    st.dataframe(filtered_trades, use_container_width=True, hide_index=True)
    
    # Download option
    csv = filtered_trades.to_csv(index=False)
    st.download_button(
        label="📥 Download Trade History as CSV",
        data=csv,
        file_name=f"ngs_trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
else:
    st.info("No trade history available.")

# --- FOOTER ---
st.markdown("---")
st.caption("nGS Historical Performance Analysis - For questions about strategy performance, contact the trading desk.")
