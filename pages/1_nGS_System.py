import streamlit as st
import pandas as pd
import datetime
from data_manager import (
    get_portfolio_metrics,
    get_strategy_performance,
    get_portfolio_performance_stats,
    get_trades_history
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="nGS System - Historical Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PAGE HEADER ---
st.title("nGulfStream Swing Trader")
st.caption("Historical Performance Track Record")

# --- VARIABLE ACCOUNT SIZE ---
st.markdown("## Historical Performance")
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- PORTFOLIO METRICS - ALL ON ONE ROW ---
metrics = get_portfolio_metrics(initial_portfolio_value=initial_value, is_historical=True)

# All metrics on one row as requested
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label="Total Portfolio Value", value=metrics['total_value'])
with col2:
    st.metric(label="Total Return", value=metrics['total_return_pct'])
with col3:
    st.metric(label="M/E Ratio (Avg)", value=metrics['historical_me_ratio'])
with col4:
    st.metric(label="MTD Return", value=metrics['mtd_return'])
with col5:
    st.metric(label="YTD Return", value=metrics['ytd_return'])

# --- STRATEGY PERFORMANCE TABLE ---
st.subheader("Strategy Performance")
strategy_df = get_strategy_performance(initial_portfolio_value=initial_value)
if not strategy_df.empty:
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
else:
    st.info("No strategy performance data available.")

# --- PERFORMANCE STATS AND EQUITY CURVE ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Portfolio Performance Statistics")
    perf_stats_df = get_portfolio_performance_stats()
    if not perf_stats_df.empty:
        st.dataframe(perf_stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("No performance statistics available.")

with col2:
    st.subheader("Equity Curve")
    # Create equity curve from trade history
    trades_df = get_trades_history()
    if not trades_df.empty:
        try:
            # Sort trades by exit date and calculate cumulative profit
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_sorted = trades_df.sort_values('exit_date')
            trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
            
            # Create the equity curve chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], linewidth=2, color='#1f77b4')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                           where=(trades_sorted['cumulative_profit'] > 0), alpha=0.3, color='green')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                           where=(trades_sorted['cumulative_profit'] <= 0), alpha=0.3, color='red')
            ax.set_title('Cumulative Profit Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Profit ($)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Format y-axis to show dollars without cents
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating equity curve: {e}")
    else:
        st.info("No trade history for equity curve.")

# --- TRADE HISTORY TABLE ---
st.subheader("Recent Trades")
if not trades_df.empty:
    # Show last 20 trades
    recent_trades = trades_df.sort_values('exit_date', ascending=False).head(20)
    
    # Format for display
    display_trades = recent_trades.copy()
    display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f"${x:.2f}")
    display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f"${x:.2f}")
    display_trades['profit'] = display_trades['profit'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_trades[['symbol', 'type', 'entry_date', 'exit_date', 
                                'entry_price', 'exit_price', 'shares', 'profit', 'exit_reason']], 
                use_container_width=True, hide_index=True)
else:
    st.info("No trade history available.")

st.markdown("---")
st.caption("nGulfStream Swing Trader - Historical Performance Dashboard")
