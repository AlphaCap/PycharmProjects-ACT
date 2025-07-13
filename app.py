import streamlit as st
import pandas as pd
import datetime
from data_manager import (
    get_portfolio_metrics,
    get_strategy_performance,
    get_portfolio_performance_stats,
    get_positions,
    get_long_positions_formatted,
    get_short_positions_formatted,
    get_signals,
    get_system_status,
    get_trades_history  # Added import
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Alpha Capture Technology AI",
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
    
    /* Hide only the auto-generated page navigation, not our custom sidebar */
    [data-testid="stSidebarNav"] {display: none;}
    
    .stAppViewContainer > .main .block-container {
        padding-top: 1rem;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Trading Systems")
    if st.button("nGulfStream Swing Trader", use_container_width=True):
        st.switch_page("pages/1_nGS_System.py")
    
    # Disabled placeholder buttons for future systems
    st.button("Alpha Capture AI", use_container_width=True, disabled=True, help="Coming Soon")
    st.button("gST DayTrader", use_container_width=True, disabled=True, help="Coming Soon")
    
    st.markdown("---")
    st.caption("Data last updated:")
    st.caption(f"{datetime.datetime.now().strftime('%m/%d/%Y %H:%M')}")

# --- PAGE HEADER ---
st.title("Alpha Capture Technology AI")
st.caption("S&P 500 Long/Short Position Trader - Historical Performance")

# --- VARIABLE ACCOUNT SIZE ---
st.markdown("## Current Portfolio Status")
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- PORTFOLIO METRICS ---
metrics = get_portfolio_metrics(initial_portfolio_value=initial_value)

# Portfolio Overview Metrics - Only 3 metrics for historical performance
col1, col2, col3 = st.columns(3)
with col1:
    # Remove cents from total value display
    total_value_clean = metrics['total_value'].replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean)
with col2:
    st.metric(label="Total Return", value=metrics['total_return_pct'])
with col3:
    st.metric(label="M/E Ratio", value=metrics['me_ratio'])

# MTD and YTD Returns
col4, col5 = st.columns(2)
with col4:
    st.metric(label="MTD Return", value=metrics['mtd_return'], delta=metrics['mtd_delta'])
with col5:
    st.metric(label="YTD Return", value=metrics['ytd_return'], delta=metrics['ytd_delta'])

# --- STRATEGY PERFORMANCE TABLE ---
st.subheader("Strategy Performance")
strategy_df = get_strategy_performance(initial_portfolio_value=initial_value)
if not strategy_df.empty:
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
else:
    st.info("No strategy performance data available.")

# --- POSITIONS SECTION ---
st.markdown("## Current Positions")

# Long Positions
st.subheader("📈 Long Positions")
long_positions_df = get_long_positions_formatted()
if not long_positions_df.empty:
    st.dataframe(long_positions_df, use_container_width=True, hide_index=True)
    
    # Long positions summary
    long_count = len(long_positions_df)
    long_total_value = long_positions_df['P&L'].str.replace('$', '').str.replace(',', '').astype(float).sum()
    st.caption(f"**Long Summary:** {long_count} positions, Total P&L: ${long_total_value:.2f}")
else:
    st.info("No active long positions.")

# Short Positions  
st.subheader("📉 Short Positions")
short_positions_df = get_short_positions_formatted()
if not short_positions_df.empty:
    st.dataframe(short_positions_df, use_container_width=True, hide_index=True)
    
    # Short positions summary
    short_count = len(short_positions_df)
    short_total_value = short_positions_df['P&L'].str.replace('$', '').str.replace(',', '').astype(float).sum()
    st.caption(f"**Short Summary:** {short_count} positions, Total P&L: ${short_total_value:.2f}")
else:
    st.info("No active short positions.")

# --- SIGNALS AND PERFORMANCE SECTION ---
col1, col2 = st.columns([1, 2])  # Make signals column narrower

with col1:
    st.subheader("Today's Signals")
    signals_df = get_signals()
    if not signals_df.empty:
        # Filter for today's signals if date column exists
        today = datetime.datetime.now().date()
        if 'date' in signals_df.columns:
            signals_df['date'] = pd.to_datetime(signals_df['date']).dt.date
            todays_signals = signals_df[signals_df['date'] == today]
        else:
            todays_signals = signals_df.head(10)  # Show recent signals
        
        if not todays_signals.empty:
            for _, signal in todays_signals.iterrows():
                st.markdown(
                    f"- **{signal['date']}** | **{signal['symbol']}** | **{signal['signal_type']}** | {signal['strategy']} at `{signal['price']}`"
                )
        else:
            st.info("No signals generated today.")
    else:
        st.info("No recent signals.")

with col2:
    # Split this column for performance stats and equity curve
    subcol1, subcol2 = st.columns(2)
    
    with subcol1:
        st.subheader("Portfolio Performance Stats")
        perf_stats_df = get_portfolio_performance_stats()
        if not perf_stats_df.empty:
            st.dataframe(perf_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No performance statistics available.")
    
    with subcol2:
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
                fig, ax = plt.subplots(figsize=(8, 4))
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
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Error creating equity curve: {e}")
        else:
            st.info("No trade history for equity curve.")

st.markdown("---")
st.caption("Alpha Trading Systems Dashboard - For additional support, please contact the trading desk.")
