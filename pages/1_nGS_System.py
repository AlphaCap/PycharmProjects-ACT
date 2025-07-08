import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import data_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import (
    get_trades_history,
    get_portfolio_metrics,
    get_strategy_performance,
    get_portfolio_performance_stats
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="nGulfStream Swing Trader - Historical Performance",
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
    if st.button("← Back to Main Dashboard", use_container_width=True):
        st.switch_page("App.py")
    
    st.markdown("---")
    
    if st.button("nGulfStream Swing Trader", use_container_width=True, type="primary"):
        st.rerun()
    
    # Disabled placeholder buttons
    st.button("Alpha Capture AI", use_container_width=True, disabled=True, help="Coming Soon")
    st.button("gST DayTrader", use_container_width=True, disabled=True, help="Coming Soon")
    
    st.markdown("---")
    st.caption("Data last updated:")
    st.caption(f"{datetime.datetime.now().strftime('%m/%d/%Y %H:%M')}")

# --- PAGE HEADER ---
st.title("nGulfStream Swing Trader")
st.caption("Historical Trading Performance & Track Record")

# --- ACCOUNT SIZE INPUT ---
initial_value = st.number_input(
    "Portfolio/Account Size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- PORTFOLIO METRICS ---
metrics = get_portfolio_metrics(initial_portfolio_value=initial_value)

# Portfolio Overview Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_value_clean = metrics['total_value'].replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean, delta=metrics['total_return_pct'])
with col2:
    st.metric(label="Total Return", value=metrics['ytd_return'], delta=metrics['ytd_delta'])
with col3:
    st.metric(label="M/E Ratio", value=metrics['me_ratio'])
with col4:
    st.metric(label="Long Exposure", value=metrics['long_exposure'])
with col5:
    st.metric(label="Short Exposure", value=metrics['short_exposure'])

# --- HISTORICAL PERFORMANCE SECTION ---
st.markdown("## Historical Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Strategy Statistics")
    perf_stats_df = get_portfolio_performance_stats()
    if not perf_stats_df.empty:
        st.dataframe(perf_stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("No performance statistics available.")

with col2:
    st.subheader("Equity Curve")
    trades_df = get_trades_history()
    if not trades_df.empty:
        try:
            # Create equity curve
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_sorted = trades_df.sort_values('exit_date')
            trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                   linewidth=2, color='#1f77b4', marker='o', markersize=3)
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                           where=(trades_sorted['cumulative_profit'] > 0), alpha=0.3, color='green')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                           where=(trades_sorted['cumulative_profit'] <= 0), alpha=0.3, color='red')
            ax.set_title('nGS Strategy Cumulative Profit - Inception to Date', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add final value annotation
            final_profit = trades_sorted['cumulative_profit'].iloc[-1]
            ax.annotate(f'Total: ${final_profit:.2f}', 
                       xy=(trades_sorted['exit_date'].iloc[-1], final_profit),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error creating equity curve: {e}")
    else:
        st.info("No trade history available for equity curve.")

# --- STRATEGY PERFORMANCE TABLE ---
st.subheader("Strategy Breakdown")
strategy_df = get_strategy_performance(initial_portfolio_value=initial_value)
if not strategy_df.empty:
    st.dataframe(strategy_df, use_container_width=True, hide_index=True)
else:
    st.info("No strategy performance data available.")

# --- COMPLETE TRADE HISTORY ---
st.markdown("## Complete Trade History")
st.caption("All trades from strategy inception - maintained for track record purposes")

trades_df = get_trades_history()
if not trades_df.empty:
    # Sort by exit date (most recent first)
    trades_display = trades_df.sort_values('exit_date', ascending=False).copy()
    
    # Format for better display
    trades_display['entry_date'] = pd.to_datetime(trades_display['entry_date']).dt.strftime('%Y-%m-%d')
    trades_display['exit_date'] = pd.to_datetime(trades_display['exit_date']).dt.strftime('%Y-%m-%d')
    trades_display['entry_price'] = trades_display['entry_price'].apply(lambda x: f"${x:.2f}")
    trades_display['exit_price'] = trades_display['exit_price'].apply(lambda x: f"${x:.2f}")
    trades_display['profit'] = trades_display['profit'].apply(lambda x: f"${x:.2f}")
    
    # Rename columns for display
    display_columns = {
        'symbol': 'Symbol',
        'type': 'Type',
        'entry_date': 'Entry Date',
        'exit_date': 'Exit Date',
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'shares': 'Shares',
        'profit': 'Profit',
        'exit_reason': 'Exit Reason'
    }
    
    trades_display = trades_display.rename(columns=display_columns)
    
    # Color code profitable vs losing trades
    def color_profit(val):
        if val.startswith('-'):
            return 'color: red'
        elif val.startswith('$') and float(val.replace('$', '').replace(',', '')) > 0:
            return 'color: green'
        else:
            return ''
    
    # Apply styling
    styled_trades = trades_display.style.applymap(color_profit, subset=['Profit'])
    
    st.dataframe(styled_trades, use_container_width=True, hide_index=True)
    
    # Trade history summary
    st.markdown("### Trade History Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
    with col2:
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        st.metric("Winning Trades", winning_trades)
    with col3:
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        st.metric("Losing Trades", losing_trades)
    with col4:
        win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1%}")

else:
    st.info("No historical trades available.")

st.markdown("---")
st.caption("nGulfStream Swing Trader - Neural Grid Strategy for S&P 500 Long/Short Trading")