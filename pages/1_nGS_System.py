import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import (
    get_portfolio_metrics,
    get_strategy_performance,
    get_portfolio_performance_stats,
    get_positions,
    get_long_positions_formatted,
    get_short_positions_formatted,
    get_signals,
    get_system_status,
    get_trades_history
)

# Import the real portfolio calculator
try:
    from portfolio_calculator import calculate_real_portfolio_metrics, get_enhanced_strategy_performance
    USE_REAL_METRICS = True
except ImportError:
    USE_REAL_METRICS = False

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

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Trading Systems")
    if st.button("← Back to Main Dashboard", use_container_width=True, key="back_to_main"):
        st.switch_page("app.py")
    
    st.markdown("---")
    st.caption("Historical Performance")
    st.caption(f"{datetime.datetime.now().strftime('%m/%d/%Y %H:%M')}")

# --- PAGE HEADER ---
st.title("📊 nGulfStream Swing Trader - Historical Performance")
st.caption("Detailed Performance Analytics & Trade History")

# --- VARIABLE ACCOUNT SIZE ---
st.markdown("## Portfolio Performance Analysis")
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d"
)

# --- GET PORTFOLIO METRICS WITH ERROR HANDLING ---
try:
    if USE_REAL_METRICS:
        metrics = calculate_real_portfolio_metrics(initial_portfolio_value=initial_value)
    else:
        metrics = get_portfolio_metrics(initial_portfolio_value=initial_value)
except Exception as e:
    st.error(f"Error getting portfolio metrics: {e}")
    metrics = {
        'total_value': f"${initial_value:,.0f}",
        'total_return_pct': "+0.0%",
        'daily_pnl': "$0.00",
        'me_ratio': "0.00",
        'net_exposure': "$0",
        'mtd_return': "+0.0%",
        'mtd_delta': "+0.0%",
        'ytd_return': "+0.0%",
        'ytd_delta': "+0.0%"
    }

# Ensure all required metrics exist with safe defaults
safe_metrics = {
    'total_value': f"${initial_value:,.0f}",
    'total_return_pct': "+0.0%",
    'daily_pnl': "$0.00",
    'me_ratio': "0.00",
    'net_exposure': "$0",
    'mtd_return': "+0.0%",
    'mtd_delta': "+0.0%",
    'ytd_return': "+0.0%",
    'ytd_delta': "+0.0%"
}

# Update with actual metrics if available
for key, default_value in safe_metrics.items():
    if key not in metrics:
        metrics[key] = default_value

# --- DETAILED PORTFOLIO METRICS ---
st.subheader("📈 Detailed Portfolio Metrics")

# Show debug info if using real metrics
if USE_REAL_METRICS and metrics.get('total_trades', 0) > 0:
    st.success(f"✅ Real portfolio metrics calculated from {metrics['total_trades']} trades")
    st.info(f"💰 Total profit: ${metrics.get('total_profit_raw', 0):,.2f} | Winners: {metrics.get('winning_trades', 0)} | Losers: {metrics.get('losing_trades', 0)}")

# Portfolio Overview Metrics - Consolidated Layout
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_value_clean = str(metrics['total_value']).replace('.00', '').replace(',', '')
    st.metric(label="Total Portfolio Value", value=total_value_clean, delta=metrics['total_return_pct'])
with col2:
    st.metric(label="Daily P&L", value=metrics['daily_pnl'])
with col3:
    st.metric(label="M/E Ratio", value=metrics['me_ratio'])
with col4:
    st.metric(label="Net Exposure", value=metrics['net_exposure'])
with col5:
    st.metric(label="MTD Return", value=metrics['mtd_return'], delta=metrics['mtd_delta'])

# Second row
col6, col7 = st.columns([1, 1])
with col6:
    st.metric(label="YTD Return", value=metrics['ytd_return'], delta=metrics['ytd_delta'])
with col7:
    if st.button("🔄 Refresh Historical Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# --- STRATEGY PERFORMANCE TABLE ---
st.markdown("---")
st.subheader("🎯 Strategy Performance")

try:
    if USE_REAL_METRICS:
        strategy_df = get_enhanced_strategy_performance(initial_portfolio_value=initial_value)
    else:
        strategy_df = get_strategy_performance(initial_portfolio_value=initial_value)

    if not strategy_df.empty:
        st.dataframe(strategy_df, use_container_width=True, hide_index=True)
    else:
        st.info("No strategy performance data available.")
except Exception as e:
    st.error(f"Error loading strategy performance: {e}")

# --- PERFORMANCE STATISTICS ---
st.markdown("---")
st.subheader("📊 Performance Statistics")

col1, col2 = st.columns([1, 2])

with col1:
    try:
        perf_stats_df = get_portfolio_performance_stats()
        if not perf_stats_df.empty:
            st.dataframe(perf_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No performance statistics available.")
    except Exception as e:
        st.error(f"Error loading performance stats: {e}")

with col2:
    st.subheader("📈 Equity Curve")
    try:
        # Create equity curve from trade history
        trades_df = get_trades_history()
        if not trades_df.empty:
            # Sort trades by exit date and calculate cumulative profit
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            trades_sorted = trades_df.sort_values('exit_date')
            trades_sorted['cumulative_profit'] = trades_sorted['profit'].cumsum()
            
            # Create the equity curve chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], linewidth=2, color='#1f77b4')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                           where=(trades_sorted['cumulative_profit'] > 0), alpha=0.3, color='green')
            ax.fill_between(trades_sorted['exit_date'], trades_sorted['cumulative_profit'], 
                           where=(trades_sorted['cumulative_profit'] <= 0), alpha=0.3, color='red')
            ax.set_title('Cumulative Profit Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Profit ($)')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No trade history available for equity curve.")
    except Exception as e:
        st.error(f"Error creating equity curve: {e}")

# --- TRADE HISTORY ---
st.markdown("---")
st.subheader("📋 Complete Trade History")

try:
    trades_df = get_trades_history()
    if not trades_df.empty:
        # Add some basic stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades_df))
        with col2:
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            st.metric("Winning Trades", winning_trades)
        with col3:
            win_rate = (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        with col4:
            total_profit = trades_df['profit'].sum()
            st.metric("Total Profit", f"${total_profit:,.2f}")
        
        # Show trade history table
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
        
        # Download option
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade History CSV",
            data=csv,
            file_name=f"trade_history_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No trade history available.")
except Exception as e:
    st.error(f"Error loading trade history: {e}")

# --- SIGNALS HISTORY ---
st.markdown("---")
st.subheader("🎯 Signal History")

try:
    signals_df = get_signals()
    if not signals_df.empty:
        # Show recent signals
        st.dataframe(signals_df.head(50), use_container_width=True, hide_index=True)
        
        if len(signals_df) > 50:
            st.caption(f"Showing latest 50 signals out of {len(signals_df)} total")
    else:
        st.info("No signal history available.")
except Exception as e:
    st.error(f"Error loading signals: {e}")

# --- SYSTEM STATUS ---
st.markdown("---")
st.subheader("⚙️ System Status")

try:
    system_status = get_system_status()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ System Online")
        st.info(f"Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    with col2:
        if USE_REAL_METRICS:
            st.success("✅ Real Metrics Active")
        else:
            st.warning("⚠️ Using Placeholder Metrics")
    
    with col3:
        st.info("📊 Data Sources Connected")
        
except Exception as e:
    st.error(f"Error getting system status: {e}")

st.markdown("---")
st.caption("nGulfStream Swing Trader - Historical Performance Analytics")
