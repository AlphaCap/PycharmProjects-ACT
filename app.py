import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add the current directory to the path so we can import data_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import data_manager as dm
except ImportError as e:
    st.error(f"Could not import data_manager: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="nGS Trading Dashboard", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .status-green {
        color: #27ae60;
        font-weight: bold;
    }
    .status-red {
        color: #e74c3c;
        font-weight: bold;
    }
    .status-yellow {
        color: #f39c12;
        font-weight: bold;
    }
    .consolidated-metrics {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .consolidated-metrics .metric-label {
        color: white;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
        opacity: 0.9;
    }
    .consolidated-metrics .metric-value {
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("## 🚀 nGS Dashboard")
    st.markdown("---")
    
    if st.button("📊 Performance Analytics", use_container_width=True):
        st.switch_page("pages/1_nGS_System.py")
    
    st.markdown("---")
    st.markdown("### System Status")
    
    try:
        system_status = dm.get_system_status()
        if not system_status.empty:
            latest_status = system_status.iloc[0]
            st.success(f"✅ {latest_status['system']}: {latest_status['message']}")
            st.caption(f"Last updated: {latest_status['timestamp']}")
        else:
            st.info("No system status available")
    except Exception as e:
        st.warning(f"Status check failed: {e}")
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    # Get S&P 500 symbol count
    try:
        symbols = dm.get_sp500_symbols()
        symbol_count = len(symbols)
        if symbol_count >= 490:
            st.success(f"📈 {symbol_count} S&P 500 symbols loaded")
        else:
            st.warning(f"⚠️ Only {symbol_count} symbols loaded")
    except:
        st.error("❌ Symbol loading failed")
    
    # Data retention info
    st.info(f"📅 Data retention: {dm.RETENTION_DAYS} days")

# --- HEADER ---
st.markdown('<h1 class="main-header">🚀 nGS Trading System Dashboard</h1>', unsafe_allow_html=True)

# --- LOAD DATA ---
try:
    # Initialize data manager
    dm.initialize()
    
    # Get portfolio metrics (current M/E ratio for main page)
    portfolio_metrics = dm.get_portfolio_metrics(initial_portfolio_value=100000, is_historical=False)
    
except Exception as e:
    st.error(f"Error loading portfolio data: {e}")
    st.stop()

# --- CONSOLIDATED PERFORMANCE METRICS (SINGLE ROW) ---
st.markdown('<div class="section-header">Portfolio Performance</div>', unsafe_allow_html=True)

# Single row with 5 key metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="💰 Total Portfolio Value",
        value=portfolio_metrics['total_value'],
        help="Current total portfolio value including cash and positions"
    )

with col2:
    st.metric(
        label="📈 Total Return",
        value=portfolio_metrics['total_return_pct'],
        help="Total return since system inception"
    )

with col3:
    st.metric(
        label="📅 MTD Return", 
        value=portfolio_metrics['mtd_return'],
        delta=portfolio_metrics['mtd_delta'],
        help="Month-to-date performance"
    )

with col4:
    st.metric(
        label="⚡ Daily P&L",
        value=portfolio_metrics['daily_pnl'],
        help="Unrealized profit/loss from current positions"
    )

with col5:
    st.metric(
        label="⚖️ M/E Ratio",
        value=portfolio_metrics['me_ratio'],
        help="Market Exposure to Equity ratio - risk indicator"
    )

# --- CURRENT POSITIONS ---
st.markdown('<div class="section-header">Current Positions</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Long Positions")
    long_positions = dm.get_long_positions_formatted()
    
    if not long_positions.empty:
        st.dataframe(
            long_positions,
            use_container_width=True,
            hide_index=True
        )
        
        # Long positions summary
        total_long_value = len(long_positions)
        long_pnl = long_positions['P&L'].str.replace('$', '').str.replace(',', '').astype(float).sum()
        st.caption(f"Total: {total_long_value} positions | Combined P&L: ${long_pnl:,.0f}")
    else:
        st.info("No long positions currently held")

with col2:
    st.subheader("🔴 Short Positions")
    short_positions = dm.get_short_positions_formatted()
    
    if not short_positions.empty:
        st.dataframe(
            short_positions,
            use_container_width=True,
            hide_index=True
        )
        
        # Short positions summary
        total_short_value = len(short_positions)
        short_pnl = short_positions['P&L'].str.replace('$', '').str.replace(',', '').astype(float).sum()
        st.caption(f"Total: {total_short_value} positions | Combined P&L: ${short_pnl:,.0f}")
    else:
        st.info("No short positions currently held")

# --- EXPOSURE SUMMARY ---
total_positions = len(long_positions) + len(short_positions)
if total_positions > 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Positions", total_positions)
    with col2:
        st.metric("🟢 Long Exposure", portfolio_metrics['long_exposure'])
    with col3:
        st.metric("🔴 Short Exposure", portfolio_metrics['short_exposure'])

# --- TODAY'S TRADES ---
st.markdown('<div class="section-header">Today\'s Trades</div>', unsafe_allow_html=True)

try:
    # Get today's trades (entries and exits)
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check for today's completed trades (exits)
    recent_trades = dm.get_trades_history_formatted()
    today_trades = pd.DataFrame()
    
    if not recent_trades.empty:
        recent_trades['Date'] = pd.to_datetime(recent_trades['Date'])
        today_trades = recent_trades[recent_trades['Date'].dt.strftime('%Y-%m-%d') == today].copy()
        today_trades['Date'] = today_trades['Date'].dt.strftime('%Y-%m-%d')
    
    # Also check for new signals/entries from signals file
    signals_df = dm.get_signals()
    today_signals = pd.DataFrame()
    
    if not signals_df.empty and 'date' in signals_df.columns:
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        today_signals = signals_df[signals_df['date'].dt.strftime('%Y-%m-%d') == today].copy()
    
    # Combine today's activity
    if not today_trades.empty or not today_signals.empty:
        if not today_trades.empty:
            st.subheader("🔄 Today's Completed Trades")
            st.dataframe(
                today_trades,
                use_container_width=True,
                hide_index=True
            )
            
            # Today's trades summary
            today_pnl = today_trades['P&L'].str.replace('
, '').str.replace(',', '').astype(float).sum()
            trade_count = len(today_trades)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Today's Trades", trade_count)
            with col2:
                st.metric("💰 Today's P&L", f"${today_pnl:,.0f}")
        
        if not today_signals.empty:
            st.subheader("📡 Today's New Signals")
            st.dataframe(
                today_signals,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("No trading activity today yet")
        
except Exception as e:
    st.warning(f"Could not load today's trades: {e}")

# --- FOOTER ---
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🚀 nGS Trading System**")
    st.caption("Neural Grid Strategy Dashboard")

with col2:
    st.markdown("**📊 Live Data**")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.markdown("**📅 Data Retention**")
    st.caption(f"{dm.RETENTION_DAYS} days (6 months)")
