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

# --- RECENT SIGNALS ---
st.markdown('<div class="section-header">Recent Trading Signals</div>', unsafe_allow_html=True)

try:
    signals_df = dm.get_signals()
    if not signals_df.empty:
        # Show last 10 signals
        recent_signals = signals_df.head(10)
        st.dataframe(
            recent_signals,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent trading signals available")
except Exception as e:
    st.warning(f"Could not load signals: {e}")

# --- TRADE HISTORY PREVIEW ---
st.markdown('<div class="section-header">Recent Trade History</div>', unsafe_allow_html=True)

try:
    # Get recent trades (last 10)
    recent_trades = dm.get_trades_history_formatted()
    
    if not recent_trades.empty:
        # Filter to last 6 months and show recent 10 trades
        cutoff_date = datetime.now() - timedelta(days=dm.RETENTION_DAYS)
        recent_trades['Date'] = pd.to_datetime(recent_trades['Date'])
        recent_trades = recent_trades[recent_trades['Date'] >= cutoff_date]
        recent_trades['Date'] = recent_trades['Date'].dt.strftime('%Y-%m-%d')
        
        if not recent_trades.empty:
            st.dataframe(
                recent_trades.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Trade summary
            total_trades = len(recent_trades)
            avg_days = recent_trades['Days'].mean() if 'Days' in recent_trades.columns else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Showing last 10 of {total_trades} trades (6 months)")
            with col2:
                st.caption(f"Average hold time: {avg_days:.1f} days")
        else:
            st.info("No trades in the last 6 months")
    else:
        st.info("No trade history available yet")
        
except Exception as e:
    st.warning(f"Could not load trade history: {e}")

# --- PERFORMANCE CHART PREVIEW ---
st.markdown('<div class="section-header">Portfolio Performance Chart</div>', unsafe_allow_html=True)

try:
    trades_df = dm.get_trades_history()
    
    if not trades_df.empty:
        # Filter to last 6 months
        cutoff_date = datetime.now() - timedelta(days=dm.RETENTION_DAYS)
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df = trades_df[trades_df['exit_date'] >= cutoff_date]
        
        if not trades_df.empty:
            # Calculate cumulative returns
            trades_df_sorted = trades_df.sort_values('exit_date')
            trades_df_sorted['cumulative_pnl'] = trades_df_sorted['profit'].cumsum()
            trades_df_sorted['portfolio_value'] = 100000 + trades_df_sorted['cumulative_pnl']
            
            # Create simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df_sorted['exit_date'],
                y=trades_df_sorted['portfolio_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve (Last 6 Months)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades in the last 6 months to display")
    else:
        st.info("Portfolio chart will be displayed once trades are executed")
        
except Exception as e:
    st.warning(f"Could not create performance chart: {e}")

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
