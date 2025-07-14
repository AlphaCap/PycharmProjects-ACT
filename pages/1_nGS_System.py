import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import data_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import data_manager as dm
except ImportError as e:
    st.error(f"Could not import data_manager: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="nGS Performance Analytics", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("---")
    
    if st.button("🏠 HOME", use_container_width=True):
        st.switch_page("app.py")

# --- HEADER ---
st.markdown('<h1 class="main-header">📈 nGS System Performance Analytics</h1>', unsafe_allow_html=True)

# --- LOAD DATA ---
try:
    # Initialize data manager
    dm.initialize()
    
    # Get portfolio metrics with historical M/E ratio
    portfolio_metrics = dm.get_portfolio_metrics(initial_portfolio_value=100000, is_historical=True)
    trades_df = dm.get_trades_history()
    
    if trades_df.empty:
        st.warning("No trade history available yet. Analytics will be displayed once trades are executed.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- PERFORMANCE OVERVIEW ---
st.markdown('<div class="section-header">Performance Overview (Last 6 Months)</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Return",
        value=portfolio_metrics['total_return_pct'],
        help="Total return since system inception"
    )

with col2:
    st.metric(
        label="YTD Return", 
        value=portfolio_metrics['ytd_return'],
        delta=portfolio_metrics['ytd_delta'],
        help="Year-to-date performance"
    )

with col3:
    st.metric(
        label="MTD Return",
        value=portfolio_metrics['mtd_return'], 
        delta=portfolio_metrics['mtd_delta'],
        help="Month-to-date performance"
    )

with col4:
    st.metric(
        label="Historical M/E Ratio",
        value=portfolio_metrics['historical_me_ratio'],
        help="Historical average Market/Equity ratio"
    )

# --- PERFORMANCE CHARTS ---
st.markdown('<div class="section-header">Performance Analytics</div>', unsafe_allow_html=True)

# Create tabs for different chart views
tab1, tab2, tab3 = st.tabs(["📊 Equity Curve", "📈 M/E Ratio History", "🎯 Trade Analysis"])

with tab1:
    # Equity Curve Chart
    if not trades_df.empty:
        # Calculate cumulative returns
        trades_df_sorted = trades_df.sort_values('exit_date')
        trades_df_sorted['cumulative_pnl'] = trades_df_sorted['profit'].cumsum()
        trades_df_sorted['portfolio_value'] = 100000 + trades_df_sorted['cumulative_pnl']
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=trades_df_sorted['exit_date'],
            y=trades_df_sorted['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.0f}<extra></extra>'
        ))
        
        fig_equity.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.info("Equity curve will be displayed once trades are executed.")

with tab2:
    # M/E Ratio History Chart
    with st.spinner("Calculating historical M/E ratios..."):
        me_history = dm.get_me_ratio_history()
    
    if not me_history.empty:
        fig_me = go.Figure()
        fig_me.add_trace(go.Scatter(
            x=me_history['Date'],
            y=me_history['ME_Ratio'],
            mode='lines+markers',
            name='M/E Ratio',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4),
            hovertemplate='<b>Date:</b> %{x}<br><b>M/E Ratio:</b> %{y:.1f}%<extra></extra>'
        ))
        
        # Add horizontal line for average
        if len(me_history) > 1:
            avg_me = me_history['ME_Ratio'].mean()
            fig_me.add_hline(
                y=avg_me, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Average: {avg_me:.1f}%"
            )
        
        fig_me.update_layout(
            title="Historical M/E Ratio (Risk Exposure Over Time)",
            xaxis_title="Date",
            yaxis_title="M/E Ratio (%)",
            hovermode='x unified',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_me, use_container_width=True)
        
        # Show summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Days Tracked", len(me_history))
        with col2:
            st.metric("Average M/E", f"{me_history['ME_Ratio'].mean():.1f}%")
        with col3:
            st.metric("Max M/E", f"{me_history['ME_Ratio'].max():.1f}%")
        with col4:
            st.metric("Current M/E", portfolio_metrics['me_ratio'])
    else:
        st.info("M/E ratio history will be calculated once sufficient trade data is available.")

with tab3:
    # Trade Analysis Charts
    if not trades_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit/Loss Distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=trades_df['profit'],
                nbinsx=20,
                name='Trade P&L',
                marker_color='#1f77b4',
                opacity=0.7
            ))
            
            fig_dist.update_layout(
                title="Trade P&L Distribution",
                xaxis_title="Profit/Loss ($)",
                yaxis_title="Number of Trades",
                height=350
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Monthly Performance
            trades_df['month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
            monthly_pnl = trades_df.groupby('month')['profit'].sum().reset_index()
            monthly_pnl['month_str'] = monthly_pnl['month'].astype(str)
            
            fig_monthly = go.Figure()
            colors = ['green' if x >= 0 else 'red' for x in monthly_pnl['profit']]
            
            fig_monthly.add_trace(go.Bar(
                x=monthly_pnl['month_str'],
                y=monthly_pnl['profit'],
                marker_color=colors,
                name='Monthly P&L'
            ))
            
            fig_monthly.update_layout(
                title="Monthly Performance",
                xaxis_title="Month",
                yaxis_title="P&L ($)",
                height=350
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.info("Trade analysis will be displayed once trades are executed.")

# --- DETAILED STATISTICS ---
st.markdown('<div class="section-header">Detailed Performance Statistics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Performance Stats Table
    performance_stats = dm.get_portfolio_performance_stats()
    if not performance_stats.empty:
        st.subheader("📊 Key Metrics")
        st.dataframe(
            performance_stats,
            use_container_width=True,
            hide_index=True
        )

with col2:
    # Strategy Performance
    strategy_performance = dm.get_strategy_performance()
    if not strategy_performance.empty:
        st.subheader("🎯 Strategy Breakdown")
        st.dataframe(
            strategy_performance,
            use_container_width=True,
            hide_index=True
        )

# --- TRADE HISTORY TABLE ---
st.markdown('<div class="section-header">Trade History (Last 6 Months)</div>', unsafe_allow_html=True)

# Get formatted trade history (already filtered to 6 months)
recent_trades = dm.get_trades_history_formatted()

if not recent_trades.empty:
    # Show all trades within 6-month period
    st.dataframe(
        recent_trades,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(recent_trades))
    
    with col2:
        winning_trades = len(recent_trades[recent_trades['P&L'].str.replace('$', '').str.replace(',', '').astype(float) > 0])
        win_rate = winning_trades / len(recent_trades) * 100 if len(recent_trades) > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        avg_days = recent_trades['Days'].mean() if 'Days' in recent_trades.columns else 0
        st.metric("Avg Hold Time", f"{avg_days:.1f} days")
    
    with col4:
        # Calculate average P&L
        pnl_values = recent_trades['P&L'].str.replace('$', '').str.replace(',', '').astype(float)
        avg_pnl = pnl_values.mean()
        st.metric("Avg P&L", f"${avg_pnl:.0f}")

else:
    st.info("No trade history available yet.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>nGS Trading System - Performance Analytics Dashboard</p>", 
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.8rem;'>* Data retention: 6 months (180 days)</p>", 
    unsafe_allow_html=True
)
