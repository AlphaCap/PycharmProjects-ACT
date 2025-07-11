import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ACT Trading Dashboard",
    page_icon="📈",
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
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .system-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    .system-card:hover {
        transform: translateY(-5px);
    }
    .status-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .nav-button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 0.5rem;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Import data manager functions with path fixes
try:
    from data_manager import (
        get_portfolio_metrics,
        get_strategy_performance,
        get_portfolio_performance_stats,
        get_current_positions,  # ✅ Correct function name
        get_recent_signals,     # ✅ Correct function name
        get_trades_history      # ✅ Correct function name
    )
    DATA_MANAGER_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Data manager import error: {e}")
    st.info("🔧 Please ensure data_manager.py is properly configured.")
    DATA_MANAGER_AVAILABLE = False

# --- MAIN HEADER ---
st.markdown('<div class="main-header">🚀 ACT Trading Systems Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Algorithmic Trading Platform</div>', unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🎯 Navigation")
    
    st.markdown("### 📊 Trading Systems")
    
    # System navigation buttons with unique keys
    if st.button("🔥 nGS Strategy Dashboard", use_container_width=True, key="nav_ngs_system"):
        st.switch_page("pages/1_nGS_System.py")
    
    if st.button("📈 Portfolio Analysis", use_container_width=True, key="nav_portfolio"):
        st.info("Portfolio analysis page coming soon!")
    
    if st.button("⚙️ System Settings", use_container_width=True, key="nav_settings"):
        st.info("Settings page coming soon!")
    
    st.markdown("---")
    
    # Quick stats in sidebar
    st.markdown("### 📋 Quick Stats")
    
    if DATA_MANAGER_AVAILABLE:
        try:
            metrics = get_portfolio_metrics()
            st.metric("Total Trades", f"{metrics.get('total_trades', 0):,}", key="sidebar_total_trades")
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%", key="sidebar_win_rate")
            
            profit = metrics.get('total_profit', 0)
            profit_color = "🟢" if profit > 0 else "🔴" if profit < 0 else "⚪"
            st.metric("Total P&L", f"{profit_color} ${profit:,.2f}", key="sidebar_pnl")
            
        except Exception as e:
            st.error(f"Error loading sidebar metrics: {e}")
    else:
        st.info("Metrics unavailable - check data manager")

# --- MAIN DASHBOARD CONTENT ---

# System Status Overview
st.markdown("## 🔄 System Status Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="system-card status-success">
        <h3>🤖 nGS System</h3>
        <p><strong>Status:</strong> Active</p>
        <p><strong>Mode:</strong> Live Trading</p>
        <p><strong>Uptime:</strong> 99.8%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="system-card">
        <h3>📊 Data Feed</h3>
        <p><strong>Status:</strong> Connected</p>
        <p><strong>Latency:</strong> 12ms</p>
        <p><strong>Last Update:</strong> Live</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="system-card status-warning">
        <h3>⚡ Portfolio Monitor</h3>
        <p><strong>Status:</strong> Monitoring</p>
        <p><strong>Positions:</strong> Active</p>
        <p><strong>Risk Level:</strong> Moderate</p>
    </div>
    """, unsafe_allow_html=True)

# Performance Overview
st.markdown("## 📈 Performance Overview")

if DATA_MANAGER_AVAILABLE:
    try:
        # Get performance data
        portfolio_metrics = get_portfolio_metrics()
        strategy_performance = get_strategy_performance()
        
        # Display key metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            total_profit = portfolio_metrics.get('total_profit', 0)
            profit_delta = f"{(total_profit/100000)*100:.2f}%" if total_profit != 0 else "0%"
            st.metric(
                label="💰 Total Profit/Loss",
                value=f"${total_profit:,.2f}",
                delta=profit_delta,
                key="main_total_profit"
            )
        
        with metric_col2:
            total_trades = portfolio_metrics.get('total_trades', 0)
            st.metric(
                label="📊 Total Trades",
                value=f"{total_trades:,}",
                key="main_total_trades"
            )
        
        with metric_col3:
            win_rate = portfolio_metrics.get('win_rate', 0)
            st.metric(
                label="🎯 Win Rate",
                value=f"{win_rate:.1f}%",
                delta=f"{win_rate-50:.1f}% vs 50%",
                key="main_win_rate"
            )
        
        with metric_col4:
            sharpe_ratio = strategy_performance.get('sharpe_ratio', 0)
            st.metric(
                label="📈 Sharpe Ratio",
                value=f"{sharpe_ratio:.2f}",
                delta=f"{sharpe_ratio-1:.2f} vs 1.0",
                key="main_sharpe_ratio"
            )
            
        # Performance Chart
        st.markdown("### 📊 Portfolio Performance Chart")
        
        try:
            performance_stats = get_portfolio_performance_stats()
            
            if performance_stats and 'cumulative_returns' in performance_stats:
                # Create sample performance chart
                dates = pd.date_range(start='2024-01-01', periods=len(performance_stats['cumulative_returns']), freq='D')
                returns = performance_stats['cumulative_returns']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=returns,
                    mode='lines',
                    name='Portfolio Performance',
                    line=dict(color='#1f77b4', width=3),
                    fill='tonexty' if len(returns) > 1 else None,
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ))
                
                fig.update_layout(
                    title="Portfolio Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True, key="main_performance_chart")
            else:
                st.info("📊 Performance chart will be available once trading data is generated.")
                
        except Exception as e:
            st.error(f"Error creating performance chart: {e}")
            
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        st.info("📊 Using sample data for demonstration")
        
        # Show sample metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Profit/Loss", "$12,450.00", "+2.45%", key="sample_profit")
        with col2:
            st.metric("📊 Total Trades", "147", key="sample_trades")
        with col3:
            st.metric("🎯 Win Rate", "68.2%", "+18.2%", key="sample_win_rate")
        with col4:
            st.metric("📈 Sharpe Ratio", "1.34", "+0.34", key="sample_sharpe")

else:
    st.warning("⚠️ Data manager not available. Please check configuration.")
    
    # Show placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Profit/Loss", "---", key="placeholder_profit")
    with col2:
        st.metric("📊 Total Trades", "---", key="placeholder_trades")
    with col3:
        st.metric("🎯 Win Rate", "---", key="placeholder_win_rate")
    with col4:
        st.metric("📈 Sharpe Ratio", "---", key="placeholder_sharpe")

# Current Market Status
st.markdown("## 🌍 Current Market Status")

# Sample market data (replace with real data feed)
market_col1, market_col2, market_col3 = st.columns(3)

with market_col1:
    st.metric("📈 S&P 500", "4,385.24", "+0.67%", key="market_sp500")

with market_col2:
    st.metric("💹 NASDAQ", "13,567.98", "+1.23%", key="market_nasdaq")

with market_col3:
    st.metric("💵 VIX", "18.45", "-2.1%", key="market_vix")

# Recent Activity
if DATA_MANAGER_AVAILABLE:
    st.markdown("## 📋 Recent Activity")
    
    try:
        # Get recent data
        recent_signals = get_recent_signals()
        recent_trades = get_trades_history()
        current_positions = get_current_positions()
        
        activity_col1, activity_col2 = st.columns(2)
        
        with activity_col1:
            st.markdown("### 🔔 Recent Signals")
            if not recent_signals.empty:
                for idx, signal in recent_signals.head(3).iterrows():
                    signal_type = signal.get('signal', 'Unknown')
                    symbol = signal.get('symbol', 'Unknown')
                    confidence = signal.get('confidence', 0)
                    
                    signal_emoji = "🟢" if signal_type.upper() == "BUY" else "🔴" if signal_type.upper() == "SELL" else "🟡"
                    st.markdown(f"{signal_emoji} **{signal_type.upper()}** {symbol} - Confidence: {confidence:.1%}")
            else:
                st.info("No recent signals")
        
        with activity_col2:
            st.markdown("### 💼 Current Positions")
            if not current_positions.empty:
                st.dataframe(current_positions.head(5), use_container_width=True, key="main_current_positions")
            else:
                st.info("No current positions")
                
    except Exception as e:
        st.error(f"Error loading recent activity: {e}")

# Quick Actions
st.markdown("## ⚡ Quick Actions")

action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("🔄 Refresh Dashboard", use_container_width=True, key="action_refresh"):
        st.rerun()

with action_col2:
    if st.button("📊 View nGS System", use_container_width=True, key="action_view_ngs"):
        st.switch_page("pages/1_nGS_System.py")

with action_col3:
    if st.button("📈 Export Data", use_container_width=True, key="action_export"):
        st.info("Export functionality coming soon!")

with action_col4:
    if st.button("⚙️ System Settings", use_container_width=True, key="action_settings"):
        st.info("Settings panel coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.9em; padding: 2rem;'>
    <p><strong>🚀 ACT Trading Systems</strong> | Advanced Algorithmic Trading Platform</p>
    <p>📊 Real-time Analytics | 🤖 Automated Execution | ⚡ High-Performance Computing</p>
    <p><em>Disclaimer: Trading involves risk. Past performance does not guarantee future results.</em></p>
</div>
""", unsafe_allow_html=True)

