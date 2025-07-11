import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data manager functions with error handling
try:
    from data_manager import (
        get_portfolio_metrics,
        get_strategy_performance,
        get_portfolio_performance_stats,
        get_current_positions,
        get_recent_signals,
        get_trades_history
    )
except ImportError as e:
    st.error(f"Error importing data manager functions: {e}")
    st.stop()

# Import the real portfolio calculator
try:
    from portfolio_calculator import calculate_real_portfolio_metrics, calculate_portfolio_over_time
    USE_REAL_METRICS = True
except ImportError:
    st.warning("Portfolio calculator not found. Using sample data.")
    USE_REAL_METRICS = False

# Page configuration
st.set_page_config(
    page_title="nGS Trading System", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Trading Systems")
    if st.button("← Back to Main Dashboard", use_container_width=True, key="ngs_back_to_main"):
        st.switch_page("app.py")
    
    st.markdown("---")
    
    # System status
    st.subheader("🔄 System Status")
    st.success("✅ nGS System Active")
    st.info("📊 Real-time Analysis")
    st.info("🔔 Alerts Enabled")

# --- MAIN CONTENT ---
st.markdown('<div class="main-header">🚀 nGS Trading System Dashboard</div>', unsafe_allow_html=True)

# --- VARIABLE ACCOUNT SIZE ---
st.markdown("## Portfolio Performance Analysis")
initial_value = st.number_input(
    "Set initial portfolio/account size:",
    min_value=1000,
    value=100000,
    step=1000,
    format="%d",
    help="Enter your starting portfolio value for performance calculations",
    key="ngs_portfolio_initial_value"
)

# --- REAL-TIME METRICS ---
st.markdown("## 📊 Current Performance Metrics")

# Get portfolio metrics
try:
    if USE_REAL_METRICS:
        portfolio_metrics = calculate_real_portfolio_metrics(initial_value)
    else:
        portfolio_metrics = get_portfolio_metrics()
        
    strategy_performance = get_strategy_performance()
except Exception as e:
    st.error(f"Error loading portfolio metrics: {e}")
    portfolio_metrics = {'total_profit': 0, 'total_trades': 0, 'win_rate': 0, 'avg_profit_per_trade': 0}
    strategy_performance = {'sharpe_ratio': 0, 'max_drawdown': 0, 'total_return': 0, 'volatility': 0}

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    profit_color = "success-metric" if portfolio_metrics.get('total_profit', 0) > 0 else "danger-metric"
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="💰 Total Profit/Loss",
        value=f"${portfolio_metrics.get('total_profit', 0):,.2f}",
        delta=f"{(portfolio_metrics.get('total_profit', 0)/initial_value)*100:.2f}%" if initial_value > 0 else "0%"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="📈 Total Trades",
        value=f"{portfolio_metrics.get('total_trades', 0):,}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    win_rate = portfolio_metrics.get('win_rate', 0)
    win_color = "success-metric" if win_rate > 60 else "warning-metric" if win_rate > 40 else "danger-metric"
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="🎯 Win Rate",
        value=f"{win_rate:.1f}%"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    sharpe = strategy_performance.get('sharpe_ratio', 0)
    sharpe_color = "success-metric" if sharpe > 1 else "warning-metric" if sharpe > 0.5 else "danger-metric"
    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        label="📊 Sharpe Ratio",
        value=f"{sharpe:.2f}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- PORTFOLIO PERFORMANCE CHART ---
st.markdown("## 📈 Portfolio Performance Over Time")

try:
    if USE_REAL_METRICS:
        performance_data = calculate_portfolio_over_time(initial_value)
    else:
        performance_data = get_portfolio_performance_stats()
    
    if performance_data and 'cumulative_returns' in performance_data:
        # Create performance chart
        dates = pd.date_range(start='2023-01-01', periods=len(performance_data['cumulative_returns']), freq='D')
        portfolio_values = [initial_value * ret for ret in performance_data['cumulative_returns']]
        
        fig = go.Figure()
        
        # Portfolio performance line
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Benchmark comparison if available
        if 'benchmark_returns' in performance_data and performance_data['benchmark_returns']:
            benchmark_values = [initial_value * ret for ret in performance_data['benchmark_returns']]
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_values,
                mode='lines',
                name='Benchmark (S&P 500)',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> $%{y:,.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Performance data will appear here once trading begins.")
        
except Exception as e:
    st.error(f"Error displaying performance chart: {e}")

# --- CURRENT POSITIONS ---
st.markdown("## 💼 Current Positions")

try:
    positions_df = get_current_positions()
    
    if not positions_df.empty:
        # Format the positions dataframe for display
        display_positions = positions_df.copy()
        if 'unrealized_pnl' in display_positions.columns:
            display_positions['unrealized_pnl'] = display_positions['unrealized_pnl'].apply(lambda x: f"${x:,.2f}")
        if 'entry_price' in display_positions.columns:
            display_positions['entry_price'] = display_positions['entry_price'].apply(lambda x: f"${x:.2f}")
        if 'current_price' in display_positions.columns:
            display_positions['current_price'] = display_positions['current_price'].apply(lambda x: f"${x:.2f}")
            
        st.dataframe(display_positions, use_container_width=True, key="ngs_positions_table")
        
        # Position summary
        if 'unrealized_pnl' in positions_df.columns:
            total_unrealized = positions_df['unrealized_pnl'].sum()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Unrealized P&L", f"${total_unrealized:,.2f}")
            with col2:
                open_positions = len(positions_df)
                st.metric("Open Positions", open_positions)
    else:
        st.info("💼 No current positions. Ready for new opportunities!")
        
except Exception as e:
    st.error(f"Error loading current positions: {e}")

# --- RECENT SIGNALS ---
st.markdown("## 🔔 Recent Trading Signals")

try:
    signals_df = get_recent_signals()
    
    if not signals_df.empty:
        # Display recent signals
        for idx, signal in signals_df.head(5).iterrows():
            signal_type = signal.get('signal', 'Unknown')
            symbol = signal.get('symbol', 'Unknown')
            confidence = signal.get('confidence', 0)
            timestamp = signal.get('timestamp', 'Unknown')
            
            signal_color = "🟢" if signal_type.upper() == "BUY" else "🔴" if signal_type.upper() == "SELL" else "🟡"
            
            st.markdown(f"""
            **{signal_color} {signal_type.upper()} Signal - {symbol}**  
            Confidence: {confidence:.1%} | Time: {timestamp}
            """)
    else:
        st.info("🔔 No recent signals. System is monitoring for opportunities.")
        
except Exception as e:
    st.error(f"Error loading recent signals: {e}")

# --- RECENT TRADES ---
st.markdown("## 📋 Recent Trades")

try:
    trades_df = get_trades_history()
    
    if not trades_df.empty:
        # Show last 10 trades
        recent_trades = trades_df.tail(10).copy()
        
        # Format for display
        if 'profit' in recent_trades.columns:
            recent_trades['profit'] = recent_trades['profit'].apply(lambda x: f"${x:,.2f}")
        if 'entry_price' in recent_trades.columns:
            recent_trades['entry_price'] = recent_trades['entry_price'].apply(lambda x: f"${x:.2f}")
        if 'exit_price' in recent_trades.columns:
            recent_trades['exit_price'] = recent_trades['exit_price'].apply(lambda x: f"${x:.2f}")
            
        st.dataframe(recent_trades, use_container_width=True, key="ngs_recent_trades_table")
        
        # Trade summary
        if 'profit' in trades_df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_profit = trades_df['profit'].mean()
                st.metric("Average Profit per Trade", f"${avg_profit:,.2f}")
            with col2:
                winning_trades = len(trades_df[trades_df['profit'] > 0])
                total_trades = len(trades_df)
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                total_profit = trades_df['profit'].sum()
                st.metric("Total Profit", f"${total_profit:,.2f}")
    else:
        st.info("📋 No trade history available yet.")
        
except Exception as e:
    st.error(f"Error loading trade history: {e}")

# --- SYSTEM CONTROLS ---
st.markdown("## ⚙️ System Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True, key="ngs_refresh_data"):
        st.rerun()

with col2:
    trading_enabled = st.checkbox("🤖 Auto Trading", value=True, key="ngs_auto_trading")
    if trading_enabled:
        st.success("✅ Auto trading is enabled")
    else:
        st.warning("⚠️ Auto trading is disabled")

with col3:
    risk_level = st.selectbox(
        "📊 Risk Level",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        key="ngs_risk_level"
    )

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🚀 nGS Trading System | Real-time Market Analysis & Automated Trading</p>
    <p>⚡ Powered by Advanced Algorithms | 📊 Live Data Integration</p>
</div>
<<<<<<< HEAD
""", unsafe_allow_html=True)
=======
""", unsafe_allow_html=True)
>>>>>>> a389841400885a00253bbc8a3262df690d8e552d
